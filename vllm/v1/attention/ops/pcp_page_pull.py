# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-aware one-sided KV pull for PCP runahead.

KV memory geometry is exchanged once after stable cache registration. Per-layer
readiness is then a tiny NIXL notification. CUDA cache writes are ordered with
an event queried by the progress thread, so the model thread never synchronizes
the device merely to publish READY.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group
from vllm.platforms import current_platform
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range


@dataclass(frozen=True)
class PCPPagePlan:
    """Per-step logical segment ownership and precompiled physical-page routes."""

    segment_to_rank: tuple[int, ...]
    blocks_by_segment: tuple[tuple[int, ...], ...]
    block_size: int
    _required_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _required_source_sets_by_rank: tuple[frozenset[int], ...] = field(
        init=False, repr=False, compare=False
    )
    _consumers_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_ids_by_rank: tuple[tuple[tuple[int, ...], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_arrays_by_rank: tuple[tuple[np.ndarray, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_max_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.segment_to_rank:
            raise ValueError("page-pull plan requires at least one logical segment")
        if len(self.blocks_by_segment) != len(self.segment_to_rank):
            raise ValueError("page map must match logical segment count")
        if self.block_size <= 0:
            raise ValueError("page-pull block_size must be positive")

        world_size = self.world_size
        missing = set(range(world_size)) - set(self.segment_to_rank)
        if missing:
            raise ValueError(
                "page-pull segment map must cover every PCP rank; "
                f"missing={sorted(missing)}"
            )

        required_sources_by_rank: list[tuple[int, ...]] = []
        transfer_ids_by_rank: list[tuple[tuple[int, ...], ...]] = []
        transfer_arrays_by_rank: list[tuple[np.ndarray, ...]] = []
        transfer_max_by_rank: list[tuple[int, ...]] = []
        for rank in range(world_size):
            owned = [
                segment_idx
                for segment_idx, owner in enumerate(self.segment_to_rank)
                if owner == rank
            ]
            max_segment = owned[-1]
            source_blocks: list[list[int]] = [[] for _ in range(world_size)]
            source_order: list[int] = []
            seen_sources: set[int] = set()
            for segment_idx in range(max_segment):
                source_rank = self.segment_to_rank[segment_idx]
                if source_rank == rank:
                    continue
                if source_rank not in seen_sources:
                    seen_sources.add(source_rank)
                    source_order.append(source_rank)
                source_blocks[source_rank].extend(self.blocks_by_segment[segment_idx])

            ids_by_source = tuple(tuple(blocks) for blocks in source_blocks)
            arrays_by_source = tuple(
                np.asarray(blocks, dtype=np.int64) for blocks in ids_by_source
            )
            max_by_source = tuple(
                max(blocks) if blocks else -1 for blocks in ids_by_source
            )
            required_sources_by_rank.append(tuple(source_order))
            transfer_ids_by_rank.append(ids_by_source)
            transfer_arrays_by_rank.append(arrays_by_source)
            transfer_max_by_rank.append(max_by_source)

        required_sources = tuple(required_sources_by_rank)
        required_source_sets = tuple(frozenset(items) for items in required_sources)
        consumers = tuple(
            tuple(
                rank
                for rank in range(world_size)
                if rank != source_rank
                and source_rank in required_source_sets[rank]
            )
            for source_rank in range(world_size)
        )
        object.__setattr__(self, "_required_sources_by_rank", required_sources)
        object.__setattr__(
            self, "_required_source_sets_by_rank", required_source_sets
        )
        object.__setattr__(self, "_consumers_by_rank", consumers)
        object.__setattr__(self, "_transfer_ids_by_rank", tuple(transfer_ids_by_rank))
        object.__setattr__(
            self, "_transfer_arrays_by_rank", tuple(transfer_arrays_by_rank)
        )
        object.__setattr__(self, "_transfer_max_by_rank", tuple(transfer_max_by_rank))

    @property
    def num_segments(self) -> int:
        return len(self.segment_to_rank)

    @property
    def world_size(self) -> int:
        return max(self.segment_to_rank) + 1

    def owned_segments(self, rank: int) -> tuple[int, ...]:
        return tuple(
            segment_idx
            for segment_idx, owner in enumerate(self.segment_to_rank)
            if owner == rank
        )

    def max_owned_segment(self, rank: int) -> int:
        owned = self.owned_segments(rank)
        if not owned:
            raise ValueError(f"physical PCP rank {rank} owns no logical segment")
        return owned[-1]

    def required_segments(self, rank: int) -> tuple[int, ...]:
        max_segment = self.max_owned_segment(rank)
        return tuple(
            segment_idx
            for segment_idx in range(max_segment)
            if self.segment_to_rank[segment_idx] != rank
        )

    def required_source_ranks(self, rank: int) -> tuple[int, ...]:
        return self._required_sources_by_rank[rank]

    def requires_source(self, rank: int, source_rank: int) -> bool:
        return source_rank in self._required_source_sets_by_rank[rank]

    def consumer_ranks(self, source_rank: int) -> tuple[int, ...]:
        return self._consumers_by_rank[source_rank]

    def transfer_block_ids(
        self, destination_rank: int, source_rank: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return stable local and remote descriptor IDs for one producer."""
        ids = self._transfer_ids_by_rank[destination_rank][source_rank]
        return ids, ids

    def transfer_block_arrays(
        self, destination_rank: int, source_rank: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return prebuilt int64 descriptor arrays and their maximum block ID."""
        ids = self._transfer_arrays_by_rank[destination_rank][source_rank]
        return ids, ids, self._transfer_max_by_rank[destination_rank][source_rank]


@dataclass
class _MemoryRegistration:
    base_addr: int
    block_bytes: int
    num_blocks: int
    device_id: int
    reg_descs: Any
    local_xfer_handle: int


@dataclass
class _InflightRead:
    layer_id: int
    source_rank: int
    handle: int


@dataclass
class _PendingReady:
    layer_id: int
    event: Any | None


class PCPPagePullTransport:
    _POLL_INTERVAL_S = 0.00005
    _NOTIF_PREFIX = "PCP_READY"

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
        max_inflight_reads: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
        pcp_group: Any | None = None,
    ) -> None:
        if max_inflight_reads <= 0:
            raise ValueError("max_inflight_reads must be positive")
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.max_inflight_reads = max_inflight_reads
        self.nixl_backends = nixl_backends
        self.group = pcp_group

        self._wrapper: Any | None = None
        self._memory_type: str | None = None
        self._remote_agents: dict[int, str] = {}
        self._registered_descs: list[Any] = []
        self._memory_by_ptr: dict[int, _MemoryRegistration] = {}
        self._layer_names: tuple[str, ...] = ()
        self._layer_memory: list[_MemoryRegistration] = []
        self._layer_id_by_ptr: dict[int, int] = {}
        self._ready_events: list[Any | None] = []
        self._remote_layer_handles: dict[tuple[int, int], int] = {}
        self._remote_num_blocks: dict[tuple[int, int], int] = {}
        self._metadata_exchanged = False

        self._epoch = 0
        self._plan: PCPPagePlan | None = None
        self._step_finished = True
        self._pending_ready: deque[_PendingReady] = deque()
        self._ready_waiting: deque[tuple[int, int]] = deque()
        self._inflight: dict[tuple[int, int], _InflightRead] = {}
        self._done_pairs: set[tuple[int, int]] = set()

        self._progress_stop = threading.Event()
        self._progress_wakeup = threading.Event()
        self._progress_lock = threading.Lock()
        self._progress_thread: threading.Thread | None = None
        self._progress_error: BaseException | None = None

    def _group(self):
        return self.group if self.group is not None else get_pcp_group()

    @property
    def enabled(self) -> bool:
        return self._plan is not None

    @property
    def registered_layer_names(self) -> tuple[str, ...]:
        return self._layer_names

    def _ensure_wrapper(self) -> None:
        if self._wrapper is not None:
            return
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise RuntimeError(
                "PCP transport=page_pull requires NIXL, but NIXL is unavailable"
            )
        non_ucx = [backend for backend in self.nixl_backends if backend != "UCX"]
        if nixl_agent_config is None:
            config = None
        elif non_ucx:
            config = nixl_agent_config(
                backends=list(self.nixl_backends), capture_telemetry=True
            )
        else:
            config = nixl_agent_config(num_threads=4, capture_telemetry=True)
        self._wrapper = NixlWrapper(
            f"pcp-page-pull-r{self.rank}-{uuid.uuid4()}", config
        )
        memory_type = current_platform.get_nixl_memory_type()
        if memory_type is None:
            memory_type = "VRAM" if self.device.type in ("cuda", "xpu") else "DRAM"
        self._memory_type = memory_type

    def _ensure_remote_agents(self) -> None:
        if self._remote_agents or self.world_size <= 1:
            return
        self._ensure_wrapper()
        assert self._wrapper is not None
        group = self._group()
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            self._wrapper.get_agent_metadata(),
            group=group.cpu_group,
        )
        for source_rank, metadata in enumerate(gathered):
            if source_rank != self.rank:
                self._remote_agents[source_rank] = self._wrapper.add_remote_agent(metadata)

    def configure_step(self, *, epoch: int, plan: PCPPagePlan) -> None:
        self.finish_step()
        self._check_progress_error()
        if plan.world_size != self.world_size:
            raise ValueError(
                "page plan PCP world size mismatch: "
                f"plan={plan.world_size}, runtime={self.world_size}"
            )
        self._epoch = epoch
        self._plan = plan
        self._step_finished = False
        self._pending_ready.clear()
        self._ready_waiting.clear()
        self._inflight.clear()
        self._done_pairs.clear()
        if self._layer_memory:
            self._ensure_remote_agents()
            self._start_progress_thread()
        self._progress_wakeup.set()

    def disable_step(self) -> None:
        self.finish_step()

    @staticmethod
    def _block_descriptors(
        *, base_addr: int, block_bytes: int, num_blocks: int, device_id: int
    ) -> np.ndarray:
        blocks = np.arange(num_blocks, dtype=np.uint64)
        result = np.empty((num_blocks, 3), dtype=np.uint64)
        result[:, 0] = np.uint64(base_addr) + blocks * np.uint64(block_bytes)
        result[:, 1] = np.uint64(block_bytes)
        result[:, 2] = np.uint64(device_id)
        return result

    @staticmethod
    def _physical_page_geometry(kv_cache: torch.Tensor) -> tuple[int, int]:
        if kv_cache.ndim < 2:
            raise RuntimeError(
                f"PCP page-pull expects block-major KV cache, got {kv_cache.shape}"
            )
        num_blocks = int(kv_cache.shape[0])
        page_elements = math.prod(kv_cache.shape[1:])
        if kv_cache.stride(0) != page_elements:
            raise NotImplementedError(
                "PCP page_pull requires dense block-major KV pages; "
                f"shape={tuple(kv_cache.shape)}, stride={tuple(kv_cache.stride())}"
            )
        return num_blocks, int(page_elements * kv_cache.element_size())

    def _register_memory(self, kv_cache: torch.Tensor) -> _MemoryRegistration:
        self._ensure_wrapper()
        assert self._wrapper is not None and self._memory_type is not None
        ptr = kv_cache.data_ptr()
        existing = self._memory_by_ptr.get(ptr)
        if existing is not None:
            return existing
        if kv_cache.device.type != self.device.type:
            raise RuntimeError(
                "PCP page-pull KV cache device changed unexpectedly: "
                f"runtime={self.device}, cache={kv_cache.device}"
            )
        num_blocks, block_bytes = self._physical_page_geometry(kv_cache)
        if num_blocks <= 0 or block_bytes <= 0:
            raise RuntimeError("PCP page-pull cannot register an empty KV cache")
        device_id = max(kv_cache.get_device(), 0)
        reg_descs = self._wrapper.get_reg_descs(
            [(ptr, num_blocks * block_bytes, device_id, "")], self._memory_type
        )
        self._wrapper.register_memory(reg_descs, backends=list(self.nixl_backends))
        block_data = self._block_descriptors(
            base_addr=ptr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
        )
        xfer_descs = self._wrapper.get_xfer_descs(block_data, self._memory_type)
        local_handle = self._wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
        registration = _MemoryRegistration(
            base_addr=ptr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
            reg_descs=reg_descs,
            local_xfer_handle=local_handle,
        )
        self._registered_descs.append(reg_descs)
        self._memory_by_ptr[ptr] = registration
        return registration

    @staticmethod
    def _discover_bound_layer_caches() -> dict[str, torch.Tensor]:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        result: dict[str, torch.Tensor] = {}
        for layer_name, layer in forward_context.no_compile_layers.items():
            kv_cache = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor):
                continue
            if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
                continue
            result[layer_name] = kv_cache
        return result

    @staticmethod
    def _wire_meta(registration: _MemoryRegistration) -> tuple[int, int, int, int]:
        return (
            registration.base_addr,
            registration.block_bytes,
            registration.num_blocks,
            registration.device_id,
        )

    def _exchange_layer_metadata(self) -> None:
        if self._metadata_exchanged or not self._layer_memory:
            return
        self._ensure_remote_agents()
        assert self._wrapper is not None and self._memory_type is not None
        group = self._group()
        local_meta = tuple(self._wire_meta(item) for item in self._layer_memory)
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, local_meta, group=group.cpu_group)
        for source_rank, remote_layers in enumerate(gathered):
            if len(remote_layers) != len(self._layer_memory):
                raise RuntimeError(
                    "PCP page-pull layer registration differs across ranks: "
                    f"rank={source_rank}, layers={len(remote_layers)}, "
                    f"local={len(self._layer_memory)}"
                )
            if source_rank == self.rank:
                continue
            for layer_id, meta in enumerate(remote_layers):
                base_addr, block_bytes, num_blocks, device_id = map(int, meta)
                local = self._layer_memory[layer_id]
                if block_bytes != local.block_bytes:
                    raise RuntimeError(
                        "PCP page-pull requires homogeneous KV page bytes: "
                        f"local={local.block_bytes}, remote={block_bytes}, "
                        f"source_rank={source_rank}, layer_id={layer_id}"
                    )
                block_data = self._block_descriptors(
                    base_addr=base_addr,
                    block_bytes=block_bytes,
                    num_blocks=num_blocks,
                    device_id=device_id,
                )
                descs = self._wrapper.get_xfer_descs(block_data, self._memory_type)
                self._remote_layer_handles[(source_rank, layer_id)] = (
                    self._wrapper.prep_xfer_dlist(
                        self._remote_agents[source_rank], descs
                    )
                )
                self._remote_num_blocks[(source_rank, layer_id)] = num_blocks
        self._metadata_exchanged = True

    def register_layer_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise RuntimeError("PCP page_pull received no tensor KV caches to register")
        layer_names = tuple(sorted(kv_caches))
        registrations = [self._register_memory(kv_caches[name]) for name in layer_names]
        if self._layer_names:
            if layer_names != self._layer_names:
                raise RuntimeError(
                    "PCP page-pull layer set changed after registration: "
                    f"old={self._layer_names}, new={layer_names}"
                )
            for layer_id, registration in enumerate(registrations):
                if self._layer_memory[layer_id].base_addr != registration.base_addr:
                    raise RuntimeError(
                        "PCP page-pull layer cache address changed: "
                        f"layer={layer_names[layer_id]}"
                    )
        else:
            self._layer_names = layer_names
            self._layer_memory = registrations
            self._layer_id_by_ptr = {
                registration.base_addr: layer_id
                for layer_id, registration in enumerate(registrations)
            }
            self._ready_events = (
                [torch.cuda.Event() for _ in layer_names]
                if self.device.type == "cuda"
                else [None] * len(layer_names)
            )
        self._exchange_layer_metadata()
        self._start_progress_thread()
        if self._plan is not None:
            self._progress_wakeup.set()

    def register_current_layer(self, kv_cache: torch.Tensor) -> int:
        if self._plan is None:
            raise RuntimeError("page-pull layer registration requires an active step")
        if not self._layer_names:
            discovered = self._discover_bound_layer_caches()
            if not discovered:
                raise RuntimeError("PCP page-pull requires stable bound layer KV caches")
            self.register_layer_caches(discovered)

        layer_id = self._layer_id_by_ptr.get(kv_cache.data_ptr())
        if layer_id is None:
            raise RuntimeError(
                "PCP page-pull cache is not part of stable layer registration"
            )
        self._check_progress_error()
        return layer_id

    def publish_ready(self, layer_id: int) -> None:
        if self._plan is None:
            return
        if not 0 <= layer_id < len(self._layer_memory):
            raise ValueError(f"invalid page-pull layer id: {layer_id}")
        event = self._ready_events[layer_id]
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        self._pending_ready.append(_PendingReady(layer_id=layer_id, event=event))
        self._start_progress_thread()
        self._progress_wakeup.set()
        self._check_progress_error()

    def _send_ready(self, layer_id: int) -> None:
        assert self._wrapper is not None and self._plan is not None
        msg = f"{self._NOTIF_PREFIX}:{self._epoch}:{layer_id}:{self.rank}".encode()
        for destination_rank in self._plan.consumer_ranks(self.rank):
            self._wrapper.send_notif(
                self._remote_agents[destination_rank], notif_msg=msg
            )

    def _publish_completed_ready(self) -> None:
        while self._pending_ready:
            pending = self._pending_ready[0]
            if pending.event is not None and not pending.event.query():
                break
            self._pending_ready.popleft()
            self._send_ready(pending.layer_id)

    def _poll_ready_notifications(self) -> None:
        assert self._wrapper is not None
        for notifications in self._wrapper.get_new_notifs().values():
            for raw in notifications:
                try:
                    prefix, epoch, layer_id, source_rank = raw.decode().split(":")
                    epoch_i = int(epoch)
                    layer_i = int(layer_id)
                    source_i = int(source_rank)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError(
                        f"invalid PCP page-pull NIXL notification: {raw!r}"
                    ) from exc
                if prefix != self._NOTIF_PREFIX:
                    continue
                if epoch_i != self._epoch:
                    raise RuntimeError(
                        "PCP page-pull READY epoch mismatch: "
                        f"got={epoch_i}, expected={self._epoch}"
                    )
                if self._plan is None or not self._plan.requires_source(
                    self.rank, source_i
                ):
                    raise RuntimeError(
                        f"unexpected PCP READY source rank {source_i} for rank {self.rank}"
                    )
                key = (layer_i, source_i)
                if key not in self._done_pairs and key not in self._inflight:
                    self._ready_waiting.append(key)

    def _start_read(self, layer_id: int, source_rank: int) -> None:
        assert self._wrapper is not None and self._plan is not None
        local = self._layer_memory[layer_id]
        destination_ids, source_ids, max_block_id = self._plan.transfer_block_arrays(
            self.rank, source_rank
        )
        key = (layer_id, source_rank)
        if destination_ids.size == 0:
            self._done_pairs.add(key)
            return
        if max_block_id >= local.num_blocks:
            raise RuntimeError(
                "PCP page-pull destination block id exceeds local cache: "
                f"max={max_block_id}, num_blocks={local.num_blocks}"
            )
        remote_num_blocks = self._remote_num_blocks[(source_rank, layer_id)]
        if max_block_id >= remote_num_blocks:
            raise RuntimeError(
                "PCP page-pull source block id exceeds remote cache: "
                f"max={max_block_id}, num_blocks={remote_num_blocks}"
            )
        with pcp_nvtx_range("pcp.page_pull_read_submit"):
            handle = self._wrapper.make_prepped_xfer(
                "READ",
                local.local_xfer_handle,
                destination_ids,
                self._remote_layer_handles[(source_rank, layer_id)],
                source_ids,
            )
            self._wrapper.transfer(handle)
        self._inflight[key] = _InflightRead(layer_id, source_rank, handle)

    def _progress_once(self) -> None:
        with self._progress_lock:
            if self._plan is None:
                return
            assert self._wrapper is not None
            self._publish_completed_ready()
            self._poll_ready_notifications()
            while self._ready_waiting and len(self._inflight) < self.max_inflight_reads:
                layer_id, source_rank = self._ready_waiting.popleft()
                self._start_read(layer_id, source_rank)

            for key, transfer in list(self._inflight.items()):
                state = self._wrapper.check_xfer_state(transfer.handle)
                if state == "PROC":
                    continue
                self._wrapper.release_xfer_handle(transfer.handle)
                del self._inflight[key]
                if state != "DONE":
                    raise RuntimeError(
                        "PCP page-pull NIXL READ failed: "
                        f"state={state}, layer_id={transfer.layer_id}, "
                        f"source_rank={transfer.source_rank}"
                    )
                self._done_pairs.add(key)

    def _progress_loop(self) -> None:
        try:
            while not self._progress_stop.is_set():
                if self._plan is None:
                    self._progress_wakeup.wait()
                    self._progress_wakeup.clear()
                    continue
                self._progress_once()
                self._progress_wakeup.wait(self._POLL_INTERVAL_S)
                self._progress_wakeup.clear()
        except BaseException as exc:
            self._progress_error = exc
            self._progress_stop.set()

    def _start_progress_thread(self) -> None:
        if self._progress_thread is not None:
            return
        if not self._layer_memory:
            return
        self._ensure_remote_agents()
        self._progress_stop.clear()
        self._progress_thread = threading.Thread(
            target=self._progress_loop,
            name=f"pcp-page-pull-r{self.rank}",
            daemon=True,
        )
        self._progress_thread.start()

    def _check_progress_error(self) -> None:
        if self._progress_error is not None:
            raise RuntimeError("PCP page-pull progress engine failed") from self._progress_error

    def progress(self) -> None:
        self._check_progress_error()
        if self._progress_thread is None:
            self._progress_once()

    def wait_layer(self, layer_id: int) -> None:
        plan = self._plan
        if plan is None:
            return
        required_sources = plan.required_source_ranks(self.rank)
        with pcp_nvtx_range("pcp.page_pull_wait"):
            while any(
                (layer_id, source_rank) not in self._done_pairs
                for source_rank in required_sources
            ):
                self._check_progress_error()
                if self._progress_thread is None:
                    self._progress_once()
                time.sleep(self._POLL_INTERVAL_S)
        self._check_progress_error()

    def _expected_pairs(self) -> set[tuple[int, int]]:
        if self._plan is None:
            return set()
        required_sources = self._plan.required_source_ranks(self.rank)
        return {
            (layer_id, source_rank)
            for layer_id in range(len(self._layer_memory))
            for source_rank in required_sources
        }

    def finish_step(self) -> None:
        if self._step_finished:
            return
        if self._plan is not None and self._layer_memory:
            expected = self._expected_pairs()
            while self._pending_ready or not expected.issubset(self._done_pairs):
                self._check_progress_error()
                if self._progress_thread is None:
                    self._progress_once()
                time.sleep(self._POLL_INTERVAL_S)

        self._check_progress_error()
        self._plan = None
        self._progress_wakeup.set()
        with self._progress_lock:
            self._pending_ready.clear()
            self._ready_waiting.clear()
            self._inflight.clear()
            self._done_pairs.clear()
            self._step_finished = True


__all__ = ["PCPPagePlan", "PCPPagePullTransport"]

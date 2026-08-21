# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-aware one-sided KV pull for experimental PCP runahead.

The control plane is expressed in logical PCP segments while the data plane is
expressed in physical KV-cache block IDs. Producers publish only tiny READY
messages after their local cache write is visible. Consumers decide which READY
source to service next and issue NIXL READ operations directly into their local
paged KV cache.

This module intentionally does not depend on the request-level KVConnector
scheduler. PCP is a layer-level protocol and only reuses the NIXL registered
memory / one-sided transfer primitives.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group
from vllm.platforms import current_platform
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range


@dataclass(frozen=True)
class PCPPagePlan:
    """Per-step logical-page ownership and local/remote physical block mapping."""

    segment_to_rank: tuple[int, ...]
    source_blocks_by_segment: tuple[tuple[int, ...], ...]
    destination_blocks_by_segment: tuple[tuple[int, ...], ...]
    block_size: int

    def __post_init__(self) -> None:
        num_segments = len(self.segment_to_rank)
        if num_segments == 0:
            raise ValueError("page-pull plan requires at least one logical segment")
        if len(self.source_blocks_by_segment) != num_segments:
            raise ValueError("source block map must match logical segment count")
        if len(self.destination_blocks_by_segment) != num_segments:
            raise ValueError("destination block map must match logical segment count")
        if self.block_size <= 0:
            raise ValueError("page-pull block_size must be positive")
        for segment_idx in range(num_segments - 1):
            source = self.source_blocks_by_segment[segment_idx]
            destination = self.destination_blocks_by_segment[segment_idx]
            if len(source) != len(destination):
                raise ValueError(
                    "page-pull source/destination block counts differ for logical "
                    f"segment {segment_idx}: src={len(source)}, dst={len(destination)}"
                )

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
        """Remote logical segments needed by this rank's latest local query."""
        max_segment = self.max_owned_segment(rank)
        return tuple(
            segment_idx
            for segment_idx in range(max_segment)
            if self.segment_to_rank[segment_idx] != rank
        )

    def required_source_ranks(self, rank: int) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                self.segment_to_rank[segment_idx]
                for segment_idx in self.required_segments(rank)
            )
        )

    def consumer_ranks(self, source_rank: int) -> tuple[int, ...]:
        consumers = []
        for rank in range(self.world_size):
            if rank == source_rank:
                continue
            if source_rank in self.required_source_ranks(rank):
                consumers.append(rank)
        return tuple(consumers)

    def transfer_block_ids(
        self, destination_rank: int, source_rank: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return destination/source physical block IDs for one source rank."""
        dst: list[int] = []
        src: list[int] = []
        for segment_idx in self.required_segments(destination_rank):
            if self.segment_to_rank[segment_idx] != source_rank:
                continue
            dst.extend(self.destination_blocks_by_segment[segment_idx])
            src.extend(self.source_blocks_by_segment[segment_idx])
        if len(dst) != len(src):
            raise RuntimeError(
                "PCP page plan produced mismatched transfer block counts: "
                f"dst={len(dst)}, src={len(src)}, source_rank={source_rank}"
            )
        return tuple(dst), tuple(src)


@dataclass
class _MemoryRegistration:
    base_addr: int
    block_bytes: int
    num_blocks: int
    device_id: int
    reg_descs: Any
    local_xfer_handle: int


@dataclass
class _ReadyRecv:
    tensor: torch.Tensor
    work: Any


@dataclass
class _InflightRead:
    layer_ordinal: int
    source_rank: int
    handle: int


class PCPPagePullTransport:
    """Consumer-driven PCP page pull using NIXL READ.

    Layer cache registrations are retained across scheduler steps. After the
    first full pass, later steps can post READY receives for already-known
    future layers and prefetch them opportunistically when ``progress`` runs at
    earlier layer boundaries.
    """

    _READY_FIELDS = 6

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
        max_inflight_reads: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
    ) -> None:
        if max_inflight_reads <= 0:
            raise ValueError("max_inflight_reads must be positive")
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.max_inflight_reads = max_inflight_reads
        self.nixl_backends = nixl_backends

        self._wrapper: Any | None = None
        self._memory_type: str | None = None
        self._remote_agents: dict[int, str] = {}
        self._registered_descs: list[Any] = []
        self._memory_by_ptr: dict[int, _MemoryRegistration] = {}
        self._layer_memory: list[_MemoryRegistration] = []
        self._remote_layer_handles: dict[tuple[int, int, int], int] = {}

        self._epoch = 0
        self._plan: PCPPagePlan | None = None
        self._layer_cursor = 0
        self._ready_recvs: dict[tuple[int, int], _ReadyRecv] = {}
        self._ready_waiting: deque[tuple[int, int, tuple[int, ...]]] = deque()
        self._pending_ready_sends: list[tuple[Any, torch.Tensor]] = []
        self._inflight: dict[tuple[int, int], _InflightRead] = {}
        self._done_pairs: set[tuple[int, int]] = set()

    @property
    def enabled(self) -> bool:
        return self._plan is not None

    def _ensure_agents(self) -> None:
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

        group = get_pcp_group()
        local_agent_metadata = self._wrapper.get_agent_metadata()
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            gathered, local_agent_metadata, group=group.cpu_group
        )
        for source_rank, metadata in enumerate(gathered):
            if source_rank == self.rank:
                continue
            self._remote_agents[source_rank] = self._wrapper.add_remote_agent(metadata)

    def configure_step(self, *, epoch: int, plan: PCPPagePlan) -> None:
        self.finish_step()
        if plan.world_size != self.world_size:
            raise ValueError(
                "page plan PCP world size mismatch: "
                f"plan={plan.world_size}, runtime={self.world_size}"
            )
        self._ensure_agents()
        self._epoch = epoch
        self._plan = plan
        self._layer_cursor = 0
        self._ready_recvs.clear()
        self._ready_waiting.clear()
        self._inflight.clear()
        self._done_pairs.clear()

        # Registrations survive between steps. Pre-post receives for all layers
        # seen previously so READY for a future layer can arrive while this rank
        # is still computing an earlier layer.
        for layer_ordinal in range(len(self._layer_memory)):
            self._post_ready_recvs(layer_ordinal)

    def disable_step(self) -> None:
        self.finish_step()
        self._plan = None
        self._layer_cursor = 0

    def _block_descriptors(
        self,
        *,
        base_addr: int,
        block_bytes: int,
        num_blocks: int,
        device_id: int,
    ) -> np.ndarray:
        blocks = np.arange(num_blocks, dtype=np.uint64)
        result = np.empty((num_blocks, 3), dtype=np.uint64)
        result[:, 0] = np.uint64(base_addr) + blocks * np.uint64(block_bytes)
        result[:, 1] = np.uint64(block_bytes)
        result[:, 2] = np.uint64(device_id)
        return result

    def _register_memory(self, kv_cache: torch.Tensor) -> _MemoryRegistration:
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
        if kv_cache.ndim < 2:
            raise RuntimeError(
                f"PCP page-pull expects block-major KV cache, got {kv_cache.shape}"
            )

        num_blocks = int(kv_cache.shape[0])
        block_bytes = int(kv_cache.stride(0) * kv_cache.element_size())
        if num_blocks <= 0 or block_bytes <= 0:
            raise RuntimeError(
                "PCP page-pull cannot register an empty KV cache: "
                f"shape={tuple(kv_cache.shape)}, stride={tuple(kv_cache.stride())}"
            )
        device_id = max(kv_cache.get_device(), 0)
        region_bytes = num_blocks * block_bytes
        reg_descs = self._wrapper.get_reg_descs(
            [(ptr, region_bytes, device_id, "")], self._memory_type
        )
        self._wrapper.register_memory(
            reg_descs, backends=list(self.nixl_backends)
        )
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

    def register_current_layer(self, kv_cache: torch.Tensor) -> int:
        if self._plan is None:
            raise RuntimeError("page-pull layer registration requires an active step")
        self._ensure_agents()
        layer_ordinal = self._layer_cursor
        self._layer_cursor += 1
        registration = self._register_memory(kv_cache)
        if layer_ordinal == len(self._layer_memory):
            self._layer_memory.append(registration)
        else:
            expected = self._layer_memory[layer_ordinal]
            if expected.base_addr != registration.base_addr:
                raise RuntimeError(
                    "PCP page-pull layer cache address changed between steps: "
                    f"layer={layer_ordinal}, old={expected.base_addr}, "
                    f"new={registration.base_addr}"
                )
        self._post_ready_recvs(layer_ordinal)
        self.progress()
        return layer_ordinal

    def _post_ready_recvs(self, layer_ordinal: int) -> None:
        if self._plan is None:
            return
        group = get_pcp_group()
        for source_rank in self._plan.required_source_ranks(self.rank):
            key = (layer_ordinal, source_rank)
            if key in self._ready_recvs or key in self._done_pairs or key in self._inflight:
                continue
            tensor = torch.empty(self._READY_FIELDS, dtype=torch.int64, device="cpu")
            work = dist.irecv(
                tensor,
                src=group.ranks[source_rank],
                group=group.cpu_group,
            )
            self._ready_recvs[key] = _ReadyRecv(tensor=tensor, work=work)

    def _drain_ready_sends(self) -> None:
        remaining: list[tuple[Any, torch.Tensor]] = []
        for work, tensor in self._pending_ready_sends:
            if work.is_completed():
                work.wait()
            else:
                remaining.append((work, tensor))
        self._pending_ready_sends = remaining

    def publish_ready(self, layer_ordinal: int) -> None:
        if self._plan is None:
            return
        if not 0 <= layer_ordinal < len(self._layer_memory):
            raise ValueError(f"invalid page-pull layer ordinal: {layer_ordinal}")
        registration = self._layer_memory[layer_ordinal]

        # The READY message must be ordered after the producer's cache write.
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            with pcp_nvtx_range("pcp.page_pull_local_ready_wait"):
                event.synchronize()

        meta_values = (
            self._epoch,
            layer_ordinal,
            registration.base_addr,
            registration.block_bytes,
            registration.num_blocks,
            registration.device_id,
        )
        group = get_pcp_group()
        for destination_rank in self._plan.consumer_ranks(self.rank):
            tensor = torch.tensor(meta_values, dtype=torch.int64, device="cpu")
            work = dist.isend(
                tensor,
                dst=group.ranks[destination_rank],
                group=group.cpu_group,
            )
            self._pending_ready_sends.append((work, tensor))
        self._drain_ready_sends()

    def _remote_handle(
        self,
        *,
        source_rank: int,
        layer_ordinal: int,
        meta: tuple[int, ...],
    ) -> int:
        assert self._wrapper is not None and self._memory_type is not None
        _, _, base_addr, block_bytes, num_blocks, device_id = meta
        cache_key = (source_rank, layer_ordinal, base_addr)
        handle = self._remote_layer_handles.get(cache_key)
        if handle is not None:
            return handle
        local = self._layer_memory[layer_ordinal]
        if block_bytes != local.block_bytes:
            raise RuntimeError(
                "PCP page-pull requires homogeneous KV page bytes: "
                f"local={local.block_bytes}, remote={block_bytes}, "
                f"source_rank={source_rank}, layer={layer_ordinal}"
            )
        block_data = self._block_descriptors(
            base_addr=base_addr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
        )
        descs = self._wrapper.get_xfer_descs(block_data, self._memory_type)
        handle = self._wrapper.prep_xfer_dlist(
            self._remote_agents[source_rank], descs
        )
        self._remote_layer_handles[cache_key] = handle
        return handle

    def _start_read(
        self,
        layer_ordinal: int,
        source_rank: int,
        meta: tuple[int, ...],
    ) -> None:
        assert self._wrapper is not None and self._plan is not None
        local = self._layer_memory[layer_ordinal]
        dst_ids, src_ids = self._plan.transfer_block_ids(self.rank, source_rank)
        key = (layer_ordinal, source_rank)
        if not dst_ids:
            self._done_pairs.add(key)
            return
        if max(dst_ids) >= local.num_blocks:
            raise RuntimeError(
                "PCP page-pull destination block id exceeds local cache: "
                f"max={max(dst_ids)}, num_blocks={local.num_blocks}"
            )
        remote_handle = self._remote_handle(
            source_rank=source_rank,
            layer_ordinal=layer_ordinal,
            meta=meta,
        )
        remote_num_blocks = meta[4]
        if max(src_ids) >= remote_num_blocks:
            raise RuntimeError(
                "PCP page-pull source block id exceeds remote cache: "
                f"max={max(src_ids)}, num_blocks={remote_num_blocks}"
            )
        dst_desc_ids = np.asarray(dst_ids, dtype=np.int64)
        src_desc_ids = np.asarray(src_ids, dtype=np.int64)
        with pcp_nvtx_range("pcp.page_pull_read_submit"):
            handle = self._wrapper.make_prepped_xfer(
                "READ",
                local.local_xfer_handle,
                dst_desc_ids,
                remote_handle,
                src_desc_ids,
            )
            self._wrapper.transfer(handle)
        self._inflight[key] = _InflightRead(
            layer_ordinal=layer_ordinal,
            source_rank=source_rank,
            handle=handle,
        )

    def progress(self) -> None:
        if self._plan is None:
            return
        assert self._wrapper is not None

        for key, recv in list(self._ready_recvs.items()):
            if not recv.work.is_completed():
                continue
            recv.work.wait()
            meta = tuple(int(value) for value in recv.tensor.tolist())
            epoch, layer_ordinal = meta[:2]
            expected_layer, source_rank = key
            if epoch != self._epoch or layer_ordinal != expected_layer:
                raise RuntimeError(
                    "PCP page-pull READY message is out of order: "
                    f"got epoch/layer={epoch}/{layer_ordinal}, "
                    f"expected={self._epoch}/{expected_layer}, source={source_rank}"
                )
            self._ready_waiting.append((expected_layer, source_rank, meta))
            del self._ready_recvs[key]

        # READY arrival determines service order. A layer not registered locally
        # yet remains queued; this enables later-step future-layer prefetch once
        # its persistent cache registration is already known.
        waiting_rounds = len(self._ready_waiting)
        for _ in range(waiting_rounds):
            if len(self._inflight) >= self.max_inflight_reads:
                break
            layer_ordinal, source_rank, meta = self._ready_waiting.popleft()
            if layer_ordinal >= len(self._layer_memory):
                self._ready_waiting.append((layer_ordinal, source_rank, meta))
                continue
            self._start_read(layer_ordinal, source_rank, meta)

        for key, transfer in list(self._inflight.items()):
            state = self._wrapper.check_xfer_state(transfer.handle)
            if state == "PROC":
                continue
            if state != "DONE":
                self._wrapper.release_xfer_handle(transfer.handle)
                del self._inflight[key]
                raise RuntimeError(
                    "PCP page-pull NIXL READ failed: "
                    f"state={state}, layer={transfer.layer_ordinal}, "
                    f"source_rank={transfer.source_rank}"
                )
            self._wrapper.release_xfer_handle(transfer.handle)
            del self._inflight[key]
            self._done_pairs.add(key)

        self._drain_ready_sends()

    def wait_layer(self, layer_ordinal: int) -> None:
        if self._plan is None:
            return
        required = {
            (layer_ordinal, source_rank)
            for source_rank in self._plan.required_source_ranks(self.rank)
        }
        with pcp_nvtx_range("pcp.page_pull_wait"):
            while not required.issubset(self._done_pairs):
                self.progress()
                time.sleep(0.00005)
        # Run one extra progress pass so already-arrived future-layer READY
        # messages can start prefetch before attention consumes this layer.
        self.progress()

    def finish_step(self) -> None:
        if self._wrapper is not None:
            # Every layer actually consumed by the model is waited explicitly.
            # Drain any speculative future reads that were already submitted.
            while self._inflight:
                self.progress()
                time.sleep(0.00005)
        for work, _tensor in self._pending_ready_sends:
            work.wait()
        self._pending_ready_sends.clear()
        self._ready_recvs.clear()
        self._ready_waiting.clear()
        self._done_pairs.clear()


__all__ = ["PCPPagePlan", "PCPPagePullTransport"]
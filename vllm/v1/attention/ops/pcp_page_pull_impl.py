# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-aware one-sided KV-pull runtime for PCP runahead."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion, PCPNixlPeerTransport
from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_mark, pcp_nvtx_range


@dataclass
class _InflightRead:
    layer_id: int
    source_rank: int
    handle: int
    num_pages: int


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
        self._peer = PCPNixlPeerTransport(
            world_size=world_size,
            rank=rank,
            device=device,
            nixl_backends=nixl_backends,
            pcp_group=pcp_group,
        )

        self._layer_names: tuple[str, ...] = ()
        self._layer_memory: list[NixlMemoryRegion] = []
        self._layer_id_by_ptr: dict[int, int] = {}
        self._ready_events: list[Any | None] = []

        self._epoch = 0
        self._plan: PCPPagePlan | None = None
        self._step_finished = True
        self._pending_ready: deque[_PendingReady] = deque()
        self._ready_waiting: deque[tuple[int, int]] = deque()
        self._inflight: dict[tuple[int, int], _InflightRead] = {}
        self._done_pairs: set[tuple[int, int]] = set()
        self._demand_layer: int | None = None

        self._progress_stop = threading.Event()
        self._progress_wakeup = threading.Event()
        self._progress_lock = threading.Lock()
        self._progress_thread: threading.Thread | None = None
        self._progress_error: BaseException | None = None

    @property
    def enabled(self) -> bool:
        with self._progress_lock:
            return self._plan is not None

    @property
    def registered_layer_names(self) -> tuple[str, ...]:
        return self._layer_names

    def _layer_label(self, layer_id: int) -> str | int:
        if 0 <= layer_id < len(self._layer_names):
            return self._layer_names[layer_id]
        return layer_id

    @staticmethod
    def _physical_page_geometry(kv_cache: torch.Tensor) -> tuple[int, int]:
        return PCPNixlPeerTransport.physical_page_geometry(kv_cache)

    def _clear_step_state_locked(self) -> None:
        self._pending_ready.clear()
        self._ready_waiting.clear()
        self._inflight.clear()
        self._done_pairs.clear()
        self._demand_layer = None

    def configure_step(self, *, epoch: int, plan: PCPPagePlan) -> None:
        self.finish_step()
        self._check_progress_error()
        if not self._layer_memory:
            discovered = self._discover_bound_layer_caches()
            if discovered:
                self.register_layer_caches(discovered)
        if plan.world_size != self.world_size:
            raise ValueError(
                "page plan PCP world size mismatch: "
                f"plan={plan.world_size}, runtime={self.world_size}"
            )
        with self._progress_lock:
            self._epoch = epoch
            self._plan = plan
            self._step_finished = False
            self._clear_step_state_locked()
        if self._layer_memory:
            self._peer.ensure_peers()
            self._start_progress_thread()
        self._progress_wakeup.set()
        pcp_nvtx_mark("pcp.page_pull_step_begin", e=self._epoch, rank=self.rank)

    def disable_step(self) -> None:
        self.finish_step()

    @staticmethod
    def _discover_bound_layer_caches() -> dict[str, torch.Tensor]:
        try:
            forward_context = (
                get_current_vllm_config().compilation_config.static_forward_context
            )
        except (AttributeError, RuntimeError):
            return {}
        result: dict[str, torch.Tensor] = {}
        for layer_name, layer in forward_context.items():
            kv_cache = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor):
                continue
            if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
                continue
            result[layer_name] = kv_cache
        return result

    def register_layer_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise RuntimeError("PCP page_pull received no tensor KV caches to register")
        layer_names = tuple(sorted(kv_caches))
        registrations = [self._peer.register_tensor(kv_caches[name]) for name in layer_names]
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
        self._peer.exchange_regions(self._layer_memory)
        with self._progress_lock:
            active = self._plan is not None
        if active:
            self._start_progress_thread()
            self._progress_wakeup.set()

    def register_current_layer(self, kv_cache: torch.Tensor) -> int:
        with self._progress_lock:
            active = self._plan is not None
        if not active:
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
        with self._progress_lock:
            if self._plan is None:
                return
        if not 0 <= layer_id < len(self._layer_memory):
            raise ValueError(f"invalid page-pull layer id: {layer_id}")
        event = self._ready_events[layer_id]
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        with self._progress_lock:
            if self._plan is None:
                return
            self._pending_ready.append(_PendingReady(layer_id=layer_id, event=event))
        self._start_progress_thread()
        self._progress_wakeup.set()
        self._check_progress_error()

    def _send_ready_locked(self, layer_id: int) -> None:
        assert self._plan is not None
        destinations = self._plan.consumer_ranks(self.rank)
        msg = f"{self._NOTIF_PREFIX}:{self._epoch}:{layer_id}:{self.rank}".encode()
        self._peer.send_notification(destinations, msg)
        for destination_rank in destinations:
            pcp_nvtx_mark(
                "pcp.page_pull_ready_send",
                e=self._epoch,
                l=self._layer_label(layer_id),
                src=self.rank,
                dst=destination_rank,
            )

    def _publish_completed_ready_locked(self) -> None:
        while self._pending_ready:
            pending = self._pending_ready[0]
            if pending.event is not None and not pending.event.query():
                break
            self._pending_ready.popleft()
            pcp_nvtx_mark(
                "pcp.page_pull_ready",
                e=self._epoch,
                l=self._layer_label(pending.layer_id),
                src=self.rank,
            )
            self._send_ready_locked(pending.layer_id)

    def _poll_ready_notifications_locked(self) -> None:
        assert self._plan is not None
        for raw in self._peer.iter_notifications():
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
            if not self._plan.requires_source(self.rank, source_i):
                raise RuntimeError(
                    f"unexpected PCP READY source rank {source_i} for rank {self.rank}"
                )
            key = (layer_i, source_i)
            if (
                key not in self._done_pairs
                and key not in self._inflight
                and key not in self._ready_waiting
            ):
                self._ready_waiting.append(key)
                pcp_nvtx_mark(
                    "pcp.page_pull_ready_recv",
                    e=self._epoch,
                    l=self._layer_label(layer_i),
                    src=source_i,
                    dst=self.rank,
                )

    def _pop_ready_locked(self) -> tuple[int, int] | None:
        if not self._ready_waiting:
            return None
        demand_layer = self._demand_layer
        if demand_layer is None:
            return self._ready_waiting.popleft()
        for index, key in enumerate(self._ready_waiting):
            if key[0] != demand_layer:
                continue
            self._ready_waiting.rotate(-index)
            selected = self._ready_waiting.popleft()
            self._ready_waiting.rotate(index)
            return selected
        return self._ready_waiting.popleft()

    def _start_read_locked(self, layer_id: int, source_rank: int) -> None:
        assert self._plan is not None
        local = self._layer_memory[layer_id]
        route = self._plan.transfer_route(self.rank, source_rank)
        key = (layer_id, source_rank)
        num_pages = route.num_pages
        if num_pages == 0:
            self._done_pairs.add(key)
            return
        with pcp_nvtx_range(
            "pcp.page_pull_read_submit",
            e=self._epoch,
            l=self._layer_label(layer_id),
            src=source_rank,
            dst=self.rank,
            pages=num_pages,
        ):
            handle = self._peer.submit_prepared_read(
                local_region=local,
                local_block_ids=route.destination_block_array,
                local_max_block_id=route.destination_max_block_id,
                source_rank=source_rank,
                remote_region_id=layer_id,
                remote_block_ids=route.source_block_array,
                remote_max_block_id=route.source_max_block_id,
            )
        self._inflight[key] = _InflightRead(
            layer_id=layer_id,
            source_rank=source_rank,
            handle=handle,
            num_pages=num_pages,
        )

    def _progress_once(self) -> None:
        with self._progress_lock:
            if self._plan is None:
                return
            self._publish_completed_ready_locked()
            self._poll_ready_notifications_locked()
            for key, transfer in list(self._inflight.items()):
                state = self._peer.check_read(transfer.handle)
                if state == "PROC":
                    continue
                del self._inflight[key]
                if state != "DONE":
                    raise RuntimeError(
                        "PCP page-pull NIXL READ failed: "
                        f"state={state}, layer_id={transfer.layer_id}, "
                        f"source_rank={transfer.source_rank}"
                    )
                self._done_pairs.add(key)
                pcp_nvtx_mark(
                    "pcp.page_pull_read_done",
                    e=self._epoch,
                    l=self._layer_label(transfer.layer_id),
                    src=transfer.source_rank,
                    dst=self.rank,
                    pages=transfer.num_pages,
                )
            while len(self._inflight) < self.max_inflight_reads:
                ready = self._pop_ready_locked()
                if ready is None:
                    break
                layer_id, source_rank = ready
                self._start_read_locked(layer_id, source_rank)

    def _progress_loop(self) -> None:
        try:
            while not self._progress_stop.is_set():
                with self._progress_lock:
                    active = self._plan is not None
                if not active:
                    self._progress_wakeup.wait()
                    self._progress_wakeup.clear()
                    continue
                self._progress_once()
                self._progress_wakeup.wait(self._POLL_INTERVAL_S)
                self._progress_wakeup.clear()
        except BaseException as exc:
            self._progress_error = exc
            self._progress_stop.set()
            self._progress_wakeup.set()

    def _start_progress_thread(self) -> None:
        if self._progress_thread is not None:
            return
        if not self._layer_memory:
            return
        self._peer.ensure_peers()
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
        with self._progress_lock:
            if self._plan is None:
                return
            self._demand_layer = layer_id
        self._progress_wakeup.set()
        try:
            while True:
                self._check_progress_error()
                with self._progress_lock:
                    plan = self._plan
                    if plan is None:
                        return
                    required = plan.required_source_ranks(self.rank)
                    complete = all(
                        (layer_id, source_rank) in self._done_pairs
                        for source_rank in required
                    )
                if complete:
                    return
                if self._progress_thread is None:
                    self._progress_once()
                else:
                    self._progress_wakeup.set()
                time.sleep(self._POLL_INTERVAL_S)
        finally:
            with self._progress_lock:
                if self._demand_layer == layer_id:
                    self._demand_layer = None

    def _expected_pairs_locked(self) -> set[tuple[int, int]]:
        assert self._plan is not None
        required_sources = self._plan.required_source_ranks(self.rank)
        return {
            (layer_id, source_rank)
            for layer_id in range(len(self._layer_memory))
            for source_rank in required_sources
        }

    def _finish_step_if_ready(self) -> bool:
        with self._progress_lock:
            if self._step_finished:
                return True
            if self._plan is not None and self._layer_memory:
                expected = self._expected_pairs_locked()
                if self._pending_ready or not expected.issubset(self._done_pairs):
                    return False
            self._plan = None
            self._clear_step_state_locked()
            self._step_finished = True
            return True

    def finish_step(self) -> None:
        if self._step_finished:
            return
        while not self._finish_step_if_ready():
            self._check_progress_error()
            if self._progress_thread is None:
                self._progress_once()
            else:
                self._progress_wakeup.set()
            time.sleep(self._POLL_INTERVAL_S)
        self._check_progress_error()
        pcp_nvtx_mark("pcp.page_pull_step_end", e=self._epoch, rank=self.rank)
        self._progress_wakeup.set()

    def close(self) -> None:
        self.finish_step()
        self._progress_stop.set()
        self._progress_wakeup.set()
        if self._progress_thread is not None:
            self._progress_thread.join()
            self._progress_thread = None
        self._check_progress_error()


__all__ = ["PCPPagePullTransport"]

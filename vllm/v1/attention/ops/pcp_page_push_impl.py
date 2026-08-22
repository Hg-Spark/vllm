# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-push KV-page pipeline for PCP runahead.

The data path is owner-driven NIXL WRITE:

* CURRENT pushes satisfy same-step causal dependencies.
* REPLICA pushes materialize finalized full pages for later chunks.
* Historical demand is an assertion only; there is no READ fallback.

Writes are drained before the restore/scheduler boundary, so NIXL DMA never
outlives vLLM's physical KV-block lifetime fence.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from vllm.v1.attention.ops.pcp_nixl import (
    NixlMemoryRegion,
    NixlWrite,
    PCPNixlPeerTransport,
)
from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan, PCPPageRoute
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_mark, pcp_nvtx_range


_RouteKind = Literal["current", "replica"]
_RouteKey = tuple[int, int, _RouteKind]


@dataclass
class _PendingReady:
    layer_id: int
    event: Any | None


@dataclass
class _InflightWriteBatch:
    keys: tuple[_RouteKey, ...]
    destination_rank: int
    kind: _RouteKind
    handle: int
    num_pages: int


class PCPPagePushTransport:
    """Per-process producer-push page pipeline.

    Ready routes are opportunistically coalesced by destination and route kind.
    The pipeline never delays a CURRENT route waiting for future layers: it only
    batches layers that are already CUDA-ready when a submission slot opens.
    """

    _POLL_INTERVAL_S = 0.00005
    _NOTIF_PREFIX = "PCP_VISIBLE"

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
        max_inflight_writes: int = 4,
        max_batch_layers: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
        pcp_group: Any | None = None,
        static_forward_context: dict[str, Any] | None = None,
    ) -> None:
        if max_inflight_writes <= 0:
            raise ValueError("max_inflight_writes must be positive")
        if max_batch_layers <= 0:
            raise ValueError("max_batch_layers must be positive")
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.max_inflight_writes = max_inflight_writes
        self.max_batch_layers = max_batch_layers
        self._static_forward_context = static_forward_context
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
        self._slot_mapping_configured = False
        self._replica_routes: dict[tuple[int, int], PCPPageRoute] = {}

        self._pending_ready: deque[_PendingReady] = deque()
        self._current_waiting: deque[_RouteKey] = deque()
        self._replica_waiting: deque[_RouteKey] = deque()
        self._queued_outgoing: set[_RouteKey] = set()
        self._inflight: dict[int, _InflightWriteBatch] = {}
        self._outgoing_done: set[_RouteKey] = set()
        self._incoming_done: set[_RouteKey] = set()
        self._deferred_notifications: deque[bytes] = deque()

        # The manager commits page-state only after restore. Keep one transport
        # proof table across adjacent steps so a historical route can assert that
        # the preceding CURRENT/REPLICA WRITE populated the exact physical block.
        self._persistent_visible_blocks: dict[tuple[int, int], set[int]] = {}

        self._progress_stop = threading.Event()
        self._progress_wakeup = threading.Event()
        self._progress_lock = threading.Lock()
        self._completion_cv = threading.Condition(self._progress_lock)
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

    def _clear_step_state_locked(self) -> None:
        self._slot_mapping_configured = False
        self._replica_routes.clear()
        self._pending_ready.clear()
        self._current_waiting.clear()
        self._replica_waiting.clear()
        self._queued_outgoing.clear()
        self._inflight.clear()
        self._outgoing_done.clear()
        self._incoming_done.clear()
        self._deferred_notifications.clear()

    def configure_step(self, *, epoch: int, plan: PCPPagePlan) -> None:
        self.finish_step()
        self._check_progress_error()
        if not self._layer_memory:
            caches = self._discover_bound_layer_caches()
            if caches:
                self.register_layer_caches(caches)
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
            try:
                self._validate_history_replicas_locked()
            except BaseException:
                self._plan = None
                self._step_finished = True
                raise
        if self._layer_memory:
            self._peer.ensure_peers()
            self._start_progress_thread()
        self._progress_wakeup.set()
        pcp_nvtx_mark("pcp.page_push_step_begin", e=self._epoch, rank=self.rank)

    def disable_step(self) -> None:
        self.finish_step()

    def _discover_bound_layer_caches(self) -> dict[str, torch.Tensor]:
        if self._static_forward_context is None:
            return {}
        result: dict[str, torch.Tensor] = {}
        for layer_name, layer in self._static_forward_context.items():
            kv_cache = getattr(layer, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor):
                continue
            if getattr(layer, "kv_sharing_target_layer_name", None) is not None:
                continue
            result[layer_name] = kv_cache
        return result

    def register_layer_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise RuntimeError("PCP page-push received no tensor KV caches to register")

        # static_forward_context insertion order is model execution order.
        layer_names = tuple(kv_caches)
        registrations = [
            self._peer.register_tensor(kv_caches[name]) for name in layer_names
        ]
        if self._layer_names:
            if layer_names != self._layer_names:
                raise RuntimeError(
                    "PCP page-push layer order changed after registration: "
                    f"old={self._layer_names}, new={layer_names}"
                )
            for layer_id, registration in enumerate(registrations):
                if self._layer_memory[layer_id].base_addr != registration.base_addr:
                    raise RuntimeError(
                        "PCP page-push layer cache address changed: "
                        f"layer={layer_names[layer_id]}"
                    )
        else:
            self._layer_names = layer_names
            self._layer_memory = registrations
            self._layer_id_by_ptr = {
                region.base_addr: layer_id
                for layer_id, region in enumerate(registrations)
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
                self._validate_history_replicas_locked()
        if active:
            self._start_progress_thread()
            self._progress_wakeup.set()

    def register_current_layer(self, kv_cache: torch.Tensor) -> int:
        with self._progress_lock:
            active = self._plan is not None
        if not active:
            raise RuntimeError("page-push layer registration requires an active step")
        if not self._layer_names:
            caches = self._discover_bound_layer_caches()
            if not caches:
                raise RuntimeError("PCP page-push requires stable bound layer KV caches")
            self.register_layer_caches(caches)
        layer_id = self._layer_id_by_ptr.get(kv_cache.data_ptr())
        if layer_id is None:
            raise RuntimeError(
                "PCP page-push cache is not part of stable layer registration"
            )
        self._check_progress_error()
        return layer_id

    def _validate_history_replicas_locked(self) -> None:
        plan = self._plan
        if plan is None or not self._layer_memory:
            return
        for layer_id in range(len(self._layer_memory)):
            for source_rank in plan.historical_source_ranks(self.rank):
                route = plan.history_transfer_route(self.rank, source_rank)
                visible = self._persistent_visible_blocks.get(
                    (layer_id, source_rank), set()
                )
                missing = [
                    block_id
                    for block_id in route.destination_block_ids
                    if block_id not in visible
                ]
                if missing:
                    raise RuntimeError(
                        "PCP page-push historical replica is missing; READ fallback "
                        "is disabled: "
                        f"epoch={self._epoch}, rank={self.rank}, "
                        f"layer={self._layer_label(layer_id)}, "
                        f"source_rank={source_rank}, missing_blocks={missing[:8]}, "
                        f"missing_count={len(missing)}"
                    )
                pcp_nvtx_mark(
                    "pcp.page_push_history_hit",
                    e=self._epoch,
                    l=self._layer_label(layer_id),
                    src=source_rank,
                    dst=self.rank,
                    pages=route.num_pages,
                )

    def configure_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        rank_offsets: Sequence[int],
    ) -> None:
        """Compile finalized-page REPLICA routes once per scheduler step."""
        with self._progress_lock:
            plan = self._plan
            if plan is None:
                raise RuntimeError("page-push slot mapping requires an active plan")
            if self._slot_mapping_configured:
                return
            epoch = self._epoch
            block_size = plan.block_size

        offsets = tuple(int(value) for value in rank_offsets)
        if len(offsets) != self.world_size + 1:
            raise ValueError(
                "PCP page-push rank offsets must have world_size + 1 entries"
            )
        if offsets[0] != 0 or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError(f"invalid PCP rank offsets: {offsets}")
        total_rows = offsets[-1]
        if slot_mapping.shape[0] < total_rows:
            raise ValueError(
                "PCP page-push slot mapping is shorter than rank-major rows: "
                f"slots={slot_mapping.shape[0]}, rows={total_rows}"
            )

        slots = slot_mapping[:total_rows].detach()
        if slots.device.type != "cpu":
            slots = slots.to(device="cpu")
        slots_np = np.asarray(slots, dtype=np.int64)

        finalized_by_source: list[tuple[int, ...]] = []
        for source_rank in range(self.world_size):
            start, stop = offsets[source_rank], offsets[source_rank + 1]
            values = slots_np[start:stop]
            values = values[values >= 0]
            page_ends = values[values % block_size == block_size - 1]
            finalized_by_source.append(
                tuple(dict.fromkeys(int(value) for value in (page_ends // block_size)))
            )

        replica_routes: dict[tuple[int, int], PCPPageRoute] = {}
        assert plan is not None
        for source_rank, finalized_blocks in enumerate(finalized_by_source):
            if not finalized_blocks:
                continue
            for destination_rank in range(self.world_size):
                if destination_rank == source_rank:
                    continue
                current_blocks: set[int] = set()
                if plan.requires_current_source(destination_rank, source_rank):
                    current_blocks.update(
                        plan.current_transfer_route(
                            destination_rank, source_rank
                        ).source_block_ids
                    )
                replica_blocks = tuple(
                    block_id
                    for block_id in finalized_blocks
                    if block_id not in current_blocks
                )
                if replica_blocks:
                    replica_routes[(destination_rank, source_rank)] = PCPPageRoute(
                        destination_rank=destination_rank,
                        source_rank=source_rank,
                        destination_block_ids=replica_blocks,
                        source_block_ids=replica_blocks,
                    )

        with self._completion_cv:
            if self._plan is None or self._epoch != epoch:
                raise RuntimeError("PCP page-push step changed while compiling slots")
            if self._slot_mapping_configured:
                return
            self._replica_routes = replica_routes
            self._slot_mapping_configured = True
            self._process_deferred_notifications_locked()
            self._completion_cv.notify_all()
        self._progress_wakeup.set()
        pcp_nvtx_mark(
            "pcp.page_push_replica_plan",
            e=epoch,
            rank=self.rank,
            routes=len(replica_routes),
            pages=sum(route.num_pages for route in replica_routes.values()),
        )

    def _route_locked(self, key: _RouteKey, *, incoming: bool) -> PCPPageRoute:
        assert self._plan is not None
        _, peer_rank, kind = key
        destination_rank = self.rank if incoming else peer_rank
        source_rank = peer_rank if incoming else self.rank
        if kind == "current":
            return self._plan.current_transfer_route(destination_rank, source_rank)
        route = self._replica_routes.get((destination_rank, source_rank))
        if route is None:
            raise RuntimeError(
                "missing PCP replica route: "
                f"source={source_rank}, destination={destination_rank}"
            )
        return route

    def _queue_outgoing_locked(self, key: _RouteKey) -> None:
        if key in self._outgoing_done or key in self._queued_outgoing:
            return
        if any(key in batch.keys for batch in self._inflight.values()):
            return
        self._queued_outgoing.add(key)
        (self._current_waiting if key[2] == "current" else self._replica_waiting).append(
            key
        )

    def publish_ready(self, layer_id: int) -> None:
        with self._progress_lock:
            plan = self._plan
            if plan is None:
                return
            if not self._slot_mapping_configured:
                raise RuntimeError(
                    "PCP page-push layer became ready before slot plan configuration"
                )
            has_current = bool(plan.consumer_ranks(self.rank))
            has_replica = any(source_rank == self.rank for _, source_rank in self._replica_routes)
            if not has_current and not has_replica:
                return
        if not 0 <= layer_id < len(self._layer_memory):
            raise ValueError(f"invalid page-push layer id: {layer_id}")
        event = self._ready_events[layer_id]
        if event is not None:
            event.record(torch.cuda.current_stream(self.device))
        with self._progress_lock:
            if self._plan is not None:
                self._pending_ready.append(_PendingReady(layer_id, event))
        self._start_progress_thread()
        self._progress_wakeup.set()
        self._check_progress_error()

    def _publish_completed_ready_locked(self) -> None:
        assert self._plan is not None
        while self._pending_ready:
            pending = self._pending_ready[0]
            if pending.event is not None and not pending.event.query():
                break
            self._pending_ready.popleft()
            layer_id = pending.layer_id
            for destination_rank in self._plan.consumer_ranks(self.rank):
                self._queue_outgoing_locked((layer_id, destination_rank, "current"))
            for destination_rank, source_rank in self._replica_routes:
                if source_rank == self.rank:
                    self._queue_outgoing_locked((layer_id, destination_rank, "replica"))
            pcp_nvtx_mark(
                "pcp.page_push_local_ready",
                e=self._epoch,
                l=self._layer_label(layer_id),
                src=self.rank,
            )

    def _take_batch_locked(self, queue: deque[_RouteKey]) -> tuple[_RouteKey, ...]:
        first = queue.popleft()
        selected = [first]
        if self.max_batch_layers == 1 or not queue:
            return (first,)

        destination_rank = first[1]
        remaining: deque[_RouteKey] = deque()
        while queue:
            key = queue.popleft()
            if len(selected) < self.max_batch_layers and key[1] == destination_rank:
                selected.append(key)
            else:
                remaining.append(key)
        queue.extend(remaining)
        return tuple(selected)

    def _start_batch_locked(self, keys: tuple[_RouteKey, ...]) -> None:
        destination_rank = keys[0][1]
        kind = keys[0][2]
        writes: list[NixlWrite] = []
        num_pages = 0
        for key in keys:
            if key[1] != destination_rank or key[2] != kind:
                raise AssertionError("PCP WRITE batch mixed destination or route kind")
            layer_id = key[0]
            route = self._route_locked(key, incoming=False)
            self._queued_outgoing.discard(key)
            if route.num_pages == 0:
                self._outgoing_done.add(key)
                continue
            writes.append(
                NixlWrite(
                    local_region_id=layer_id,
                    remote_region_id=layer_id,
                    local_block_ids=route.source_block_array,
                    remote_block_ids=route.destination_block_array,
                )
            )
            num_pages += route.num_pages
        if not writes:
            return

        with pcp_nvtx_range(
            "pcp.page_push_write_submit",
            e=self._epoch,
            src=self.rank,
            dst=destination_rank,
            layers=len(writes),
            pages=num_pages,
            kind=kind,
        ):
            handle = self._peer.submit_write_batch(
                destination_rank=destination_rank,
                writes=writes,
            )
        self._inflight[handle] = _InflightWriteBatch(
            keys=keys,
            destination_rank=destination_rank,
            kind=kind,
            handle=handle,
            num_pages=num_pages,
        )

    def _send_visible_locked(self, batch: _InflightWriteBatch) -> None:
        layers = ",".join(str(key[0]) for key in batch.keys)
        payload = (
            f"{self._NOTIF_PREFIX}:{self._epoch}:{self.rank}:"
            f"{batch.kind}:{layers}"
        ).encode()
        self._peer.send_notification(batch.destination_rank, payload)
        pcp_nvtx_mark(
            "pcp.page_push_write_done",
            e=self._epoch,
            src=self.rank,
            dst=batch.destination_rank,
            layers=len(batch.keys),
            pages=batch.num_pages,
            kind=batch.kind,
        )

    def _accept_visible_locked(
        self,
        *,
        layer_id: int,
        source_rank: int,
        kind: _RouteKind,
    ) -> None:
        assert self._plan is not None
        if not 0 <= layer_id < len(self._layer_memory):
            raise RuntimeError(f"invalid PCP visible layer id: {layer_id}")
        key: _RouteKey = (layer_id, source_rank, kind)
        if key in self._incoming_done:
            return
        if kind == "current":
            if not self._plan.requires_current_source(self.rank, source_rank):
                raise RuntimeError(
                    "unexpected PCP CURRENT visibility source: "
                    f"source={source_rank}, destination={self.rank}"
                )
        elif not self._slot_mapping_configured:
            raise RuntimeError("replica visibility arrived before slot plan")

        route = self._route_locked(key, incoming=True)
        self._persistent_visible_blocks.setdefault((layer_id, source_rank), set()).update(
            route.destination_block_ids
        )
        self._incoming_done.add(key)
        pcp_nvtx_mark(
            "pcp.page_push_visible",
            e=self._epoch,
            l=self._layer_label(layer_id),
            src=source_rank,
            dst=self.rank,
            pages=route.num_pages,
            kind=kind,
        )

    def _parse_notification_locked(self, raw: bytes) -> bool:
        try:
            prefix, epoch, source_rank, kind, layers = raw.decode().split(":")
            epoch_i = int(epoch)
            source_i = int(source_rank)
            layer_ids = tuple(int(layer) for layer in layers.split(",") if layer)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"invalid PCP page-push NIXL notification: {raw!r}"
            ) from exc
        if prefix != self._NOTIF_PREFIX:
            return True
        if epoch_i != self._epoch:
            raise RuntimeError(
                "PCP page-push visibility epoch mismatch: "
                f"got={epoch_i}, expected={self._epoch}"
            )
        if kind not in ("current", "replica") or not layer_ids:
            raise RuntimeError(f"invalid PCP visibility notification: {raw!r}")
        if kind == "replica" and not self._slot_mapping_configured:
            return False
        for layer_id in layer_ids:
            self._accept_visible_locked(
                layer_id=layer_id,
                source_rank=source_i,
                kind=kind,  # type: ignore[arg-type]
            )
        self._completion_cv.notify_all()
        return True

    def _process_deferred_notifications_locked(self) -> None:
        if not self._deferred_notifications:
            return
        deferred = tuple(self._deferred_notifications)
        self._deferred_notifications.clear()
        for raw in deferred:
            if not self._parse_notification_locked(raw):
                self._deferred_notifications.append(raw)

    def _poll_notifications_locked(self) -> None:
        for raw in self._peer.iter_notifications():
            if not self._parse_notification_locked(raw):
                self._deferred_notifications.append(raw)
        if self._slot_mapping_configured:
            self._process_deferred_notifications_locked()

    def _schedule_writes_locked(self) -> None:
        while self._current_waiting and len(self._inflight) < self.max_inflight_writes:
            self._start_batch_locked(self._take_batch_locked(self._current_waiting))
        if self._current_waiting:
            return

        # Reserve one handle for a CURRENT route that becomes ready while
        # background replication is using the transport.
        replica_limit = max(1, self.max_inflight_writes - 1)
        replica_inflight = sum(
            batch.kind == "replica" for batch in self._inflight.values()
        )
        while (
            self._replica_waiting
            and len(self._inflight) < self.max_inflight_writes
            and replica_inflight < replica_limit
        ):
            self._start_batch_locked(self._take_batch_locked(self._replica_waiting))
            replica_inflight += 1

    def _progress_once(self) -> None:
        with self._completion_cv:
            if self._plan is None:
                return
            self._publish_completed_ready_locked()
            self._poll_notifications_locked()
            for handle, batch in list(self._inflight.items()):
                state = self._peer.check_transfer(handle)
                if state == "PROC":
                    continue
                del self._inflight[handle]
                if state != "DONE":
                    raise RuntimeError(
                        "PCP page-push NIXL WRITE failed: "
                        f"state={state}, destination_rank={batch.destination_rank}, "
                        f"kind={batch.kind}, layers={[key[0] for key in batch.keys]}"
                    )
                self._outgoing_done.update(batch.keys)
                self._send_visible_locked(batch)
            self._schedule_writes_locked()
            self._completion_cv.notify_all()

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
            with self._completion_cv:
                self._progress_error = exc
                self._completion_cv.notify_all()
            self._progress_stop.set()
            self._progress_wakeup.set()

    def _start_progress_thread(self) -> None:
        if self._progress_thread is not None or not self._layer_memory:
            return
        self._peer.ensure_peers()
        self._progress_stop.clear()
        self._progress_thread = threading.Thread(
            target=self._progress_loop,
            name=f"pcp-page-push-r{self.rank}",
            daemon=True,
        )
        self._progress_thread.start()

    def _check_progress_error(self) -> None:
        if self._progress_error is not None:
            raise RuntimeError(
                "PCP page-push progress engine failed"
            ) from self._progress_error

    def _layer_visible_locked(self, layer_id: int) -> bool:
        plan = self._plan
        return plan is None or all(
            (layer_id, source_rank, "current") in self._incoming_done
            for source_rank in plan.current_source_ranks(self.rank)
        )

    def layer_visible(self, layer_id: int) -> bool:
        with self._progress_lock:
            return self._layer_visible_locked(layer_id)

    def wait_layer(self, layer_id: int) -> None:
        self._progress_wakeup.set()
        while True:
            self._check_progress_error()
            with self._completion_cv:
                if self._layer_visible_locked(layer_id):
                    return
                self._completion_cv.wait(timeout=0.1)

    def _expected_outgoing_locked(self) -> set[_RouteKey]:
        assert self._plan is not None
        expected: set[_RouteKey] = set()
        for layer_id in range(len(self._layer_memory)):
            expected.update(
                (layer_id, destination_rank, "current")
                for destination_rank in self._plan.consumer_ranks(self.rank)
            )
            expected.update(
                (layer_id, destination_rank, "replica")
                for destination_rank, source_rank in self._replica_routes
                if source_rank == self.rank
            )
        return expected

    def _expected_incoming_locked(self) -> set[_RouteKey]:
        assert self._plan is not None
        expected: set[_RouteKey] = set()
        for layer_id in range(len(self._layer_memory)):
            expected.update(
                (layer_id, source_rank, "current")
                for source_rank in self._plan.current_source_ranks(self.rank)
            )
            expected.update(
                (layer_id, source_rank, "replica")
                for destination_rank, source_rank in self._replica_routes
                if destination_rank == self.rank
            )
        return expected

    def _finish_step_if_ready(self) -> bool:
        with self._completion_cv:
            if self._step_finished:
                return True
            if self._plan is not None and self._layer_memory:
                if not self._slot_mapping_configured:
                    raise RuntimeError(
                        "PCP page-push step ended before slot mapping was configured"
                    )
                if (
                    self._pending_ready
                    or self._current_waiting
                    or self._replica_waiting
                    or self._inflight
                    or not self._expected_outgoing_locked().issubset(self._outgoing_done)
                    or not self._expected_incoming_locked().issubset(self._incoming_done)
                    or self._deferred_notifications
                ):
                    return False
            self._plan = None
            self._clear_step_state_locked()
            self._step_finished = True
            self._completion_cv.notify_all()
            return True

    def finish_step(self) -> None:
        if self._step_finished:
            return
        with pcp_nvtx_range("pcp.page_push_restore_drain", e=self._epoch, rank=self.rank):
            while not self._finish_step_if_ready():
                self._check_progress_error()
                if self._progress_thread is None:
                    self._progress_once()
                else:
                    self._progress_wakeup.set()
                with self._completion_cv:
                    self._completion_cv.wait(timeout=self._POLL_INTERVAL_S)
        self._check_progress_error()
        pcp_nvtx_mark("pcp.page_push_step_end", e=self._epoch, rank=self.rank)
        self._progress_wakeup.set()

    def close(self) -> None:
        self.finish_step()
        self._progress_stop.set()
        self._progress_wakeup.set()
        if self._progress_thread is not None:
            self._progress_thread.join()
            self._progress_thread = None
        self._check_progress_error()


__all__ = ["PCPPagePushTransport"]

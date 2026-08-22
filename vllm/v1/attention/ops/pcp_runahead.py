# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step transport state for PCP causal-prefix runahead."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.parallel_state import Handle, get_pcp_group
from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan, PCPPagePullTransport
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_mark, pcp_nvtx_range


@dataclass
class _PendingSend:
    works: list[Handle]
    tensors: tuple[torch.Tensor, ...]

    def completed(self) -> bool:
        return all(work.is_completed() for work in self.works)

    def wait(self) -> None:
        with pcp_nvtx_range("pcp.send_wait"):
            for work in self.works:
                work.wait()


class PCPRunaheadRuntime:
    """Per-process state for one logical PCP communicator.

    One-segment-per-rank bindings are compiled into the primary PCP communicator
    member order, so ``rank`` is the logical causal segment index for tensor
    transports. Repeated logical ownership is page-pull-only and lives in
    ``PCPPagePlan``.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        max_inflight_sends: int = 4,
        max_inflight_reads: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
        pcp_group: Any | None = None,
    ) -> None:
        if max_inflight_sends <= 0:
            raise ValueError("max_inflight_sends must be positive")
        if max_inflight_reads <= 0:
            raise ValueError("max_inflight_reads must be positive")
        self.world_size = pcp_world_size
        self.rank = pcp_rank
        self.device = device
        self.max_inflight_sends = max_inflight_sends
        self.max_inflight_reads = max_inflight_reads
        self.nixl_backends = nixl_backends
        self.group = pcp_group

        # Runahead is created while model initialization owns the vLLM config
        # context. Retain only the mutable static-forward-context reference that
        # page-pull actually needs. KV-cache tensors are bound later on the same
        # layer objects, so the reference must not be copied.
        vllm_config = get_current_vllm_config_or_none()
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
            if vllm_config is not None
            else None
        )

        self.active = False
        self.transport: str | None = None
        self.rows_per_rank: tuple[int, ...] = ()
        self.rank_offsets: tuple[int, ...] = ()
        self._pending_sends: deque[_PendingSend] = deque()
        self._page_pull: PCPPagePullTransport | None = None
        self._pending_page_layers: dict[int, int] = {}
        self._epoch = 0

    def _group(self):
        return self.group if self.group is not None else get_pcp_group()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def local_rows(self) -> int:
        return self.rows_per_rank[self.rank] if self.rows_per_rank else 0

    @property
    def prefix_rows(self) -> int:
        return self.rank_offsets[self.rank] if self.rank_offsets else 0

    @property
    def visible_rows(self) -> int:
        return self.rank_offsets[self.rank + 1] if self.rank_offsets else 0

    @property
    def total_rows(self) -> int:
        return self.rank_offsets[-1] if self.rank_offsets else 0

    @property
    def prev_rank(self) -> int | None:
        return self.rank - 1 if self.rank > 0 else None

    @property
    def next_rank(self) -> int | None:
        return self.rank + 1 if self.rank + 1 < self.world_size else None

    def begin_step(
        self,
        rows_per_rank: Sequence[int],
        *,
        transport: str = "prefix_p2p",
    ) -> None:
        self.flush()
        if transport not in (
            "full_kv_collective",
            "prefix_p2p",
            "direct_p2p",
            "page_pull",
        ):
            raise ValueError(f"unsupported PCP step transport: {transport!r}")
        rows = tuple(int(value) for value in rows_per_rank)
        if len(rows) != self.world_size:
            raise ValueError(
                "runahead PCP rows must match PCP world size: "
                f"rows={rows}, world_size={self.world_size}"
            )
        if any(value <= 0 for value in rows):
            raise ValueError(f"runahead PCP requires positive rows per rank: {rows}")

        offsets = [0]
        for value in rows:
            offsets.append(offsets[-1] + value)
        self.rows_per_rank = rows
        self.rank_offsets = tuple(offsets)
        self.transport = transport
        self.active = True
        self._epoch += 1
        pcp_nvtx_mark(
            "pcp.runahead_step_begin",
            e=self._epoch,
            rank=self.rank,
            transport=transport,
            local_rows=self.local_rows,
            total_rows=self.total_rows,
        )

    def configure_page_plan(self, plan: PCPPagePlan) -> None:
        if not self.active or self.transport != "page_pull":
            raise RuntimeError("page plan requires an active page_pull step")
        if plan.world_size != self.world_size:
            raise ValueError(
                f"page plan world size {plan.world_size} != runtime {self.world_size}"
            )
        if self._page_pull is None:
            self._page_pull = PCPPagePullTransport(
                world_size=self.world_size,
                rank=self.rank,
                device=self.device,
                max_inflight_reads=self.max_inflight_reads,
                nixl_backends=self.nixl_backends,
                pcp_group=self._group(),
                static_forward_context=self._static_forward_context,
            )
        self._page_pull.configure_step(epoch=self._epoch, plan=plan)

    def disable_step(self) -> None:
        self.flush()
        if self._page_pull is not None:
            self._page_pull.disable_step()
        if self.active:
            pcp_nvtx_mark(
                "pcp.runahead_step_end",
                e=self._epoch,
                rank=self.rank,
                transport=self.transport,
            )
        self.active = False
        self.transport = None
        self.rows_per_rank = ()
        self.rank_offsets = ()
        self._pending_page_layers.clear()

    def rank_local_slot_mapping(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        if not self.rows_per_rank:
            return slot_mapping
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "rank-major slot mapping is shorter than configured PCP rows: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )
        start = self.rank_offsets[self.rank]
        stop = self.rank_offsets[self.rank + 1]
        return slot_mapping[start:stop]

    def exchange_cache_inputs(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Exchange rank-local cache rows using the active tensor transport."""
        if self.transport == "full_kv_collective":
            return self.exchange_full(tensors, slot_mapping)
        if self.transport == "prefix_p2p":
            return self.exchange_prefix(tensors, slot_mapping)
        if self.transport == "direct_p2p":
            return self.exchange_direct(tensors, slot_mapping)
        if self.transport == "page_pull":
            raise RuntimeError(
                "page_pull uses KV-cache page transport, not tensor exchange"
            )
        raise RuntimeError(f"unsupported active PCP transport: {self.transport!r}")

    def exchange_full(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Gather all rank-local rows, preserving compact rank-major order."""
        if not self.active or self.transport != "full_kv_collective":
            raise RuntimeError(
                "exchange_full requires full_kv_collective, "
                f"got {self.transport!r}"
            )
        self._validate_group()
        if not tensors or any(t.shape[0] != self.local_rows for t in tensors):
            raise ValueError(
                "full collective PCP expects configured local rows: "
                f"rank={self.rank}, expected={self.local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "rank-major slot mapping is shorter than configured PCP rows: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )

        group = self._group()
        contiguous = tuple(tensor.contiguous() for tensor in tensors)
        sizes = list(self.rows_per_rank)
        if len(set(sizes)) == 1:
            gathered = tuple(group.all_gather(tensor, dim=0) for tensor in contiguous)
        elif len(contiguous) == 1:
            gathered = (group.all_gatherv(contiguous[0], dim=0, sizes=sizes),)
        else:
            gathered = tuple(group.all_gatherv(list(contiguous), dim=0, sizes=sizes))
        return gathered, slot_mapping[: self.total_rows]

    def _validate_group(self) -> None:
        group = self._group()
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise RuntimeError(
                "runahead PCP logical process-group membership changed after initialization"
            )

    def _drain_sends(self) -> None:
        while self._pending_sends and self._pending_sends[0].completed():
            self._pending_sends.popleft().wait()

    def _bound_pending_sends(self) -> None:
        self._drain_sends()
        while len(self._pending_sends) > self.max_inflight_sends:
            self._pending_sends.popleft().wait()

    def _p2p(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        peer: int,
        recv: bool,
    ) -> list[Handle]:
        group = self._group()
        if not 0 <= peer < group.world_size:
            raise ValueError(f"invalid PCP peer rank: {peer}")
        global_peer = group.ranks[peer]
        op = dist.irecv if recv else dist.isend
        ops = [
            dist.P2POp(op, tensor, global_peer, group=group.device_group)
            for tensor in tensors
        ]
        works = dist.batch_isend_irecv(ops)
        if not recv:
            for tensor in tensors:
                if tensor.is_cuda:
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
        return works

    def exchange_prefix(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if not self.active:
            return tensors, slot_mapping
        if self.transport != "prefix_p2p":
            raise RuntimeError(
                f"exchange_prefix requires prefix_p2p, got {self.transport!r}"
            )
        self._validate_group()
        if not tensors or any(t.shape[0] != self.local_rows for t in tensors):
            raise ValueError(
                "runahead PCP expects configured local rows: "
                f"rank={self.rank}, rows={self.local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )

        if self.prev_rank is None:
            visible = tuple(tensor.contiguous() for tensor in tensors)
        else:
            visible = tuple(
                tensor.new_empty((self.visible_rows, *tensor.shape[1:]))
                for tensor in tensors
            )
            recv_views = tuple(tensor[: self.prefix_rows] for tensor in visible)
            works = self._p2p(recv_views, peer=self.prev_rank, recv=True)
            with pcp_nvtx_range(
                "pcp.prefix_recv_wait",
                e=self._epoch,
                rank=self.rank,
                src=self.prev_rank,
                rows=self.prefix_rows,
            ):
                for work in works:
                    work.wait()
            for output, local in zip(visible, tensors, strict=True):
                output[self.prefix_rows :].copy_(local)

        visible_slots = slot_mapping[: self.visible_rows]
        if self.next_rank is not None:
            works = self._p2p(visible, peer=self.next_rank, recv=False)
            pcp_nvtx_mark(
                "pcp.prefix_send",
                e=self._epoch,
                src=self.rank,
                dst=self.next_rank,
                rows=self.visible_rows,
            )
            self._pending_sends.append(_PendingSend(works, visible))
            self._bound_pending_sends()
        return visible, visible_slots

    def exchange_direct(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if not self.active or self.transport != "direct_p2p":
            raise RuntimeError(
                f"exchange_direct requires direct_p2p, got {self.transport!r}"
            )
        self._validate_group()
        if any(t.shape[0] != self.local_rows for t in tensors):
            raise ValueError(
                "direct PCP expects configured local rows: "
                f"rank={self.rank}, expected={self.local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )

        visible = tuple(
            tensor.new_empty((self.visible_rows, *tensor.shape[1:]))
            for tensor in tensors
        )
        local_start = self.rank_offsets[self.rank]
        local_stop = self.rank_offsets[self.rank + 1]
        for output, local in zip(visible, tensors, strict=True):
            output[local_start:local_stop].copy_(local)

        recv_works: list[Handle] = []
        for source_rank in range(self.rank):
            start = self.rank_offsets[source_rank]
            stop = self.rank_offsets[source_rank + 1]
            views = tuple(output[start:stop] for output in visible)
            recv_works.extend(self._p2p(views, peer=source_rank, recv=True))
            pcp_nvtx_mark(
                "pcp.direct_recv_submit",
                e=self._epoch,
                src=source_rank,
                dst=self.rank,
                rows=stop - start,
            )

        for destination_rank in range(self.rank + 1, self.world_size):
            works = self._p2p(tensors, peer=destination_rank, recv=False)
            pcp_nvtx_mark(
                "pcp.direct_send",
                e=self._epoch,
                src=self.rank,
                dst=destination_rank,
                rows=self.local_rows,
            )
            self._pending_sends.append(_PendingSend(works, tensors))
        self._bound_pending_sends()

        with pcp_nvtx_range(
            "pcp.direct_recv_wait",
            e=self._epoch,
            rank=self.rank,
            sources=self.rank,
            rows=self.prefix_rows,
        ):
            for work in recv_works:
                work.wait()
        return visible, slot_mapping[: self.visible_rows]

    def page_pull_prepare_layer(self, kv_cache: torch.Tensor) -> None:
        if self.transport != "page_pull" or self._page_pull is None:
            raise RuntimeError("page-pull layer requested before page plan setup")
        ptr = kv_cache.data_ptr()
        if ptr in self._pending_page_layers:
            raise RuntimeError("page-pull layer cache update was prepared twice")
        self._pending_page_layers[ptr] = self._page_pull.register_current_layer(kv_cache)

    def page_pull_after_cache_write(self, kv_cache: torch.Tensor) -> None:
        if self.transport != "page_pull" or self._page_pull is None:
            return
        ptr = kv_cache.data_ptr()
        layer_id = self._pending_page_layers.pop(ptr, None)
        if layer_id is None:
            raise RuntimeError("page-pull native cache write has no prepared layer")
        self._page_pull.publish_ready(layer_id)
        with pcp_nvtx_range(
            "pcp.page_pull_wait",
            e=self._epoch,
            rank=self.rank,
            layer_id=layer_id,
        ):
            self._page_pull.wait_layer(layer_id)

    def flush(self) -> None:
        with pcp_nvtx_range("pcp.flush", e=self._epoch, rank=self.rank):
            while self._pending_sends:
                self._pending_sends.popleft().wait()
            if self._pending_page_layers:
                raise RuntimeError(
                    "page-pull step ended before native KV cache writes completed: "
                    f"pending={len(self._pending_page_layers)}"
                )
            if self._page_pull is not None and self._page_pull.enabled:
                self._page_pull.finish_step()


_RUNTIME: PCPRunaheadRuntime | None = None


def register_pcp_runahead_runtime(runtime: PCPRunaheadRuntime | None) -> None:
    global _RUNTIME
    if _RUNTIME is not None and _RUNTIME is not runtime:
        _RUNTIME.flush()
    _RUNTIME = runtime


def get_pcp_runahead_runtime() -> PCPRunaheadRuntime | None:
    if _RUNTIME is None or not _RUNTIME.active:
        return None
    return _RUNTIME

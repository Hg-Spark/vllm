# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step transport state for experimental PCP causal-prefix runahead."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import Handle, get_pcp_group
from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan, PCPPagePullTransport
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range


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
    """Per-process state for logical-segment PCP transport.

    Logical segment order defines causal dependency. Physical PCP rank defines
    process-group membership and device placement. ``segment_to_rank`` bridges
    the two coordinate systems. Tensor transports currently require a rank
    permutation; ``page_pull`` additionally supports repeated physical owners.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        max_inflight_sends: int = 4,
        max_inflight_reads: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
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
        self.active = False
        self.transport: str | None = None
        self.rows_per_rank: tuple[int, ...] = ()
        self.rank_offsets: tuple[int, ...] = ()
        self.segment_to_rank: tuple[int, ...] = tuple(range(pcp_world_size))
        self.segments_by_rank: tuple[tuple[int, ...], ...] = tuple(
            (rank,) for rank in range(pcp_world_size)
        )
        self.rank_to_segment: tuple[int, ...] = tuple(range(pcp_world_size))
        self.segment_rows: tuple[int, ...] = ()
        self.segment_offsets: tuple[int, ...] = ()
        self._pending_sends: deque[_PendingSend] = deque()
        self._page_pull: PCPPagePullTransport | None = None
        self._epoch = 0

    @property
    def mapping_is_permutation(self) -> bool:
        return len(self.segment_to_rank) == self.world_size and sorted(
            self.segment_to_rank
        ) == list(range(self.world_size))

    @property
    def owned_segments(self) -> tuple[int, ...]:
        return self.segments_by_rank[self.rank]

    @property
    def segment_idx(self) -> int:
        owned = self.owned_segments
        if len(owned) != 1:
            raise RuntimeError(
                "single segment_idx is undefined when one rank owns multiple "
                f"logical segments: rank={self.rank}, segments={owned}"
            )
        return owned[0]

    @property
    def local_rows(self) -> int:
        return self.rows_per_rank[self.rank] if self.rows_per_rank else 0

    @property
    def prefix_rows(self) -> int:
        return self.segment_offsets[self.segment_idx] if self.segment_offsets else 0

    @property
    def visible_rows(self) -> int:
        if not self.segment_offsets:
            return 0
        max_segment = max(self.owned_segments)
        return self.segment_offsets[max_segment + 1]

    @property
    def total_rows(self) -> int:
        return self.rank_offsets[-1] if self.rank_offsets else 0

    @property
    def prev_rank(self) -> int | None:
        if self.segment_idx == 0:
            return None
        return self.segment_to_rank[self.segment_idx - 1]

    @property
    def next_rank(self) -> int | None:
        if self.segment_idx + 1 >= len(self.segment_to_rank):
            return None
        return self.segment_to_rank[self.segment_idx + 1]

    def begin_step(
        self,
        rows_per_rank: Sequence[int],
        *,
        transport: str = "prefix_p2p",
        segment_to_rank: Sequence[int] | None = None,
        segment_rows: Sequence[int] | None = None,
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

        mapping = (
            tuple(range(self.world_size))
            if segment_to_rank is None
            else tuple(int(value) for value in segment_to_rank)
        )
        if not mapping or any(not 0 <= rank < self.world_size for rank in mapping):
            raise ValueError(
                "segment_to_rank contains an invalid PCP rank: "
                f"mapping={mapping}, world_size={self.world_size}"
            )
        segments_by_rank = tuple(
            tuple(
                segment_idx
                for segment_idx, owner in enumerate(mapping)
                if owner == rank
            )
            for rank in range(self.world_size)
        )
        if any(not segments for segments in segments_by_rank):
            raise ValueError(
                "every PCP rank must own at least one logical segment: "
                f"mapping={mapping}"
            )
        permutation = len(mapping) == self.world_size and sorted(mapping) == list(
            range(self.world_size)
        )
        if transport != "page_pull" and not permutation:
            raise ValueError(
                f"transport={transport} requires a logical-segment rank permutation"
            )

        if segment_rows is None:
            if not permutation:
                raise ValueError("repeated rank mapping requires explicit segment_rows")
            seg_rows = tuple(rows[rank] for rank in mapping)
        else:
            seg_rows = tuple(int(value) for value in segment_rows)
            if len(seg_rows) != len(mapping):
                raise ValueError(
                    "segment_rows must match logical segment count: "
                    f"rows={seg_rows}, mapping={mapping}"
                )
            if any(value < 0 for value in seg_rows):
                raise ValueError(f"segment_rows must be non-negative: {seg_rows}")
            if sum(seg_rows) != sum(rows):
                raise ValueError(
                    "segment rows and physical rows describe different token counts: "
                    f"segments={seg_rows}, ranks={rows}"
                )

        rank_offsets = [0]
        for value in rows:
            rank_offsets.append(rank_offsets[-1] + value)
        segment_offsets = [0]
        for value in seg_rows:
            segment_offsets.append(segment_offsets[-1] + value)

        rank_to_segment = [-1] * self.world_size
        if permutation:
            for segment_idx, rank in enumerate(mapping):
                rank_to_segment[rank] = segment_idx

        self.rows_per_rank = rows
        self.rank_offsets = tuple(rank_offsets)
        self.segment_to_rank = mapping
        self.segments_by_rank = segments_by_rank
        self.rank_to_segment = tuple(rank_to_segment)
        self.segment_rows = seg_rows
        self.segment_offsets = tuple(segment_offsets)
        self.transport = transport
        self.active = True
        self._epoch += 1

    def configure_page_plan(self, plan: PCPPagePlan) -> None:
        if not self.active or self.transport != "page_pull":
            raise RuntimeError("page plan requires an active page_pull step")
        if plan.segment_to_rank != self.segment_to_rank:
            raise ValueError(
                "page plan mapping differs from active PCP mapping: "
                f"plan={plan.segment_to_rank}, runtime={self.segment_to_rank}"
            )
        if self._page_pull is None:
            self._page_pull = PCPPagePullTransport(
                world_size=self.world_size,
                rank=self.rank,
                device=self.device,
                max_inflight_reads=self.max_inflight_reads,
                nixl_backends=self.nixl_backends,
            )
        self._page_pull.configure_step(epoch=self._epoch, plan=plan)

    def disable_step(self) -> None:
        self.flush()
        if self._page_pull is not None:
            self._page_pull.disable_step()
        self.active = False
        self.transport = None
        self.rows_per_rank = ()
        self.rank_offsets = ()
        self.segment_to_rank = tuple(range(self.world_size))
        self.segments_by_rank = tuple((rank,) for rank in range(self.world_size))
        self.rank_to_segment = tuple(range(self.world_size))
        self.segment_rows = ()
        self.segment_offsets = ()

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

    def rank_major_to_segment_major(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reorder one-segment-per-rank compact rows into logical order."""
        if not self.rows_per_rank:
            return tensor
        if tensor.shape[0] < self.total_rows:
            raise ValueError(
                "rank-major tensor is shorter than configured PCP rows: "
                f"rows={tensor.shape[0]}, expected={self.total_rows}"
            )
        if not self.mapping_is_permutation:
            raise RuntimeError(
                "rank_major_to_segment_major requires one logical segment per rank"
            )
        if self.segment_to_rank == tuple(range(self.world_size)):
            return tensor[: self.total_rows]

        pieces = [
            tensor[self.rank_offsets[rank] : self.rank_offsets[rank + 1]]
            for rank in self.segment_to_rank
        ]
        return torch.cat(pieces, dim=0)

    def _validate_group(self) -> None:
        group = get_pcp_group()
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise RuntimeError(
                "runahead PCP process-group membership changed after initialization"
            )

    def _drain_sends(self) -> None:
        while self._pending_sends and self._pending_sends[0].completed():
            self._pending_sends.popleft().wait()

    def _bound_pending_sends(self) -> None:
        self._drain_sends()
        while len(self._pending_sends) > self.max_inflight_sends:
            self._pending_sends.popleft().wait()

    @staticmethod
    def _p2p(
        tensors: tuple[torch.Tensor, ...],
        *,
        peer: int,
        recv: bool,
    ) -> list[Handle]:
        group = get_pcp_group()
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
        """Receive an accumulated causal prefix and forward it one hop."""
        if not self.active:
            return tensors, slot_mapping
        if self.transport != "prefix_p2p":
            raise RuntimeError(
                f"exchange_prefix requires prefix_p2p, got {self.transport!r}"
            )

        self._validate_group()
        local_rows = self.local_rows
        if not tensors:
            raise ValueError("runahead PCP requires at least one tensor")
        if any(tensor.shape[0] != local_rows for tensor in tensors):
            raise ValueError(
                "runahead PCP expects configured local rows: "
                f"rank={self.rank}, rows={local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "runahead PCP slot mapping is shorter than configured compact rows: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )

        if self.prev_rank is None:
            with pcp_nvtx_range("pcp.prefix_local_prepare"):
                visible = tuple(tensor.contiguous() for tensor in tensors)
        else:
            with pcp_nvtx_range("pcp.prefix_visible_alloc"):
                visible = tuple(
                    tensor.new_empty((self.visible_rows, *tensor.shape[1:]))
                    for tensor in tensors
                )
            recv_views = tuple(tensor[: self.prefix_rows] for tensor in visible)
            works = self._p2p(recv_views, peer=self.prev_rank, recv=True)
            with pcp_nvtx_range("pcp.prefix_recv_wait"):
                for work in works:
                    work.wait()
            with pcp_nvtx_range("pcp.prefix_local_append"):
                for output, local in zip(visible, tensors, strict=True):
                    output[self.prefix_rows :].copy_(local)

        logical_slots = self.rank_major_to_segment_major(slot_mapping)
        visible_slots = logical_slots[: self.visible_rows]
        if self.next_rank is not None:
            with pcp_nvtx_range("pcp.prefix_send_enqueue"):
                works = self._p2p(visible, peer=self.next_rank, recv=False)
            self._pending_sends.append(_PendingSend(works, visible))
            self._bound_pending_sends()
        return visible, visible_slots

    def exchange_direct(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Direct logical suffix fanout without accumulated-prefix relays."""
        if not self.active or self.transport != "direct_p2p":
            raise RuntimeError(
                f"exchange_direct requires direct_p2p, got {self.transport!r}"
            )
        self._validate_group()
        if not self.mapping_is_permutation:
            raise RuntimeError("direct_p2p currently requires a rank permutation")
        if any(tensor.shape[0] != self.local_rows for tensor in tensors):
            raise ValueError(
                "direct PCP expects configured local rows: "
                f"rank={self.rank}, expected={self.local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )

        segment_idx = self.segment_idx
        visible = tuple(
            tensor.new_empty((self.visible_rows, *tensor.shape[1:]))
            for tensor in tensors
        )
        local_start = self.segment_offsets[segment_idx]
        local_stop = self.segment_offsets[segment_idx + 1]
        for output, local in zip(visible, tensors, strict=True):
            output[local_start:local_stop].copy_(local)

        recv_works: list[Handle] = []
        with pcp_nvtx_range("pcp.direct_recv_enqueue"):
            for source_segment in range(segment_idx):
                source_rank = self.segment_to_rank[source_segment]
                start = self.segment_offsets[source_segment]
                stop = self.segment_offsets[source_segment + 1]
                views = tuple(output[start:stop] for output in visible)
                recv_works.extend(self._p2p(views, peer=source_rank, recv=True))

        with pcp_nvtx_range("pcp.direct_send_enqueue"):
            for destination_segment in range(segment_idx + 1, len(self.segment_to_rank)):
                destination_rank = self.segment_to_rank[destination_segment]
                works = self._p2p(tensors, peer=destination_rank, recv=False)
                self._pending_sends.append(_PendingSend(works, tensors))
            self._bound_pending_sends()

        with pcp_nvtx_range("pcp.direct_recv_wait"):
            for work in recv_works:
                work.wait()

        logical_slots = self.rank_major_to_segment_major(slot_mapping)
        return visible, logical_slots[: self.visible_rows]

    def page_pull_register_layer(self, kv_cache: torch.Tensor) -> int:
        if self.transport != "page_pull" or self._page_pull is None:
            raise RuntimeError("page-pull layer requested before page plan setup")
        return self._page_pull.register_current_layer(kv_cache)

    def page_pull_publish_and_wait(self, layer_ordinal: int) -> None:
        if self.transport != "page_pull" or self._page_pull is None:
            raise RuntimeError("page-pull wait requested before page plan setup")
        self._page_pull.publish_ready(layer_ordinal)
        self._page_pull.wait_layer(layer_ordinal)

    def flush(self) -> None:
        """Drain outstanding transport work."""
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_sends:
                self._pending_sends.popleft().wait()
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
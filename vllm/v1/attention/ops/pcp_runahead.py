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
    """Per-process step state and causal-prefix P2P runtime.

    Physical PCP rank owns device/process-group membership. Logical segment order
    owns token order and causal P2P adjacency. ``segment_to_rank`` bridges them.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        max_inflight_sends: int = 4,
    ) -> None:
        if max_inflight_sends <= 0:
            raise ValueError("max_inflight_sends must be positive")
        self.world_size = pcp_world_size
        self.rank = pcp_rank
        self.device = device
        self.max_inflight_sends = max_inflight_sends
        self.active = False
        self.transport: str | None = None
        self.rows_per_rank: tuple[int, ...] = ()
        self.rank_offsets: tuple[int, ...] = ()
        self.segment_to_rank: tuple[int, ...] = tuple(range(pcp_world_size))
        self.rank_to_segment: tuple[int, ...] = tuple(range(pcp_world_size))
        self.segment_offsets: tuple[int, ...] = ()
        self._pending_sends: deque[_PendingSend] = deque()

    @property
    def segment_idx(self) -> int:
        return self.rank_to_segment[self.rank]

    @property
    def local_rows(self) -> int:
        return self.rows_per_rank[self.rank] if self.rows_per_rank else 0

    @property
    def prefix_rows(self) -> int:
        return self.segment_offsets[self.segment_idx] if self.segment_offsets else 0

    @property
    def visible_rows(self) -> int:
        return (
            self.segment_offsets[self.segment_idx + 1]
            if self.segment_offsets
            else 0
        )

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
        if self.segment_idx + 1 >= self.world_size:
            return None
        return self.segment_to_rank[self.segment_idx + 1]

    def begin_step(
        self,
        rows_per_rank: Sequence[int],
        *,
        transport: str = "prefix_p2p",
        segment_to_rank: Sequence[int] | None = None,
    ) -> None:
        self.flush()
        if transport not in ("full_kv_collective", "prefix_p2p"):
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
        if len(mapping) != self.world_size or sorted(mapping) != list(
            range(self.world_size)
        ):
            raise ValueError(
                "segment_to_rank must be a permutation of PCP ranks: "
                f"mapping={mapping}, world_size={self.world_size}"
            )
        rank_to_segment = [0] * self.world_size
        for segment_idx, rank in enumerate(mapping):
            rank_to_segment[rank] = segment_idx

        rank_offsets = [0]
        for value in rows:
            rank_offsets.append(rank_offsets[-1] + value)
        segment_offsets = [0]
        for rank in mapping:
            segment_offsets.append(segment_offsets[-1] + rows[rank])

        self.rows_per_rank = rows
        self.rank_offsets = tuple(rank_offsets)
        self.segment_to_rank = mapping
        self.rank_to_segment = tuple(rank_to_segment)
        self.segment_offsets = tuple(segment_offsets)
        self.transport = transport
        self.active = True

    def disable_step(self) -> None:
        self.flush()
        self.active = False
        self.transport = None
        self.rows_per_rank = ()
        self.rank_offsets = ()
        self.segment_to_rank = tuple(range(self.world_size))
        self.rank_to_segment = tuple(range(self.world_size))
        self.segment_offsets = ()

    def rank_major_to_segment_major(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reorder a compact rank-major row tensor into causal segment order."""
        if not self.rows_per_rank:
            return tensor
        if tensor.shape[0] < self.total_rows:
            raise ValueError(
                "rank-major tensor is shorter than configured PCP rows: "
                f"rows={tensor.shape[0]}, expected={self.total_rows}"
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
        """Receive earlier logical segments, append local rows, and forward."""
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

    def flush(self) -> None:
        """Drain outstanding prefix sends. Persistent KV remains untouched."""
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_sends:
                self._pending_sends.popleft().wait()


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

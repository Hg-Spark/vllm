# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal-prefix runahead runtime for prefill context parallelism.

The runtime carries only the causal-visible K/V prefix between PCP ranks.
Persistent KV cache replication is intentionally omitted: after a runahead
prefill, each rank retains only the causal-visible cache image it produced.
"""

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
    """Per-process runtime for causal-prefix PCP runahead."""

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
    ) -> None:
        self.world_size = pcp_world_size
        self.rank = pcp_rank
        self.device = device
        self.active = False
        self.rows_per_rank: tuple[int, ...] = ()
        self.rank_offsets: tuple[int, ...] = ()
        self._pending_sends: deque[_PendingSend] = deque()

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

    def begin_step(self, rows_per_rank: Sequence[int]) -> None:
        self.flush()
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
        self.active = True

    def disable_step(self) -> None:
        self.flush()
        self.active = False
        self.rows_per_rank = ()
        self.rank_offsets = ()

    def _validate_group(self) -> None:
        group = get_pcp_group()
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise RuntimeError(
                "runahead PCP process-group membership changed after initialization"
            )

    def _drain_sends(self) -> None:
        while self._pending_sends and self._pending_sends[0].completed():
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
        """Receive ranks [0, r), append local rows, and forward to rank r+1."""
        if not self.active:
            return tensors, slot_mapping

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
        if slot_mapping.shape[0] < self.visible_rows:
            raise ValueError(
                "runahead PCP slot mapping is shorter than the causal-visible "
                f"prefix: slots={slot_mapping.shape[0]}, visible={self.visible_rows}"
            )

        if self.rank == 0:
            with pcp_nvtx_range("pcp.prefix_local_prepare"):
                visible = tuple(tensor.contiguous() for tensor in tensors)
        else:
            with pcp_nvtx_range("pcp.prefix_visible_alloc"):
                visible = tuple(
                    tensor.new_empty((self.visible_rows, *tensor.shape[1:]))
                    for tensor in tensors
                )
            recv_views = tuple(tensor[: self.prefix_rows] for tensor in visible)
            works = self._p2p(recv_views, peer=self.rank - 1, recv=True)
            with pcp_nvtx_range("pcp.prefix_recv_wait"):
                for work in works:
                    work.wait()
            with pcp_nvtx_range("pcp.prefix_local_append"):
                for output, local in zip(visible, tensors, strict=True):
                    output[self.prefix_rows :].copy_(local)

        visible_slots = slot_mapping[: self.visible_rows]
        if self.rank + 1 < self.world_size:
            works = self._p2p(visible, peer=self.rank + 1, recv=False)
            self._pending_sends.append(_PendingSend(works, visible))
            self._drain_sends()
        return visible, visible_slots

    def flush(self) -> None:
        """Drain outstanding prefix sends. Persistent KV remains sharded."""
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

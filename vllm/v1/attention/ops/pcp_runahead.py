# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental KV-runahead transport for prefill context parallelism.

The critical path forwards compact causal KV prefixes left-to-right with P2P
sends. A separate asynchronous PCP all-gather restores the replicated KV-cache
image used by later decode steps. Request eligibility and batch partition policy
live in ``RunaheadPCPManager``; this module only owns the per-layer runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from vllm.distributed.parallel_state import Handle, get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_transport import (
    all_gather_variable_into_tensor_async,
    batch_irecv_tensors,
    batch_isend_tensors,
)

CacheUpdate = Callable[[tuple[torch.Tensor, ...], torch.Tensor], None]


@dataclass
class _PendingSend:
    works: list[Handle]

    def completed(self) -> bool:
        return all(work.is_completed() for work in self.works)

    def wait(self) -> None:
        with pcp_nvtx_range("pcp.send_wait"):
            for work in self.works:
                work.wait()


@dataclass
class _PendingReplica:
    works: list[Handle]
    local_inputs: tuple[torch.Tensor, ...]
    gathered: tuple[torch.Tensor, ...]
    slot_mapping: torch.Tensor
    apply: CacheUpdate

    def completed(self) -> bool:
        return all(work.is_completed() for work in self.works)

    def finish(self) -> None:
        with pcp_nvtx_range("pcp.replica_commit"):
            with pcp_nvtx_range("pcp.replica_wait"):
                for work in self.works:
                    work.wait()
            with pcp_nvtx_range("pcp.replica_cache_update"):
                self.apply(self.gathered, self.slot_mapping)


class PCPRunaheadRuntime:
    """Per-process runtime for causal-prefix PCP runahead.

    The runtime consumes a known variable-width token slab from each PCP rank.
    It starts asynchronous compact full-cache replication, propagates the
    causal-visible prefix across PCP ranks, and commits completed replicas lazily
    with bounded backpressure. Higher-level topology and workload eligibility are
    governed by ``RunaheadPCPManager``.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        max_pending_replica_layers: int = 4,
    ) -> None:
        self.world_size = pcp_world_size
        self.rank = pcp_rank
        self.device = device
        self.max_pending_replica_layers = max_pending_replica_layers
        self.active = False
        self.rows_per_rank: tuple[int, ...] = ()
        self.rank_offsets: tuple[int, ...] = ()
        self._pending_sends: deque[_PendingSend] = deque()
        self._pending_replicas: deque[_PendingReplica] = deque()

    @property
    def local_rows(self) -> int:
        if not self.rows_per_rank:
            return 0
        return self.rows_per_rank[self.rank]

    @property
    def prefix_rows(self) -> int:
        if not self.rank_offsets:
            return 0
        return self.rank_offsets[self.rank]

    @property
    def visible_rows(self) -> int:
        if not self.rank_offsets:
            return 0
        return self.rank_offsets[self.rank + 1]

    @property
    def total_rows(self) -> int:
        if not self.rank_offsets:
            return 0
        return self.rank_offsets[-1]

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

    def _validate_groups(self) -> None:
        pcp_group = get_pcp_group()
        if pcp_group.world_size != self.world_size:
            raise RuntimeError(
                "runahead PCP world-size changed after runtime initialization"
            )
        if pcp_group.rank_in_group != self.rank:
            raise RuntimeError(
                "runahead PCP rank changed after runtime initialization"
            )

    def _drain_sends(self) -> None:
        while self._pending_sends and self._pending_sends[0].completed():
            self._pending_sends.popleft().wait()

    def _drain_replicas(self) -> None:
        while self._pending_replicas and self._pending_replicas[0].completed():
            self._pending_replicas.popleft().finish()

    def _apply_backpressure(self) -> None:
        with pcp_nvtx_range("pcp.replica_backpressure"):
            self._drain_sends()
            self._drain_replicas()
            while len(self._pending_replicas) >= self.max_pending_replica_layers:
                self._pending_replicas.popleft().finish()
                self._drain_replicas()

    def exchange_prefix(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return the local causal prefix and forward it to the next PCP rank.

        Rank r receives the compact prefix owned by ranks [0, r), appends its
        local variable-width slab, starts a nonblocking send of the aggregate to
        rank r+1, and returns after the receive dependency is satisfied.
        """
        if not self.active:
            return tensors, slot_mapping

        self._validate_groups()
        local_rows = self.local_rows
        if not tensors:
            raise ValueError("runahead PCP requires at least one tensor")
        if any(tensor.shape[0] != local_rows for tensor in tensors):
            raise ValueError(
                "runahead PCP expects the configured local row count: "
                f"rank={self.rank}, rows={local_rows}, "
                f"shapes={[tuple(t.shape) for t in tensors]}"
            )

        pcp_group = get_pcp_group()
        if self.rank == 0:
            with pcp_nvtx_range("pcp.prefix_local_prepare"):
                visible = tuple(tensor.contiguous() for tensor in tensors)
        else:
            prefix_rows = self.prefix_rows
            recv_tensors = tuple(
                tensor.new_empty((prefix_rows, *tensor.shape[1:]))
                for tensor in tensors
            )
            works = batch_irecv_tensors(pcp_group, recv_tensors, self.rank - 1)
            with pcp_nvtx_range("pcp.prefix_recv_wait"):
                for work in works:
                    work.wait()
            with pcp_nvtx_range("pcp.prefix_concat"):
                visible = tuple(
                    torch.cat((prefix, local.contiguous()), dim=0)
                    for prefix, local in zip(recv_tensors, tensors, strict=True)
                )

        visible_slot_mapping = slot_mapping[: self.visible_rows]

        if self.rank + 1 < self.world_size:
            works = batch_isend_tensors(pcp_group, visible, self.rank + 1)
            self._pending_sends.append(_PendingSend(works))
            self._drain_sends()

        return visible, visible_slot_mapping

    def enqueue_replication(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
    ) -> None:
        """Asynchronously materialize the compact full PCP cache image."""
        if not self.active:
            return

        self._validate_groups()
        self._apply_backpressure()

        local_rows = self.local_rows
        if any(tensor.shape[0] != local_rows for tensor in tensors):
            raise ValueError(
                "runahead PCP replication expects the configured local rows: "
                f"rank={self.rank}, rows={local_rows}"
            )
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "runahead PCP slot mapping is shorter than the compact layout: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )

        pcp_group = get_pcp_group()
        local_inputs: list[torch.Tensor] = []
        gathered: list[torch.Tensor] = []
        with pcp_nvtx_range("pcp.replica_buffer_prepare"):
            for tensor in tensors:
                local = tensor.contiguous()
                output = tensor.new_empty((self.total_rows, *tensor.shape[1:]))
                local_inputs.append(local)
                gathered.append(output)

        works: list[Handle] = []
        with pcp_nvtx_range("pcp.replica_allgather_enqueue"):
            for local, output in zip(local_inputs, gathered, strict=True):
                works.append(
                    all_gather_variable_into_tensor_async(
                        pcp_group,
                        output,
                        local,
                        self.rows_per_rank,
                    )
                )

        self._pending_replicas.append(
            _PendingReplica(
                works=works,
                local_inputs=tuple(local_inputs),
                gathered=tuple(gathered),
                slot_mapping=slot_mapping[: self.total_rows],
                apply=apply,
            )
        )

    def update_and_replicate(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
    ) -> None:
        """Launch full replication, then commit only the causal prefix locally."""
        with pcp_nvtx_range("pcp.runahead_kv_update"):
            with pcp_nvtx_range("pcp.replica_launch"):
                self.enqueue_replication(tensors, slot_mapping, apply)
            with pcp_nvtx_range("pcp.prefix_exchange"):
                visible, visible_slot_mapping = self.exchange_prefix(
                    tensors, slot_mapping
                )
            with pcp_nvtx_range("pcp.visible_cache_update"):
                apply(visible, visible_slot_mapping)
            self._drain_replicas()

    def flush(self) -> None:
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_replicas:
                self._pending_replicas.popleft().finish()
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

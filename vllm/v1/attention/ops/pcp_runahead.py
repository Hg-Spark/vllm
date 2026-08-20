# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental KV-runahead transport for prefill context parallelism.

The critical path forwards compact causal KV prefixes left-to-right with P2P
sends. Full replicated-cache repair is deferred until the forward boundary so
layer L repair traffic cannot queue ahead of layer L+1 prefix traffic on the
same PCP ProcessGroup. Request eligibility and batch partition policy live in
``RunaheadPCPManager``; this module owns the per-layer runtime and deferred
repair queue.
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
class _DeferredReplica:
    local_inputs: tuple[torch.Tensor, ...]
    slot_mapping: torch.Tensor
    apply: CacheUpdate


class PCPRunaheadRuntime:
    """Per-process runtime for causal-prefix PCP runahead.

    The runtime consumes a known variable-width token slab from each PCP rank.
    During transformer-layer execution it only propagates the causal-visible
    prefix and records enough state to repair the replicated cache later. Full
    PCP all-gathers are launched by ``flush`` after the model forward, which
    keeps them out of the cross-layer prefix critical path.

    Deferred repair intentionally retains the local current-step tensors for
    each participating layer until ``flush``. This is an MVP tradeoff: it proves
    the communication schedule without requiring paged-cache pack/unpack. A
    later implementation can source repair payloads from persistent KV cache.
    """

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
        self._deferred_replicas: deque[_DeferredReplica] = deque()

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

    @property
    def num_deferred_replica_layers(self) -> int:
        return len(self._deferred_replicas)

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

    def defer_replication(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
    ) -> None:
        """Record one layer's full-cache repair without launching communication."""
        if not self.active:
            return

        self._validate_groups()
        local_rows = self.local_rows
        if any(tensor.shape[0] != local_rows for tensor in tensors):
            raise ValueError(
                "runahead PCP deferred repair expects the configured local rows: "
                f"rank={self.rank}, rows={local_rows}"
            )
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "runahead PCP slot mapping is shorter than the compact layout: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )

        with pcp_nvtx_range("pcp.replica_defer"):
            self._deferred_replicas.append(
                _DeferredReplica(
                    local_inputs=tuple(tensor.contiguous() for tensor in tensors),
                    slot_mapping=slot_mapping[: self.total_rows],
                    apply=apply,
                )
            )

    def _repair_replica(self, replica: _DeferredReplica) -> None:
        pcp_group = get_pcp_group()
        gathered: list[torch.Tensor] = []
        works: list[Handle] = []

        with pcp_nvtx_range("pcp.replica_buffer_prepare"):
            for local in replica.local_inputs:
                gathered.append(
                    local.new_empty((self.total_rows, *local.shape[1:]))
                )

        with pcp_nvtx_range("pcp.replica_allgather_enqueue"):
            for local, output in zip(replica.local_inputs, gathered, strict=True):
                works.append(
                    all_gather_variable_into_tensor_async(
                        pcp_group,
                        output,
                        local,
                        self.rows_per_rank,
                    )
                )

        with pcp_nvtx_range("pcp.replica_wait"):
            for work in works:
                work.wait()

        with pcp_nvtx_range("pcp.replica_cache_update"):
            replica.apply(tuple(gathered), replica.slot_mapping)

    def update_and_replicate(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
    ) -> None:
        """Commit the causal prefix now and defer full replication to flush."""
        with pcp_nvtx_range("pcp.runahead_kv_update"):
            with pcp_nvtx_range("pcp.prefix_exchange"):
                visible, visible_slot_mapping = self.exchange_prefix(
                    tensors, slot_mapping
                )
            with pcp_nvtx_range("pcp.visible_cache_update"):
                apply(visible, visible_slot_mapping)
            self.defer_replication(tensors, slot_mapping, apply)

    def flush(self) -> None:
        """Complete P2P sends, then repair all deferred replicated cache images."""
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_sends:
                self._pending_sends.popleft().wait()
            while self._deferred_replicas:
                with pcp_nvtx_range("pcp.replica_commit"):
                    self._repair_replica(self._deferred_replicas.popleft())


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

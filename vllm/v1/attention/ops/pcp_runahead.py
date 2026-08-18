# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental KV-runahead transport for prefill context parallelism.

The critical path forwards the causal KV prefix left-to-right with P2P sends.
A separate asynchronous PCP all-gather keeps the existing replicated KV-cache
semantics for subsequent decode steps. Completed gathers are committed lazily
and a bounded queue provides backpressure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group

CacheUpdate = Callable[[tuple[torch.Tensor, ...], torch.Tensor], None]


@dataclass
class _PendingSend:
    works: list[Any]
    tensors: tuple[torch.Tensor, ...]

    def completed(self) -> bool:
        return all(work.is_completed() for work in self.works)

    def wait(self) -> None:
        for work in self.works:
            work.wait()


@dataclass
class _PendingReplica:
    works: list[Any]
    local_inputs: tuple[torch.Tensor, ...]
    gathered: tuple[torch.Tensor, ...]
    slot_mapping: torch.Tensor
    apply: CacheUpdate

    def completed(self) -> bool:
        return all(work.is_completed() for work in self.works)

    def finish(self) -> None:
        for work in self.works:
            work.wait()
        self.apply(self.gathered, self.slot_mapping)


class PCPRunaheadRuntime:
    """Per-process runtime for the experimental runahead PCP path.

    Scope is intentionally narrow:
      * one fresh full-prefill request;
      * equal contiguous PCP chunks (last rank may contain padding);
      * existing replicated PCP KV cache is restored asynchronously.

    Transport is scoped to the PCP process group. Higher-level topology support
    is still governed by ``RunaheadPCPManager.validate_config``.
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
        self.local_num_tokens_padded = 0
        self._pending_sends: deque[_PendingSend] = deque()
        self._pending_replicas: deque[_PendingReplica] = deque()

    def begin_step(self, local_num_tokens_padded: int) -> None:
        self.flush()
        if local_num_tokens_padded <= 0:
            raise ValueError("runahead PCP requires a positive padded token count")
        self.local_num_tokens_padded = local_num_tokens_padded
        self.active = True

    def disable_step(self) -> None:
        self.flush()
        self.active = False
        self.local_num_tokens_padded = 0

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

        Rank r receives r fixed-size padded chunks from rank r-1, appends its
        local chunk, starts a nonblocking send of the aggregate to rank r+1,
        and returns immediately after the receive dependency is satisfied.
        """
        if not self.active:
            return tensors, slot_mapping

        self._validate_groups()
        rows = self.local_num_tokens_padded
        if not tensors:
            raise ValueError("runahead PCP requires at least one tensor")
        if any(tensor.shape[0] != rows for tensor in tensors):
            raise ValueError(
                "runahead PCP expects fixed padded rows on every PCP rank: "
                f"rows={rows}, shapes={[tuple(t.shape) for t in tensors]}"
            )

        pcp_group = get_pcp_group()
        p2p_group = pcp_group.device_group
        if self.rank == 0:
            visible = tuple(tensor.contiguous() for tensor in tensors)
        else:
            prefix_rows = self.rank * rows
            recv_tensors = tuple(
                tensor.new_empty((prefix_rows, *tensor.shape[1:]))
                for tensor in tensors
            )
            src_rank = pcp_group.ranks[self.rank - 1]
            ops = [
                dist.P2POp(dist.irecv, recv_tensor, src_rank, group=p2p_group)
                for recv_tensor in recv_tensors
            ]
            works = dist.batch_isend_irecv(ops)
            for work in works:
                work.wait()
            visible = tuple(
                torch.cat((prefix, local.contiguous()), dim=0)
                for prefix, local in zip(recv_tensors, tensors, strict=True)
            )

        visible_rows = (self.rank + 1) * rows
        visible_slot_mapping = slot_mapping[:visible_rows]

        if self.rank + 1 < self.world_size:
            dst_rank = pcp_group.ranks[self.rank + 1]
            ops = [
                dist.P2POp(dist.isend, tensor, dst_rank, group=p2p_group)
                for tensor in visible
            ]
            works = dist.batch_isend_irecv(ops)
            # Hold the tensors until NCCL has finished reading from them.
            self._pending_sends.append(_PendingSend(works, visible))
            self._drain_sends()

        return visible, visible_slot_mapping

    def enqueue_replication(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
    ) -> None:
        """Asynchronously materialize the full PCP cache image.

        The result is not on the current-layer attention critical path. The
        bounded pending queue ensures memory cannot grow with model depth.
        """
        if not self.active:
            return

        self._validate_groups()
        self._apply_backpressure()

        rows = self.local_num_tokens_padded
        if any(tensor.shape[0] != rows for tensor in tensors):
            raise ValueError(
                "runahead PCP replication expects padded local inputs "
                f"with {rows} rows"
            )

        pcp_group = get_pcp_group().device_group
        local_inputs: list[torch.Tensor] = []
        gathered: list[torch.Tensor] = []
        works: list[Any] = []
        for tensor in tensors:
            local = tensor.contiguous()
            output = tensor.new_empty((self.world_size * rows, *tensor.shape[1:]))
            work = dist.all_gather_into_tensor(
                output,
                local,
                group=pcp_group,
                async_op=True,
            )
            local_inputs.append(local)
            gathered.append(output)
            works.append(work)

        self._pending_replicas.append(
            _PendingReplica(
                works=works,
                local_inputs=tuple(local_inputs),
                gathered=tuple(gathered),
                slot_mapping=slot_mapping,
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
        self.enqueue_replication(tensors, slot_mapping, apply)
        visible, visible_slot_mapping = self.exchange_prefix(tensors, slot_mapping)
        apply(visible, visible_slot_mapping)
        self._drain_replicas()

    def flush(self) -> None:
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

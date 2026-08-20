# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal-prefix runahead runtime for prefill context parallelism.

Layer execution only carries the causal-visible K/V prefix. Replicated-cache
repair is a forward-boundary operation sourced from persistent paged KV storage,
so no per-layer K/V activation is retained for repair.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import Handle, get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range

_REPAIR_CHUNK_BYTES = 64 * 1024 * 1024


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
    """Per-process runtime for compact causal-prefix PCP runahead."""

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
        self._cache_blocks: dict[int, torch.Tensor] = {}
        self._repair_block_ids: torch.Tensor | None = None

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
    def num_cache_block_views(self) -> int:
        return len(self._cache_blocks)

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
        self._repair_block_ids = None
        self._cache_blocks.clear()
        self.active = True

    def disable_step(self) -> None:
        self.flush()
        self.active = False
        self.rows_per_rank = ()
        self.rank_offsets = ()
        self._repair_block_ids = None

    def _validate_group(self) -> None:
        group = get_pcp_group()
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise RuntimeError(
                "runahead PCP process-group membership changed after initialization"
            )

    @staticmethod
    def _raw_cache_block_view(kv_cache: torch.Tensor) -> torch.Tensor:
        """View a block-major KV backing allocation as raw byte pages."""
        if kv_cache.ndim == 0 or kv_cache.shape[0] <= 0:
            raise ValueError(
                f"runahead PCP requires block-major KV cache: {kv_cache.shape}"
            )
        blocks = torch.empty(0, dtype=torch.uint8, device=kv_cache.device)
        blocks.set_(kv_cache.untyped_storage())
        num_blocks = int(kv_cache.shape[0])
        if blocks.numel() % num_blocks != 0:
            raise RuntimeError(
                "runahead PCP requires block-major KV backing storage: "
                f"bytes={blocks.numel()}, num_blocks={num_blocks}"
            )
        return blocks.view(num_blocks, -1)

    def register_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Register one persistent block-major KV backing allocation."""
        if not self.active:
            return
        ptr = kv_cache.untyped_storage().data_ptr()
        existing = self._cache_blocks.get(ptr)
        if existing is None:
            self._cache_blocks[ptr] = self._raw_cache_block_view(kv_cache)
            return
        if existing.shape[0] != kv_cache.shape[0]:
            raise RuntimeError(
                "runahead PCP found incompatible KV views sharing one backing "
                f"allocation: registered_blocks={existing.shape[0]}, "
                f"new_blocks={kv_cache.shape[0]}"
            )

    def set_repair_block_ids(self, block_ids: torch.Tensor | None) -> None:
        """Set the step-level set of kernel-cache blocks touched by this forward."""
        if not self.active or block_ids is None:
            self._repair_block_ids = None
            return
        self._repair_block_ids = block_ids.to(device=self.device, dtype=torch.long)

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
            recv_tensors = tuple(
                tensor.new_empty((self.prefix_rows, *tensor.shape[1:]))
                for tensor in tensors
            )
            works = self._p2p(recv_tensors, peer=self.rank - 1, recv=True)
            with pcp_nvtx_range("pcp.prefix_recv_wait"):
                for work in works:
                    work.wait()
            with pcp_nvtx_range("pcp.prefix_concat"):
                visible = tuple(
                    torch.cat((prefix, local.contiguous()), dim=0)
                    for prefix, local in zip(recv_tensors, tensors, strict=True)
                )

        visible_slots = slot_mapping[: self.visible_rows]
        if self.rank + 1 < self.world_size:
            works = self._p2p(visible, peer=self.rank + 1, recv=False)
            self._pending_sends.append(_PendingSend(works, visible))
            self._drain_sends()
        return visible, visible_slots

    def _repair_cache_blocks(self, cache_blocks: torch.Tensor) -> None:
        block_ids = self._repair_block_ids
        if block_ids is None or block_ids.numel() == 0:
            return

        block_ids = block_ids[
            (block_ids >= 0) & (block_ids < cache_blocks.shape[0])
        ]
        if block_ids.numel() == 0:
            return

        group = get_pcp_group()
        source_rank = self.world_size - 1
        bytes_per_block = int(cache_blocks.shape[1])
        blocks_per_chunk = max(1, _REPAIR_CHUNK_BYTES // bytes_per_block)

        for start in range(0, block_ids.numel(), blocks_per_chunk):
            chunk_ids = block_ids[start : start + blocks_per_chunk]
            with pcp_nvtx_range("pcp.replica_buffer_prepare"):
                if self.rank == source_rank:
                    payload = cache_blocks.index_select(0, chunk_ids).contiguous()
                else:
                    payload = torch.empty(
                        (chunk_ids.numel(), bytes_per_block),
                        dtype=torch.uint8,
                        device=cache_blocks.device,
                    )
            with pcp_nvtx_range("pcp.replica_broadcast"):
                group.broadcast(payload, src=source_rank)
            if self.rank != source_rank:
                with pcp_nvtx_range("pcp.replica_cache_update"):
                    cache_blocks.index_copy_(0, chunk_ids, payload)

    def flush(self) -> None:
        """Drain prefix sends and restore touched persistent KV pages."""
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_sends:
                self._pending_sends.popleft().wait()

            if self._repair_block_ids is not None and self._cache_blocks:
                with pcp_nvtx_range("pcp.replica_forward_boundary"):
                    get_pcp_group().barrier()
                for cache_blocks in self._cache_blocks.values():
                    with pcp_nvtx_range("pcp.replica_commit"):
                        self._repair_cache_blocks(cache_blocks)

            self._repair_block_ids = None
            self._cache_blocks.clear()


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

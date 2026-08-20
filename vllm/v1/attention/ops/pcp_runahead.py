# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental KV-runahead transport for prefill context parallelism.

The critical path forwards compact causal KV prefixes left-to-right with P2P
sends. Full replicated-cache repair is deferred until the forward boundary so
layer L repair traffic cannot queue ahead of layer L+1 prefix traffic on the
same PCP ProcessGroup.

Deferred repair is sourced from persistent paged KV cache storage. The runtime
records raw cache block views and the block IDs touched by the current step;
it does not retain per-layer K/V activation tensors until the end of forward.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from vllm.distributed.parallel_state import Handle, get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_transport import (
    batch_irecv_tensors,
    batch_isend_tensors,
)

CacheUpdate = Callable[[tuple[torch.Tensor, ...], torch.Tensor], None]

# Bound temporary memory used by one raw-page broadcast. Large packed KV
# backings can put many layers in one physical page, so chunk by bytes rather
# than by an arbitrary number of blocks.
_REPAIR_CHUNK_BYTES = 64 * 1024 * 1024


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
class _DeferredPagedRepair:
    """Persistent KV storage plus the physical cache blocks to repair."""

    cache_blocks: torch.Tensor  # uint8 [num_blocks, bytes_per_block]
    block_ids: torch.Tensor  # int64, device resident


class PCPRunaheadRuntime:
    """Per-process runtime for causal-prefix PCP runahead.

    The runtime consumes a known variable-width token slab from each PCP rank.
    During transformer-layer execution it only propagates the causal-visible
    prefix, commits that visible prefix to the persistent KV cache, and records
    which persistent cache blocks need replication later.

    At ``flush`` all outstanding prefix sends are drained, PCP ranks rendezvous
    through the CPU group, and the last PCP rank broadcasts the touched raw KV
    pages. The last rank is the cache authority because the causal prefix chain
    gives it every current-step KV row before its layer cache write. Broadcasting
    complete pages also makes rank boundaries inside one page safe.
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
        # Keyed by backing-storage data pointer. Packed cross-layer KV caches
        # therefore register once and can be repaired as one raw page image.
        self._deferred_repairs: dict[int, _DeferredPagedRepair] = {}

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
    def num_deferred_repair_buffers(self) -> int:
        return len(self._deferred_repairs)

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

    @staticmethod
    def _raw_cache_block_view(
        kv_cache: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """View a block-major KV backing allocation as raw byte pages.

        This mirrors the storage-level block view used by
        ``copy_kv_cache_blocks_inplace``. For packed cross-layer allocations,
        ``untyped_storage`` exposes the complete backing allocation and one raw
        row contains every layer packed into that physical cache block.
        """
        if num_blocks <= 0:
            raise ValueError(f"KV cache must expose positive block count: {num_blocks}")
        blocks = torch.empty(0, dtype=torch.uint8, device=kv_cache.device)
        blocks.set_(kv_cache.untyped_storage())
        if blocks.numel() % num_blocks != 0:
            raise RuntimeError(
                "runahead PCP requires block-major KV backing storage: "
                f"bytes={blocks.numel()}, blocks={num_blocks}"
            )
        return blocks.view(num_blocks, -1)

    def defer_paged_repair(
        self,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache_block_size: int,
    ) -> None:
        """Record raw paged-cache blocks touched by this runahead step.

        ``cache_block_size`` is the number of token slots represented by one
        physical block in ``kv_cache``. No K/V activation tensor is retained.
        """
        if not self.active:
            return
        self._validate_groups()
        if cache_block_size <= 0:
            raise ValueError(
                f"runahead PCP cache block size must be positive: {cache_block_size}"
            )
        if kv_cache.ndim == 0 or kv_cache.shape[0] <= 0:
            raise ValueError(
                f"runahead PCP requires a block-major KV cache view: {kv_cache.shape}"
            )
        if slot_mapping.shape[0] < self.total_rows:
            raise ValueError(
                "runahead PCP slot mapping is shorter than the compact layout: "
                f"slots={slot_mapping.shape[0]}, rows={self.total_rows}"
            )

        with pcp_nvtx_range("pcp.replica_defer"):
            slots = slot_mapping[: self.total_rows]
            valid_slots = slots[slots >= 0]
            if valid_slots.numel() == 0:
                return
            block_ids = torch.unique(
                torch.div(valid_slots, cache_block_size, rounding_mode="floor")
            ).to(dtype=torch.long)
            num_blocks = int(kv_cache.shape[0])
            if bool((block_ids >= num_blocks).any().item()):
                raise RuntimeError(
                    "runahead PCP slot mapping addresses past the KV cache block view: "
                    f"num_blocks={num_blocks}, max_block={int(block_ids.max().item())}"
                )

            storage_ptr = kv_cache.untyped_storage().data_ptr()
            cache_blocks = self._raw_cache_block_view(kv_cache, num_blocks)
            existing = self._deferred_repairs.get(storage_ptr)
            if existing is None:
                self._deferred_repairs[storage_ptr] = _DeferredPagedRepair(
                    cache_blocks=cache_blocks,
                    block_ids=block_ids,
                )
                return

            if existing.cache_blocks.shape != cache_blocks.shape:
                raise RuntimeError(
                    "runahead PCP found incompatible block views sharing one KV "
                    f"backing storage: {existing.cache_blocks.shape} vs "
                    f"{cache_blocks.shape}"
                )
            existing.block_ids = torch.unique(
                torch.cat((existing.block_ids, block_ids), dim=0)
            )

    def update_visible_and_defer_repair(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        apply: CacheUpdate,
        kv_cache: torch.Tensor,
        cache_block_size: int,
    ) -> None:
        """Commit the causal prefix and record paged-cache repair metadata."""
        with pcp_nvtx_range("pcp.runahead_kv_update"):
            with pcp_nvtx_range("pcp.prefix_exchange"):
                visible, visible_slot_mapping = self.exchange_prefix(
                    tensors, slot_mapping
                )
            with pcp_nvtx_range("pcp.visible_cache_update"):
                apply(visible, visible_slot_mapping)
            self.defer_paged_repair(kv_cache, slot_mapping, cache_block_size)

    def _repair_paged_cache(self, repair: _DeferredPagedRepair) -> None:
        pcp_group = get_pcp_group()
        source_rank = self.world_size - 1
        block_ids = repair.block_ids
        cache_blocks = repair.cache_blocks
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
                pcp_group.broadcast(payload, src=source_rank)

            if self.rank != source_rank:
                with pcp_nvtx_range("pcp.replica_cache_update"):
                    cache_blocks.index_copy_(0, chunk_ids, payload)

    def flush(self) -> None:
        """Drain prefix P2P and repair touched paged-KV blocks from last rank."""
        with pcp_nvtx_range("pcp.flush"):
            while self._pending_sends:
                self._pending_sends.popleft().wait()

            if self._deferred_repairs:
                # Every rank first leaves the layer-level device-group P2P
                # sequence. GroupCoordinator.barrier() intentionally uses the
                # CPU/Gloo group, so no repair collective can be inserted into
                # the NCCL stream while a later rank is still submitting prefix
                # P2P on that same device group.
                pcp_group = get_pcp_group()
                with pcp_nvtx_range("pcp.replica_forward_boundary"):
                    pcp_group.barrier()

            for repair in self._deferred_repairs.values():
                with pcp_nvtx_range("pcp.replica_commit"):
                    self._repair_paged_cache(repair)
            self._deferred_repairs.clear()


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable causal page-routing plans for PCP runahead."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PCPPagePlan:
    """Per-step ownership plus precompiled source/page routes."""

    segment_to_rank: tuple[int, ...]
    blocks_by_segment: tuple[tuple[int, ...], ...]
    block_size: int
    _required_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _required_source_sets_by_rank: tuple[frozenset[int], ...] = field(
        init=False, repr=False, compare=False
    )
    _consumers_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_ids_by_rank: tuple[tuple[tuple[int, ...], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_arrays_by_rank: tuple[tuple[np.ndarray, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _transfer_max_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.segment_to_rank:
            raise ValueError("page-pull plan requires at least one logical segment")
        if len(self.blocks_by_segment) != len(self.segment_to_rank):
            raise ValueError("page map must match logical segment count")
        if self.block_size <= 0:
            raise ValueError("page-pull block_size must be positive")
        if min(self.segment_to_rank) < 0:
            raise ValueError("page-pull physical ranks must be non-negative")

        world_size = self.world_size
        missing = set(range(world_size)) - set(self.segment_to_rank)
        if missing:
            raise ValueError(
                "page-pull segment map must cover every PCP rank; "
                f"missing={sorted(missing)}"
            )

        required_sources_by_rank: list[tuple[int, ...]] = []
        transfer_ids_by_rank: list[tuple[tuple[int, ...], ...]] = []
        transfer_arrays_by_rank: list[tuple[np.ndarray, ...]] = []
        transfer_max_by_rank: list[tuple[int, ...]] = []
        for rank in range(world_size):
            owned = [
                segment_idx
                for segment_idx, owner in enumerate(self.segment_to_rank)
                if owner == rank
            ]
            max_segment = owned[-1]
            source_blocks: list[list[int]] = [[] for _ in range(world_size)]
            source_order: list[int] = []
            seen_sources: set[int] = set()
            for segment_idx in range(max_segment):
                source_rank = self.segment_to_rank[segment_idx]
                if source_rank == rank:
                    continue
                if source_rank not in seen_sources:
                    seen_sources.add(source_rank)
                    source_order.append(source_rank)
                source_blocks[source_rank].extend(self.blocks_by_segment[segment_idx])

            ids_by_source = tuple(tuple(blocks) for blocks in source_blocks)
            arrays_by_source = tuple(
                np.asarray(blocks, dtype=np.int64) for blocks in ids_by_source
            )
            max_by_source = tuple(
                max(blocks) if blocks else -1 for blocks in ids_by_source
            )
            required_sources_by_rank.append(tuple(source_order))
            transfer_ids_by_rank.append(ids_by_source)
            transfer_arrays_by_rank.append(arrays_by_source)
            transfer_max_by_rank.append(max_by_source)

        required_sources = tuple(required_sources_by_rank)
        required_source_sets = tuple(frozenset(items) for items in required_sources)
        consumers = tuple(
            tuple(
                rank
                for rank in range(world_size)
                if rank != source_rank
                and source_rank in required_source_sets[rank]
            )
            for source_rank in range(world_size)
        )
        object.__setattr__(self, "_required_sources_by_rank", required_sources)
        object.__setattr__(
            self, "_required_source_sets_by_rank", required_source_sets
        )
        object.__setattr__(self, "_consumers_by_rank", consumers)
        object.__setattr__(self, "_transfer_ids_by_rank", tuple(transfer_ids_by_rank))
        object.__setattr__(
            self, "_transfer_arrays_by_rank", tuple(transfer_arrays_by_rank)
        )
        object.__setattr__(self, "_transfer_max_by_rank", tuple(transfer_max_by_rank))

    @property
    def num_segments(self) -> int:
        return len(self.segment_to_rank)

    @property
    def world_size(self) -> int:
        return max(self.segment_to_rank) + 1

    def owned_segments(self, rank: int) -> tuple[int, ...]:
        return tuple(
            segment_idx
            for segment_idx, owner in enumerate(self.segment_to_rank)
            if owner == rank
        )

    def max_owned_segment(self, rank: int) -> int:
        owned = self.owned_segments(rank)
        if not owned:
            raise ValueError(f"physical PCP rank {rank} owns no logical segment")
        return owned[-1]

    def required_segments(self, rank: int) -> tuple[int, ...]:
        max_segment = self.max_owned_segment(rank)
        return tuple(
            segment_idx
            for segment_idx in range(max_segment)
            if self.segment_to_rank[segment_idx] != rank
        )

    def required_source_ranks(self, rank: int) -> tuple[int, ...]:
        return self._required_sources_by_rank[rank]

    def requires_source(self, rank: int, source_rank: int) -> bool:
        return source_rank in self._required_source_sets_by_rank[rank]

    def consumer_ranks(self, source_rank: int) -> tuple[int, ...]:
        return self._consumers_by_rank[source_rank]

    def transfer_block_ids(
        self, destination_rank: int, source_rank: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        ids = self._transfer_ids_by_rank[destination_rank][source_rank]
        return ids, ids

    def transfer_block_arrays(
        self, destination_rank: int, source_rank: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        ids = self._transfer_arrays_by_rank[destination_rank][source_rank]
        return ids, ids, self._transfer_max_by_rank[destination_rank][source_rank]


__all__ = ["PCPPagePlan"]

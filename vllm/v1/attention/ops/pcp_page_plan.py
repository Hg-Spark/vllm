# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable causal page-routing plans for PCP runahead."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PCPPageRoute:
    """One precompiled page transfer between two PCP ranks."""

    destination_rank: int
    source_rank: int
    destination_block_ids: tuple[int, ...]
    source_block_ids: tuple[int, ...]
    destination_block_array: np.ndarray = field(init=False, repr=False, compare=False)
    source_block_array: np.ndarray = field(init=False, repr=False, compare=False)
    destination_max_block_id: int = field(init=False)
    source_max_block_id: int = field(init=False)

    def __post_init__(self) -> None:
        if len(self.destination_block_ids) != len(self.source_block_ids):
            raise ValueError("page route source/destination page counts differ")
        source = np.asarray(self.source_block_ids, dtype=np.int64)
        destination = (
            source
            if self.destination_block_ids == self.source_block_ids
            else np.asarray(self.destination_block_ids, dtype=np.int64)
        )
        object.__setattr__(self, "destination_block_array", destination)
        object.__setattr__(self, "source_block_array", source)
        object.__setattr__(
            self,
            "destination_max_block_id",
            max(self.destination_block_ids) if self.destination_block_ids else -1,
        )
        object.__setattr__(
            self,
            "source_max_block_id",
            max(self.source_block_ids) if self.source_block_ids else -1,
        )

    @property
    def num_pages(self) -> int:
        return len(self.source_block_ids)


@dataclass(frozen=True)
class PCPPagePlan:
    """Per-step ownership plus precompiled source/page routes.

    ``blocks_by_segment`` names the source-rank physical blocks that own each
    logical segment. ``destination_blocks_by_segment`` may override the local
    destination blocks when allocator identities differ across ranks. The
    current replicated scheduler allocation leaves it unset, preserving the
    zero-metadata identical-ID fast path.
    """

    segment_to_rank: tuple[int, ...]
    blocks_by_segment: tuple[tuple[int, ...], ...]
    block_size: int
    destination_blocks_by_segment: tuple[tuple[int, ...], ...] | None = None
    _required_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _required_source_sets_by_rank: tuple[frozenset[int], ...] = field(
        init=False, repr=False, compare=False
    )
    _consumers_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _routes_by_rank: tuple[tuple[PCPPageRoute | None, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.segment_to_rank:
            raise ValueError("page-pull plan requires at least one logical segment")
        if len(self.blocks_by_segment) != len(self.segment_to_rank):
            raise ValueError("page map must match logical segment count")
        destination_by_segment = self.destination_blocks_by_segment
        if destination_by_segment is None:
            destination_by_segment = self.blocks_by_segment
        if len(destination_by_segment) != len(self.segment_to_rank):
            raise ValueError("destination page map must match logical segment count")
        for segment_idx, (source_ids, destination_ids) in enumerate(
            zip(self.blocks_by_segment, destination_by_segment, strict=True)
        ):
            if len(source_ids) != len(destination_ids):
                raise ValueError(
                    "source/destination page counts differ for logical segment "
                    f"{segment_idx}: source={len(source_ids)}, "
                    f"destination={len(destination_ids)}"
                )
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
        routes_by_rank: list[tuple[PCPPageRoute | None, ...]] = []
        for destination_rank in range(world_size):
            owned = [
                segment_idx
                for segment_idx, owner in enumerate(self.segment_to_rank)
                if owner == destination_rank
            ]
            max_segment = owned[-1]
            source_blocks: list[list[int]] = [[] for _ in range(world_size)]
            destination_blocks: list[list[int]] = [[] for _ in range(world_size)]
            source_order: list[int] = []
            seen_sources: set[int] = set()
            for segment_idx in range(max_segment):
                source_rank = self.segment_to_rank[segment_idx]
                if source_rank == destination_rank:
                    continue
                if source_rank not in seen_sources:
                    seen_sources.add(source_rank)
                    source_order.append(source_rank)
                source_blocks[source_rank].extend(self.blocks_by_segment[segment_idx])
                destination_blocks[source_rank].extend(
                    destination_by_segment[segment_idx]
                )

            required_sources_by_rank.append(tuple(source_order))
            routes_by_rank.append(
                tuple(
                    PCPPageRoute(
                        destination_rank=destination_rank,
                        source_rank=source_rank,
                        destination_block_ids=tuple(destination_blocks[source_rank]),
                        source_block_ids=tuple(source_blocks[source_rank]),
                    )
                    if source_blocks[source_rank]
                    else None
                    for source_rank in range(world_size)
                )
            )

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
        object.__setattr__(self, "_routes_by_rank", tuple(routes_by_rank))

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

    def transfer_route(self, destination_rank: int, source_rank: int) -> PCPPageRoute:
        route = self._routes_by_rank[destination_rank][source_rank]
        if route is None:
            raise ValueError(
                "no page-pull route for "
                f"source={source_rank}, destination={destination_rank}"
            )
        return route

    def transfer_block_ids(
        self, destination_rank: int, source_rank: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        route = self.transfer_route(destination_rank, source_rank)
        return route.destination_block_ids, route.source_block_ids

    def transfer_block_arrays(
        self, destination_rank: int, source_rank: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compatibility accessor for identical source/destination mappings."""
        route = self.transfer_route(destination_rank, source_rank)
        if route.destination_max_block_id != route.source_max_block_id:
            raise RuntimeError(
                "independent source/destination mappings require transfer_route()"
            )
        return (
            route.destination_block_array,
            route.source_block_array,
            route.source_max_block_id,
        )


__all__ = ["PCPPagePlan", "PCPPageRoute"]

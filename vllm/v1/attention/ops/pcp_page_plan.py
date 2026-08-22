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


_RouteMatrix = tuple[tuple[PCPPageRoute | None, ...], ...]


@dataclass(frozen=True)
class PCPPagePlan:
    """Per-step page demand plan.

    The legacy constructor (``segment_to_rank`` + ``blocks_by_segment``)
    compiles fresh-prefill causal routes exactly as before. Chunked-prefill
    planning can instead provide explicit ``history_routes_by_rank`` and
    ``current_routes_by_rank`` matrices. Historical routes are immediately
    readable at step start; current routes wait for the producer's same-layer
    READY notification.
    """

    segment_to_rank: tuple[int, ...]
    blocks_by_segment: tuple[tuple[int, ...], ...]
    block_size: int
    destination_blocks_by_segment: tuple[tuple[int, ...], ...] | None = None
    history_routes_by_rank: _RouteMatrix | None = None
    current_routes_by_rank: _RouteMatrix | None = None
    explicit_world_size: int | None = None
    _world_size: int = field(init=False, repr=False, compare=False)
    _history_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _current_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _required_sources_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _required_source_sets_by_rank: tuple[frozenset[int], ...] = field(
        init=False, repr=False, compare=False
    )
    _current_source_sets_by_rank: tuple[frozenset[int], ...] = field(
        init=False, repr=False, compare=False
    )
    _consumers_by_rank: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _history_routes: _RouteMatrix = field(init=False, repr=False, compare=False)
    _current_routes: _RouteMatrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("page-pull block_size must be positive")
        if (
            self.history_routes_by_rank is not None
            or self.current_routes_by_rank is not None
        ):
            self._init_explicit_routes()
            return
        self._init_legacy_segments()

    @staticmethod
    def _validate_route_matrix(matrix: _RouteMatrix, world_size: int) -> None:
        if len(matrix) != world_size:
            raise ValueError("page route matrix must match PCP world size")
        for destination_rank, row in enumerate(matrix):
            if len(row) != world_size:
                raise ValueError("page route row must match PCP world size")
            for source_rank, route in enumerate(row):
                if route is None:
                    continue
                if route.destination_rank != destination_rank:
                    raise ValueError("page route destination rank does not match matrix")
                if route.source_rank != source_rank:
                    raise ValueError("page route source rank does not match matrix")

    def _set_route_state(
        self,
        *,
        world_size: int,
        history_routes: _RouteMatrix,
        current_routes: _RouteMatrix,
    ) -> None:
        history_sources = tuple(
            tuple(source for source, route in enumerate(row) if route is not None)
            for row in history_routes
        )
        current_sources = tuple(
            tuple(source for source, route in enumerate(row) if route is not None)
            for row in current_routes
        )
        required_sources = tuple(
            tuple(dict.fromkeys((*history_sources[rank], *current_sources[rank])))
            for rank in range(world_size)
        )
        required_sets = tuple(frozenset(items) for items in required_sources)
        current_sets = tuple(frozenset(items) for items in current_sources)
        consumers = tuple(
            tuple(
                rank
                for rank in range(world_size)
                if rank != source_rank and source_rank in current_sets[rank]
            )
            for source_rank in range(world_size)
        )
        object.__setattr__(self, "_world_size", world_size)
        object.__setattr__(self, "_history_routes", history_routes)
        object.__setattr__(self, "_current_routes", current_routes)
        object.__setattr__(self, "_history_sources_by_rank", history_sources)
        object.__setattr__(self, "_current_sources_by_rank", current_sources)
        object.__setattr__(self, "_required_sources_by_rank", required_sources)
        object.__setattr__(self, "_required_source_sets_by_rank", required_sets)
        object.__setattr__(self, "_current_source_sets_by_rank", current_sets)
        object.__setattr__(self, "_consumers_by_rank", consumers)

    def _init_explicit_routes(self) -> None:
        history = self.history_routes_by_rank
        current = self.current_routes_by_rank
        if history is None and current is None:
            raise AssertionError("unreachable")
        inferred = len(history) if history is not None else len(current or ())
        world_size = self.explicit_world_size or inferred
        if world_size <= 0:
            raise ValueError("explicit page plan requires a positive PCP world size")
        empty: _RouteMatrix = tuple(
            tuple(None for _ in range(world_size)) for _ in range(world_size)
        )
        history = history if history is not None else empty
        current = current if current is not None else empty
        self._validate_route_matrix(history, world_size)
        self._validate_route_matrix(current, world_size)
        self._set_route_state(
            world_size=world_size,
            history_routes=history,
            current_routes=current,
        )

    def _init_legacy_segments(self) -> None:
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
        if min(self.segment_to_rank) < 0:
            raise ValueError("page-pull physical ranks must be non-negative")

        world_size = max(self.segment_to_rank) + 1
        missing = set(range(world_size)) - set(self.segment_to_rank)
        if missing:
            raise ValueError(
                "page-pull segment map must cover every PCP rank; "
                f"missing={sorted(missing)}"
            )

        current_rows: list[tuple[PCPPageRoute | None, ...]] = []
        for destination_rank in range(world_size):
            owned = [
                segment_idx
                for segment_idx, owner in enumerate(self.segment_to_rank)
                if owner == destination_rank
            ]
            max_segment = owned[-1]
            source_blocks: list[list[int]] = [[] for _ in range(world_size)]
            destination_blocks: list[list[int]] = [[] for _ in range(world_size)]
            for segment_idx in range(max_segment):
                source_rank = self.segment_to_rank[segment_idx]
                if source_rank == destination_rank:
                    continue
                source_blocks[source_rank].extend(self.blocks_by_segment[segment_idx])
                destination_blocks[source_rank].extend(
                    destination_by_segment[segment_idx]
                )
            current_rows.append(
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

        empty_history: _RouteMatrix = tuple(
            tuple(None for _ in range(world_size)) for _ in range(world_size)
        )
        self._set_route_state(
            world_size=world_size,
            history_routes=empty_history,
            current_routes=tuple(current_rows),
        )

    @property
    def num_segments(self) -> int:
        return len(self.segment_to_rank)

    @property
    def world_size(self) -> int:
        return self._world_size

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

    def historical_source_ranks(self, rank: int) -> tuple[int, ...]:
        return self._history_sources_by_rank[rank]

    def current_source_ranks(self, rank: int) -> tuple[int, ...]:
        return self._current_sources_by_rank[rank]

    def required_source_ranks(self, rank: int) -> tuple[int, ...]:
        return self._required_sources_by_rank[rank]

    def requires_source(self, rank: int, source_rank: int) -> bool:
        return source_rank in self._required_source_sets_by_rank[rank]

    def requires_current_source(self, rank: int, source_rank: int) -> bool:
        return source_rank in self._current_source_sets_by_rank[rank]

    def consumer_ranks(self, source_rank: int) -> tuple[int, ...]:
        """Ranks that need this source's current-chunk pages after READY."""
        return self._consumers_by_rank[source_rank]

    def _route(
        self,
        matrix: _RouteMatrix,
        destination_rank: int,
        source_rank: int,
        kind: str,
    ) -> PCPPageRoute:
        route = matrix[destination_rank][source_rank]
        if route is None:
            raise ValueError(
                f"no {kind} page-pull route for "
                f"source={source_rank}, destination={destination_rank}"
            )
        return route

    def history_transfer_route(
        self, destination_rank: int, source_rank: int
    ) -> PCPPageRoute:
        return self._route(
            self._history_routes, destination_rank, source_rank, "historical"
        )

    def current_transfer_route(
        self, destination_rank: int, source_rank: int
    ) -> PCPPageRoute:
        return self._route(
            self._current_routes, destination_rank, source_rank, "current"
        )

    def transfer_route(self, destination_rank: int, source_rank: int) -> PCPPageRoute:
        """Compatibility accessor for fresh-prefill/current routes."""
        return self.current_transfer_route(destination_rank, source_rank)

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

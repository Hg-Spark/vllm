# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-side planning for sharing MLA history across PCP segments."""

from dataclasses import dataclass
from typing import Sequence

from vllm.utils.math_utils import round_down


@dataclass(frozen=True)
class PCPSharedContextMaterialization:
    """One historical KV range materialized once for its query slices."""

    source_row: int
    kv_start: int
    kv_length: int
    consumers: tuple[slice, ...]


def combine_pcp_consumer_token_slices(consumers: tuple[slice, ...]) -> slice | None:
    """Return one contiguous token slice for the supplied query slices."""
    if not consumers:
        return None
    slices = sorted(consumers, key=lambda item: item.start)
    start = slices[0].start
    stop = slices[0].stop
    for token_slice in slices[1:]:
        if token_slice.start != stop:
            return None
        stop = token_slice.stop
    return slice(start, stop)


def _split_materialization(
    *,
    source_row: int,
    kv_start: int,
    kv_end: int,
    consumers: tuple[slice, ...],
    row_budget: int,
    split_alignment: int,
) -> list[PCPSharedContextMaterialization]:
    assert 0 <= kv_start <= kv_end
    assert kv_start == 0 or kv_start % split_alignment == 0
    if kv_start == kv_end:
        return []

    aligned_budget = round_down(row_budget, split_alignment)
    if kv_end - kv_start > row_budget:
        assert aligned_budget > 0

    jobs: list[PCPSharedContextMaterialization] = []
    start = kv_start
    while kv_end - start > row_budget:
        jobs.append(
            PCPSharedContextMaterialization(
                source_row=source_row,
                kv_start=start,
                kv_length=aligned_budget,
                consumers=consumers,
            )
        )
        start += aligned_budget

    if start < kv_end:
        jobs.append(
            PCPSharedContextMaterialization(
                source_row=source_row,
                kv_start=start,
                kv_length=kv_end - start,
                consumers=consumers,
            )
        )
    return jobs


def plan_pcp_shared_context(
    *,
    source_row: int,
    context_lens: Sequence[int],
    query_start_locs: Sequence[int],
    row_budget: int,
    split_alignment: int,
) -> list[PCPSharedContextMaterialization] | None:
    """Plan shared historical KV materialization for one PCP=2 source pair."""
    if row_budget <= 0 or split_alignment <= 0:
        raise ValueError("row_budget and split_alignment must be positive")
    if source_row < 0:
        return None
    if len(context_lens) != 2 or len(query_start_locs) != 3:
        return None
    if any(length < 0 for length in context_lens):
        return None
    if not (query_start_locs[0] <= query_start_locs[1] <= query_start_locs[2]):
        return None
    if row_budget < split_alignment and max(context_lens, default=0) > row_budget:
        return None

    consumers = tuple(
        slice(query_start_locs[i], query_start_locs[i + 1]) for i in range(2)
    )
    low_idx, high_idx = sorted(range(2), key=lambda i: (context_lens[i], i))
    low_len = int(context_lens[low_idx])
    high_len = int(context_lens[high_idx])

    if low_len == high_len:
        return _split_materialization(
            source_row=source_row,
            kv_start=0,
            kv_end=low_len,
            consumers=(consumers[0], consumers[1]),
            row_budget=row_budget,
            split_alignment=split_alignment,
        )

    jobs: list[PCPSharedContextMaterialization] = []
    shared_end = round_down(low_len, split_alignment)
    if shared_end > 0:
        jobs.extend(
            _split_materialization(
                source_row=source_row,
                kv_start=0,
                kv_end=shared_end,
                consumers=(consumers[low_idx], consumers[high_idx]),
                row_budget=row_budget,
                split_alignment=split_alignment,
            )
        )

    if low_len > shared_end:
        jobs.extend(
            _split_materialization(
                source_row=source_row,
                kv_start=shared_end,
                kv_end=low_len,
                consumers=(consumers[low_idx],),
                row_budget=row_budget,
                split_alignment=split_alignment,
            )
        )

    if high_len > shared_end:
        jobs.extend(
            _split_materialization(
                source_row=source_row,
                kv_start=shared_end,
                kv_end=high_len,
                consumers=(consumers[high_idx],),
                row_budget=row_budget,
                split_alignment=split_alignment,
            )
        )
    return jobs

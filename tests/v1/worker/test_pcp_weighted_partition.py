# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pcp_weighted_partition import (
    WeightedContiguousPCPManager,
    WeightedDualChunkPCPManager,
    parse_weighted_contiguous_partition,
    parse_weighted_dual_chunk_partition,
    weighted_partition_lengths,
)


def _absolute_boundaries(
    lengths: tuple[int, ...], start_pos: int
) -> tuple[int, ...]:
    boundaries = []
    running = start_pos
    for length in lengths[:-1]:
        running += length
        boundaries.append(running)
    return tuple(boundaries)


def _layout_inputs(
    num_tokens: int = 4096,
    *,
    is_prefilling: bool = True,
    num_computed_tokens: int = 0,
) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([num_computed_tokens], dtype=np.int32),
        np.asarray([is_prefilling], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def _block_tables() -> SimpleNamespace:
    return SimpleNamespace(
        kernel_block_sizes=[128],
        num_kv_cache_groups=0,
    )


def _global_slices(segments) -> list[tuple[int, int]]:
    return sorted(
        (segment.global_batch_slice.start, segment.global_batch_slice.stop)
        for segment in segments
    )


def test_weighted_partition_uses_cumulative_page_rounding() -> None:
    lengths = weighted_partition_lengths(
        4096,
        (2.0, 1.0, 1.0, 2.0),
        alignment=128,
    )
    assert lengths == (1408, 640, 640, 1408)
    assert sum(lengths) == 4096
    assert all(
        boundary % 128 == 0 for boundary in _absolute_boundaries(lengths, 0)
    )


def test_weighted_partition_aligns_continued_prefill_absolute_positions() -> None:
    lengths = weighted_partition_lengths(
        4096,
        (1.0, 1.0, 1.0, 1.0),
        start_pos=32,
        alignment=128,
    )
    assert sum(lengths) == 4096
    assert all(
        boundary % 128 == 0
        for boundary in _absolute_boundaries(lengths, start_pos=32)
    )


def test_contiguous_parser_requires_explicit_impl_and_valid_weights() -> None:
    assert parse_weighted_contiguous_partition(
        {"pcp_partition": {"impl": "weighted_contiguous"}}, 2
    ) == (1.0, 1.0)
    assert parse_weighted_contiguous_partition(
        {
            "pcp_partition": {
                "impl": "weighted_contiguous",
                "weights": [1.25, 0.75],
            }
        },
        2,
    ) == (1.25, 0.75)

    with pytest.raises(ValueError, match="impl must be 'weighted_contiguous'"):
        parse_weighted_contiguous_partition(
            {"pcp_partition": {"weights": [1.0, 1.0]}},
            2,
        )
    with pytest.raises(ValueError, match="requires 2 positive values"):
        parse_weighted_contiguous_partition(
            {
                "pcp_partition": {
                    "impl": "weighted_contiguous",
                    "weights": [1.0],
                }
            },
            2,
        )
    with pytest.raises(ValueError, match="finite positive"):
        parse_weighted_contiguous_partition(
            {
                "pcp_partition": {
                    "impl": "weighted_contiguous",
                    "weights": [1.0, 0.0],
                }
            },
            2,
        )


def test_dual_chunk_parser_uses_pcp_rank_weights_directly() -> None:
    assert parse_weighted_dual_chunk_partition(
        {
            "pcp_partition": {
                "impl": "weighted_dual_chunk",
                "weights": [1.0, 2.0, 3.0, 4.0],
            }
        },
        4,
    ) == (1.0, 2.0, 3.0, 4.0)


def test_removed_logical_order_config_fails_fast() -> None:
    with pytest.raises(ValueError, match="unsupported pcp_partition keys"):
        parse_weighted_dual_chunk_partition(
            {
                "pcp_partition": {
                    "impl": "weighted_dual_chunk",
                    "logical_to_physical": [1, 0],
                }
            },
            2,
        )


def test_baseline_pcp_manager_keeps_dual_chunk_swap() -> None:
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs()

    assert _global_slices(manager._get_rank_segments(0, *args)) == [
        (0, 1024),
        (3072, 4096),
    ]
    assert _global_slices(manager._get_rank_segments(1, *args)) == [
        (1024, 2048),
        (2048, 3072),
    ]


def test_weighted_contiguous_uses_pcp_rank_as_causal_order() -> None:
    manager = WeightedContiguousPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        partition_weights=(2.0, 1.0),
    )
    args = _layout_inputs()

    rank0 = manager._get_rank_segments(0, *args)
    rank1 = manager._get_rank_segments(1, *args)

    assert len(rank0) == 1
    assert len(rank1) == 1
    assert rank0[0].global_batch_slice == slice(0, 2688)
    assert rank1[0].global_batch_slice == slice(2688, 4096)
    assert rank0[0].num_tokens == 2688
    assert rank1[0].num_tokens == 1408
    assert rank0[0].global_batch_slice.stop == rank1[0].global_batch_slice.start
    assert rank0[0].global_batch_slice.stop % 128 == 0


def test_equal_weight_contiguous_prefill_has_no_overlap_or_gap() -> None:
    manager = WeightedContiguousPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs()

    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(0, *args)
    ] == [slice(0, 2048)]
    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(1, *args)
    ] == [slice(2048, 4096)]


def test_short_contiguous_prefill_falls_back_to_token_alignment() -> None:
    manager = WeightedContiguousPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs(num_tokens=2)

    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(0, *args)
    ] == [slice(0, 1)]
    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(1, *args)
    ] == [slice(1, 2)]


def test_weighted_dual_chunk_matches_rank_ordered_weighted_layout() -> None:
    manager = WeightedDualChunkPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        partition_weights=(2.0, 1.0),
    )
    args = _layout_inputs()

    rank0 = manager._get_rank_segments(0, *args)
    rank1 = manager._get_rank_segments(1, *args)

    assert _global_slices(rank0) == [(0, 1408), (2688, 4096)]
    assert _global_slices(rank1) == [(1408, 2048), (2048, 2688)]
    assert sum(segment.num_tokens for segment in rank0) == 2816
    assert sum(segment.num_tokens for segment in rank1) == 1280
    assert all(boundary % 128 == 0 for boundary in (1408, 2048, 2688))


def test_dual_chunk_segment_weights_follow_pcp_rank_order() -> None:
    manager = WeightedDualChunkPCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        partition_weights=(1.0, 2.0, 3.0, 4.0),
    )
    assert manager._segment_weights == (
        1.0,
        2.0,
        3.0,
        4.0,
        4.0,
        3.0,
        2.0,
        1.0,
    )


def test_short_dual_chunk_prefill_falls_back_to_token_alignment() -> None:
    manager = WeightedDualChunkPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs(num_tokens=2)

    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(0, *args)
    ] == [slice(0, 1)]
    assert [
        segment.global_batch_slice
        for segment in manager._get_rank_segments(1, *args)
    ] == [slice(1, 2)]


def test_dual_chunk_decode_remains_replicated_on_pcp_ranks() -> None:
    manager = WeightedDualChunkPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs(num_tokens=1, is_prefilling=False)

    for pcp_rank in range(2):
        segments = manager._get_rank_segments(pcp_rank, *args)
        assert [segment.global_batch_slice for segment in segments] == [slice(0, 1)]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pcp_weighted_partition import (
    WeightedPCPManager,
    parse_pcp_partition_weights,
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


def test_partition_weights_parser_uses_top_level_key() -> None:
    assert parse_pcp_partition_weights({}, 2) == (1.0, 1.0)
    assert parse_pcp_partition_weights({"other_option": True}, 2) == (1.0, 1.0)
    assert parse_pcp_partition_weights(
        {"pcp_partition_weights": [1.25, 0.75]},
        2,
    ) == (1.25, 0.75)


def test_partition_weights_parser_validates_only_owned_value() -> None:
    assert parse_pcp_partition_weights(
        {
            "pcp_partition_weights": [1.0, 1.0],
            "future_option": {"enabled": True},
        },
        2,
    ) == (1.0, 1.0)

    with pytest.raises(ValueError, match="requires 2 positive values"):
        parse_pcp_partition_weights(
            {"pcp_partition_weights": [1.0]},
            2,
        )
    with pytest.raises(ValueError, match="finite positive"):
        parse_pcp_partition_weights(
            {"pcp_partition_weights": [1.0, 0.0]},
            2,
        )
    with pytest.raises(ValueError, match="must be numeric"):
        parse_pcp_partition_weights(
            {"pcp_partition_weights": [1.0, "bad"]},
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


def test_weighted_manager_uses_pcp_rank_as_causal_order() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        pcp_partition_weights=(2.0, 1.0),
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


def test_equal_weight_prefill_has_no_overlap_or_gap() -> None:
    manager = WeightedPCPManager(
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


def test_short_prefill_falls_back_to_token_alignment() -> None:
    manager = WeightedPCPManager(
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


def test_decode_is_owned_only_by_last_pcp_rank() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    args = _layout_inputs(num_tokens=1, is_prefilling=False)

    assert manager._get_rank_segments(0, *args) == []
    rank1_segments = manager._get_rank_segments(1, *args)
    assert [segment.global_batch_slice for segment in rank1_segments] == [slice(0, 1)]

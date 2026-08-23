# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pcp_manager import (
    PCPManager,
    _parse_partition_weights,
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


def test_weighted_partition_uses_cumulative_page_rounding() -> None:
    lengths = weighted_partition_lengths(
        4096,
        (2.0, 1.0, 1.0, 2.0),
        alignment=128,
    )
    assert lengths == (1408, 640, 640, 1408)
    assert sum(lengths) == 4096
    assert all(boundary % 128 == 0 for boundary in _absolute_boundaries(lengths, 0))


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


def test_partition_weights_default_and_validation() -> None:
    assert _parse_partition_weights(None, 2) == (1.0, 1.0)
    assert _parse_partition_weights({}, 2) == (1.0, 1.0)
    assert _parse_partition_weights(
        {"pcp_partition": {"weights": [1.25, 0.75]}}, 2
    ) == (1.25, 0.75)

    with pytest.raises(ValueError, match="requires 2 positive values"):
        _parse_partition_weights({"pcp_partition": {"weights": [1.0]}}, 2)
    with pytest.raises(ValueError, match="finite positive"):
        _parse_partition_weights(
            {"pcp_partition": {"weights": [1.0, 0.0]}}, 2
        )


def test_pcp_prefill_uses_one_contiguous_weighted_segment_per_rank() -> None:
    block_tables = SimpleNamespace(
        kernel_block_sizes=[128],
        num_kv_cache_groups=0,
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=block_tables,
        partition_weights=(2.0, 1.0),
    )

    num_scheduled_tokens = np.asarray([4096], dtype=np.int32)
    num_computed_tokens = np.asarray([0], dtype=np.int32)
    is_prefilling = np.asarray([True], dtype=np.bool_)
    query_start_loc = np.asarray([0, 4096], dtype=np.int32)

    rank0 = manager._get_rank_segments(
        0,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )
    rank1 = manager._get_rank_segments(
        1,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )

    assert len(rank0) == 1
    assert len(rank1) == 1
    assert rank0[0].global_batch_slice == slice(0, 2688)
    assert rank1[0].global_batch_slice == slice(2688, 4096)
    assert rank0[0].num_tokens == 2688
    assert rank1[0].num_tokens == 1408
    assert rank0[0].global_batch_slice.stop == rank1[0].global_batch_slice.start
    assert rank0[0].global_batch_slice.stop % 128 == 0


def test_equal_weight_contiguous_prefill_has_no_overlap_or_gap() -> None:
    block_tables = SimpleNamespace(
        kernel_block_sizes=[128],
        num_kv_cache_groups=0,
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=block_tables,
    )

    num_scheduled_tokens = np.asarray([4096], dtype=np.int32)
    num_computed_tokens = np.asarray([0], dtype=np.int32)
    is_prefilling = np.asarray([True], dtype=np.bool_)
    query_start_loc = np.asarray([0, 4096], dtype=np.int32)

    rank0 = manager._get_rank_segments(
        0,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )
    rank1 = manager._get_rank_segments(
        1,
        num_scheduled_tokens,
        num_computed_tokens,
        is_prefilling,
        query_start_loc,
    )

    assert [segment.global_batch_slice for segment in rank0] == [slice(0, 2048)]
    assert [segment.global_batch_slice for segment in rank1] == [slice(2048, 4096)]

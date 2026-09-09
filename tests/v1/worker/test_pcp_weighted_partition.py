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


def _block_tables() -> SimpleNamespace:
    return SimpleNamespace(kernel_block_sizes=[128], num_kv_cache_groups=0)


def _layout_inputs(
    num_tokens: int = 4096,
    *,
    is_prefilling: bool = True,
    num_computed_tokens: int = 0,
):
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([num_computed_tokens], dtype=np.int32),
        np.asarray([is_prefilling], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def _absolute_boundaries(lengths: tuple[int, ...], start_pos: int):
    running = start_pos
    result = []
    for length in lengths[:-1]:
        running += length
        result.append(running)
    return tuple(result)


def test_weighted_partition_uses_largest_remainders_without_alignment() -> None:
    assert weighted_partition_lengths(9, (2.0, 1.0)) == (6, 3)
    assert weighted_partition_lengths(10, (1.0, 1.0, 1.0)) == (4, 3, 3)


def test_weighted_partition_uses_cumulative_page_rounding() -> None:
    lengths = weighted_partition_lengths(
        4096, (2.0, 1.0, 1.0, 2.0), alignment=128
    )
    assert lengths == (1408, 640, 640, 1408)
    assert sum(lengths) == 4096
    assert all(boundary % 128 == 0 for boundary in _absolute_boundaries(lengths, 0))


def test_continued_prefill_uses_absolute_page_boundaries() -> None:
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


def test_partition_weights_parser() -> None:
    assert parse_pcp_partition_weights({}, 2) == (1.0, 1.0)
    assert parse_pcp_partition_weights(
        {"pcp_partition_weights": [1.25, 0.75]}, 2
    ) == (1.25, 0.75)
    with pytest.raises(ValueError, match="requires 2 positive values"):
        parse_pcp_partition_weights({"pcp_partition_weights": [1.0]}, 2)
    with pytest.raises(ValueError, match="finite positive"):
        parse_pcp_partition_weights({"pcp_partition_weights": [1.0, 0.0]}, 2)


def test_weighted_manager_builds_all_rank_segments_in_one_policy_pass() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        pcp_partition_weights=(2.0, 1.0),
    )
    segments = manager._get_segments_by_rank(*_layout_inputs())

    assert [segment.global_batch_slice for segment in segments[0]] == [slice(0, 2688)]
    assert [segment.global_batch_slice for segment in segments[1]] == [slice(2688, 4096)]


def test_short_prefill_falls_back_to_token_alignment() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    segments = manager._get_segments_by_rank(*_layout_inputs(2))
    assert [segment.global_batch_slice for segment in segments[0]] == [slice(0, 1)]
    assert [segment.global_batch_slice for segment in segments[1]] == [slice(1, 2)]


def test_weighted_decode_is_owned_by_last_rank() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    segments = manager._get_segments_by_rank(*_layout_inputs(3, is_prefilling=False))
    assert segments[0] == ()
    assert [segment.global_batch_slice for segment in segments[1]] == [slice(0, 3)]


def test_baseline_manager_remains_available() -> None:
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    assert type(manager) is PCPManager

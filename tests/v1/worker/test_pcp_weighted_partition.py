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


def _layout_inputs(num_tokens: int = 9, *, is_prefilling: bool = True):
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([is_prefilling], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def test_weighted_partition_uses_largest_remainders() -> None:
    assert weighted_partition_lengths(9, (2.0, 1.0)) == (6, 3)
    assert weighted_partition_lengths(10, (1.0, 1.0, 1.0)) == (4, 3, 3)


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

    assert [segment.global_batch_slice for segment in segments[0]] == [slice(0, 6)]
    assert [segment.global_batch_slice for segment in segments[1]] == [slice(6, 9)]


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

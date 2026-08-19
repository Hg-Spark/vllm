# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime
from vllm.v1.worker.gpu.pcp_manager import (
    RunaheadPCPManager,
    parse_runahead_load_weights,
    runahead_batch_eligible,
    weighted_partition_lengths,
)


def _manager(
    world_size: int,
    weights: tuple[float, ...] | None = None,
) -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = world_size
    manager._standard_attention_pcp = True
    manager._use_runahead_partition = True
    manager._load_weights = weights
    return manager


def test_runahead_partition_is_contiguous_and_complete() -> None:
    manager = _manager(4)
    expected = [(0, 3), (3, 6), (6, 9), (9, 10)]
    actual: list[tuple[int, int]] = []

    for rank in range(4):
        segments = manager._get_rank_segments(
            rank,
            np.asarray([10], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True]),
            np.asarray([0, 10], dtype=np.int32),
        )
        assert len(segments) == 1
        segment = segments[0]
        actual.append(
            (segment.global_batch_slice.start, segment.global_batch_slice.stop)
        )

    assert actual == expected


def test_runahead_partition_handles_exact_division() -> None:
    manager = _manager(4)
    lengths = []

    for rank in range(4):
        (segment,) = manager._get_rank_segments(
            rank,
            np.asarray([16], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True]),
            np.asarray([0, 16], dtype=np.int32),
        )
        lengths.append(segment.num_tokens)

    assert lengths == [4, 4, 4, 4]


def test_runahead_partition_supports_multiple_prefill_requests() -> None:
    manager = _manager(4)
    scheduled = np.asarray([8, 4], dtype=np.int32)
    computed = np.asarray([0, 16384], dtype=np.int32)
    is_prefilling = np.asarray([True, True])
    query_start = np.asarray([0, 8, 12], dtype=np.int32)

    expected = [
        [(0, 2), (8, 9)],
        [(2, 4), (9, 10)],
        [(4, 6), (10, 11)],
        [(6, 8), (11, 12)],
    ]

    actual: list[list[tuple[int, int]]] = []
    for rank in range(4):
        segments = manager._get_rank_segments(
            rank,
            scheduled,
            computed,
            is_prefilling,
            query_start,
        )
        actual.append(
            [
                (segment.global_batch_slice.start, segment.global_batch_slice.stop)
                for segment in segments
            ]
        )

    assert actual == expected


def test_weighted_partition_uses_normalized_load_weights() -> None:
    weights = (4.0, 2.5, 1.9, 1.6)
    assert weighted_partition_lengths(10_000, weights) == (4000, 2500, 1900, 1600)

    manager = _manager(4, weights)
    actual: list[tuple[int, int]] = []
    for rank in range(4):
        (segment,) = manager._get_rank_segments(
            rank,
            np.asarray([10_000], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True]),
            np.asarray([0, 10_000], dtype=np.int32),
        )
        actual.append(
            (segment.global_batch_slice.start, segment.global_batch_slice.stop)
        )

    assert actual == [
        (0, 4000),
        (4000, 6500),
        (6500, 8400),
        (8400, 10_000),
    ]


def test_weighted_partition_rounding_preserves_all_tokens() -> None:
    lengths = weighted_partition_lengths(10, (4.0, 2.5, 1.9, 1.6))
    assert lengths == (4, 2, 2, 2)
    assert sum(lengths) == 10


def test_weighted_partition_is_ignored_for_mla() -> None:
    manager = _manager(4, (4.0, 2.5, 1.9, 1.6))
    manager._standard_attention_pcp = False

    assert manager._weighted_lengths(10) == (3, 3, 3, 1)
    assert not manager._use_compact_layout()


def test_parse_runahead_load_weights() -> None:
    assert parse_runahead_load_weights("4,2.5,1.9,1.6", 4) == (
        4.0,
        2.5,
        1.9,
        1.6,
    )
    assert parse_runahead_load_weights(None, 4) is None
    assert parse_runahead_load_weights("", 4) is None

    with pytest.raises(ValueError, match="requires 4 values"):
        parse_runahead_load_weights("1,1,1", 4)
    with pytest.raises(ValueError, match="finite and positive"):
        parse_runahead_load_weights("1,1,0,1", 4)
    with pytest.raises(ValueError, match="numeric list"):
        parse_runahead_load_weights("1,foo,1,1", 4)


def test_variable_width_runtime_builds_compact_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4000, 2500, 1900, 1600))

    assert runtime.rows_per_rank == (4000, 2500, 1900, 1600)
    assert runtime.rank_offsets == (0, 4000, 6500, 8400, 10_000)
    assert runtime.local_rows == 1900
    assert runtime.prefix_rows == 6500
    assert runtime.visible_rows == 8400
    assert runtime.total_rows == 10_000


def test_variable_width_runtime_rejects_empty_rank() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="positive rows"):
        runtime.begin_step((4, 3, 0, 1))


def test_runahead_batch_accepts_fresh_and_existing_context_prefill() -> None:
    assert runahead_batch_eligible(
        num_reqs=2,
        is_prefilling=np.asarray([True, True]),
        num_scheduled_tokens=np.asarray([2048, 4096], dtype=np.int32),
        pcp_world_size=4,
    )


def test_runahead_batch_accepts_chunked_prefill() -> None:
    assert runahead_batch_eligible(
        num_reqs=1,
        is_prefilling=np.asarray([True]),
        num_scheduled_tokens=np.asarray([2048], dtype=np.int32),
        pcp_world_size=4,
    )


def test_runahead_batch_rejects_mixed_decode_prefill() -> None:
    assert not runahead_batch_eligible(
        num_reqs=2,
        is_prefilling=np.asarray([False, True]),
        num_scheduled_tokens=np.asarray([1, 4096], dtype=np.int32),
        pcp_world_size=4,
    )


def test_runahead_batch_rejects_small_prefill() -> None:
    assert not runahead_batch_eligible(
        num_reqs=1,
        is_prefilling=np.asarray([True]),
        num_scheduled_tokens=np.asarray([512], dtype=np.int32),
        pcp_world_size=4,
    )

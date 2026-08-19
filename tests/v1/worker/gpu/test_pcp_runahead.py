# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.pcp_manager import (
    RunaheadPCPManager,
    runahead_batch_eligible,
)


def _manager(world_size: int) -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = world_size
    manager._use_runahead_partition = True
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

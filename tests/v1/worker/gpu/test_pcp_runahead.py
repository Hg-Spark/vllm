# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.pcp_manager import RunaheadPCPManager


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

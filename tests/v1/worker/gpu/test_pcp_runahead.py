# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    actual = []

    for rank in range(4):
        (segment,) = manager._get_rank_segments(
            rank,
            np.asarray([10], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True]),
            np.asarray([0, 10], dtype=np.int32),
        )
        actual.append(
            (segment.global_batch_slice.start, segment.global_batch_slice.stop)
        )

    assert actual == expected


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
    actual = []
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
    assert sum(weighted_partition_lengths(10, weights)) == 10


def test_mla_uses_equal_compact_runahead_partition() -> None:
    manager = _manager(4, (4.0, 2.5, 1.9, 1.6))
    manager._standard_attention_pcp = False

    assert manager._weighted_lengths(10) == (3, 3, 3, 1)
    assert manager._use_compact_layout()


def test_runahead_layout_is_always_compact_and_rank_major() -> None:
    manager = _manager(4, (4.0, 3.0, 2.0, 1.0))
    manager.device = torch.device("cpu")
    scheduled = np.asarray([8, 4], dtype=np.int32)
    computed = np.asarray([0, 16384], dtype=np.int32)
    is_prefilling = np.asarray([True, True])
    query_start = np.asarray([0, 8, 12], dtype=np.int32)

    def copy_to_cpu(
        value: np.ndarray,
        device: torch.device | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del device
        tensor = torch.as_tensor(value)
        if out is not None:
            out.copy_(tensor)
            return out
        return tensor

    with patch(
        "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
        side_effect=copy_to_cpu,
    ):
        _, rows = manager._build_batch_layout(
            scheduled,
            computed,
            is_prefilling,
            query_start,
        )

    assert rows == [5, 3, 3, 1]
    assert manager._rank_offsets == (0, 5, 8, 11, 12)
    assert manager._padded_gather_idx is not None
    assert manager._padded_gather_idx.tolist() == [
        0,
        1,
        2,
        8,
        9,
        3,
        4,
        10,
        5,
        6,
        11,
        7,
    ]
    assert manager._hidden_restore_idx is not None
    assert manager._hidden_restore_idx.tolist() == [
        0,
        1,
        2,
        5,
        6,
        8,
        9,
        11,
        3,
        4,
        7,
        10,
    ]
    assert manager._gathered_kv_write_mask is not None
    assert bool(manager._gathered_kv_write_mask.all())


def test_step_level_repair_plan_uses_kernel_block_ids() -> None:
    manager = _manager(4)
    manager.device = torch.device("cpu")
    manager._block_tables = SimpleNamespace(kernel_block_sizes=[4, 2])
    manager._global_batch = SimpleNamespace(num_tokens=6)
    manager._runahead_runtime = MagicMock()

    slot_mappings = torch.tensor(
        [
            [0, 1, 4, 7, 8, -1],
            [0, 1, 4, 5, 8, -1],
        ],
        dtype=torch.int64,
    )
    manager._plan_runahead_repair(slot_mappings)

    (block_ids,) = manager._runahead_runtime.set_repair_block_ids.call_args.args
    assert block_ids.tolist() == [0, 1, 2, 4]


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


@pytest.mark.parametrize(
    ("is_prefilling", "scheduled", "expected"),
    [
        ([True, True], [2048, 4096], True),
        ([True], [2048], True),
        ([False, True], [1, 4096], False),
        ([True], [512], False),
    ],
)
def test_runahead_batch_eligibility(
    is_prefilling: list[bool],
    scheduled: list[int],
    expected: bool,
) -> None:
    assert (
        runahead_batch_eligible(
            num_reqs=len(is_prefilling),
            is_prefilling=np.asarray(is_prefilling),
            num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
            pcp_world_size=4,
        )
        is expected
    )

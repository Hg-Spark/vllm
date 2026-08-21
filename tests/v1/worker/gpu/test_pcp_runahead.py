# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan
from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime
from vllm.v1.attention.ops.pcp_standard import _write_local_kv_cache_for_page_pull
from vllm.v1.worker.gpu.pcp_runahead_config import (
    PCPRunaheadConfig,
    parse_pcp_runahead_config,
)
from vllm.v1.worker.gpu.pcp_runahead_manager import (
    RunaheadPCPManager,
    compact_hidden_restore_idx,
    parse_runahead_weights,
    runahead_batch_eligible,
    weighted_partition_lengths,
)


def _manager(world_size: int) -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = world_size
    manager._standard_attention_pcp = True
    manager._use_custom_partition = True
    manager._config = PCPRunaheadConfig()
    manager._page_alignment = 1
    return manager


def test_nested_config_parses_independent_experiment_axes() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "full_kv_collective",
                "partition": {
                    "policy": "weighted_contiguous",
                    "weights": [4, 2.5, 1.9, 1.6],
                    "page_align": True,
                },
                "layout": "compact",
                "eligibility": {
                    "require_full_prefill": True,
                    "min_tokens": 2048,
                },
                "runtime": {"max_inflight_sends": 3},
            }
        },
        4,
    )
    assert config is not None
    assert config.transport == "full_kv_collective"
    assert config.partition_policy == "weighted_contiguous"
    assert config.weights == (4.0, 2.5, 1.9, 1.6)
    assert config.segment_to_rank == (0, 1, 2, 3)
    assert config.layout == "compact"
    assert config.min_tokens == 2048
    assert config.max_inflight_sends == 3


def test_segments_compile_logical_to_physical_permutation() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "direct_p2p",
                "partition": {
                    "policy": "weighted_contiguous",
                    "segments": [
                        {"weight": 3, "pcp_rank": 1},
                        {"weight": 1, "pcp_rank": 0},
                    ],
                },
                "layout": "compact",
            }
        },
        2,
    )
    assert config is not None
    assert config.weights == (3.0, 1.0)
    assert config.segment_to_rank == (1, 0)


def test_page_pull_accepts_repeated_rank_binding() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "page_pull",
                "partition": {
                    "policy": "weighted_contiguous",
                    "segments": [
                        {"weight": 1, "pcp_rank": 1},
                        {"weight": 1, "pcp_rank": 0},
                        {"weight": 1, "pcp_rank": 1},
                    ],
                    "page_align": True,
                },
                "layout": "compact",
                "runtime": {
                    "max_inflight_reads": 2,
                    "nixl_backends": ["UCX"],
                },
            }
        },
        2,
    )
    assert config is not None
    assert config.segment_to_rank == (1, 0, 1)
    assert config.weights == (1.0, 1.0, 1.0)
    assert config.max_inflight_reads == 2


def test_tensor_transport_rejects_repeated_rank_binding() -> None:
    with pytest.raises(ValueError, match="repeated segment bindings"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "direct_p2p",
                    "partition": {
                        "policy": "weighted_contiguous",
                        "segments": [
                            {"pcp_rank": 1},
                            {"pcp_rank": 0},
                            {"pcp_rank": 1},
                        ],
                    },
                    "layout": "compact",
                }
            },
            2,
        )


def test_segments_require_all_physical_ranks() -> None:
    with pytest.raises(ValueError, match="cover every PCP rank"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "page_pull",
                    "partition": {
                        "policy": "weighted_contiguous",
                        "segments": [
                            {"pcp_rank": 0},
                            {"pcp_rank": 0},
                        ],
                    },
                    "layout": "compact",
                }
            },
            2,
        )


def test_legacy_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_pcp_runahead_config({"pcp_runahead": True}, 4)

    with pytest.raises(ValueError, match="partition.weights"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "prefix_p2p",
                    "partition": {"policy": "weighted_contiguous"},
                },
                "pcp_runahead_weights": [4, 2, 1, 1],
            },
            4,
        )


def test_invalid_stock_prefix_combination_is_rejected() -> None:
    with pytest.raises(ValueError, match="stock partition"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "prefix_p2p",
                    "partition": {"policy": "stock"},
                    "layout": "compact",
                }
            },
            4,
        )


def test_runahead_partition_is_contiguous_and_complete() -> None:
    manager = _manager(4)
    expected = [(0, 3), (3, 6), (6, 8), (8, 10)]
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


def test_permuted_rank_owns_logical_segment_not_physical_interval() -> None:
    manager = _manager(2)
    manager._config = replace(
        manager._config,
        partition_policy="weighted_contiguous",
        weights=(3.0, 1.0),
        segment_to_rank=(1, 0),
    )
    rank0 = manager._get_rank_segments(
        0,
        np.asarray([100], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([True]),
        np.asarray([0, 100], dtype=np.int32),
    )
    rank1 = manager._get_rank_segments(
        1,
        np.asarray([100], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([True]),
        np.asarray([0, 100], dtype=np.int32),
    )
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank1] == [
        (0, 75)
    ]
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank0] == [
        (75, 100)
    ]


def test_repeated_binding_builds_multiple_local_segments() -> None:
    manager = _manager(2)
    manager._config = replace(
        manager._config,
        transport="page_pull",
        partition_policy="weighted_contiguous",
        weights=(1.0, 1.0, 1.0),
        segment_to_rank=(1, 0, 1),
    )
    rank1 = manager._get_rank_segments(
        1,
        np.asarray([12], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([True]),
        np.asarray([0, 12], dtype=np.int32),
    )
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank1] == [
        (0, 4),
        (8, 12),
    ]


def test_manual_weights_parse_and_validate() -> None:
    assert parse_runahead_weights([4, 2.5, 1.9, 1.6], 4) == (4.0, 2.5, 1.9, 1.6)

    with pytest.raises(ValueError, match="requires 4 values"):
        parse_runahead_weights([1, 1], 4)
    with pytest.raises(ValueError, match="finite and positive"):
        parse_runahead_weights([1, 1, 0, 1], 4)


def test_weighted_partition_is_page_aligned() -> None:
    lengths = weighted_partition_lengths(
        10000,
        (4.0, 2.5, 1.9, 1.6),
        start_pos=0,
        alignment=16,
    )
    assert lengths == (4000, 2496, 1904, 1600)
    cuts = np.cumsum(lengths)[:-1]
    assert all(int(cut) % 16 == 0 for cut in cuts)


def test_weighted_partition_aligns_absolute_positions() -> None:
    start_pos = 1003
    lengths = weighted_partition_lengths(
        10000,
        (4.0, 2.5, 1.9, 1.6),
        start_pos=start_pos,
        alignment=16,
    )
    cuts = np.cumsum(lengths)[:-1]
    assert all((start_pos + int(cut)) % 16 == 0 for cut in cuts)
    assert sum(lengths) == 10000


def test_weighted_partition_supports_multiple_prefill_requests() -> None:
    manager = _manager(4)
    manager._config = replace(
        manager._config,
        partition_policy="weighted_contiguous",
        weights=(4.0, 2.0, 1.0, 1.0),
    )
    scheduled = np.asarray([80, 40], dtype=np.int32)
    computed = np.asarray([0, 160], dtype=np.int32)
    is_prefilling = np.asarray([True, True])
    query_start = np.asarray([0, 80, 120], dtype=np.int32)

    rows = [0, 0, 0, 0]
    for rank in range(4):
        segments = manager._get_rank_segments(
            rank, scheduled, computed, is_prefilling, query_start
        )
        rows[rank] = sum(segment.num_tokens for segment in segments)

    assert sum(rows) == 120
    assert rows[0] > rows[1] > rows[2]
    assert rows[2] == rows[3]


def test_compact_hidden_restore_index_removes_padding() -> None:
    padded_restore = torch.tensor([0, 1, 2, 3, 4, 5, 8, 12], dtype=torch.long)
    compact = compact_hidden_restore_idx(
        padded_restore,
        padded_rows=4,
        rows_per_rank=(4, 2, 1, 1),
    )
    assert compact.tolist() == list(range(8))


def test_restore_hidden_uses_vllm_all_gatherv() -> None:
    manager = _manager(4)
    manager._compact_hidden_restore_idx = torch.tensor([0, 1, 2, 3])
    manager._rows_per_rank = (2, 1, 1, 1)
    group = MagicMock()
    group.all_gatherv.return_value = torch.arange(5, dtype=torch.float32).view(5, 1)

    with patch(
        "vllm.v1.worker.gpu.pcp_runahead_manager.get_pcp_group",
        return_value=group,
    ):
        restored = manager.restore_hidden_states(
            torch.arange(2, dtype=torch.float32).view(2, 1)
        )

    group.all_gatherv.assert_called_once()
    assert restored[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_variable_width_runtime_builds_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))

    assert runtime.rows_per_rank == (4, 3, 2, 1)
    assert runtime.rank_offsets == (0, 4, 7, 9, 10)
    assert runtime.local_rows == 2
    assert runtime.prefix_rows == 7
    assert runtime.visible_rows == 9


def test_runtime_follows_logical_peer_order_for_permutation() -> None:
    mapping = (2, 0, 3, 1)
    rows = (3, 1, 4, 2)
    runtimes = []
    for rank in range(4):
        runtime = PCPRunaheadRuntime(
            pcp_world_size=4,
            pcp_rank=rank,
            device=torch.device("cpu"),
        )
        runtime.begin_step(rows, segment_to_rank=mapping)
        runtimes.append(runtime)

    # Logical chain is physical rank 2 -> 0 -> 3 -> 1.
    assert runtimes[2].prev_rank is None
    assert runtimes[2].next_rank == 0
    assert runtimes[0].prev_rank == 2
    assert runtimes[0].next_rank == 3
    assert runtimes[3].prev_rank == 0
    assert runtimes[3].next_rank == 1
    assert runtimes[1].prev_rank == 3
    assert runtimes[1].next_rank is None


def test_page_pull_runtime_accepts_repeated_mapping_with_segment_rows() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
    )
    runtime.begin_step(
        (4, 8),
        transport="page_pull",
        segment_to_rank=(1, 0, 1),
        segment_rows=(4, 4, 4),
    )
    assert runtime.owned_segments == (0, 2)
    assert runtime.local_rows == 8
    assert runtime.visible_rows == 12


def test_page_plan_short_circuits_locally_owned_earlier_segment() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(1, 0, 1),
        source_blocks_by_segment=((10, 11), (20, 21), (30,)),
        destination_blocks_by_segment=((100, 101), (110, 111), (120,)),
        block_size=16,
    )
    # rank1 owns segment0 and segment2. For its latest query (segment2), only
    # segment1 is remote; segment0 is already local and must not be transferred.
    assert plan.required_segments(1) == (1,)
    assert plan.required_source_ranks(1) == (0,)
    assert plan.transfer_block_ids(1, 0) == ((110, 111), (20, 21))
    assert plan.consumer_ranks(0) == (1,)


def test_page_pull_local_cache_write_uses_physical_slots() -> None:
    # [blocks, heads, block_size, 2*head_size]
    cache = torch.zeros((3, 1, 4, 4), dtype=torch.float32)
    key = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    value = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    slots = torch.tensor([1, 9], dtype=torch.long)

    _write_local_kv_cache_for_page_pull(key, value, slots, cache)

    key_cache, value_cache = cache.transpose(1, 2).split(2, dim=-1)
    assert key_cache[0, 1, 0].tolist() == [1.0, 2.0]
    assert value_cache[0, 1, 0].tolist() == [5.0, 6.0]
    assert key_cache[2, 1, 0].tolist() == [3.0, 4.0]
    assert value_cache[2, 1, 0].tolist() == [7.0, 8.0]


@pytest.mark.parametrize(
    (
        "is_prefilling",
        "scheduled",
        "computed",
        "prefill_len",
        "expected",
    ),
    [
        ([True, True], [2048, 4096], [0, 0], [2048, 4096], True),
        ([True], [2048], [0], [2048], True),
        ([False, True], [1, 4096], [0, 0], [1, 4096], False),
        ([True], [512], [0], [512], False),
        ([True], [1024], [0], [2048], False),
        ([True], [1024], [1024], [2048], False),
    ],
)
def test_runahead_batch_eligibility_requires_full_fresh_prefill(
    is_prefilling: list[bool],
    scheduled: list[int],
    computed: list[int],
    prefill_len: list[int],
    expected: bool,
) -> None:
    assert (
        runahead_batch_eligible(
            num_reqs=len(is_prefilling),
            is_prefilling=np.asarray(is_prefilling),
            num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
            num_computed_tokens=np.asarray(computed, dtype=np.int32),
            prefill_len=np.asarray(prefill_len, dtype=np.int32),
            pcp_world_size=4,
        )
        is expected
    )
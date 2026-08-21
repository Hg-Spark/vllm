# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_name
from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime
from vllm.v1.worker.gpu.pcp_runahead_config import (
    PCPRunaheadConfig,
    parse_pcp_runahead_config,
)
from vllm.v1.worker.gpu.pcp_runahead_manager import (
    RunaheadPCPManager,
    parse_runahead_weights,
    runahead_batch_eligible,
    weighted_partition_lengths,
)


def _manager(world_size: int) -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = world_size
    manager.pcp_rank = 0
    manager._physical_pcp_rank = 0
    manager._logical_pcp_rank = 0
    manager._standard_attention_pcp = True
    manager._use_custom_partition = True
    manager._use_compact_layout = True
    manager._active_layout = None
    manager._config = PCPRunaheadConfig(
        pcp_world_size=world_size,
        weights=(1.0,) * world_size,
        segment_to_rank=tuple(range(world_size)),
    )
    manager._page_alignment = 1
    return manager


def _batch(
    scheduled: list[int],
    computed: list[int] | None = None,
) -> SimpleNamespace:
    if computed is None:
        computed = [0] * len(scheduled)
    query_start = np.asarray([0, *np.cumsum(scheduled)], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=len(scheduled),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        query_start_loc_np=query_start,
    )


def test_config_derives_compact_full_prefill_invariants() -> None:
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
    assert config.weights == (4.0, 2.5, 1.9, 1.6)
    assert config.segment_to_rank == (0, 1, 2, 3)
    assert config.runtime_segment_to_rank == (0, 1, 2, 3)
    assert config.min_tokens == 2048
    assert config.max_inflight_sends == 3


def test_permutation_is_compiled_into_process_group_order() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "direct_p2p",
                "partition": {
                    "segments": [
                        {"weight": 3, "pcp_rank": 1},
                        {"weight": 1, "pcp_rank": 0},
                    ]
                },
            }
        },
        2,
    )
    assert config is not None
    assert config.weights == (3.0, 1.0)
    assert config.segment_to_rank == (1, 0)
    assert config.process_group_order == (1, 0)
    assert config.runtime_segment_to_rank == (0, 1)


def test_page_pull_keeps_repeated_rank_binding_in_plan_space() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "page_pull",
                "partition": {
                    "segments": [
                        {"pcp_rank": 1},
                        {"pcp_rank": 0},
                        {"pcp_rank": 1},
                    ],
                    "page_align": True,
                },
                "runtime": {
                    "max_inflight_reads": 2,
                    "nixl_backends": ["UCX"],
                },
            }
        },
        2,
    )
    assert config is not None
    assert not config.mapping_is_permutation
    assert config.segment_to_rank == (1, 0, 1)
    assert config.runtime_segment_to_rank == (1, 0, 1)
    assert config.max_inflight_reads == 2


def test_tensor_transport_rejects_repeated_rank_binding() -> None:
    with pytest.raises(ValueError, match="repeated segment bindings"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "direct_p2p",
                    "partition": {
                        "segments": [
                            {"pcp_rank": 1},
                            {"pcp_rank": 0},
                            {"pcp_rank": 1},
                        ]
                    },
                }
            },
            2,
        )


def test_derived_invariants_reject_old_experiment_modes() -> None:
    with pytest.raises(ValueError, match="partition.policy"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "prefix_p2p",
                    "partition": {"policy": "stock"},
                }
            },
            4,
        )
    with pytest.raises(ValueError, match="layout=compact"):
        parse_pcp_runahead_config(
            {"pcp_runahead": {"transport": "prefix_p2p", "layout": "padded"}},
            4,
        )
    with pytest.raises(ValueError, match="require_full_prefill=true"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {
                    "transport": "prefix_p2p",
                    "eligibility": {"require_full_prefill": False},
                }
            },
            4,
        )


def test_legacy_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_pcp_runahead_config({"pcp_runahead": True}, 4)
    with pytest.raises(ValueError, match="partition.weights"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead": {"transport": "prefix_p2p"},
                "pcp_runahead_weights": [4, 2, 1, 1],
            },
            4,
        )


def test_segment_layout_is_compiled_once_for_all_ranks() -> None:
    manager = _manager(4)
    layout = manager._compile_segment_layout(_batch([10]))
    assert layout.rows_per_rank == (3, 3, 2, 2)
    assert [
        (segments[0].global_batch_slice.start, segments[0].global_batch_slice.stop)
        for segments in layout.segments_by_rank
    ] == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert sum(layout.rows_per_segment) == 10


def test_permutation_layout_uses_logical_group_rank() -> None:
    manager = _manager(2)
    manager._config = replace(
        manager._config,
        weights=(3.0, 1.0),
        segment_to_rank=(1, 0),
    )
    layout = manager._compile_segment_layout(_batch([100]))
    # Once the communicator is ordered [physical1, physical0], logical rank 0
    # owns the first weighted interval and no runtime reorder is required.
    rank0 = layout.segments_by_rank[0]
    rank1 = layout.segments_by_rank[1]
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank0] == [
        (0, 75)
    ]
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank1] == [
        (75, 100)
    ]


def test_repeated_binding_builds_multiple_local_segments() -> None:
    manager = _manager(2)
    manager._config = replace(
        manager._config,
        transport="page_pull",
        weights=(1.0, 1.0, 1.0),
        segment_to_rank=(1, 0, 1),
    )
    layout = manager._compile_segment_layout(_batch([12]))
    rank1 = layout.segments_by_rank[1]
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank1] == [
        (0, 4),
        (8, 12),
    ]
    assert layout.rows_per_rank == (4, 8)


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


def test_runahead_requires_full_prefill_and_threshold() -> None:
    common = dict(
        num_reqs=1,
        is_prefilling=np.asarray([True]),
        pcp_world_size=2,
        require_full_prefill=True,
        min_prefill_tokens=8,
    )
    assert runahead_batch_eligible(
        **common,
        num_scheduled_tokens=np.asarray([8]),
        num_computed_tokens=np.asarray([0]),
        prefill_len=np.asarray([8]),
    )
    assert not runahead_batch_eligible(
        **common,
        num_scheduled_tokens=np.asarray([4]),
        num_computed_tokens=np.asarray([0]),
        prefill_len=np.asarray([8]),
    )


def test_variable_width_runtime_uses_logical_rank_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))
    assert runtime.rank_offsets == (0, 4, 7, 9, 10)
    assert runtime.prev_rank == 1
    assert runtime.next_rank == 3
    assert runtime.local_rows == 2
    assert runtime.prefix_rows == 7
    assert runtime.visible_rows == 9


def test_page_plan_short_circuits_locally_owned_prefix() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(1, 0, 1),
        blocks_by_segment=((10, 11), (20, 21), (30,)),
        block_size=16,
    )
    assert plan.required_segments(1) == (1,)
    assert plan.required_source_ranks(1) == (0,)
    assert plan.transfer_block_ids(1, 0) == ((20, 21), (20, 21))
    assert plan.consumer_ranks(0) == (1,)


def test_compact_restore_uses_allgatherv_only_for_variable_width() -> None:
    manager = _manager(4)
    manager._hidden_restore_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    manager._per_rank_num_tokens = (2, 1, 1, 1)
    manager._pcp_group = MagicMock()
    manager._pcp_group.all_gatherv.return_value = torch.arange(5).view(5, 1)
    restored = manager.restore_hidden_states(torch.arange(2).view(2, 1))
    manager._pcp_group.all_gatherv.assert_called_once()
    manager._pcp_group.all_gather.assert_not_called()
    assert restored[:, 0].tolist() == [0, 1, 2, 3]


def test_pcp_nvtx_name_is_structured_and_stable() -> None:
    assert (
        pcp_nvtx_name("page_pull.read_submit", e=3, src=0, dst=2, pages=16)
        == "pcp.page_pull.read_submit[e=3,src=0,dst=2,pages=16]"
    )

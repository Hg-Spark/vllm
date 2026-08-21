# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.config.pcp_runahead import (
    PCPRunaheadConfig,
    compile_pcp_binding,
    parse_pcp_runahead_config,
)
from vllm.model_executor.layers.attention.pcp import (
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.attention.ops.pcp_page_pull import (
    PCPPagePlan,
    PCPPagePullTransport,
)
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_name
from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime
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
    manager._standard_attention_pcp = True
    manager._active_step = None
    manager._config = PCPRunaheadConfig(
        transport="prefix_p2p",
        weights=(1.0,) * world_size,
        binding=compile_pcp_binding(tuple(range(world_size)), world_size),
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


def _runahead_vllm_config(
    *,
    transport: str = "prefix_p2p",
    sparse_mla: bool = False,
) -> SimpleNamespace:
    hf_text_config = SimpleNamespace()
    if sparse_mla:
        hf_text_config.index_topk = 2048
    return SimpleNamespace(
        additional_config={"pcp_runahead": {"transport": transport}},
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_dbo=False,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            is_moe=True,
            hf_text_config=hf_text_config,
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def test_config_parses_only_runtime_axes() -> None:
    config = parse_pcp_runahead_config(
        {
            "pcp_runahead": {
                "transport": "full_kv_collective",
                "partition": {"weights": [4, 2.5, 1.9, 1.6]},
                "eligibility": {"min_tokens": 2048},
                "runtime": {"max_inflight_sends": 3},
            }
        },
        4,
    )
    assert config is not None
    assert config.transport == "full_kv_collective"
    assert config.weights == (4.0, 2.5, 1.9, 1.6)
    assert config.segment_to_physical_rank == (0, 1, 2, 3)
    assert config.segment_to_group_rank == (0, 1, 2, 3)
    assert config.pcp_group_order == (0, 1, 2, 3)
    assert config.min_tokens == 2048
    assert config.max_inflight_sends == 3


def test_permutation_compiles_physical_binding_into_primary_group_order() -> None:
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
    assert config.segment_to_physical_rank == (1, 0)
    assert config.segment_to_group_rank == (0, 1)
    assert config.pcp_group_order == (1, 0)
    assert config.binding.physical_rank_to_group_rank == (1, 0)


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
                    ]
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
    assert not config.binding.is_permutation
    assert config.segment_to_physical_rank == (1, 0, 1)
    assert config.segment_to_group_rank == (1, 0, 1)
    assert config.pcp_group_order == (0, 1)
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


@pytest.mark.parametrize(
    "config,match",
    [
        (
            {"transport": "prefix_p2p", "partition": {"policy": "stock"}},
            "unsupported partition keys",
        ),
        (
            {"transport": "prefix_p2p", "layout": "padded"},
            "unsupported pcp_runahead keys",
        ),
        (
            {
                "transport": "prefix_p2p",
                "eligibility": {"require_full_prefill": False},
            },
            "unsupported eligibility keys",
        ),
        (
            {
                "transport": "page_pull",
                "partition": {"page_align": False},
            },
            "unsupported partition keys",
        ),
    ],
)
def test_obsolete_experiment_axes_are_rejected(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_pcp_runahead_config({"pcp_runahead": config}, 4)


def test_legacy_top_level_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="was removed"):
        parse_pcp_runahead_config(
            {
                "pcp_runahead_weights": [1, 1],
                "pcp_runahead": {"transport": "prefix_p2p"},
            },
            2,
        )


def test_page_pull_accepts_dense_nhd_physical_pages() -> None:
    physical = torch.empty((8, 16, 2, 32))
    logical_nhd = physical.permute(0, 2, 1, 3)
    assert not logical_nhd[0].is_contiguous()
    num_blocks, block_bytes = PCPPagePullTransport._physical_page_geometry(logical_nhd)
    assert num_blocks == 8
    assert block_bytes == 16 * 2 * 32 * logical_nhd.element_size()


def test_boolean_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty JSON object"):
        parse_pcp_runahead_config({"pcp_runahead": True}, 4)


def test_segment_layout_is_compiled_once_for_all_ranks() -> None:
    manager = _manager(4)
    layout = manager._compile_segment_layout(_batch([10]))
    assert layout is not None
    assert layout.rows_per_rank == (3, 3, 2, 2)
    assert [
        (segments[0].global_batch_slice.start, segments[0].global_batch_slice.stop)
        for segments in layout.segments_by_rank
    ] == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert sum(
        piece.end_pos - piece.start_pos
        for pieces in layout.logical_segments
        for piece in pieces
    ) == 10


def test_permutation_layout_uses_primary_group_rank() -> None:
    manager = _manager(2)
    manager._config = replace(
        manager._config,
        weights=(3.0, 1.0),
        binding=compile_pcp_binding((1, 0), 2),
    )
    layout = manager._compile_segment_layout(_batch([100]))
    assert layout is not None
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
        binding=compile_pcp_binding((1, 0, 1), 2),
    )
    layout = manager._compile_segment_layout(_batch([12]))
    assert layout is not None
    rank1 = layout.segments_by_rank[1]
    assert [(s.global_batch_slice.start, s.global_batch_slice.stop) for s in rank1] == [
        (0, 4),
        (8, 12),
    ]
    assert layout.rows_per_rank == (4, 8)


def test_manual_weights_parse_and_validate() -> None:
    assert parse_runahead_weights([4, 2.5, 1.9, 1.6], 4) == (4.0, 2.5, 1.9, 1.6)
    with pytest.raises(ValueError, match="requires 4"):
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


def test_full_collective_supports_variable_width_rows() -> None:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = 2
    gathered = torch.arange(10, dtype=torch.float32).view(10, 1)
    group.all_gatherv.return_value = gathered
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
        pcp_group=group,
    )
    runtime.begin_step((4, 3, 2, 1), transport="full_kv_collective")
    local = torch.tensor([[7.0], [8.0]])
    slots = torch.arange(10, dtype=torch.int64)
    (result,), result_slots = runtime.exchange_full((local,), slots)
    group.all_gatherv.assert_called_once()
    group.all_gather.assert_not_called()
    assert torch.equal(result, gathered)
    assert torch.equal(result_slots, slots)


def test_mla_latent_cache_uses_runahead_prefix_transport() -> None:
    runtime = MagicMock()
    runtime.transport = "prefix_p2p"
    visible_kv = torch.arange(12, dtype=torch.float32).view(3, 4)
    visible_kpe_flat = torch.arange(6, dtype=torch.float32).view(3, 2)
    visible_slots = torch.tensor([10, 11, 12], dtype=torch.int64)
    runtime.exchange_prefix.return_value = (
        (visible_kv, visible_kpe_flat),
        visible_slots,
    )
    kv_c = torch.arange(8, dtype=torch.float32).view(2, 4)
    k_pe = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    slots = torch.tensor([10, 11, 12, 13], dtype=torch.int64)

    with patch(
        "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        cache_kv, cache_kpe, cache_slots = maybe_gather_mla_latent_cache_inputs(
            kv_c,
            k_pe,
            slots,
            num_decode_tokens=0,
            use_pcp=True,
        )

    runtime.exchange_prefix.assert_called_once()
    assert torch.equal(cache_kv, visible_kv)
    assert torch.equal(cache_kpe, visible_kpe_flat.view(3, 1, 2))
    assert torch.equal(cache_slots, visible_slots)


def test_dense_mla_moe_is_allowed_for_tensor_runahead() -> None:
    RunaheadPCPManager.validate_config(_runahead_vllm_config(), False)


def test_mla_page_pull_remains_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="MLA does not support page_pull"):
        RunaheadPCPManager.validate_config(
            _runahead_vllm_config(transport="page_pull"), False
        )


def test_sparse_mla_runahead_remains_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="sparse MLA"):
        RunaheadPCPManager.validate_config(
            _runahead_vllm_config(sparse_mla=True), False
        )


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
    group = MagicMock()
    group.all_gatherv.return_value = torch.arange(5).view(5, 1)
    with patch("vllm.v1.worker.gpu.pcp_manager.get_pcp_group", return_value=group):
        restored = manager.restore_hidden_states(torch.arange(2).view(2, 1))
    group.all_gatherv.assert_called_once()
    group.all_gather.assert_not_called()
    assert restored[:, 0].tolist() == [0, 1, 2, 3]


def test_pcp_nvtx_name_is_structured_and_stable() -> None:
    assert (
        pcp_nvtx_name("page_pull.read_submit", e=3, src=0, dst=2, pages=16)
        == "pcp.page_pull.read_submit[e=3,src=0,dst=2,pages=16]"
    )

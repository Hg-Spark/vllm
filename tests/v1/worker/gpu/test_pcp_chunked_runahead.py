# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.config.pcp_runahead import PCPRunaheadConfig, compile_pcp_binding
from vllm.v1.attention.ops.pcp_page_state import PCPPageStateTracker
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pcp_runahead_manager import (
    DecodeRequestPlacement,
    DecodeStep,
    RunaheadPCPManager,
    RunaheadStep,
    weighted_partition_lengths,
)


def _chunk_manager() -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = 2
    manager.pcp_rank = 0
    manager._standard_attention_pcp = True
    manager._active_step = None
    manager._decode_step = None
    manager._tensor_sharded_kv_history = False
    manager._config = PCPRunaheadConfig(
        transport="page_pull",
        weights=(1.0, 1.0),
        binding=compile_pcp_binding((0, 1), 2),
    )
    manager._page_alignment = 16
    manager._req_states = SimpleNamespace(index_to_req_id={0: "req"})
    manager._page_state = PCPPageStateTracker(
        rank=0,
        block_size=16,
        max_model_len=256,
    )
    manager._pending_page_valid_updates = []
    manager._pending_page_advances = []
    manager._pending_tail_invalidations = []
    return manager


def _batch(
    *,
    scheduled: int,
    computed: int,
    req_id: str = "req",
    prefilling: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        req_ids=[req_id],
        num_scheduled_tokens=np.asarray([scheduled], dtype=np.int32),
        num_computed_tokens_np=np.asarray([computed], dtype=np.int32),
        is_prefilling_np=np.asarray([prefilling], dtype=np.bool_),
        query_start_loc_np=np.asarray([0, scheduled], dtype=np.int32),
        idx_mapping_np=np.asarray([0], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, (0, 1)),
        (17, (16, 1)),
        (31, (16, 15)),
        (32, (16, 16)),
        (100, (48, 52)),
        (202752, (101376, 101376)),
    ],
)
def test_page_aligned_partition_uses_cumulative_boundaries(
    num_tokens: int, expected: tuple[int, int]
) -> None:
    assert weighted_partition_lengths(
        num_tokens,
        (1.0, 1.0),
        alignment=16,
    ) == expected


def test_page_aligned_partition_keeps_segments_nonempty_when_feasible() -> None:
    assert weighted_partition_lengths(
        17,
        (0.01, 0.99),
        alignment=16,
    ) == (16, 1)
    assert weighted_partition_lengths(
        33,
        (0.01, 0.01, 0.98),
        alignment=16,
    ) == (16, 16, 1)


@pytest.mark.parametrize("num_tokens", [1, 7, 16, 17, 31, 33, 100])
def test_page_aligned_partition_has_no_unaligned_internal_cut(
    num_tokens: int,
) -> None:
    lengths = weighted_partition_lengths(
        num_tokens,
        (1.0, 1.0, 1.0, 1.0),
        alignment=16,
    )
    assert sum(lengths) == num_tokens

    cut = 0
    for length in lengths[:-1]:
        cut += length
        if 0 < cut < num_tokens:
            assert cut % 16 == 0


def test_inactive_page_pull_rank_uses_noncompact_padding() -> None:
    manager = _chunk_manager()
    batch = _batch(scheduled=1, computed=0)

    layout = manager._compile_segment_layout(batch)
    assert layout.rows_per_rank == (0, 1)
    assert [
        (piece.start_pos, piece.end_pos, piece.owner_group_rank)
        for piece in layout.causal_segments_by_request[0]
    ] == [(0, 1, 1)]
    assert {(u.page_idx, u.owner_rank) for u in layout.page_owner_updates} == {(0, 1)}

    manager._active_step = RunaheadStep(layout=layout, transport="page_pull")
    assert manager._execution_rows_per_rank() == (1, 1)
    assert not manager._compact_layout_enabled()


def test_page_pull_keeps_compact_layout_when_all_ranks_are_active() -> None:
    manager = _chunk_manager()
    layout = manager._compile_segment_layout(_batch(scheduled=32, computed=0))

    assert layout.rows_per_rank == (16, 16)
    manager._active_step = RunaheadStep(layout=layout, transport="page_pull")
    assert manager._execution_rows_per_rank() == (16, 16)
    assert manager._compact_layout_enabled()


@pytest.mark.parametrize("req_id", ["_warmup_0_", "req"])
def test_prefill_validator_does_not_special_case_decode(req_id: str) -> None:
    manager = _chunk_manager()

    with pytest.raises(RuntimeError, match="mixed decode-prefill"):
        manager._validate_step_semantics(
            _batch(
                scheduled=1,
                computed=1,
                req_id=req_id,
                prefilling=False,
            )
        )


def test_page_pull_decode_dispatch_does_not_activate_runahead_runtime() -> None:
    manager = _chunk_manager()
    manager._runahead_runtime = MagicMock(active=False)
    decode_step = MagicMock()
    manager._build_decode_step = MagicMock(return_value=decode_step)
    batch = _batch(scheduled=1, computed=1, prefilling=False)
    local_batch = object()

    with patch.object(PCPManager, "partition_batch", return_value=local_batch) as base:
        result = manager.partition_batch(batch)

    assert result is local_batch
    assert manager._decode_step is decode_step
    base.assert_called_once_with(manager, batch)
    manager._runahead_runtime.begin_step.assert_not_called()


def _build_decode_layout(
    *, decode_rank: int, writer_enabled: bool
) -> tuple[list[bool], list[int]]:
    manager = object.__new__(PCPManager)
    manager.pcp_world_size = 2
    manager.device = torch.device("cpu")
    manager._decode_rank = lambda _: decode_rank  # type: ignore[method-assign]
    manager._decode_kv_write_enabled = (  # type: ignore[method-assign]
        lambda _: writer_enabled
    )

    def copy_to_cpu(value, **kwargs):
        del kwargs
        return torch.as_tensor(value)

    with patch(
        "vllm.v1.worker.gpu.pcp_manager.async_copy_to_gpu",
        side_effect=copy_to_cpu,
    ):
        manager._build_batch_layout(
            np.asarray([1], dtype=np.int32),
            np.asarray([16], dtype=np.int32),
            np.asarray([False], dtype=np.bool_),
            np.asarray([0, 1], dtype=np.int32),
        )
    return (
        manager._gathered_kv_write_mask.tolist(),
        manager._hidden_restore_idx.tolist(),
    )


def test_decode_rank_keeps_mla_writer_in_first_slab() -> None:
    write_mask, restore_idx = _build_decode_layout(
        decode_rank=1, writer_enabled=True
    )
    assert write_mask == [True, False]
    assert restore_idx == [1]


def test_nonselected_process_disables_decode_kv_write() -> None:
    write_mask, restore_idx = _build_decode_layout(
        decode_rank=1, writer_enabled=False
    )
    assert write_mask == [False, False]
    assert restore_idx == [1]


def test_page_state_causal_owner_tracks_last_committed_page() -> None:
    tracker = PCPPageStateTracker(rank=0, block_size=16, max_model_len=128)
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 1)
    tracker.advance(state, 5)
    assert tracker.causal_owner(state) == 1

    tracker.assign_owner(state, 1, 0)
    tracker.advance(state, 17)
    assert tracker.causal_owner(state) == 0


def test_decode_step_selects_final_causal_owner() -> None:
    manager = _chunk_manager()
    tracker = manager._page_state
    assert tracker is not None
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 0)
    tracker.assign_owner(state, 1, 1)
    tracker.advance(state, 32)

    step = manager._build_decode_step(
        _batch(scheduled=1, computed=32, prefilling=False)
    )

    assert step.rank_by_global_req_idx == (1,)
    assert step.requests[0].rank == 1


def test_decode_completion_assigns_new_page_to_decode_rank() -> None:
    manager = _chunk_manager()
    tracker = manager._page_state
    assert tracker is not None
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 1)
    tracker.advance(state, 16)

    block_tables = MagicMock()
    block_tables.get_block_ids_cpu.return_value = [10, 11]
    manager._block_tables = block_tables
    step = DecodeStep(
        requests=(
            DecodeRequestPlacement(
                global_batch_req_idx=0,
                req_state_idx=0,
                request_id="req",
                start_pos=16,
                end_pos=17,
                rank=1,
            ),
        ),
        rank_by_global_req_idx=(1,),
    )

    manager._commit_decode_step_completion(step)

    assert tracker.owner(state, 1) == 1
    assert tracker.committed_tokens(state) == 17


def test_chunked_layout_keeps_mutable_tail_on_existing_owner() -> None:
    manager = _chunk_manager()
    tracker = manager._page_state
    assert tracker is not None
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 1)
    tracker.advance(state, 5)

    layout = manager._compile_segment_layout(_batch(scheduled=64, computed=5))

    assert layout.rows_per_rank == (32, 32)
    assert [
        (piece.start_pos, piece.end_pos, piece.owner_group_rank)
        for piece in layout.causal_segments_by_request[0]
    ] == [
        (5, 16, 1),
        (16, 48, 0),
        (48, 69, 1),
    ]
    assert {
        (update.page_idx, update.owner_rank) for update in layout.page_owner_updates
    } == {(1, 0), (2, 0), (3, 1), (4, 1)}


def test_chunked_layout_rejects_untracked_nonzero_prefix() -> None:
    manager = _chunk_manager()

    with pytest.raises(RuntimeError, match="untracked nonzero prefix"):
        manager._compile_segment_layout(_batch(scheduled=64, computed=64))


def test_page_state_rejects_scheduler_progress_jump() -> None:
    tracker = PCPPageStateTracker(rank=0, block_size=16, max_model_len=128)
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 0)
    tracker.advance(state, 16)

    with pytest.raises(RuntimeError, match="progress diverged"):
        tracker.prepare_request(0, "req", 32)


def test_new_page_ownership_is_not_committed_during_planning() -> None:
    manager = _chunk_manager()
    tracker = manager._page_state
    assert tracker is not None

    layout = manager._compile_segment_layout(_batch(scheduled=32, computed=0))
    state = tracker.existing_request(0, "req")
    assert state is not None
    assert tracker.owner(state, 0) == -1
    assert tracker.owner(state, 1) == -1
    assert {(u.page_idx, u.owner_rank) for u in layout.page_owner_updates} == {
        (0, 0),
        (1, 1),
    }


def test_chunked_plan_separates_history_from_current_dependencies() -> None:
    manager = _chunk_manager()
    tracker = manager._page_state
    assert tracker is not None
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 0)
    tracker.assign_owner(state, 1, 1)
    tracker.mark_local_valid(state, 0, 10)
    tracker.advance(state, 32)

    batch = _batch(scheduled=32, computed=32)
    layout = manager._compile_segment_layout(batch)
    manager._active_step = RunaheadStep(layout=layout, transport="page_pull")
    manager._global_batch = batch

    block_tables = MagicMock()
    block_tables.get_block_ids_cpu.return_value = [10, 11, 12, 13]
    manager._block_tables = block_tables
    manager._runahead_runtime = MagicMock()

    manager._configure_chunked_page_pull_plan()

    plan = manager._runahead_runtime.configure_page_plan.call_args.args[0]
    assert plan.historical_source_ranks(0) == (1,)
    assert plan.history_transfer_route(0, 1).source_block_ids == (11,)
    assert plan.current_source_ranks(1) == (0,)
    assert plan.current_transfer_route(1, 0).source_block_ids == (12,)
    assert plan.consumer_ranks(0) == (1,)


def test_local_validity_is_bound_to_vllm_physical_block_id() -> None:
    tracker = PCPPageStateTracker(rank=0, block_size=16, max_model_len=128)
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 0)
    tracker.mark_local_valid(state, 0, 7)

    assert tracker.local_is_valid(state, 0, 7)
    assert not tracker.local_is_valid(state, 0, 9)


def test_non_owner_mutable_tail_replica_is_invalidated() -> None:
    tracker = PCPPageStateTracker(rank=0, block_size=16, max_model_len=128)
    state = tracker.prepare_request(0, "req", 0)
    tracker.assign_owner(state, 0, 1)
    tracker.mark_local_valid(state, 0, 7)
    tracker.advance(state, 5)

    tracker.invalidate_mutable_tail(state, 5)

    assert not tracker.local_is_valid(state, 0, 7)
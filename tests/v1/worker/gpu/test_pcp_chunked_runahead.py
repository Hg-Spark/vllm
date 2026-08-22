# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from vllm.config.pcp_runahead import PCPRunaheadConfig, compile_pcp_binding
from vllm.v1.attention.ops.pcp_page_state import PCPPageStateTracker
from vllm.v1.worker.gpu.pcp_runahead_manager import (
    RunaheadPCPManager,
    RunaheadStep,
    runahead_batch_eligible,
)


def _chunk_manager() -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = 2
    manager.pcp_rank = 0
    manager._standard_attention_pcp = True
    manager._active_step = None
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
    return manager


def _batch(*, scheduled: int, computed: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.asarray([scheduled], dtype=np.int32),
        num_computed_tokens_np=np.asarray([computed], dtype=np.int32),
        query_start_loc_np=np.asarray([0, scheduled], dtype=np.int32),
        idx_mapping_np=np.asarray([0], dtype=np.int32),
    )


def test_chunked_eligibility_allows_nonzero_computed_tokens() -> None:
    assert runahead_batch_eligible(
        num_reqs=1,
        is_prefilling=np.asarray([True]),
        num_scheduled_tokens=np.asarray([64]),
        num_computed_tokens=np.asarray([128]),
        prefill_len=np.asarray([256]),
        pcp_world_size=2,
        require_full_prefill=False,
        min_prefill_tokens=32,
    )


def test_chunked_layout_keeps_mutable_tail_on_existing_owner() -> None:
    manager = _chunk_manager()
    state = manager._page_state.prepare_request(0, "req", 5)
    manager._page_state.assign_owner(state, 0, 1)

    layout = manager._compile_segment_layout(_batch(scheduled=64, computed=5))

    assert layout is not None
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


def test_chunked_layout_rejects_unknown_nonzero_prefix() -> None:
    manager = _chunk_manager()

    assert manager._compile_segment_layout(_batch(scheduled=64, computed=64)) is None


def test_chunked_plan_separates_history_from_current_dependencies() -> None:
    manager = _chunk_manager()
    state = manager._page_state.prepare_request(0, "req", 32)
    manager._page_state.assign_owner(state, 0, 0)
    manager._page_state.assign_owner(state, 1, 1)
    manager._page_state.mark_local_valid(state, 0, 10)
    manager._page_state.advance(state, 32)

    batch = _batch(scheduled=32, computed=32)
    layout = manager._compile_segment_layout(batch)
    assert layout is not None
    manager._commit_page_owner_updates(layout, batch)
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
    state = tracker.prepare_request(0, "req", 5)
    tracker.assign_owner(state, 0, 1)
    tracker.mark_local_valid(state, 0, 7)

    tracker.invalidate_mutable_tail(state, 5)

    assert not tracker.local_is_valid(state, 0, 7)

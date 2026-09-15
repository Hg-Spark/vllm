# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.ops.pcp_shared_context import plan_pcp_shared_context


def _plan(context_lens, query_start_locs, row_budget, split_alignment):
    return plan_pcp_shared_context(
        source_row=0,
        context_lens=context_lens,
        query_start_locs=query_start_locs,
        row_budget=row_budget,
        split_alignment=split_alignment,
    )


def _assert_initialized_before_continuation(jobs, context_lens, query_start_locs):
    initialized = set()
    consumer_indices = {
        (query_start_locs[i], query_start_locs[i + 1]): i
        for i in range(len(query_start_locs) - 1)
    }
    for job in jobs:
        for consumer in job.consumers:
            request_index = consumer_indices[(consumer.start, consumer.stop)]
            if request_index not in initialized:
                assert job.kv_start == 0
                initialized.add(request_index)
    assert initialized == {
        request_index
        for request_index, context_len in enumerate(context_lens)
        if context_len > 0
    }


def test_accumulator_initialization_order_with_shared_prefix() -> None:
    context_lens = (1000, 2000)
    query_starts = (0, 128, 256)
    jobs = _plan(context_lens, query_starts, 512, 128)
    assert jobs is not None
    _assert_initialized_before_continuation(jobs, context_lens, query_starts)


def test_accumulator_initialization_order_without_aligned_shared_prefix() -> None:
    context_lens = (63, 2000)
    query_starts = (0, 128, 256)
    jobs = _plan(context_lens, query_starts, 512, 128)
    assert jobs is not None
    _assert_initialized_before_continuation(jobs, context_lens, query_starts)


def test_materialized_rows_drop_by_shared_prefix_rows() -> None:
    context_lens = (4096, 8192)
    jobs = _plan(context_lens, (0, 256, 512), 2048, 128)
    assert jobs is not None
    logical_rows = sum(context_lens)
    materialized_rows = sum(job.kv_length for job in jobs)
    shared_rows = sum(job.kv_length for job in jobs if len(job.consumers) == 2)
    assert shared_rows == 4096
    assert materialized_rows == logical_rows - shared_rows

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

from vllm.v1.attention.ops.pcp_shared_context import plan_pcp_shared_context


def _job_range(job):
    return job.kv_start, job.kv_start + job.kv_length


def _consumer_index(token_slice, query_start_locs):
    for index in range(len(query_start_locs) - 1):
        if (token_slice.start, token_slice.stop) == (
            query_start_locs[index],
            query_start_locs[index + 1],
        ):
            return index
    raise AssertionError(f"unexpected consumer slice: {token_slice}")


def _assert_exact_history(jobs, context_lens, query_start_locs):
    coverage = defaultdict(list)
    for job in jobs:
        for consumer in job.consumers:
            coverage[_consumer_index(consumer, query_start_locs)].append(_job_range(job))
    for request_index, context_len in enumerate(context_lens):
        cursor = 0
        for start, end in sorted(coverage[request_index]):
            assert start == cursor
            assert start < end
            cursor = end
        assert cursor == context_len


def _plan(context_lens, query_start_locs, row_budget, split_alignment):
    return plan_pcp_shared_context(
        source_row=0,
        context_lens=context_lens,
        query_start_locs=query_start_locs,
        row_budget=row_budget,
        split_alignment=split_alignment,
    )


def test_unaligned_shorter_history_uses_aligned_shared_cut() -> None:
    query_starts = (0, 128, 256)
    jobs = _plan((1000, 2000), query_starts, 4096, 128)
    assert jobs is not None
    assert [_job_range(job) for job in jobs] == [
        (0, 896),
        (896, 1000),
        (896, 2000),
    ]
    assert [tuple(c.start for c in job.consumers) for job in jobs] == [
        (0, 128),
        (0,),
        (128,),
    ]
    _assert_exact_history(jobs, (1000, 2000), query_starts)
    assert all(job.kv_start == 0 or job.kv_start % 128 == 0 for job in jobs)


def test_aligned_shorter_history_has_no_duplicate_tail() -> None:
    query_starts = (0, 32, 96)
    jobs = _plan((1024, 2048), query_starts, 4096, 128)
    assert jobs is not None
    assert [_job_range(job) for job in jobs] == [(0, 1024), (1024, 2048)]
    assert [tuple(c.start for c in job.consumers) for job in jobs] == [
        (0, 32),
        (32,),
    ]
    _assert_exact_history(jobs, (1024, 2048), query_starts)


def test_shared_and_private_ranges_split_to_workspace_budget() -> None:
    query_starts = (0, 256, 512)
    jobs = _plan((4096, 8192), query_starts, 2048, 128)
    assert jobs is not None
    assert [_job_range(job) for job in jobs] == [
        (0, 2048),
        (2048, 4096),
        (4096, 6144),
        (6144, 8192),
    ]
    assert [tuple(c.start for c in job.consumers) for job in jobs] == [
        (0, 256),
        (0, 256),
        (256,),
        (256,),
    ]
    _assert_exact_history(jobs, (4096, 8192), query_starts)


def test_equal_unaligned_histories_share_everything() -> None:
    query_starts = (0, 64, 128)
    jobs = _plan((1000, 1000), query_starts, 512, 128)
    assert jobs is not None
    assert [_job_range(job) for job in jobs] == [(0, 512), (512, 1000)]
    assert all(len(job.consumers) == 2 for job in jobs)
    _assert_exact_history(jobs, (1000, 1000), query_starts)


def test_request_order_does_not_change_history_coverage() -> None:
    query_starts = (0, 80, 160)
    jobs = _plan((2048, 1024), query_starts, 4096, 128)
    assert jobs is not None
    _assert_exact_history(jobs, (2048, 1024), query_starts)
    assert tuple(c.start for c in jobs[0].consumers) == (80, 0)


def test_invalid_source_row_falls_back() -> None:
    assert (
        plan_pcp_shared_context(
            source_row=-1,
            context_lens=(1024, 2048),
            query_start_locs=(0, 32, 64),
            row_budget=4096,
            split_alignment=128,
        )
        is None
    )


def test_non_two_segment_shape_falls_back() -> None:
    assert (
        plan_pcp_shared_context(
            source_row=0,
            context_lens=(512, 1024, 1536),
            query_start_locs=(0, 32, 64, 96),
            row_budget=4096,
            split_alignment=128,
        )
        is None
    )


def test_too_small_workspace_falls_back_safely() -> None:
    assert (
        plan_pcp_shared_context(
            source_row=0,
            context_lens=(256, 512),
            query_start_locs=(0, 32, 64),
            row_budget=64,
            split_alignment=128,
        )
        is None
    )

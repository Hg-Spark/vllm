# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU PCP job planning and real native chunk/restore contracts."""

import math

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _build_pcp_packed_context_metadata,
    _pcp_context_reuse_applicable,
    _plan_pcp_context_reuse,
    accumulate_mla_context_chunk,
    init_mla_context_partial,
    restore_pcp_context_partial,
)


def _plan(
    *,
    req_idx=(7, 7),
    num_decodes=0,
    context_lens=(64, 112),
    query_starts=(0, 16, 32),
    workspace=128,
    device=torch.device("cpu"),
):
    # Use the real CPU planner and metadata builder; no fake attention builder
    # or scheduler metadata is needed for these contracts.
    cpu_plan = _plan_pcp_context_reuse(
        list(req_idx[num_decodes : num_decodes + len(context_lens)]),
        list(context_lens),
        list(query_starts),
    )
    if cpu_plan is None:
        return None
    context = _build_pcp_packed_context_metadata(
        cpu_plan,
        chunked_prefill_workspace=torch.empty((workspace, 1), device=device),
        chunked_prefill_workspace_size=workspace,
        page_size=16,
        device=device,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
        align_indptr=False,
        block_table=torch.arange(
            len(context_lens), dtype=torch.int32, device=device
        ).unsqueeze(-1),
    )
    starts = cpu_plan.merged_starts
    query_lens = [b - a for a, b in zip(starts, starts[1:])]
    return (
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
        max(query_lens),
        context,
    )


def test_pcp_shared_tail_single_chunk_and_mixed_decode() -> None:
    plan = _plan(req_idx=(99, 7, 7), num_decodes=1)
    assert plan is not None
    starts, query_lens, max_query_len, context = plan
    assert starts.tolist() == [0, 32]
    assert query_lens.tolist() == [32]
    assert max_query_len == 32
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].num_context_tokens == 96
    assert context.chunks[0].starts.tolist() == [0, 80]
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 0]
    assert mapping.num_base_tokens == 32
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.tolist() == (
        list(range(32)) + list(range(16, 32))
    )


def test_pcp_unaligned_tail_uses_absolute_cache_offset() -> None:
    plan = _plan(context_lens=(70, 100), query_starts=(0, 8, 24))
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [0, 78]
    assert context.chunks[0].seq_lens.tolist() == [70, 22]


def test_pcp_only_shared_context_uses_contiguous_query_slice() -> None:
    plan = _plan(context_lens=(64, 80))
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert context.pcp_mapping.query_selector == slice(0, 32)
    assert context.pcp_mapping.num_base_tokens == 32


def test_pcp_mixed_singleton_and_pair_keep_explicit_query_indices() -> None:
    # The gap from an omitted singleton cancels the repeated tail's length,
    # but the resulting selector is not a contiguous slice.
    plan = _plan(
        req_idx=(10, 11, 20, 20),
        context_lens=(32, 0, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [0, 0, 80]
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 2, 2]
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.tolist() == (
        list(range(16)) + list(range(32, 64)) + list(range(48, 64))
    )


def test_pcp_two_pairs_pack_into_one_native_chunk() -> None:
    plan = _plan(
        workspace=4096,
        req_idx=(0, 0, 1, 1),
        context_lens=(64, 112, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].num_context_tokens == 192
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 2, 0, 2]
    assert mapping.num_base_tokens == 64
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.numel() == 96
    assert mapping.query_selector[64:80].tolist() == list(range(16, 32))


def test_pcp_workspace_splits_preserve_absolute_tail_offsets() -> None:
    plan = _plan(
        req_idx=(0, 0, 1, 1),
        context_lens=(64, 112, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert [c.num_context_tokens for c in context.chunks] == [128, 64]
    assert context.chunks[-1].starts.tolist() == [80, 80]


def test_pcp_no_shared_context_packs_tail_into_base() -> None:
    plan = _plan(
        req_idx=(1, 1, 2, 2),
        context_lens=(0, 48, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert context.pcp_mapping.num_base_tokens == 48
    assert context.pcp_mapping.block_table.flatten().tolist() == [0, 2, 2]
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [16, 0, 80]
    assert context.chunks[0].seq_lens.tolist() == [32, 64, 32]


def test_pcp_reuse_gates_dcp_and_invalid_pairs() -> None:
    assert not _pcp_context_reuse_applicable(2, False, False, True, True)
    assert not _pcp_context_reuse_applicable(1, True, False, True, True)
    assert not _pcp_context_reuse_applicable(1, False, True, True, True)
    assert not _pcp_context_reuse_applicable(1, False, False, False, True)
    assert not _pcp_context_reuse_applicable(1, False, False, True, False)
    assert _pcp_context_reuse_applicable(1, False, False, True, True)
    assert _plan_pcp_context_reuse([7, 7, 7], [64, 80, 96], [0, 16, 32, 48]) is None
    assert _plan_pcp_context_reuse([7, 8, 7], [64, 64, 64], [0, 16, 32, 48]) is None


def test_pcp_restore_merges_tail_and_neutralizes_absent_context() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = _plan(
        req_idx=(7, 7, 8),
        context_lens=(64, 112, 0),
        query_starts=(0, 16, 32, 48),
        device=device,
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    mapping = context.pcp_mapping
    partial = torch.cat(
        (
            torch.full((32, 1, 8), 2.0, device=device),
            torch.full((16, 1, 8), 6.0, device=device),
        )
    )
    lse = torch.zeros((1, 48), device=device)
    output, out_lse = restore_pcp_context_partial(mapping, partial, lse, 48)
    torch.testing.assert_close(output[:16], torch.full((16, 1, 8), 2.0, device=device))
    torch.testing.assert_close(
        output[16:32], torch.full((16, 1, 8), 4.0, device=device)
    )
    torch.testing.assert_close(output[32:], torch.zeros((16, 1, 8), device=device))
    torch.testing.assert_close(out_lse[0, :16], torch.zeros(16, device=device))
    torch.testing.assert_close(
        out_lse[0, 16:32], torch.full((16,), math.log(2), device=device)
    )
    assert torch.isneginf(out_lse[0, 32:]).all()

    hot_plan = _plan(device=device)
    assert hot_plan is not None
    hot_context = hot_plan[3]
    assert hot_context is not None and hot_context.pcp_mapping is not None
    hot_partial = partial.clone()
    hot_lse = lse.clone()
    hot_output, hot_out_lse = restore_pcp_context_partial(
        hot_context.pcp_mapping, hot_partial, hot_lse, 32
    )
    assert hot_output.data_ptr() == hot_partial.data_ptr()
    assert hot_out_lse.data_ptr() == hot_lse.data_ptr()
    torch.testing.assert_close(
        hot_output[16:], torch.full((16, 1, 8), 4.0, device=device)
    )


def test_pcp_context_chunk_out_equals_copy_with_continuation_and_empty_rows() -> None:
    # Exercise the real init/accumulate/restore functions. No fake backend or
    # patched kernel is needed to compare in-place and copying accumulation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = (
        ((256, 272), (0, 16, 32), (7, 7), True),
        ((64, 112, 0), (0, 16, 32, 48), (7, 7, 8), False),
    )
    for context_lens, query_starts, req_idx, multi_chunk in cases:
        plan = _plan(
            req_idx=req_idx,
            context_lens=context_lens,
            query_starts=query_starts,
            device=device,
        )
        assert plan is not None
        context = plan[3]
        assert context is not None and context.pcp_mapping is not None
        assert len(context.chunks) == (2 if multi_chunk else 1)
        assert not context.empty_token_slices
        mapping = context.pcp_mapping
        num_packed = (
            mapping.query_selector.numel()
            if isinstance(mapping.query_selector, torch.Tensor)
            else mapping.query_selector.stop - mapping.query_selector.start
        )
        results = []
        for direct in (False, True):
            single_chunk = len(context.chunks) == 1
            output = (
                torch.empty((num_packed, 1, 8), device=device)
                if direct and not single_chunk
                else None
            )
            output_lse = None
            wrote_any = False
            for chunk in context.chunks:
                tokens = chunk.token_slice.stop - chunk.token_slice.start
                wrote = output is not None and not chunk.is_continuation
                attn_output = (
                    output[chunk.token_slice]
                    if wrote
                    else torch.empty((tokens, 1, 8), device=device)
                )
                attn_output.fill_(float(chunk.index + 1))
                attn_lse = torch.zeros((1, tokens), device=device)
                wrote_any |= wrote
                if output_lse is None:
                    if single_chunk:
                        output, output_lse = attn_output, attn_lse
                        break
                    output, output_lse = init_mla_context_partial(
                        context,
                        attn_output,
                        attn_lse,
                        num_tokens=num_packed,
                        output=output,
                    )
                accumulate_mla_context_chunk(
                    chunk,
                    attn_output,
                    attn_lse,
                    output,
                    output_lse,
                    output_written=wrote,
                )
            assert wrote_any == (direct and multi_chunk)
            results.append(
                restore_pcp_context_partial(
                    mapping, output, output_lse, query_starts[-1]
                )
            )
        for before, after in zip(*results):
            torch.testing.assert_close(after, before)

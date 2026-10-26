# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU PCP job planning and real native chunk/restore contracts."""

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _build_pcp_packed_context_metadata,
    _pcp_context_reuse_applicable,
    _plan_pcp_context_reuse,
    restore_pcp_context_partial,
)


def _build_context(
    *,
    request_ids=(7, 7),
    context_lens=(64, 112),
    query_starts=(0, 16, 32),
    workspace=128,
    device=torch.device("cpu"),
):
    # Construct real native metadata from a CPU plan, without faking builder
    # state or reproducing the builder's decode/query-length calculations.
    plan = _plan_pcp_context_reuse(
        list(request_ids), list(context_lens), list(query_starts)
    )
    assert plan is not None
    context = _build_pcp_packed_context_metadata(
        plan,
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
    assert context is not None
    return plan, context


def test_pcp_shared_tail_single_chunk_mapping() -> None:
    plan, context = _build_context()
    assert plan.merged_starts == [0, 32]
    assert context.pcp_mapping is not None
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
    _, context = _build_context(context_lens=(70, 100), query_starts=(0, 8, 24))
    assert context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [0, 78]
    assert context.chunks[0].seq_lens.tolist() == [70, 22]


def test_pcp_only_shared_context_uses_contiguous_query_slice() -> None:
    _, context = _build_context(context_lens=(64, 80))
    assert context.pcp_mapping is not None
    assert context.pcp_mapping.query_selector == slice(0, 32)
    assert context.pcp_mapping.num_base_tokens == 32


def test_pcp_mixed_singleton_and_pair_keep_explicit_query_indices() -> None:
    # The gap from an omitted singleton cancels the repeated tail's length,
    # but the resulting selector is not a contiguous slice.
    _, context = _build_context(
        request_ids=(10, 11, 20, 20),
        context_lens=(32, 0, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [0, 0, 80]
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 2, 2]
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.tolist() == (
        list(range(16)) + list(range(32, 64)) + list(range(48, 64))
    )


def test_pcp_two_pairs_pack_into_one_native_chunk() -> None:
    _, context = _build_context(
        workspace=4096,
        request_ids=(0, 0, 1, 1),
        context_lens=(64, 112, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].num_context_tokens == 192
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 2, 0, 2]
    assert mapping.num_base_tokens == 64
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.numel() == 96
    assert mapping.query_selector[64:80].tolist() == list(range(16, 32))


def test_pcp_workspace_splits_preserve_absolute_tail_offsets() -> None:
    _, context = _build_context(
        request_ids=(0, 0, 1, 1),
        context_lens=(64, 112, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert context.pcp_mapping is not None
    assert [c.num_context_tokens for c in context.chunks] == [128, 64]
    assert context.chunks[-1].starts.tolist() == [80, 80]


def test_pcp_no_shared_context_packs_tail_into_base() -> None:
    _, context = _build_context(
        request_ids=(1, 1, 2, 2),
        context_lens=(0, 48, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert context.pcp_mapping is not None
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
    _, context = _build_context(
        request_ids=(7, 7, 8),
        context_lens=(64, 112, 0),
        query_starts=(0, 16, 32, 48),
        device=device,
    )
    assert context.pcp_mapping is not None
    mapping = context.pcp_mapping
    # Three heads and a non-contiguous token stride exercise restore's
    # index_select(out=) and the merge kernel's head-stride requirement.
    shared_values = torch.tensor([2.0, 3.0, 4.0], device=device)
    tail_values = torch.tensor([6.0, 7.0, 8.0], device=device)
    shared_weights = torch.tensor([1.0, 2.0, 4.0], device=device)
    packed = torch.cat(
        (
            shared_values.view(1, 3, 1).expand(32, 3, 8),
            tail_values.view(1, 3, 1).expand(16, 3, 8),
        )
    )
    partial = torch.stack((packed, packed), dim=1)[:, 0]
    assert not partial.is_contiguous()
    lse = torch.zeros((3, 48), device=device)
    lse[:, :32] = shared_weights.log().unsqueeze(-1)
    merged_values = (
        shared_values * shared_weights + tail_values
    ) / (shared_weights + 1)
    merged_lse = (shared_weights + 1).log()
    output, out_lse = restore_pcp_context_partial(mapping, partial, lse, 48)
    torch.testing.assert_close(
        output[:16], shared_values.view(1, 3, 1).expand(16, 3, 8)
    )
    torch.testing.assert_close(
        output[16:32], merged_values.view(1, 3, 1).expand(16, 3, 8)
    )
    torch.testing.assert_close(output[32:], torch.zeros((16, 3, 8), device=device))
    torch.testing.assert_close(
        out_lse[:, :16], shared_weights.log().view(3, 1).expand(3, 16)
    )
    torch.testing.assert_close(
        out_lse[:, 16:32], merged_lse.view(3, 1).expand(3, 16)
    )
    assert torch.isneginf(out_lse[:, 32:]).all()

    _, hot_context = _build_context(device=device)
    assert hot_context.pcp_mapping is not None
    hot_partial = partial.clone()
    hot_lse = lse.clone()
    hot_output, hot_out_lse = restore_pcp_context_partial(
        hot_context.pcp_mapping, hot_partial, hot_lse, 32
    )
    assert hot_output.data_ptr() == hot_partial.data_ptr()
    assert hot_out_lse.data_ptr() == hot_lse.data_ptr()
    torch.testing.assert_close(
        hot_output[16:], merged_values.view(1, 3, 1).expand(16, 3, 8)
    )
    torch.testing.assert_close(
        hot_out_lse[:, 16:], merged_lse.view(3, 1).expand(3, 16)
    )

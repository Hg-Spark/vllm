# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import (
    _build_pcp_context_reuse,
    restore_pcp_context_partial,
)
from vllm.v1.attention.ops.merge_attn_states import _merge_attn_states_torch


def _builder(workspace: int = 128):
    return SimpleNamespace(
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
        use_sparse=False,
        page_size=16,
        device=torch.device("cpu"),
        chunked_prefill_workspace_size=workspace,
        chunked_prefill_workspace=torch.empty((workspace, 1)),
        model_config=SimpleNamespace(get_sliding_window=lambda: None),
    )


def _metadata(req_idx=(7, 7)):
    return SimpleNamespace(causal=True, req_idx=np.array(req_idx, dtype=np.intp))


def _plan(
    *,
    builder=None,
    metadata=None,
    num_decodes=0,
    context_lens=(64, 112),
    query_starts=(0, 16, 32),
):
    return _build_pcp_context_reuse(
        builder or _builder(),
        metadata or _metadata(),
        num_decodes=num_decodes,
        num_prefills=len(context_lens),
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
        block_table=torch.arange(len(context_lens), dtype=torch.int32).unsqueeze(-1),
    )


def test_pcp_shared_tail_single_chunk_and_mixed_decode() -> None:
    plan = _plan(metadata=_metadata((99, 7, 7)), num_decodes=1)
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
    assert context.pcp_mapping.num_base_tokens == 24


def test_pcp_only_shared_context_uses_contiguous_query_slice() -> None:
    plan = _plan(context_lens=(64, 80))
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.pcp_mapping.query_selector == slice(0, 32)
    assert context.pcp_mapping.num_base_tokens == 32


def test_pcp_mixed_singleton_and_pair_share_one_chunk() -> None:
    # Q[16:32] has no context; the 16 repeated tail queries exactly
    # offset the gap, so packed length alone cannot imply a slice.
    plan = _plan(
        metadata=_metadata((10, 11, 20, 20)),
        context_lens=(32, 0, 64, 112),
        query_starts=(0, 16, 32, 48, 64),
    )
    assert plan is not None
    assert plan[0].tolist() == [0, 16, 32, 64]
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].starts.tolist() == [0, 0, 80]
    assert context.chunks[0].num_context_tokens == 128
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == [0, 2, 2]
    assert mapping.num_base_tokens == 48
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.tolist() == (
        list(range(16)) + list(range(32, 64)) + list(range(48, 64))
    )


def test_pcp_two_pairs_pack_into_one_native_chunk() -> None:
    pairs = 2
    plan = _plan(
        builder=_builder(4096),
        metadata=_metadata(tuple(i for i in range(pairs) for _ in range(2))),
        context_lens=(64, 112) * pairs,
        query_starts=tuple(range(0, 16 * (2 * pairs + 1), 16)),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 1
    assert context.chunks[0].num_context_tokens == 192
    mapping = context.pcp_mapping
    assert mapping.block_table.flatten().tolist() == (
        [0, 2, 0, 2]
    )
    assert mapping.num_base_tokens == 64
    assert isinstance(mapping.query_selector, torch.Tensor)
    assert mapping.query_selector.numel() == 96
    assert mapping.query_selector[64:80].tolist() == list(range(16, 32))


def test_pcp_workspace_splits_preserve_absolute_tail_offsets() -> None:
    pairs = 2
    plan = _plan(
        metadata=_metadata(tuple(i for i in range(pairs) for _ in range(2))),
        context_lens=(64, 112) * pairs,
        query_starts=tuple(range(0, 16 * (2 * pairs + 1), 16)),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    assert len(context.chunks) == 2
    assert [chunk.num_context_tokens for chunk in context.chunks] == [128, 64]
    # The second chunk uses the absolute cache offsets for both tails.
    assert context.chunks[-1].starts.tolist() == [80, 80]


def test_pcp_no_shared_context_packs_tail_into_base() -> None:
    plan = _plan(
        metadata=_metadata((1, 1, 2, 2)),
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


def test_pcp_unsupported_dcp_falls_back() -> None:
    builder = _builder()
    builder.dcp_world_size = 2
    assert _plan(builder=builder) is None


def test_pcp_restore_merges_tail_and_neutralizes_absent_context(monkeypatch) -> None:
    monkeypatch.setattr(
        mla_attention, "merge_attn_states", _merge_attn_states_torch
    )
    plan = _plan(
        metadata=_metadata((7, 7, 8)),
        context_lens=(64, 112, 0),
        query_starts=(0, 16, 32, 48),
    )
    assert plan is not None
    context = plan[3]
    assert context is not None and context.pcp_mapping is not None
    mapping = context.pcp_mapping
    # Shared Q0,Q1 (2.0) and Tail Q1 (6.0) have equal LSE;
    # singleton Q2 has no context, so it stays neutral.
    partial = torch.cat(
        (torch.full((32, 1, 8), 2.0), torch.full((16, 1, 8), 6.0))
    )
    lse = torch.zeros((1, 48))
    output, out_lse = restore_pcp_context_partial(mapping, partial, lse, 48)
    torch.testing.assert_close(output[:16], torch.full((16, 1, 8), 2.0))
    torch.testing.assert_close(output[16:32], torch.full((16, 1, 8), 4.0))
    torch.testing.assert_close(output[32:], torch.zeros((16, 1, 8)))
    torch.testing.assert_close(out_lse[0, :16], torch.zeros(16))
    torch.testing.assert_close(out_lse[0, 16:32], torch.full((16,), float(np.log(2))))
    assert torch.isneginf(out_lse[0, 32:]).all()

    # A fully covered base is already in original query order. Even with
    # a Tail, restoration should alias the packed storage, not scatter it.
    hot_plan = _plan()
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
    torch.testing.assert_close(hot_output[16:], torch.full((16, 1, 8), 4.0))
    torch.testing.assert_close(hot_out_lse[0, 16:], torch.full((16,), float(np.log(2))))


def test_pcp_context_out_matches_copy_for_continuation_and_empty_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mla_attention, "merge_attn_states", _merge_attn_states_torch)
    monkeypatch.setattr(mla_attention.current_platform, "is_cpu", lambda: True)

    def fake_gather(src_cache, dst, block_table, starts, cu_seq_lens):
        dst[: int(cu_seq_lens[-1])] = 1

    monkeypatch.setattr(
        mla_attention.ops, "gather_mla_context_cache_cpu", fake_gather
    )

    class FakeBackend:
        def __init__(self, direct):
            self.direct = direct
            self.calls = []

        def supports_out(self):
            return self.direct

        def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
            self.calls.append((chunk.is_continuation, out is not None))
            result = out if out is not None else torch.empty((q.shape[0], 1, 1))
            result.fill_(float(chunk.index + 1))
            return result, torch.zeros((1, q.shape[0]))

    layer = SimpleNamespace(
        kv_cache_dtype="auto",
        kv_lora_rank=1,
        num_heads=1,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        kv_b_proj=lambda x: (torch.cat((x, x), dim=-1), None),
        _concat_k_nope_k_pe=lambda k_nope, k_pe: torch.cat(
            (k_nope, k_pe), dim=-1
        ),
    )

    cases = (
        # Multiple chunks: the first chunk can write out=, continuations cannot.
        ((256, 272), (0, 16, 32), (7, 7), True),
        # The zero-context row is excluded from the packed single chunk.
        ((64, 112, 0), (0, 16, 32, 48), (7, 7, 8), False),
    )
    for context_lens, query_starts, req_idx, multi_chunk in cases:
        builder = _builder()
        builder.chunked_prefill_workspace = torch.empty((128, 2))
        plan = _plan(
            builder=builder,
            metadata=_metadata(req_idx),
            context_lens=context_lens,
            query_starts=query_starts,
        )
        assert plan is not None
        context = plan[3]
        assert context is not None
        assert len(context.chunks) == (2 if multi_chunk else 1)
        assert not context.empty_token_slices
        outputs = []
        for direct in (False, True):
            backend = FakeBackend(direct)
            prefill = SimpleNamespace(
                prefill_backend=backend,
                block_table=torch.arange(len(context_lens)).unsqueeze(-1),
                chunked_context=context,
                q_data_type=torch.float32,
                output_dtype=torch.float32,
            )
            outputs.append(
                mla_attention.MLACommonBaseImpl._compute_prefill_context(
                    layer,
                    torch.zeros((query_starts[-1], 1, 2)),
                    torch.empty((1, 1, 2)),
                    SimpleNamespace(prefill=prefill),
                    torch.tensor(1.0),
                )
            )
            assert any(cont for cont, _ in backend.calls) == (
                context_lens[0] == 256
            )
            assert all(not wrote for cont, wrote in backend.calls if cont)
            # Single-chunk execution returns directly without supplying out=,
            # even when the backend supports it.
            assert any(wrote for _, wrote in backend.calls) == (
                direct and multi_chunk
            )

        for before, after in zip(*outputs):
            torch.testing.assert_close(after, before)

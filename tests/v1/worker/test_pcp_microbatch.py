# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_microbatch import (
    PCPAttentionMicrobatchPlan,
    _run_decoder_microbatch_pipeline,
    _slice_single_request_input_batch,
    microbatch_slices,
    parse_pcp_microbatch_size,
    run_tokenwise_microbatches,
)


def test_parse_pcp_microbatch_size() -> None:
    assert parse_pcp_microbatch_size(None) == 0
    assert parse_pcp_microbatch_size({}) == 0
    assert parse_pcp_microbatch_size({"pcp_microbatch_size": 0}) == 0
    assert parse_pcp_microbatch_size({"pcp_microbatch_size": 1024}) == 1024

    for invalid in (-1, 1.5, True, "1024"):
        with pytest.raises(ValueError, match="pcp_microbatch_size"):
            parse_pcp_microbatch_size({"pcp_microbatch_size": invalid})


def test_microbatch_slices_bound_rank_local_rows() -> None:
    assert microbatch_slices(0, 4) == ()
    assert microbatch_slices(3, 4) == (slice(0, 3),)
    assert microbatch_slices(9, 4) == (
        slice(0, 4),
        slice(4, 8),
        slice(8, 9),
    )


def test_tokenwise_microbatch_preserves_output_order_and_chunk_bound() -> None:
    seen_chunk_sizes: list[int] = []

    def forward_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        seen_chunk_sizes.append(hidden_states.shape[0])
        return hidden_states * 3

    hidden_states = torch.arange(14, dtype=torch.float32).view(7, 2)
    output = run_tokenwise_microbatches(
        forward_fn,
        hidden_states,
        microbatch_size=3,
    )

    assert seen_chunk_sizes == [3, 3, 1]
    assert torch.equal(output, hidden_states * 3)


def test_disabled_microbatch_calls_sublayer_once() -> None:
    calls = 0

    def forward_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return hidden_states + 1

    hidden_states = torch.zeros(5, 2)
    output = run_tokenwise_microbatches(forward_fn, hidden_states, 0)

    assert calls == 1
    assert torch.equal(output, torch.ones_like(hidden_states))


def _make_single_request_input_batch(
    *,
    num_tokens: int,
    num_computed_tokens: int,
    prefill_len: int,
) -> InputBatch:
    positions = torch.arange(
        num_computed_tokens,
        num_computed_tokens + num_tokens,
        dtype=torch.int64,
    )
    return InputBatch(
        req_ids=["req"],
        num_reqs=1,
        num_reqs_after_padding=1,
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32),
        num_scheduled_tokens=np.array([num_tokens], dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        query_start_loc_np=np.array([0, num_tokens], dtype=np.int32),
        seq_lens=torch.tensor(
            [num_computed_tokens + num_tokens], dtype=torch.int32
        ),
        seq_lens_cpu_upper_bound=torch.tensor(
            [num_computed_tokens + num_tokens], dtype=torch.int32
        ),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([num_computed_tokens], dtype=np.int32),
        prefill_len_np=np.array([prefill_len], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array(
            [min(num_computed_tokens, prefill_len)], dtype=np.int32
        ),
        is_prefilling_np=np.array([num_computed_tokens < prefill_len], dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
        positions=positions,
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
        logits_indices=torch.tensor([num_tokens - 1], dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def test_attention_microbatch_slice_advances_causal_context() -> None:
    input_batch = _make_single_request_input_batch(
        num_tokens=9,
        num_computed_tokens=5,
        prefill_len=32,
    )

    first = _slice_single_request_input_batch(input_batch, slice(0, 4))
    second = _slice_single_request_input_batch(input_batch, slice(4, 8))
    last = _slice_single_request_input_batch(input_batch, slice(8, 9))

    assert first.query_start_loc_np.tolist() == [0, 4]
    assert second.query_start_loc_np.tolist() == [0, 4]
    assert last.query_start_loc_np.tolist() == [0, 1]

    assert first.num_computed_tokens_np.tolist() == [5]
    assert second.num_computed_tokens_np.tolist() == [9]
    assert last.num_computed_tokens_np.tolist() == [13]

    assert first.seq_lens.tolist() == [9]
    assert second.seq_lens.tolist() == [13]
    assert last.seq_lens.tolist() == [14]

    assert first.positions.tolist() == [5, 6, 7, 8]
    assert second.positions.tolist() == [9, 10, 11, 12]
    assert last.positions.tolist() == [13]


def test_decoder_microbatch_pipeline_interleaves_and_reuses_hidden_buffer() -> None:
    plan = PCPAttentionMicrobatchPlan(
        slices=(slice(0, 2), slice(2, 4)),
        attn_metadata=({"value": 1.0}, {"value": 3.0}),
    )
    hidden_buffer = torch.zeros(4, 2)
    residual = torch.arange(8, dtype=torch.float32).view(4, 2)
    original_residual = residual.clone()
    hidden_ptr = hidden_buffer.data_ptr()
    calls: list[str] = []

    def attention_forward(
        token_slice: slice,
        metadata: dict[str, float],
    ) -> torch.Tensor:
        calls.append(f"attn{token_slice.start}")
        rows = int(token_slice.stop) - int(token_slice.start)
        return torch.full((rows, 2), metadata["value"])

    def post_norm(
        attention: torch.Tensor,
        residual_mb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("norm")
        return attention + 2, residual_mb + 10

    def mlp_forward(hidden_mb: torch.Tensor) -> torch.Tensor:
        calls.append("mlp")
        return hidden_mb * 4

    output, output_residual = _run_decoder_microbatch_pipeline(
        plan,
        hidden_buffer,
        residual,
        attention_forward,
        post_norm,
        mlp_forward,
    )

    assert calls == ["attn0", "norm", "mlp", "attn2", "norm", "mlp"]
    assert output.data_ptr() == hidden_ptr
    assert output_residual.data_ptr() == residual.data_ptr()
    assert torch.equal(
        output,
        torch.tensor(
            [[12.0, 12.0], [12.0, 12.0], [20.0, 20.0], [20.0, 20.0]]
        ),
    )
    assert torch.equal(output_residual, original_residual + 10)

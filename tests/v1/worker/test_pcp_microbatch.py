# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.pcp_microbatch import (
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

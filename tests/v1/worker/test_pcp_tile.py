# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.pcp_tile import (
    parse_pcp_ffn_microbatch_size,
    parse_pcp_tile_size,
    run_tokenwise_microbatches,
    tile_slices,
)


def test_parse_pcp_tile_size() -> None:
    assert parse_pcp_tile_size(None) == 0
    assert parse_pcp_tile_size({}) == 0
    assert parse_pcp_tile_size({"pcp_tile_size": 0}) == 0
    assert parse_pcp_tile_size({"pcp_tile_size": 4096}) == 4096
    # No compatibility alias: the old key is intentionally ignored.
    assert parse_pcp_tile_size({"pcp_microbatch_size": 4096}) == 0

    for invalid in (-1, 1.5, True, "4096"):
        with pytest.raises(ValueError, match="pcp_tile_size"):
            parse_pcp_tile_size({"pcp_tile_size": invalid})


def test_ffn_microbatch_defaults_to_tile_size() -> None:
    assert parse_pcp_ffn_microbatch_size({}, 4096) == 4096
    assert (
        parse_pcp_ffn_microbatch_size(
            {"pcp_ffn_microbatch_size": 1024}, 4096
        )
        == 1024
    )


def test_tile_slices_bound_rank_local_rows() -> None:
    assert tile_slices(0, 4) == ()
    assert tile_slices(3, 4) == (slice(0, 3),)
    assert tile_slices(9, 4) == (
        slice(0, 4),
        slice(4, 8),
        slice(8, 9),
    )


def test_tokenwise_microbatch_preserves_output_order() -> None:
    seen: list[int] = []

    def forward_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        seen.append(hidden_states.shape[0])
        return hidden_states * 3

    hidden_states = torch.arange(14, dtype=torch.float32).view(7, 2)
    output = run_tokenwise_microbatches(
        forward_fn,
        hidden_states,
        microbatch_size=3,
    )

    assert seen == [3, 3, 1]
    assert torch.equal(output, hidden_states * 3)

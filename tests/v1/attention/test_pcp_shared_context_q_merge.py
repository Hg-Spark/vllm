# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch


def _noncausal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def test_combined_queries_equal_separate_queries_for_shared_context() -> None:
    generator = torch.Generator().manual_seed(17)
    q0 = torch.randn(7, 16, generator=generator)
    q1 = torch.randn(11, 16, generator=generator)
    k = torch.randn(23, 16, generator=generator)
    v = torch.randn(23, 12, generator=generator)

    separate = torch.cat(
        (
            _noncausal_attention(q0, k, v),
            _noncausal_attention(q1, k, v),
        ),
        dim=0,
    )
    combined = _noncausal_attention(torch.cat((q0, q1), dim=0), k, v)

    torch.testing.assert_close(combined, separate)

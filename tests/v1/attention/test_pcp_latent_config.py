# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.pcp_mla as pcp_mla
from vllm.v1.attention.backends.mla.pcp_mla import (
    TritonPCPLatentPrefixEngine,
    _validate_pcp_merge_lse,
)
from vllm.v1.attention.selector import _parse_pcp_latent_mla_config


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        (None, (False, False)),
        ({}, (False, False)),
        ({"pcp_latent_mla": False}, (False, False)),
        ({"pcp_latent_mla": True}, (True, False)),
        ({"pcp_latent_mla": {}}, (True, False)),
        ({"pcp_latent_mla": {"enabled": True}}, (True, False)),
        (
            {"pcp_latent_mla": {"enabled": True, "strict": True}},
            (True, True),
        ),
    ],
)
def test_parse_pcp_latent_mla_config(additional_config, expected):
    assert _parse_pcp_latent_mla_config(additional_config) == expected


@pytest.mark.parametrize(
    "additional_config",
    [
        {"pcp_latent_mla": "yes"},
        {"pcp_latent_mla": {"enabled": 1}},
        {"pcp_latent_mla": {"strict": 1}},
        {"pcp_latent_mla": {"enabled": False, "strict": True}},
        {"pcp_latent_mla": {"unknown": True}},
    ],
)
def test_parse_pcp_latent_mla_config_rejects_invalid(additional_config):
    with pytest.raises(ValueError):
        _parse_pcp_latent_mla_config(additional_config)


def test_pcp_merge_lse_contract_rejects_bfloat16() -> None:
    with pytest.raises(ValueError, match="must be float32"):
        _validate_pcp_merge_lse(
            "prefix",
            torch.empty(2, 5, dtype=torch.bfloat16),
            num_heads=2,
            num_tokens=5,
            device=torch.device("cpu"),
        )


def test_latent_prefix_engine_returns_float32_lse_for_bfloat16_query(
    monkeypatch,
) -> None:
    seen = {}

    def fake_decode_attention_fwd(
        _q,
        _k,
        _v,
        output,
        lse,
        _block_table,
        _context_lens,
        _attn_logits,
        *_args,
        **_kwargs,
    ) -> None:
        seen["lse_dtype"] = lse.dtype
        output.zero_()
        lse.zero_()

    monkeypatch.setattr(pcp_mla, "decode_attention_fwd", fake_decode_attention_fwd)

    impl = SimpleNamespace(
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        num_heads=2,
        kv_lora_rank=4,
        v_head_dim=2,
        scale=0.5,
    )
    num_tokens = 5
    q = torch.randn(num_tokens, 2, 4, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 5, dtype=torch.bfloat16)
    block_table = torch.zeros(num_tokens, 1, dtype=torch.int32)
    context_lens = torch.full((num_tokens,), 16, dtype=torch.int32)
    w_uk_t = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    w_uv = torch.randn(2, 4, 2, dtype=torch.bfloat16)

    output, lse = TritonPCPLatentPrefixEngine.run(
        impl,
        q,
        cache,
        block_table,
        context_lens,
        w_uk_t,
        w_uv,
        torch.tensor(1.0),
    )

    assert output.shape == (num_tokens, 2, 2)
    assert output.dtype == torch.bfloat16
    assert lse.shape == (2, num_tokens)
    assert lse.dtype == torch.float32
    assert lse.is_contiguous()
    assert seen["lse_dtype"] == torch.float32

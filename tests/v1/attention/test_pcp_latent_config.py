# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.pcp_mla as pcp_mla
from vllm.v1.attention.backends.mla.pcp_mla import (
    FlashInferPCPLatentPrefixEngine,
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


def test_flashinfer_latent_prefix_uses_full_query_and_reuses_plan(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeMLAWrapper:
        def __init__(self, workspace, *, backend):
            seen["workspace_bytes"] = workspace.numel() * workspace.element_size()
            seen["backend"] = backend
            self.plan_calls = 0
            self.run_calls = 0
            seen["wrapper"] = self

        def plan(
            self,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_data_type,
            kv_data_type,
        ):
            self.plan_calls += 1
            seen["qo_indptr"] = qo_indptr.tolist()
            seen["kv_indptr"] = kv_indptr.tolist()
            seen["kv_indices"] = kv_indices.tolist()
            seen["kv_len_arr"] = kv_len_arr.tolist()
            seen["plan_args"] = (
                num_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                causal,
                sm_scale,
                q_data_type,
                kv_data_type,
            )

        def run(
            self,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            *,
            return_lse,
            return_lse_base_on_e,
        ):
            self.run_calls += 1
            seen["q_nope_shape"] = tuple(q_nope.shape)
            seen["q_pe_shape"] = tuple(q_pe.shape)
            seen["ckv_shape"] = tuple(ckv_cache.shape)
            seen["kpe_shape"] = tuple(kpe_cache.shape)
            seen["return_lse"] = return_lse
            seen["return_lse_base_on_e"] = return_lse_base_on_e
            out = torch.zeros_like(q_nope)
            lse = torch.zeros(
                q_nope.shape[:2], dtype=torch.float32, device=q_nope.device
            )
            return out, lse

    pcp_mla._FLASHINFER_LATENT_PREFIX_RUNTIMES.clear()
    monkeypatch.setattr(pcp_mla, "_flashinfer_mla_wrapper_cls", lambda: FakeMLAWrapper)
    monkeypatch.setattr(
        pcp_mla.envs,
        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
        4096,
        raising=False,
    )

    impl = SimpleNamespace(
        qk_nope_head_dim=3,
        qk_rope_head_dim=1,
        num_heads=2,
        kv_lora_rank=4,
        v_head_dim=2,
        scale=0.5,
    )
    num_tokens = 513
    q = torch.randn(num_tokens, 2, 4, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 5, dtype=torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    context_len = torch.tensor([17], dtype=torch.int32)
    w_uk_t = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    w_uv = torch.randn(2, 4, 2, dtype=torch.bfloat16)
    plan_key = object()

    output, lse = FlashInferPCPLatentPrefixEngine.run(
        impl,
        q,
        cache,
        block_table,
        context_len,
        w_uk_t,
        w_uv,
        plan_key=plan_key,
    )
    output_2, lse_2 = FlashInferPCPLatentPrefixEngine.run(
        impl,
        q,
        cache,
        block_table,
        context_len,
        w_uk_t,
        w_uv,
        plan_key=plan_key,
    )

    wrapper = seen["wrapper"]
    assert isinstance(wrapper, FakeMLAWrapper)
    assert wrapper.plan_calls == 1
    assert wrapper.run_calls == 2
    assert seen["workspace_bytes"] == 4096
    assert seen["backend"] == "fa2"
    assert seen["qo_indptr"] == [0, num_tokens]
    assert seen["kv_indptr"] == [0, 2]
    assert seen["kv_indices"] == [0, 1]
    assert seen["kv_len_arr"] == [17]
    assert seen["q_nope_shape"] == (num_tokens, 2, 4)
    assert seen["q_pe_shape"] == (num_tokens, 2, 1)
    assert seen["ckv_shape"] == (2, 16, 4)
    assert seen["kpe_shape"] == (2, 16, 1)
    assert seen["return_lse"] is True
    assert seen["return_lse_base_on_e"] is True

    assert output.shape == (num_tokens, 2, 2)
    assert output.dtype == torch.bfloat16
    assert lse.shape == (2, num_tokens)
    assert lse.dtype == torch.float32
    assert lse.is_contiguous()
    torch.testing.assert_close(output_2, output)
    torch.testing.assert_close(lse_2, lse)

    pcp_mla._FLASHINFER_LATENT_PREFIX_RUNTIMES.clear()

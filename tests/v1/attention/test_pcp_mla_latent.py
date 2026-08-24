# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.pcp import _pad_prefill_for_collective
from vllm.v1.attention.backends.mla.pcp_mla import (
    PCPMLAImplMixin,
    TritonPCPLatentPrefixEngine,
    get_pcp_mla_backend,
    split_unquantized_mla_up_weights,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
)


def test_pcp_collective_padding_does_not_change_local_payload() -> None:
    tensor = torch.arange(6, dtype=torch.float32).view(3, 2)
    padded = _pad_prefill_for_collective(
        tensor, num_decode_tokens=0, collective_num_tokens=5
    )
    assert padded.shape == (5, 2)
    torch.testing.assert_close(padded[:3], tensor)
    torch.testing.assert_close(padded[3:], torch.zeros(2, 2))


def test_split_mla_up_weights_is_zero_copy_and_matches_layout() -> None:
    weight = torch.arange(40, dtype=torch.float32).view(10, 4)
    w_uk_t, w_uv = split_unquantized_mla_up_weights(
        weight,
        num_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        v_head_dim=2,
    )
    per_head = weight.view(2, 5, 4)
    torch.testing.assert_close(w_uk_t, per_head[:, :3])
    torch.testing.assert_close(w_uv, per_head[:, 3:].transpose(1, 2))
    assert w_uk_t.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert w_uv.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


def test_absorbed_mla_context_matches_expanded_kv_algebra() -> None:
    torch.manual_seed(0)
    weight = torch.randn(10, 4, dtype=torch.float64)
    w_uk_t, w_uv = split_unquantized_mla_up_weights(
        weight,
        num_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        v_head_dim=2,
    )
    q_nope = torch.randn(5, 2, 3, dtype=torch.float64)
    compressed_kv = torch.randn(7, 4, dtype=torch.float64)
    k_nope = torch.einsum("sl,npl->snp", compressed_kv, w_uk_t)
    v = torch.einsum("sl,nlv->snv", compressed_kv, w_uv)
    scores_expanded = torch.einsum("bnp,snp->bns", q_nope, k_nope)
    probs = torch.softmax(scores_expanded, dim=-1)
    out_expanded = torch.einsum("bns,snv->bnv", probs, v)
    q_latent = torch.einsum("bnp,npl->bnl", q_nope, w_uk_t)
    scores_latent = torch.einsum("bnl,sl->bns", q_latent, compressed_kv)
    latent_out = torch.einsum("bns,sl->bnl", probs, compressed_kv)
    out_latent = torch.einsum("bnl,nlv->bnv", latent_out, w_uv)
    torch.testing.assert_close(scores_latent, scores_expanded)
    torch.testing.assert_close(out_latent, out_expanded)


def test_pcp_wrapper_preserves_selected_backend_and_native_decode() -> None:
    wrapped = get_pcp_mla_backend(TritonMLABackend)
    assert wrapped is get_pcp_mla_backend(TritonMLABackend)
    assert issubclass(wrapped, TritonMLABackend)
    assert issubclass(wrapped.get_impl_cls(), TritonMLAImpl)
    assert wrapped.get_name() == TritonMLABackend.get_name()
    assert wrapped.get_impl_cls().forward_mqa is TritonMLAImpl.forward_mqa
    assert wrapped.get_impl_cls().pcp_prefix_engine is TritonPCPLatentPrefixEngine
    module = sys.modules[wrapped.__module__]
    assert getattr(module, wrapped.__name__) is wrapped
    impl_cls = wrapped.get_impl_cls()
    assert getattr(module, impl_cls.__name__) is impl_cls


def test_pcp_wrapper_is_not_gated_on_triton_backend_name() -> None:
    class AlternateDenseMLABackend(TritonMLABackend):
        @staticmethod
        def get_name() -> str:
            return "ALTERNATE_DENSE_MLA"

    wrapped = get_pcp_mla_backend(AlternateDenseMLABackend)
    assert issubclass(wrapped, AlternateDenseMLABackend)
    assert wrapped.get_name() == "ALTERNATE_DENSE_MLA"


def _dummy_impl(*, dcp_world_size: int = 1, weight_dtype=torch.bfloat16):
    class DummyPCPImpl(PCPMLAImplMixin):
        pass

    impl = object.__new__(DummyPCPImpl)
    impl.dcp_world_size = dcp_world_size
    impl.num_heads = 2
    impl.kv_lora_rank = 4
    impl.qk_nope_head_dim = 3
    impl.qk_rope_head_dim = 1
    impl.v_head_dim = 2
    impl.kv_cache_dtype = "fp8"
    impl.kv_b_proj = SimpleNamespace(weight=torch.empty(10, 4, dtype=weight_dtype))
    return impl


def test_empty_query_defense_precedes_metadata_and_native_fallback() -> None:
    class NativeFallback:
        def forward_mha(self, *_args, **_kwargs):
            raise AssertionError("empty Q reached native MLA backend")

    class DummyPCPImpl(PCPMLAImplMixin, NativeFallback):
        pass

    impl = object.__new__(DummyPCPImpl)
    q = torch.empty(0, 2, 4, dtype=torch.bfloat16)
    kv = torch.empty(0, 4, dtype=torch.bfloat16)
    k_pe = torch.empty(0, 1, dtype=torch.bfloat16)
    output = torch.empty(0, 4, dtype=torch.bfloat16)

    impl.forward_mha(
        q,
        kv,
        k_pe,
        torch.empty(0, dtype=torch.bfloat16),
        SimpleNamespace(prefill=None),
        torch.tensor(1.0),
        output,
    )


def test_pcp_latent_weights_allow_fp8_kv_cache() -> None:
    impl = _dummy_impl()
    w_uk_t, w_uv = impl._pcp_latent_context_weights()
    assert w_uk_t.shape == (2, 3, 4)
    assert w_uv.shape == (2, 4, 2)


def test_pcp_runtime_rejects_dcp_after_resolution() -> None:
    impl = _dummy_impl(dcp_world_size=2)
    with pytest.raises(NotImplementedError, match="requires DCP=1"):
        impl._validate_pcp_latent_context_runtime(
            torch.empty(1, 2, 4, dtype=torch.bfloat16), None
        )


def test_pcp_model_weight_layout_fails_instead_of_silent_fallback() -> None:
    impl = _dummy_impl(weight_dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="BF16/FP16 kv_b_proj"):
        impl._pcp_latent_context_weights()

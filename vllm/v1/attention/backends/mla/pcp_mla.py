# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP-specific dense MLA wrapping and latent-prefix execution.

The selected upstream dense MLA backend remains authoritative for local prefill
and normal decode. PCP replaces only cached-prefix chunked-context
materialization with a common compressed-KV latent-prefix engine.
"""

from functools import cache
import sys

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

_PCP_LATENT_CONTEXT_QUERY_CHUNK = 256


def split_unquantized_mla_up_weights(
    weight: torch.Tensor,
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-copy W_UK^T and W_UV views from ``kv_b_proj.weight``."""
    expected = (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    if tuple(weight.shape) != expected:
        raise ValueError(
            "Unexpected MLA kv_b_proj weight shape for latent PCP context: "
            f"got={tuple(weight.shape)}, expected={expected}"
        )
    per_head = weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    return (
        per_head[:, :qk_nope_head_dim, :],
        per_head[:, qk_nope_head_dim:, :].transpose(1, 2),
    )


class TritonPCPLatentPrefixEngine:
    """Compressed-MLA prefix engine reusable by dense top-level backends."""

    @staticmethod
    def run(
        impl,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        w_uk_t: torch.Tensor,
        w_uv: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_nope, q_pe = q.split(
            [impl.qk_nope_head_dim, impl.qk_rope_head_dim], dim=-1
        )
        ql_nope = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)
        latent_q = torch.cat((ql_nope, q_pe), dim=-1)

        batch = q.shape[0]
        latent_out = torch.empty(
            batch,
            impl.num_heads,
            impl.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.empty(batch, impl.num_heads, dtype=q.dtype, device=q.device)
        attn_logits = torch.empty(
            batch,
            impl.num_heads,
            1,
            impl.kv_lora_rank + 1,
            dtype=torch.float32,
            device=q.device,
        )
        cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = cache[..., : impl.kv_lora_rank]
        page_size = cache.size(1)

        empty_context = context_lens == 0
        safe_context_lens = context_lens.clamp_min(1)
        decode_attention_fwd(
            latent_q,
            cache,
            kv_c_cache,
            latent_out,
            lse,
            block_table,
            safe_context_lens,
            attn_logits,
            1,
            impl.scale,
            page_size,
            k_scale=k_scale,
            v_scale=k_scale,
            is_mla=True,
        )
        latent_out.masked_fill_(empty_context[:, None, None], 0)
        lse.masked_fill_(empty_context[:, None], float("-inf"))

        context_output = torch.empty(
            batch,
            impl.num_heads,
            impl.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        torch.bmm(
            latent_out.transpose(0, 1),
            w_uv,
            out=context_output.transpose(0, 1),
        )
        return context_output, lse.transpose(0, 1)


class PCPMLAImplMixin:
    """Dense MLA PCP override shared by wrapped upstream MLA implementations."""

    pcp_prefix_engine = TritonPCPLatentPrefixEngine

    def _pcp_latent_context_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self.kv_b_proj, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise NotImplementedError(
                "PCP latent MLA currently requires a directly addressable "
                "kv_b_proj.weight Tensor. Packed model-weight layouts need a "
                "load-time canonical W_UK/W_UV binding and are not enabled yet."
            )
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "PCP latent MLA currently requires BF16/FP16 kv_b_proj model "
                f"weights, got {weight.dtype}. FP8 KV cache is supported; this "
                "restriction applies to model weights."
            )
        try:
            return split_unquantized_mla_up_weights(
                weight,
                num_heads=self.num_heads,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                v_head_dim=self.v_head_dim,
            )
        except ValueError as exc:
            raise NotImplementedError(
                "PCP latent MLA does not support this kv_b_proj model-weight layout."
            ) from exc

    def _validate_pcp_latent_context_runtime(
        self,
        q: torch.Tensor,
        output_scale: torch.Tensor | None,
    ) -> None:
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "PCP latent MLA cached-prefix attention currently requires DCP=1; "
                f"got dcp_world_size={self.dcp_world_size}."
            )
        if output_scale is not None:
            raise NotImplementedError(
                "PCP latent MLA cached-prefix attention does not support "
                "quantized/scaled attention output yet."
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "PCP latent MLA requires BF16/FP16 model query input, "
                f"got {q.dtype}."
            )

    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        if q.shape[0] == 0:
            return

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill

        if prefill_metadata.chunked_context is None:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        self._validate_pcp_latent_context_runtime(q, output_scale)
        assert prefill_metadata.prefill_backend is not None
        w_uk_t, w_uv = self._pcp_latent_context_weights()

        # Local/new KV follows the upstream-selected prefill backend.
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k = self._concat_k_nope_k_pe(k_nope, k_pe)
        suffix_q, suffix_k, suffix_v = q, k, v
        if prefill_metadata.q_data_type == current_platform.fp8_dtype():
            suffix_q = q.to(prefill_metadata.q_data_type)
            suffix_k = k.to(prefill_metadata.q_data_type)
            suffix_v = v.to(prefill_metadata.q_data_type)
        suffix_output, suffix_lse = (
            prefill_metadata.prefill_backend.run_prefill_new_tokens(
                q=suffix_q,
                k=suffix_k,
                v=suffix_v,
                return_softmax_lse=True,
            )
        )
        suffix_output = suffix_output[..., : self.v_head_dim]

        # Cached prefix uses the conservative fixed-size execution shape.
        chunked = prefill_metadata.chunked_context
        query_lens = (
            prefill_metadata.query_start_loc[1:]
            - prefill_metadata.query_start_loc[:-1]
        )
        seq_ids = torch.arange(
            attn_metadata.num_prefills, dtype=torch.long, device=q.device
        )
        query_to_seq = torch.repeat_interleave(
            seq_ids, query_lens, output_size=q.shape[0]
        )
        context_lens = chunked.context_lens
        final_output = output.view(-1, self.num_heads, self.v_head_dim)

        for start in range(0, q.shape[0], _PCP_LATENT_CONTEXT_QUERY_CHUNK):
            end = min(start + _PCP_LATENT_CONTEXT_QUERY_CHUNK, q.shape[0])
            seq_idx = query_to_seq[start:end]
            block_table_chunk = prefill_metadata.block_table.index_select(0, seq_idx)
            context_lens_chunk = context_lens.index_select(0, seq_idx)
            context_output, context_lse = self.pcp_prefix_engine.run(
                self,
                q[start:end],
                kv_c_and_k_pe_cache,
                block_table_chunk,
                context_lens_chunk,
                w_uk_t,
                w_uv,
                k_scale,
            )
            merge_attn_states(
                output=final_output[start:end],
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output[start:end],
                suffix_lse=suffix_lse[:, start:end],
            )


@cache
def get_pcp_mla_backend(
    base_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    """Wrap any dense ``MLACommonBackend`` while retaining native decode."""
    if not issubclass(base_backend, MLACommonBackend):
        raise NotImplementedError(
            "PCP dense MLA wrapping requires an MLACommonBackend; got "
            f"{base_backend.__module__}.{base_backend.__qualname__}."
        )
    base_impl = base_backend.get_impl_cls()
    if not issubclass(base_impl, MLACommonImpl):
        raise NotImplementedError(
            "PCP dense MLA wrapping requires an MLACommonImpl; got "
            f"{base_impl.__module__}.{base_impl.__qualname__}."
        )

    class PCPMLAImpl(PCPMLAImplMixin, base_impl):  # type: ignore[misc, valid-type]
        pass

    class PCPMLABackend(base_backend):  # type: ignore[misc, valid-type]
        @staticmethod
        def get_impl_cls():
            return PCPMLAImpl

    PCPMLAImpl.__name__ = f"PCP{base_impl.__name__}"
    PCPMLAImpl.__qualname__ = PCPMLAImpl.__name__
    PCPMLAImpl.__module__ = __name__
    PCPMLABackend.__name__ = f"PCP{base_backend.__name__}"
    PCPMLABackend.__qualname__ = PCPMLABackend.__name__
    PCPMLABackend.__module__ = __name__
    module = sys.modules[__name__]
    setattr(module, PCPMLAImpl.__name__, PCPMLAImpl)
    setattr(module, PCPMLABackend.__name__, PCPMLABackend)
    return PCPMLABackend

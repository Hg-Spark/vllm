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

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)

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


def _validate_pcp_merge_lse(
    name: str,
    lse: torch.Tensor,
    *,
    num_heads: int,
    num_tokens: int,
    device: torch.device,
) -> None:
    """Validate the LSE contract before PCP calls the existing merge op."""
    expected_shape = (num_heads, num_tokens)
    if tuple(lse.shape) != expected_shape:
        raise ValueError(
            f"{name} LSE shape must be {expected_shape}, got {tuple(lse.shape)}"
        )
    if lse.dtype != torch.float32:
        raise ValueError(f"{name} LSE must be float32, got {lse.dtype}")
    if lse.device != device:
        raise ValueError(f"{name} LSE must be on {device}, got {lse.device}")


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
        batch = q.shape[0]
        if batch <= 0 or batch > _PCP_LATENT_CONTEXT_QUERY_CHUNK:
            raise ValueError(
                "PCP latent-prefix engine batch must be in [1, 256], "
                f"got {batch}"
            )
        if block_table.shape[0] != batch:
            raise ValueError(
                "PCP latent-prefix block-table rows must match query rows: "
                f"block_table={block_table.shape[0]} query={batch}"
            )
        if context_lens.shape[0] != batch:
            raise ValueError(
                "PCP latent-prefix context lengths must match query rows: "
                f"context_lens={context_lens.shape[0]} query={batch}"
            )

        q_nope, q_pe = q.split(
            [impl.qk_nope_head_dim, impl.qk_rope_head_dim], dim=-1
        )
        ql_nope = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)
        latent_q = torch.cat((ql_nope, q_pe), dim=-1)

        # Use a fixed 256-row allocation shape. The caching allocator can reuse
        # these same-sized buffers across continuation tiles, avoiding a sequence
        # of differently-sized workspaces at the tail of each scheduler step.
        capacity = _PCP_LATENT_CONTEXT_QUERY_CHUNK
        latent_out_storage = torch.empty(
            capacity,
            impl.num_heads,
            impl.kv_lora_rank,
            dtype=q.dtype,
            device=q.device,
        )
        # merge_attn_states' CUDA path consumes LSE values as float32. Keep the
        # latent producer on that contract even when model activations are BF16.
        lse_storage = torch.empty(
            capacity, impl.num_heads, dtype=torch.float32, device=q.device
        )
        attn_logits_storage = torch.empty(
            capacity,
            impl.num_heads,
            1,
            impl.kv_lora_rank + 1,
            dtype=torch.float32,
            device=q.device,
        )
        context_output_storage = torch.empty(
            capacity,
            impl.num_heads,
            impl.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        latent_out = latent_out_storage[:batch]
        lse = lse_storage[:batch]
        attn_logits = attn_logits_storage[:batch]

        cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = cache[..., : impl.kv_lora_rank]
        page_size = cache.size(1)

        # The Triton stage-1 kernel indexes B_Seqlen as a contiguous vector, so
        # context_lens must be materialized even when all rows share one value.
        context_lens = context_lens.contiguous()
        empty_context = context_lens == 0
        safe_context_lens = context_lens.clamp_min(1).contiguous()
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

        context_output = context_output_storage[:batch]
        torch.bmm(
            latent_out.transpose(0, 1),
            w_uv,
            out=context_output.transpose(0, 1),
        )
        # merge_attn_states consumes [heads, tokens]. Make the transposed LSE
        # explicit and contiguous so it cannot inherit a fragile stride pattern.
        return context_output, lse.transpose(0, 1).contiguous()


class PCPMLAImplMixin:
    """Dense MLA PCP override shared by wrapped upstream MLA implementations."""

    pcp_prefix_engine = TritonPCPLatentPrefixEngine
    pcp_latent_strict = False

    def _pcp_latent_context_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self.kv_b_proj, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise NotImplementedError(
                "kv_b_proj.weight is not a directly addressable Tensor; packed "
                "model-weight layouts require a canonical W_UK/W_UV binding"
            )
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "model kv_b_proj weights must be BF16/FP16, "
                f"got {weight.dtype}"
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
                "kv_b_proj model-weight layout is unsupported by latent-prefix MLA"
            ) from exc

    def _validate_pcp_latent_context_runtime(
        self,
        q: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output_scale: torch.Tensor | None,
    ) -> None:
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                f"DCP must be 1 during correctness bring-up, got {self.dcp_world_size}"
            )
        if output_scale is not None:
            raise NotImplementedError(
                "quantized/scaled attention output is not supported yet"
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"query dtype must be BF16/FP16, got {q.dtype}"
            )
        if attn_metadata.num_prefills != 1:
            raise NotImplementedError(
                "multi-request latent continuation is disabled during correctness "
                f"bring-up, got num_prefills={attn_metadata.num_prefills}"
            )

    def _pcp_fallback_forward(
        self,
        reason: str,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
    ) -> None:
        message = (
            "PCP LATENT MLA FALLBACK: requested=enabled "
            f"strict={self.pcp_latent_strict} reason={reason}; "
            f"impl={type(self).__module__}.{type(self).__qualname__}; "
            "fallback_path=native-expanded-prefix"
        )
        if self.pcp_latent_strict:
            raise RuntimeError(message)
        logger.warning_once(message)
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

        # Fresh prefill and normal decode remain entirely native. The latent
        # path is only relevant once chunked_context describes a cached prefix.
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

        try:
            self._validate_pcp_latent_context_runtime(
                q, attn_metadata, output_scale
            )
            assert prefill_metadata.prefill_backend is not None
            if prefill_metadata.block_table.shape[0] < 1:
                raise NotImplementedError("cached-prefix block table has no rows")
            chunked = prefill_metadata.chunked_context
            if chunked.context_lens.shape[0] < 1:
                raise NotImplementedError("cached-prefix context_lens has no rows")
            w_uk_t, w_uv = self._pcp_latent_context_weights()
        except (AssertionError, NotImplementedError, ValueError) as exc:
            return self._pcp_fallback_forward(
                str(exc),
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        # Local/new KV follows the upstream-selected prefill backend. Only this
        # scheduler-step suffix is projected through kv_b_proj.
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
        try:
            _validate_pcp_merge_lse(
                "suffix",
                suffix_lse,
                num_heads=self.num_heads,
                num_tokens=q.shape[0],
                device=q.device,
            )
        except ValueError as exc:
            return self._pcp_fallback_forward(
                str(exc),
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        # Correctness-first single-request continuation path. Avoid the former
        # per-query block_table.index_select completely: one canonical block-table
        # row is exposed with stride 0, which the Triton kernel supports because
        # it receives stride_req_to_tokens_b explicitly. Context lengths are
        # materialized contiguously because that kernel indexes B_Seqlen directly.
        block_table_row = prefill_metadata.block_table[:1]
        context_len_row = chunked.context_lens[:1]
        final_output = output.view(-1, self.num_heads, self.v_head_dim)

        for start in range(0, q.shape[0], _PCP_LATENT_CONTEXT_QUERY_CHUNK):
            end = min(start + _PCP_LATENT_CONTEXT_QUERY_CHUNK, q.shape[0])
            tile_rows = end - start
            block_table_chunk = block_table_row.expand(tile_rows, -1)
            context_lens_chunk = context_len_row.expand(tile_rows).contiguous()
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
            try:
                _validate_pcp_merge_lse(
                    "prefix",
                    context_lse,
                    num_heads=self.num_heads,
                    num_tokens=tile_rows,
                    device=q.device,
                )
            except ValueError as exc:
                return self._pcp_fallback_forward(
                    str(exc),
                    q,
                    kv_c_normed,
                    k_pe,
                    kv_c_and_k_pe_cache,
                    attn_metadata,
                    k_scale,
                    output,
                    output_scale,
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
    strict: bool = False,
) -> type[AttentionBackend]:
    """Wrap any dense ``MLACommonBackend`` while retaining native decode."""
    if not issubclass(base_backend, MLACommonBackend):
        raise NotImplementedError(
            "selected dense MLA backend is not an MLACommonBackend: "
            f"{base_backend.__module__}.{base_backend.__qualname__}"
        )
    base_impl = base_backend.get_impl_cls()
    if not issubclass(base_impl, MLACommonImpl):
        raise NotImplementedError(
            "selected dense MLA implementation is not an MLACommonImpl: "
            f"{base_impl.__module__}.{base_impl.__qualname__}"
        )

    class PCPMLAImpl(PCPMLAImplMixin, base_impl):  # type: ignore[misc, valid-type]
        pcp_latent_strict = strict

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

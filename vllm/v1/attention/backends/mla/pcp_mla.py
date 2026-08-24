# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP-specific dense MLA wrapping and latent-prefix execution.

The selected upstream dense MLA backend remains authoritative for local prefill
and normal decode. PCP replaces only cached-prefix chunked-context
materialization with a common compressed-KV latent-prefix engine.
"""

from dataclasses import dataclass
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

_PCP_LATENT_CONTEXT_SCRATCH_BYTES = 256 * 1024 * 1024
_PCP_LATENT_CONTEXT_MAX_QUERY_CHUNK = 4096
_PCP_LATENT_CONTEXT_QUERY_GRANULARITY = 128


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


def choose_pcp_latent_query_chunk(
    *,
    num_queries: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_element_size: int,
    block_table_width: int,
    block_table_element_size: int,
    context_output_head_stride: int | None = None,
    scratch_budget_bytes: int = _PCP_LATENT_CONTEXT_SCRATCH_BYTES,
) -> int:
    """Choose a memory-bounded query batch instead of a fixed 256-row loop."""
    if num_queries <= 0:
        return 0

    output_width = (
        v_head_dim if context_output_head_stride is None else context_output_head_stride
    )
    # Reused tensor scratch per row: latent_q (L+R), latent_out (L), padded
    # context output, LSE, split-attention logits, and optional copied page table.
    q_values_per_row = num_heads * (
        2 * kv_lora_rank + qk_rope_head_dim + output_width + 1
    )
    q_bytes_per_row = q_values_per_row * q_element_size
    logits_bytes_per_row = num_heads * (kv_lora_rank + 1) * 4
    block_table_bytes_per_row = block_table_width * block_table_element_size
    scalar_row_bytes = 2 * 4 + 1  # context len, safe len, empty mask
    bytes_per_row = max(
        1,
        q_bytes_per_row
        + logits_bytes_per_row
        + block_table_bytes_per_row
        + scalar_row_bytes,
    )

    budget_rows = max(1, scratch_budget_bytes // bytes_per_row)
    chunk = min(num_queries, budget_rows, _PCP_LATENT_CONTEXT_MAX_QUERY_CHUNK)
    if chunk >= num_queries:
        return num_queries

    rounded = (
        chunk // _PCP_LATENT_CONTEXT_QUERY_GRANULARITY
    ) * _PCP_LATENT_CONTEXT_QUERY_GRANULARITY
    return max(1, rounded)


@dataclass
class PCPLatentPrefixWorkspace:
    latent_q: torch.Tensor
    latent_out: torch.Tensor
    lse: torch.Tensor
    attn_logits: torch.Tensor
    context_output: torch.Tensor
    block_table_rows: torch.Tensor | None
    context_lens_rows: torch.Tensor
    safe_context_lens_rows: torch.Tensor
    empty_context_rows: torch.Tensor


class TritonPCPLatentPrefixEngine:
    """Compressed-MLA prefix engine reusable by dense top-level backends."""

    @staticmethod
    def allocate_workspace(
        impl,
        q: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        chunk_size: int,
        *,
        copy_block_table_rows: bool,
        output_head_stride: int,
    ) -> PCPLatentPrefixWorkspace:
        if output_head_stride < impl.v_head_dim:
            raise ValueError(
                "PCP MLA output head stride cannot be smaller than v_head_dim: "
                f"stride={output_head_stride}, v_head_dim={impl.v_head_dim}."
            )
        block_rows = (
            torch.empty(
                chunk_size,
                block_table.shape[1],
                dtype=block_table.dtype,
                device=block_table.device,
            )
            if copy_block_table_rows
            else None
        )
        context_output_storage = torch.empty(
            chunk_size,
            impl.num_heads,
            output_head_stride,
            dtype=q.dtype,
            device=q.device,
        )
        return PCPLatentPrefixWorkspace(
            latent_q=torch.empty(
                chunk_size,
                impl.num_heads,
                impl.kv_lora_rank + impl.qk_rope_head_dim,
                dtype=q.dtype,
                device=q.device,
            ),
            latent_out=torch.empty(
                chunk_size,
                impl.num_heads,
                impl.kv_lora_rank,
                dtype=q.dtype,
                device=q.device,
            ),
            lse=torch.empty(
                chunk_size,
                impl.num_heads,
                dtype=q.dtype,
                device=q.device,
            ),
            attn_logits=torch.empty(
                chunk_size,
                impl.num_heads,
                1,
                impl.kv_lora_rank + 1,
                dtype=torch.float32,
                device=q.device,
            ),
            # Preserve a possibly padded suffix head stride. merge_attn_states
            # requires prefix/suffix output head strides to match exactly.
            context_output=context_output_storage[..., : impl.v_head_dim],
            block_table_rows=block_rows,
            context_lens_rows=torch.empty(
                chunk_size, dtype=context_lens.dtype, device=context_lens.device
            ),
            safe_context_lens_rows=torch.empty(
                chunk_size, dtype=context_lens.dtype, device=context_lens.device
            ),
            empty_context_rows=torch.empty(
                chunk_size, dtype=torch.bool, device=context_lens.device
            ),
        )

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
        workspace: PCPLatentPrefixWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_nope, q_pe = q.split(
            [impl.qk_nope_head_dim, impl.qk_rope_head_dim], dim=-1
        )
        batch = q.shape[0]
        latent_q = workspace.latent_q[:batch]
        latent_q_nope = latent_q[..., : impl.kv_lora_rank]
        torch.bmm(
            q_nope.transpose(0, 1),
            w_uk_t,
            out=latent_q_nope.transpose(0, 1),
        )
        latent_q[..., impl.kv_lora_rank :].copy_(q_pe)

        latent_out = workspace.latent_out[:batch]
        lse = workspace.lse[:batch]
        attn_logits = workspace.attn_logits[:batch]
        context_output = workspace.context_output[:batch]
        safe_context_lens = workspace.safe_context_lens_rows[:batch]
        empty_context = workspace.empty_context_rows[:batch]
        torch.clamp(context_lens, min=1, out=safe_context_lens)
        torch.eq(context_lens, 0, out=empty_context)

        cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = cache[..., : impl.kv_lora_rank]
        page_size = cache.size(1)
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
        # Defense-in-depth only: the root fix represents an idle PCP rank as an
        # empty local batch upstream, so forward_mha should not be dispatched for
        # it. Keep this guard first so any future zero-row caller cannot fall
        # through to a native MLA/FlashAttention backend.
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

        # Local/new KV stays on the upstream-selected MLA prefill backend.
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

        chunked = prefill_metadata.chunked_context
        context_lens = chunked.context_lens
        single_request = attn_metadata.num_prefills == 1
        copied_block_table_width = (
            0 if single_request else prefill_metadata.block_table.shape[1]
        )
        output_head_stride = suffix_output.stride(1)
        chunk_size = choose_pcp_latent_query_chunk(
            num_queries=q.shape[0],
            num_heads=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_element_size=q.element_size(),
            block_table_width=copied_block_table_width,
            block_table_element_size=prefill_metadata.block_table.element_size(),
            context_output_head_stride=output_head_stride,
        )
        workspace = self.pcp_prefix_engine.allocate_workspace(
            self,
            q,
            prefill_metadata.block_table,
            context_lens,
            chunk_size,
            copy_block_table_rows=not single_request,
            output_head_stride=output_head_stride,
        )

        query_to_seq = None
        if not single_request:
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

        final_output = output.view(-1, self.num_heads, self.v_head_dim)
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            batch = end - start
            context_lens_chunk = workspace.context_lens_rows[:batch]

            if single_request:
                # decode_attention_fwd honors block-table stride(0); use a
                # stride-zero view instead of copying the page table for every Q.
                block_table_chunk = prefill_metadata.block_table[:1].expand(batch, -1)
                # Its seq_lens input is indexed as a dense vector, so materialize
                # that scalar into the reusable physical buffer.
                context_lens_chunk.copy_(context_lens[:1].expand(batch))
            else:
                assert query_to_seq is not None
                assert workspace.block_table_rows is not None
                seq_idx = query_to_seq[start:end]
                block_table_chunk = workspace.block_table_rows[:batch]
                torch.index_select(
                    prefill_metadata.block_table,
                    0,
                    seq_idx,
                    out=block_table_chunk,
                )
                torch.index_select(
                    context_lens,
                    0,
                    seq_idx,
                    out=context_lens_chunk,
                )

            context_output, context_lse = self.pcp_prefix_engine.run(
                self,
                q[start:end],
                kv_c_and_k_pe_cache,
                block_table_chunk,
                context_lens_chunk,
                w_uk_t,
                w_uv,
                k_scale,
                workspace,
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

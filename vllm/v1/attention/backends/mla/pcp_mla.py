# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP-specific dense MLA wrapping and latent-prefix execution.

The selected upstream dense MLA backend remains authoritative for local prefill
and normal decode. PCP replaces only cached-prefix chunked-context
materialization with FlashInfer's absorbed paged-MLA incremental-prefill path.
"""

import sys
from functools import cache

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

logger = init_logger(__name__)


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


def _flashinfer_mla_backend(device: torch.device) -> str:
    """Pick the LSE-capable FlashInfer MLA backend for this device."""
    if device.type != "cuda":
        return "fa2"
    major, _ = torch.cuda.get_device_capability(device)
    return "fa3" if major == 9 else "fa2"


@cache
def _flashinfer_mla_wrapper_cls():
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            "FlashInfer BatchMLAPagedAttentionWrapper is unavailable"
        ) from exc
    return BatchMLAPagedAttentionWrapper


class _FlashInferLatentPrefixRuntime:
    """One shared FlashInfer MLA runtime for sequential attention layers."""

    def __init__(
        self,
        *,
        device: torch.device,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        scale: float,
    ) -> None:
        workspace_bytes = int(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE)
        self.workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        wrapper_cls = _flashinfer_mla_wrapper_cls()
        self.wrapper = wrapper_cls(
            self.workspace,
            backend=_flashinfer_mla_backend(device),
        )
        self.q_dtype = q_dtype
        self.kv_dtype = kv_dtype
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = page_size
        self.scale = float(scale)
        self.planned_metadata = None
        self.planned_signature: tuple[int, int, int, int] | None = None

    def ensure_plan(
        self,
        *,
        plan_key: object,
        q_len: int,
        context_len: int,
        block_table_row: torch.Tensor,
    ) -> None:
        num_pages = (context_len + self.page_size - 1) // self.page_size
        if num_pages > block_table_row.shape[1]:
            raise ValueError(
                "cached-prefix page table is shorter than the context: "
                f"pages={num_pages} table_cols={block_table_row.shape[1]}"
            )
        if block_table_row.dtype != torch.int32:
            raise ValueError(
                f"cached-prefix block table must be int32, got {block_table_row.dtype}"
            )

        signature = (
            q_len,
            context_len,
            block_table_row.data_ptr(),
            num_pages,
        )
        if self.planned_metadata is plan_key and self.planned_signature == signature:
            return

        device = block_table_row.device
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        kv_indices = block_table_row[0, :num_pages].contiguous()
        kv_len_arr = torch.tensor([context_len], dtype=torch.int32, device=device)

        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            False,  # cached prefix is entirely before the local query span
            self.scale,
            self.q_dtype,
            self.kv_dtype,
        )
        # Keep a strong reference so object-id reuse cannot make a stale plan look
        # current. All layers sharing this metadata reuse the same FlashInfer plan.
        self.planned_metadata = plan_key
        self.planned_signature = signature


_FLASHINFER_LATENT_PREFIX_RUNTIMES: dict[
    tuple[
        str,
        int | None,
        torch.dtype,
        torch.dtype,
        int,
        int,
        int,
        int,
        float,
    ],
    _FlashInferLatentPrefixRuntime,
] = {}


def _get_flashinfer_latent_prefix_runtime(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    scale: float,
) -> _FlashInferLatentPrefixRuntime:
    key = (
        device.type,
        device.index,
        q_dtype,
        kv_dtype,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        float(scale),
    )
    runtime = _FLASHINFER_LATENT_PREFIX_RUNTIMES.get(key)
    if runtime is None:
        runtime = _FlashInferLatentPrefixRuntime(
            device=device,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            scale=scale,
        )
        _FLASHINFER_LATENT_PREFIX_RUNTIMES[key] = runtime
    return runtime


class FlashInferPCPLatentPrefixEngine:
    """Absorbed paged-MLA incremental-prefill for a cached PCP prefix."""

    @staticmethod
    def run(
        impl,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        context_len_row: torch.Tensor,
        w_uk_t: torch.Tensor,
        w_uv: torch.Tensor,
        *,
        plan_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        if num_tokens <= 0:
            raise ValueError("PCP latent-prefix query must contain at least one token")
        if tuple(block_table_row.shape[:1]) != (1,):
            raise ValueError(
                "PCP latent-prefix requires one canonical block-table row, "
                f"got shape={tuple(block_table_row.shape)}"
            )
        if context_len_row.numel() != 1:
            raise ValueError(
                "PCP latent-prefix requires one canonical context length, "
                f"got shape={tuple(context_len_row.shape)}"
            )
        if kv_c_and_k_pe_cache.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "FlashInfer latent-prefix currently supports BF16/FP16 KV cache, "
                f"got {kv_c_and_k_pe_cache.dtype}"
            )
        expected_head_size = impl.kv_lora_rank + impl.qk_rope_head_dim
        if kv_c_and_k_pe_cache.shape[-1] != expected_head_size:
            raise ValueError(
                "Unexpected MLA KV-cache head size for latent PCP context: "
                f"got={kv_c_and_k_pe_cache.shape[-1]}, expected={expected_head_size}"
            )

        q_nope, q_pe = q.split(
            [impl.qk_nope_head_dim, impl.qk_rope_head_dim], dim=-1
        )
        ql_nope = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)

        page_size = kv_c_and_k_pe_cache.shape[1]
        context_len = int(context_len_row.reshape(-1)[0].item())
        if context_len < 0:
            raise ValueError(
                "cached-prefix context length must be >= 0, "
                f"got {context_len}"
            )
        if context_len == 0:
            context_output = q.new_zeros(
                (num_tokens, impl.num_heads, impl.v_head_dim)
            )
            context_lse = torch.full(
                (impl.num_heads, num_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            return context_output, context_lse

        runtime = _get_flashinfer_latent_prefix_runtime(
            device=q.device,
            q_dtype=q.dtype,
            kv_dtype=kv_c_and_k_pe_cache.dtype,
            num_heads=impl.num_heads,
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            page_size=page_size,
            scale=impl.scale,
        )
        runtime.ensure_plan(
            plan_key=plan_key,
            q_len=num_tokens,
            context_len=context_len,
            block_table_row=block_table_row,
        )

        ckv_cache = kv_c_and_k_pe_cache[..., : impl.kv_lora_rank]
        kpe_cache = kv_c_and_k_pe_cache[..., impl.kv_lora_rank :]
        latent_out, lse = runtime.wrapper.run(
            ql_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            return_lse=True,
            return_lse_base_on_e=True,
        )
        if tuple(latent_out.shape) != (
            num_tokens,
            impl.num_heads,
            impl.kv_lora_rank,
        ):
            raise ValueError(
                "FlashInfer latent-prefix output shape mismatch: "
                f"got={tuple(latent_out.shape)} expected="
                f"{(num_tokens, impl.num_heads, impl.kv_lora_rank)}"
            )
        if tuple(lse.shape) != (num_tokens, impl.num_heads):
            raise ValueError(
                "FlashInfer latent-prefix LSE shape mismatch: "
                f"got={tuple(lse.shape)} expected={(num_tokens, impl.num_heads)}"
            )

        context_output = torch.empty(
            num_tokens,
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
        return context_output, lse.transpose(0, 1).contiguous()


class PCPMLAImplMixin:
    """Dense MLA PCP override shared by wrapped upstream MLA implementations."""

    pcp_prefix_engine = FlashInferPCPLatentPrefixEngine
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

        # Single-request PCP exposes one canonical page-table row and one cached
        # prefix length. FlashInfer consumes the whole local query span at once;
        # its internal planner owns split-K / incremental-prefill scheduling.
        block_table_row = prefill_metadata.block_table[:1]
        context_len_row = chunked.context_lens[:1]
        try:
            context_output, context_lse = self.pcp_prefix_engine.run(
                self,
                q,
                kv_c_and_k_pe_cache,
                block_table_row,
                context_len_row,
                w_uk_t,
                w_uv,
                plan_key=prefill_metadata,
            )
            _validate_pcp_merge_lse(
                "prefix",
                context_lse,
                num_heads=self.num_heads,
                num_tokens=q.shape[0],
                device=q.device,
            )
        except (NotImplementedError, ValueError) as exc:
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
            output=output.view(-1, self.num_heads, self.v_head_dim),
            prefix_output=context_output,
            prefix_lse=context_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
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

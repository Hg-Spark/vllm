# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP tiling for MLA latent production, transport, and FFN memory bounds.

A tile bounds transient token-row work. Each tile produces compressed MLA
latent state and immediately stages its KV rows to cache/P2P. MLA attention is
still invoked once with the original full metadata, so query tiling never
repeats prefix/context work.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

_TILE_KEY = "pcp_tile_size"
_FFN_KEY = "pcp_ffn_microbatch_size"
_PATCHED = False
_NVTX_ENABLED = os.getenv("VLLM_PCP_WAVEFRONT_NVTX", "0") == "1"
_LOGGED_PATHS: set[str] = set()


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    enabled = _NVTX_ENABLED and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _log_path_once(path: str, **fields: Any) -> None:
    if path in _LOGGED_PATHS:
        return
    _LOGGED_PATHS.add(path)
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("PCP MLA path=%s %s", path, details)


def _parse_size(additional_config: object, key: str, default: int) -> int:
    if not isinstance(additional_config, dict):
        return default
    raw = additional_config.get(key, default)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a non-negative integer, got {raw}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0 or value != raw:
        raise ValueError(
            f"{key} must be a non-negative integer, got {raw!r}"
        )
    return value


def parse_pcp_tile_size(additional_config: object) -> int:
    return _parse_size(additional_config, _TILE_KEY, 0)


def parse_pcp_ffn_microbatch_size(
    additional_config: object,
    tile_size: int,
) -> int:
    value = _parse_size(additional_config, _FFN_KEY, tile_size)
    return tile_size if tile_size > 0 and value == 0 else value


def tile_slices(num_tokens: int, tile_size: int) -> tuple[slice, ...]:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if tile_size <= 0 or num_tokens <= tile_size:
        return (slice(0, num_tokens),) if num_tokens > 0 else ()
    return tuple(
        slice(start, min(start + tile_size, num_tokens))
        for start in range(0, num_tokens, tile_size)
    )


def run_tokenwise_microbatches(
    forward_fn: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    microbatch_size: int,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    slices = tile_slices(hidden_states.shape[0], microbatch_size)
    if len(slices) <= 1:
        return forward_fn(hidden_states, *args, **kwargs)

    output: torch.Tensor | None = None
    for token_slice in slices:
        chunk = forward_fn(hidden_states[token_slice], *args, **kwargs)
        if output is None:
            output = chunk.new_empty(
                (hidden_states.shape[0], *chunk.shape[1:])
            )
        output[token_slice].copy_(chunk)
    assert output is not None
    return output


def _validate_config(
    vllm_config: VllmConfig,
    tile_size: int,
    ffn_size: int,
) -> None:
    if tile_size == 0:
        return

    pc = vllm_config.parallel_config
    sc = vllm_config.scheduler_config
    mc = vllm_config.model_config
    ac = vllm_config.additional_config
    if pc.prefill_context_parallel_size != 2:
        raise NotImplementedError("pcp_tile_size currently requires PCP=2.")
    if pc.tensor_parallel_size != 1 or pc.data_parallel_size != 1:
        raise NotImplementedError(
            "pcp_tile_size currently requires TP=1 and DP=1."
        )
    if pc.decode_context_parallel_size != 1:
        raise NotImplementedError("pcp_tile_size currently requires DCP=1.")
    if pc.use_sequence_parallel_moe:
        raise NotImplementedError(
            "pcp_tile_size does not support sequence-parallel MoE."
        )
    if sc.max_num_seqs != 1:
        raise NotImplementedError(
            "pcp_tile_size currently requires max_num_seqs=1."
        )
    if not isinstance(ac, dict) or "pcp_partition_weights" not in ac:
        raise NotImplementedError(
            "pcp_tile_size currently requires weighted PCP."
        )
    if not mc.use_mla or hasattr(mc.hf_text_config, "index_topk"):
        raise NotImplementedError(
            "pcp_tile_size currently supports dense MLA only."
        )
    if vllm_config.cache_config.calculate_kv_scales:
        raise NotImplementedError(
            "pcp_tile_size does not support runtime KV scales."
        )
    if ffn_size <= 0:
        raise ValueError(
            f"{_FFN_KEY} must be positive when tiling is enabled"
        )


def _attention_metadata(mla_attn: Any) -> Any:
    ctx = get_forward_context()
    raw = ctx.attn_metadata
    if isinstance(raw, dict):
        return raw[mla_attn.layer_name]
    if isinstance(raw, list):
        return raw[0][mla_attn.layer_name]
    return raw


def _forward_state(mla_attn: Any) -> tuple[Any, torch.Tensor | None]:
    ctx = get_forward_context()
    metadata = _attention_metadata(mla_attn)
    if not isinstance(ctx.slot_mapping, dict):
        raise RuntimeError("PCP tiled MLA requires dict slot mappings.")
    return metadata, ctx.slot_mapping.get(mla_attn.layer_name)


def _transport_rows(
    mla_attn: Any,
    slot_mapping: torch.Tensor | None,
    num_tokens: int,
) -> int:
    if not mla_attn.use_pcp or slot_mapping is None:
        return num_tokens
    if slot_mapping.numel() % 2:
        raise RuntimeError(
            "PCP slot mapping does not contain two equal rank slabs."
        )
    return slot_mapping.numel() // 2


def _supports_profile_tiled_mla(wrapper: Any) -> bool:
    return (
        wrapper.q_lora_rank is not None
        and not wrapper.is_sparse
        and wrapper.indexer is None
        and not wrapper.dcp_q_replicate
        and not wrapper.mla_attn.calculate_kv_scales
    )


def _supports_tiled_mla(
    wrapper: Any,
    num_tokens: int,
    tile_size: int,
) -> bool:
    if not _supports_profile_tiled_mla(wrapper):
        return False

    metadata, slot_mapping = _forward_state(wrapper.mla_attn)
    if metadata is None:
        return False
    if int(getattr(metadata, "num_prefills", 0) or 0) <= 0:
        return False

    # Every supported prefill uses the tile protocol. Short prefills are a
    # single tile rather than falling back to a separate layer-level protocol.
    return _transport_rows(
        wrapper.mla_attn,
        slot_mapping,
        num_tokens,
    ) > 0


def _produce_mla_latent_tile(
    wrapper: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    token_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert wrapper.q_lora_rank is not None
    assert wrapper.fused_qkv_a_proj is not None
    assert wrapper.q_a_layernorm is not None

    qkv = wrapper.fused_qkv_a_proj(hidden_states[token_slice])[0]
    q_c_view, kv_lora = qkv.split(
        [
            wrapper.q_lora_rank,
            wrapper.kv_lora_rank + wrapper.qk_rope_head_dim,
        ],
        dim=-1,
    )
    q_c_tile = wrapper.q_a_layernorm(q_c_view)
    kv_c, k_pe = kv_lora.split(
        [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim],
        dim=-1,
    )
    kv_c_tile = wrapper.kv_a_layernorm(kv_c)
    k_pe_tile = k_pe.unsqueeze(1)
    if wrapper.rotary_emb is not None:
        k_pe_tile, _ = wrapper.rotary_emb(
            positions[token_slice],
            k_pe_tile,
            None,
        )
    return q_c_tile, kv_c_tile, k_pe_tile


def _expand_mla_query(
    wrapper: Any,
    positions: torch.Tensor,
    q_c_buffer: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
) -> torch.Tensor:
    assert wrapper.q_b_proj is not None
    q = wrapper.q_b_proj(q_c_buffer)[0]
    q = q.view(-1, wrapper.num_heads, wrapper.qk_head_dim)
    if wrapper.rotary_emb is not None:
        q_pe = q[..., wrapper.qk_nope_head_dim :]
        q_pe_rot, _ = wrapper.rotary_emb(positions, q_pe, None)
        q_pe.copy_(q_pe_rot)
    if llama_4_scaling is not None:
        q *= llama_4_scaling
    return q


def _finish_mla(
    wrapper: Any,
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    kv_c_buffer: torch.Tensor,
    k_pe_buffer: torch.Tensor,
    metadata: Any,
) -> torch.Tensor:
    mla_attn = wrapper.mla_attn
    num_tokens = hidden_states.shape[0]
    attn_out = torch.empty(
        (num_tokens, wrapper.num_heads * wrapper.v_head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    mla_attn.forward_impl(
        q,
        kv_c_buffer,
        k_pe_buffer,
        mla_attn.kv_cache,
        metadata,
        output=attn_out,
    )
    if wrapper.g_proj is not None:
        attn_out = attn_out * wrapper.g_proj(hidden_states)[0].sigmoid()
    return wrapper.o_proj(attn_out)[0]


def _run_profile_tiled_mla(
    wrapper: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    tile_size: int,
) -> torch.Tensor:
    """Profile MLA with tiled Phase-A and the native worst-case workspace."""
    assert wrapper.q_lora_rank is not None
    assert wrapper.q_b_proj is not None

    num_tokens = hidden_states.shape[0]
    slices = tile_slices(num_tokens, tile_size)
    if not slices:
        raise RuntimeError("PCP tiled MLA profile has no model rows.")

    _log_path_once(
        "profile_tiled",
        layer=wrapper.mla_attn.layer_name,
        profile_rows=num_tokens,
        tile_size=tile_size,
        tile_count=len(slices),
    )

    q_c_buffer: torch.Tensor | None = None
    kv_c_buffer: torch.Tensor | None = None
    k_pe_buffer: torch.Tensor | None = None

    with _nvtx_range("pcp_mla.profile_tiled"):
        for tile_idx, token_slice in enumerate(slices):
            with _nvtx_range(
                f"pcp_mla.profile_tiled.phase_a.tile_{tile_idx}"
            ):
                q_c_tile, kv_c_tile, k_pe_tile = _produce_mla_latent_tile(
                    wrapper,
                    positions,
                    hidden_states,
                    token_slice,
                )

            if q_c_buffer is None:
                q_c_buffer = q_c_tile.new_empty(
                    (num_tokens, *q_c_tile.shape[1:])
                )
                kv_c_buffer = kv_c_tile.new_empty(
                    (num_tokens, *kv_c_tile.shape[1:])
                )
                k_pe_buffer = k_pe_tile.new_empty(
                    (num_tokens, *k_pe_tile.shape[1:])
                )

            q_c_buffer[token_slice].copy_(q_c_tile)
            assert kv_c_buffer is not None and k_pe_buffer is not None
            kv_c_buffer[token_slice].copy_(kv_c_tile)
            k_pe_buffer[token_slice].copy_(k_pe_tile)

        assert q_c_buffer is not None
        assert kv_c_buffer is not None
        assert k_pe_buffer is not None

        q = _expand_mla_query(
            wrapper,
            positions,
            q_c_buffer,
            llama_4_scaling,
        )
        del q_c_buffer

        # attn_metadata=None deliberately preserves vLLM's native profile-only
        # worst-case MLA workspace allocation in MLAAttention.forward_impl().
        return _finish_mla(
            wrapper,
            hidden_states,
            q,
            kv_c_buffer,
            k_pe_buffer,
            None,
        )


def _run_tiled_mla(
    wrapper: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    tile_size: int,
) -> torch.Tensor:
    """Produce/cache/transfer latent tiles, then run one MLA attention."""
    assert wrapper.q_lora_rank is not None
    assert wrapper.q_b_proj is not None

    mla_attn = wrapper.mla_attn
    metadata, slot_mapping = _forward_state(mla_attn)
    num_tokens = hidden_states.shape[0]
    work_rows = _transport_rows(mla_attn, slot_mapping, num_tokens)
    tile_count = (work_rows + tile_size - 1) // tile_size

    _log_path_once(
        "prefill_tiled",
        layer=mla_attn.layer_name,
        local_tokens=num_tokens,
        work_rows=work_rows,
        tile_size=tile_size,
        tile_count=tile_count,
    )

    q_c_buffer: torch.Tensor | None = None
    kv_c_buffer: torch.Tensor | None = None
    k_pe_buffer: torch.Tensor | None = None

    from vllm.model_executor.layers.attention.pcp import (
        iter_tiled_mla_cache_inputs,
    )

    rank_slots = None
    if mla_attn.use_pcp and metadata.num_decode_tokens is not None:
        assert slot_mapping is not None
        rank_slots = slot_mapping.view(2, work_rows)

    # Both PCP ranks iterate the same shared slab. Each real local tile is
    # produced once, copied into compact full latent buffers for the single
    # attention invocation, and immediately staged to KV cache/P2P.
    with _nvtx_range("pcp_mla.prefill_tiled"):
        for tile_idx, start in enumerate(range(0, work_rows, tile_size)):
            stop = min(start + tile_size, work_rows)
            local_stop = min(stop, num_tokens)

            if start < local_stop:
                token_slice = slice(start, local_stop)
                with _nvtx_range(
                    f"pcp_mla.prefill_tiled.phase_a.tile_{tile_idx}"
                ):
                    q_c_tile, kv_c_tile, k_pe_tile = _produce_mla_latent_tile(
                        wrapper,
                        positions,
                        hidden_states,
                        token_slice,
                    )

                if q_c_buffer is None:
                    q_c_buffer = q_c_tile.new_empty(
                        (num_tokens, *q_c_tile.shape[1:])
                    )
                    kv_c_buffer = kv_c_tile.new_empty(
                        (num_tokens, *kv_c_tile.shape[1:])
                    )
                    k_pe_buffer = k_pe_tile.new_empty(
                        (num_tokens, *k_pe_tile.shape[1:])
                    )

                q_c_buffer[token_slice].copy_(q_c_tile)
                assert kv_c_buffer is not None and k_pe_buffer is not None
                kv_c_buffer[token_slice].copy_(kv_c_tile)
                k_pe_buffer[token_slice].copy_(k_pe_tile)

            if q_c_buffer is None or kv_c_buffer is None or k_pe_buffer is None:
                raise RuntimeError("PCP tile execution has no local model rows.")

            local_kv = kv_c_buffer[start:local_stop]
            local_kpe = k_pe_buffer[start:local_stop]
            if rank_slots is None:
                tile_slots = (
                    None
                    if slot_mapping is None
                    else slot_mapping[start:local_stop]
                )
            else:
                tile_slots = torch.cat(
                    (rank_slots[0, start:stop], rank_slots[1, start:stop]),
                    dim=0,
                )

            for cache_kv, cache_kpe, cache_slots in iter_tiled_mla_cache_inputs(
                local_kv,
                local_kpe,
                tile_slots,
                metadata.num_decode_tokens,
                mla_attn.use_pcp,
                stop - start,
            ):
                mla_attn.impl.do_kv_cache_update(
                    cache_kv,
                    cache_kpe,
                    mla_attn.kv_cache,
                    cache_slots,
                    mla_attn.kv_cache_dtype,
                    mla_attn._k_scale,
                )

        assert q_c_buffer is not None
        assert kv_c_buffer is not None
        assert k_pe_buffer is not None

        # Query expansion and attention are intentionally not tiled. This is one
        # full-metadata invocation, avoiding repeated MLA context expansion.
        q = _expand_mla_query(
            wrapper,
            positions,
            q_c_buffer,
            llama_4_scaling,
        )
        del q_c_buffer

        return _finish_mla(
            wrapper,
            hidden_states,
            q,
            kv_c_buffer,
            k_pe_buffer,
            metadata,
        )


def configure_pcp_tiling(vllm_config: VllmConfig) -> int:
    global _PATCHED
    tile_size = parse_pcp_tile_size(vllm_config.additional_config)
    ffn_size = parse_pcp_ffn_microbatch_size(
        vllm_config.additional_config,
        tile_size,
    )
    _validate_config(vllm_config, tile_size, ffn_size)
    if tile_size == 0 or _PATCHED:
        return tile_size

    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2MLP,
        DeepseekV2MoE,
    )
    from vllm.model_executor.models.glm4_moe import Glm4MoE, Glm4MoeMLP

    original_mla = MultiHeadLatentAttentionWrapper.forward
    original_ds_mlp = DeepseekV2MLP.forward
    original_ds_moe = DeepseekV2MoE.forward
    original_glm_mlp = Glm4MoeMLP.forward
    original_glm_moe = Glm4MoE.forward

    def mla_forward(
        self,
        positions,
        hidden_states,
        llama_4_scaling=None,
    ):
        metadata = _attention_metadata(self.mla_attn)

        if metadata is None and _supports_profile_tiled_mla(self):
            return _run_profile_tiled_mla(
                self,
                positions,
                hidden_states,
                llama_4_scaling,
                tile_size,
            )

        if _supports_tiled_mla(
            self,
            hidden_states.shape[0],
            tile_size,
        ):
            return _run_tiled_mla(
                self,
                positions,
                hidden_states,
                llama_4_scaling,
                tile_size,
            )

        _log_path_once(
            "original",
            layer=self.mla_attn.layer_name,
            metadata_none=metadata is None,
            num_prefills=getattr(metadata, "num_prefills", None),
        )
        return original_mla(
            self,
            positions,
            hidden_states,
            llama_4_scaling,
        )

    def ds_mlp_forward(self, hidden_states):
        return run_tokenwise_microbatches(
            lambda x: original_ds_mlp(self, x),
            hidden_states,
            ffn_size,
        )

    def ds_moe_forward(
        self,
        hidden_states,
        already_sequence_parallel=False,
    ):
        if already_sequence_parallel:
            return original_ds_moe(
                self,
                hidden_states,
                already_sequence_parallel=already_sequence_parallel,
            )
        return run_tokenwise_microbatches(
            lambda x: original_ds_moe(
                self,
                x,
                already_sequence_parallel=False,
            ),
            hidden_states,
            ffn_size,
        )

    def glm_mlp_forward(self, hidden_states):
        return run_tokenwise_microbatches(
            lambda x: original_glm_mlp(self, x),
            hidden_states,
            ffn_size,
        )

    def glm_moe_forward(self, hidden_states):
        return run_tokenwise_microbatches(
            lambda x: original_glm_moe(self, x),
            hidden_states,
            ffn_size,
        )

    MultiHeadLatentAttentionWrapper.forward = mla_forward
    DeepseekV2MLP.forward = ds_mlp_forward
    DeepseekV2MoE.forward = ds_moe_forward
    Glm4MoeMLP.forward = glm_mlp_forward
    Glm4MoE.forward = glm_moe_forward
    _PATCHED = True
    logger.info(
        "Enabled PCP tiling: tile_size=%d, ffn_microbatch_size=%d; "
        "MLA Phase-A is tiled for prefill/profile, attention remains "
        "full-metadata at runtime",
        tile_size,
        ffn_size,
    )
    return tile_size

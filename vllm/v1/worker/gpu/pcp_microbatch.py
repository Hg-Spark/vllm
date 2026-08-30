# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP memory microbatching for MLA and token-wise feed-forward sublayers.

Wavefront communication remains a full-layer rank0->rank1 latent handoff.
Within each rank, expanded MLA query/attention/output work and MLP/MoE work can
run in bounded token microbatches to cap transient GPU memory.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

_CONFIG_KEY = "pcp_microbatch_size"
_ATTN_PLAN_KEY = "__pcp_memory_microbatch_plan__"
_PATCHED = False


@dataclass(frozen=True)
class PCPAttentionMicrobatchPlan:
    """Per-layer-local attention metadata for one rank-local PCP batch."""

    slices: tuple[slice, ...]
    attn_metadata: tuple[dict[str, Any], ...]


def parse_pcp_microbatch_size(additional_config: object) -> int:
    """Return the configured rank-local memory microbatch size.

    ``0`` means disabled. The value is intentionally PCP-specific instead of
    reusing DBO/``--ubatch-size``: DBO changes model-forward scheduling, while
    this knob only bounds transient allocations inside each layer.
    """
    if not isinstance(additional_config, dict):
        return 0
    raw = additional_config.get(_CONFIG_KEY, 0)
    if raw is None:
        return 0
    if isinstance(raw, bool):
        raise ValueError(f"{_CONFIG_KEY} must be a non-negative integer, got {raw}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_CONFIG_KEY} must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0 or value != raw:
        raise ValueError(
            f"{_CONFIG_KEY} must be a non-negative integer, got {raw!r}"
        )
    return value


def _validate_config(vllm_config: VllmConfig, microbatch_size: int) -> None:
    if microbatch_size == 0:
        return

    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    additional_config = vllm_config.additional_config

    if parallel_config.prefill_context_parallel_size != 2:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires prefill_context_parallel_size=2."
        )
    if parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires tensor_parallel_size=1."
        )
    if parallel_config.data_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires data_parallel_size=1."
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires decode_context_parallel_size=1."
        )
    if parallel_config.use_sequence_parallel_moe:
        raise NotImplementedError(
            "pcp_microbatch_size does not support sequence-parallel MoE yet."
        )
    if scheduler_config.max_num_seqs != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires max_num_seqs=1 so the long "
            "prefill memory bound is unambiguous."
        )
    if (
        not isinstance(additional_config, dict)
        or "pcp_partition_weights" not in additional_config
    ):
        raise NotImplementedError(
            "pcp_microbatch_size currently requires weighted PCP "
            "(pcp_partition_weights)."
        )
    if not model_config.use_mla:
        raise NotImplementedError("pcp_microbatch_size currently requires MLA.")
    if hasattr(model_config.hf_text_config, "index_topk"):
        raise NotImplementedError(
            "pcp_microbatch_size currently supports dense MLA only."
        )
    if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires eager execution "
            "(-cc.cudagraph_mode=NONE)."
        )
    if vllm_config.cache_config.calculate_kv_scales:
        raise NotImplementedError(
            "pcp_microbatch_size does not support runtime KV-scale calculation yet."
        )


def microbatch_slices(num_tokens: int, microbatch_size: int) -> tuple[slice, ...]:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if microbatch_size <= 0 or num_tokens <= microbatch_size:
        return (slice(0, num_tokens),) if num_tokens > 0 else ()
    return tuple(
        slice(start, min(start + microbatch_size, num_tokens))
        for start in range(0, num_tokens, microbatch_size)
    )


def run_tokenwise_microbatches(
    forward_fn: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    microbatch_size: int,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run a token-wise sublayer in bounded chunks and restore row order."""
    slices = microbatch_slices(hidden_states.shape[0], microbatch_size)
    if len(slices) <= 1:
        return forward_fn(hidden_states, *args, **kwargs)

    output: torch.Tensor | None = None
    for token_slice in slices:
        chunk_output = forward_fn(hidden_states[token_slice], *args, **kwargs)
        if output is None:
            output = chunk_output.new_empty(
                (hidden_states.shape[0], *chunk_output.shape[1:])
            )
        output[token_slice].copy_(chunk_output)
    assert output is not None
    return output


def _slice_single_request_input_batch(
    input_batch: InputBatch,
    token_slice: slice,
) -> InputBatch:
    """Create one causal continuation microbatch from a single local request.

    Weighted PCP gives each rank one contiguous prefill interval. A later local
    microbatch therefore starts with all preceding local microbatches as context.
    """
    if input_batch.num_reqs != 1:
        raise NotImplementedError(
            "PCP attention microbatching currently requires one local request."
        )
    start = int(token_slice.start or 0)
    stop = int(token_slice.stop or input_batch.num_tokens)
    if not (0 <= start < stop <= input_batch.num_tokens):
        raise ValueError(
            f"Invalid PCP token slice {token_slice} for {input_batch.num_tokens} tokens"
        )

    num_tokens = stop - start
    base_computed = int(input_batch.num_computed_tokens_np[0])
    computed = base_computed + start
    seq_len = base_computed + stop
    prefill_len = int(input_batch.prefill_len_np[0])
    computed_prefill = min(computed, prefill_len)

    query_start_loc_np = np.array([0, num_tokens], dtype=np.int32)
    query_start_loc = input_batch.query_start_loc.new_tensor(query_start_loc_np)
    seq_lens = input_batch.seq_lens.new_tensor([seq_len])
    seq_lens_cpu_upper_bound = torch.tensor([seq_len], dtype=torch.int32)

    return replace(
        input_batch,
        num_scheduled_tokens=np.array([num_tokens], dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        num_computed_tokens_np=np.array([computed], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array(
            [computed_prefill], dtype=np.int32
        ),
        is_prefilling_np=np.array([computed_prefill < prefill_len], dtype=np.bool_),
        input_ids=input_batch.input_ids[token_slice],
        positions=input_batch.positions[token_slice],
        is_padding=input_batch.is_padding[token_slice],
    )


def _clone_mla_prefill_backends(attn_metadata: dict[str, Any]) -> None:
    """Give every MB its own stateful MLA prefill backend instance."""
    seen: set[int] = set()
    for metadata in attn_metadata.values():
        if id(metadata) in seen:
            continue
        seen.add(id(metadata))
        prefill = getattr(metadata, "prefill", None)
        backend = getattr(prefill, "prefill_backend", None)
        if prefill is None or backend is None:
            continue
        cloned_backend = backend.clone()
        prefill.prefill_backend = cloned_backend
        cloned_backend.prepare_metadata(prefill)


def _local_pcp_slot_mappings(
    slot_mappings: torch.Tensor,
    num_local_tokens: int,
) -> torch.Tensor:
    """Extract this rank's real rows from the full two-rank PCP slot slab."""
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2:
        raise NotImplementedError("PCP attention microbatching currently requires PCP=2.")
    if slot_mappings.shape[1] % 2 != 0:
        raise RuntimeError(
            "PCP slot mappings do not contain two equal rank slabs: "
            f"shape={tuple(slot_mappings.shape)}"
        )
    slab_width = slot_mappings.shape[1] // 2
    if num_local_tokens > slab_width:
        raise RuntimeError(
            "PCP local token count exceeds rank slab width: "
            f"{num_local_tokens} > {slab_width}"
        )
    rank_start = pcp_group.rank_in_group * slab_width
    return slot_mappings[:, rank_start : rank_start + num_local_tokens]


def _attach_attention_microbatch_plan(
    full_attn_metadata: dict[str, Any],
    original_prepare_attn: Callable[..., dict[str, Any]],
    model_state: Any,
    input_batch: InputBatch,
    cudagraph_mode: CUDAGraphMode,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[Any]],
    kv_cache_config: Any,
    for_capture: bool,
    microbatch_size: int,
) -> dict[str, Any]:
    slices = microbatch_slices(input_batch.num_tokens, microbatch_size)
    if (
        len(slices) <= 1
        or input_batch.num_reqs != 1
        or not bool(input_batch.is_prefilling_np[0])
    ):
        return full_attn_metadata
    if cudagraph_mode != CUDAGraphMode.NONE or for_capture:
        raise NotImplementedError(
            "PCP attention microbatching currently supports eager execution only."
        )

    local_slot_mappings = _local_pcp_slot_mappings(
        slot_mappings, input_batch.num_tokens
    )
    metadata_per_mb: list[dict[str, Any]] = []
    for token_slice in slices:
        mb_input_batch = _slice_single_request_input_batch(input_batch, token_slice)
        mb_slot_mappings = local_slot_mappings[:, token_slice]
        mb_metadata = original_prepare_attn(
            model_state,
            mb_input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            mb_slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=False,
        )
        _clone_mla_prefill_backends(mb_metadata)
        metadata_per_mb.append(mb_metadata)

    full_attn_metadata[_ATTN_PLAN_KEY] = PCPAttentionMicrobatchPlan(
        slices=slices,
        attn_metadata=tuple(metadata_per_mb),
    )
    return full_attn_metadata


def _slice_scaling(
    scaling: torch.Tensor | None,
    token_slice: slice,
    full_num_tokens: int,
) -> torch.Tensor | None:
    if scaling is None:
        return None
    if scaling.ndim > 0 and scaling.shape[0] == full_num_tokens:
        return scaling[token_slice]
    return scaling


def _run_mla_microbatches(
    wrapper: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    plan: PCPAttentionMicrobatchPlan,
) -> torch.Tensor:
    """Run full compressed latent preparation followed by bounded MLA MBs."""
    if wrapper.q_lora_rank is None:
        raise NotImplementedError(
            "PCP MLA microbatching currently requires q_lora_rank to be set."
        )
    if wrapper.is_sparse or wrapper.indexer is not None:
        raise NotImplementedError(
            "PCP MLA microbatching currently supports dense MLA only."
        )
    if wrapper.dcp_q_replicate:
        raise NotImplementedError(
            "PCP MLA microbatching does not support DCP Q replication yet."
        )

    mla_attn = wrapper.mla_attn
    num_tokens = hidden_states.shape[0]
    if not plan.slices or plan.slices[-1].stop != num_tokens:
        raise RuntimeError(
            "PCP attention microbatch plan does not cover the rank-local batch."
        )

    assert wrapper.fused_qkv_a_proj is not None
    assert wrapper.q_a_layernorm is not None
    assert wrapper.q_b_proj is not None

    # Phase A: keep only compressed resident state for the full local layer.
    qkv_lora = wrapper.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv_lora.split(
        [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
        dim=-1,
    )
    q_c = wrapper.q_a_layernorm(q_c)
    kv_c, k_pe = kv_lora.split(
        [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = wrapper.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)

    if wrapper.rotary_emb is not None:
        k_pe, _ = wrapper.rotary_emb(positions, k_pe, None)
        # k_pe may otherwise keep the fused qkv_lora storage alive.
        k_pe = k_pe.clone()

    # Stage the complete compressed layer in KV cache once. The existing PCP
    # transport remains full-layer and can overlap rank0's subsequent local work.
    forward_context = get_forward_context()
    full_metadata_raw = forward_context.attn_metadata
    if not isinstance(full_metadata_raw, dict):
        raise RuntimeError("PCP MLA microbatching requires dict attention metadata.")
    full_metadata = full_metadata_raw[mla_attn.layer_name]

    slot_mapping = forward_context.slot_mapping
    if not isinstance(slot_mapping, dict):
        raise RuntimeError("PCP MLA microbatching requires dict slot mappings.")
    layer_slot_mapping = slot_mapping.get(mla_attn.layer_name)

    from vllm.model_executor.layers.attention.pcp import (
        maybe_transfer_mla_cache_inputs,
    )

    kv_for_cache, kpe_for_cache, cache_slot_mapping = maybe_transfer_mla_cache_inputs(
        kv_c_normed,
        k_pe,
        layer_slot_mapping,
        full_metadata.num_decode_tokens if full_metadata is not None else None,
        mla_attn.use_pcp,
    )
    mla_attn.impl.do_kv_cache_update(
        kv_for_cache,
        kpe_for_cache,
        mla_attn.kv_cache,
        cache_slot_mapping,
        mla_attn.kv_cache_dtype,
        mla_attn._k_scale,
    )
    del kv_for_cache, kpe_for_cache, cache_slot_mapping
    del qkv_lora, kv_lora, kv_c

    output = hidden_states.new_empty((num_tokens, wrapper.hidden_size))
    for token_slice, mb_metadata_dict in zip(plan.slices, plan.attn_metadata):
        q = wrapper.q_b_proj(q_c[token_slice])[0]
        q = q.view(-1, wrapper.num_heads, wrapper.qk_head_dim)

        if wrapper.rotary_emb is not None:
            q_pe = q[..., wrapper.qk_nope_head_dim :]
            q_pe_rot, _ = wrapper.rotary_emb(
                positions[token_slice], q_pe, None
            )
            q_pe.copy_(q_pe_rot)

        scaling = _slice_scaling(llama_4_scaling, token_slice, num_tokens)
        if scaling is not None:
            q *= scaling

        mb_metadata = mb_metadata_dict[mla_attn.layer_name]
        mb_output_shape = (
            q.shape[0],
            wrapper.num_heads * wrapper.v_head_dim,
        )
        attn_out = torch.empty(
            mb_output_shape,
            dtype=q.dtype,
            device=q.device,
        )
        mla_attn.forward_impl(
            q,
            kv_c_normed[token_slice],
            k_pe[token_slice],
            mla_attn.kv_cache,
            mb_metadata,
            output=attn_out,
        )

        if wrapper.g_proj is not None:
            attn_out = (
                attn_out
                * wrapper.g_proj(hidden_states[token_slice])[0].sigmoid()
            )
        chunk_output = wrapper.o_proj(attn_out)[0]
        output[token_slice].copy_(chunk_output)

    return output


def configure_pcp_memory_microbatching(vllm_config: VllmConfig) -> int:
    """Validate the knob and install the PCP layer-local memory wrappers."""
    global _PATCHED

    microbatch_size = parse_pcp_microbatch_size(vllm_config.additional_config)
    _validate_config(vllm_config, microbatch_size)
    if microbatch_size == 0 or _PATCHED:
        return microbatch_size

    # Patch model classes rather than duplicating model-specific decoder loops.
    # GLM-4.7-Flash inherits the DeepSeek MLA attention class, but its MLP/MoE
    # classes come from glm4_moe and need their own feed-forward wrappers.
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2MLP,
        DeepseekV2MoE,
    )
    from vllm.model_executor.models.glm4_moe import Glm4MoE, Glm4MoeMLP
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    original_mla_forward = MultiHeadLatentAttentionWrapper.forward
    original_deepseek_mlp_forward = DeepseekV2MLP.forward
    original_deepseek_moe_forward = DeepseekV2MoE.forward
    original_glm_mlp_forward = Glm4MoeMLP.forward
    original_glm_moe_forward = Glm4MoE.forward
    original_prepare_attn = DefaultModelState.prepare_attn

    def mla_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        plan = (
            attn_metadata.get(_ATTN_PLAN_KEY)
            if isinstance(attn_metadata, dict)
            else None
        )
        if plan is None or self.mla_attn.calculate_kv_scales:
            return original_mla_forward(
                self, positions, hidden_states, llama_4_scaling
            )
        assert isinstance(plan, PCPAttentionMicrobatchPlan)
        return _run_mla_microbatches(
            self, positions, hidden_states, llama_4_scaling, plan
        )

    def deepseek_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return run_tokenwise_microbatches(
            lambda chunk: original_deepseek_mlp_forward(self, chunk),
            hidden_states,
            microbatch_size,
        )

    def deepseek_moe_forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if already_sequence_parallel:
            return original_deepseek_moe_forward(
                self,
                hidden_states,
                already_sequence_parallel=already_sequence_parallel,
            )
        return run_tokenwise_microbatches(
            lambda chunk: original_deepseek_moe_forward(
                self,
                chunk,
                already_sequence_parallel=False,
            ),
            hidden_states,
            microbatch_size,
        )

    def glm_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return run_tokenwise_microbatches(
            lambda chunk: original_glm_mlp_forward(self, chunk),
            hidden_states,
            microbatch_size,
        )

    def glm_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return run_tokenwise_microbatches(
            lambda chunk: original_glm_moe_forward(self, chunk),
            hidden_states,
            microbatch_size,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[Any]],
        kv_cache_config: Any,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        full_attn_metadata = original_prepare_attn(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )
        return _attach_attention_microbatch_plan(
            full_attn_metadata,
            original_prepare_attn,
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            microbatch_size,
        )

    MultiHeadLatentAttentionWrapper.forward = mla_forward
    DeepseekV2MLP.forward = deepseek_mlp_forward
    DeepseekV2MoE.forward = deepseek_moe_forward
    Glm4MoeMLP.forward = glm_mlp_forward
    Glm4MoE.forward = glm_moe_forward
    DefaultModelState.prepare_attn = prepare_attn

    _PATCHED = True
    logger.info(
        "Enabled PCP layer-local memory microbatching: size=%d "
        "(full compressed MLA latent, microbatched Q/attention/o_proj + MLP/MoE)",
        microbatch_size,
    )
    return microbatch_size

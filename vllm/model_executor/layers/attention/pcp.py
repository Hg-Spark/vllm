# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)


def _stage_prefill_for_collective(
    tensor: torch.Tensor,
    collective_width: int,
) -> torch.Tensor:
    """Pad only the communication slab, never the model execution batch."""
    local_width = tensor.shape[0]
    if local_width > collective_width:
        raise RuntimeError(
            "PCP local cache-input width exceeds collective slab: "
            f"{local_width} > {collective_width}"
        )
    if local_width == collective_width:
        return tensor.contiguous()
    staged = tensor.new_zeros((collective_width, *tensor.shape[1:]))
    if local_width > 0:
        staged[:local_width].copy_(tensor)
    return staged


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Gather fixed-width slabs for rank-owned PCP cache inputs.

    Wavefront PCP gives every semantic token exactly one execution owner. The
    slot mapping already uses the same rank-major collective layout, including
    PAD slots for unused rows. Gather each rank-local model slab as-is so ranks
    with different decode ownership still execute an identical collective ABI.
    ``num_decode_tokens`` is retained for call-site compatibility during the
    staged migration to one-way layer P2P.
    """
    model_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == model_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= model_num_tokens

    pcp_group = get_pcp_group()
    pcp_size = pcp_group.world_size
    if slot_mapping.numel() % pcp_size != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain equal collective slabs: "
            f"numel={slot_mapping.numel()}, pcp_size={pcp_size}"
        )
    collective_width = slot_mapping.numel() // pcp_size

    staged_inputs = tuple(
        _stage_prefill_for_collective(tensor, collective_width) for tensor in tensors
    )
    gathered_inputs = tuple(
        pcp_group.all_gather(staged, dim=0) for staged in staged_inputs
    )
    return gathered_inputs, slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.model_executor.layers.attention.pcp_runahead import (
    post_layer_transfer,
    recv_layer_payload,
)


def _pad_to_rank_slab(
    tensor: torch.Tensor,
    rank_slab_width: int,
) -> torch.Tensor:
    """Pad only the communication slab, never the model execution batch."""
    local_width = tensor.shape[0]
    if local_width > rank_slab_width:
        raise RuntimeError(
            "PCP local cache-input width exceeds rank slab: "
            f"{local_width} > {rank_slab_width}"
        )
    if local_width == rank_slab_width:
        return tensor.contiguous()
    padded = tensor.new_zeros((rank_slab_width, *tensor.shape[1:]))
    if local_width > 0:
        padded[:local_width].copy_(tensor)
    return padded


def _transfer_mla_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Transfer one full layer of MLA cache inputs from rank0 to rank1.

    Communication uses fixed-width rank slabs. Rank0 computes only its real
    prefix rows, pads the transport slab, posts the complete layer payload, then
    immediately returns to local cache update/attention. Rank1 consumes rank0's
    slab before entering attention for that layer.
    """
    model_num_rows = tensors[0].shape[0]
    assert all(tensor.shape[0] == model_num_rows for tensor in tensors)

    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2:
        raise NotImplementedError("Wavefront MLA transfer currently requires PCP=2.")
    if slot_mapping.numel() % 2 != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain two equal rank slabs: "
            f"numel={slot_mapping.numel()}"
        )

    rank_slab_width = slot_mapping.numel() // 2
    if model_num_rows > rank_slab_width:
        raise RuntimeError(
            "PCP local model rows exceed rank slab: "
            f"{model_num_rows} > {rank_slab_width}"
        )

    rank_slot_mappings = slot_mapping.view(2, rank_slab_width)
    rank = pcp_group.rank_in_group

    if rank == 0:
        send_payload = tuple(
            _pad_to_rank_slab(tensor, rank_slab_width) for tensor in tensors
        )
        post_layer_transfer(send_payload)
        local_slots = rank_slot_mappings[0, :model_num_rows]
        return tensors, local_slots

    if rank != 1:
        raise RuntimeError(f"Unexpected PCP rank for PCP=2 wavefront: {rank}")

    recv_templates = tuple(
        tensor.new_empty((rank_slab_width, *tensor.shape[1:])) for tensor in tensors
    )
    remote_inputs = recv_layer_payload(recv_templates)

    local_slots = rank_slot_mappings[1, :model_num_rows]
    cache_inputs = tuple(
        torch.cat((remote, local), dim=0)
        for remote, local in zip(remote_inputs, tensors)
    )
    cache_slot_mapping = torch.cat((rank_slot_mappings[0], local_slots), dim=0)
    return cache_inputs, cache_slot_mapping


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Legacy collective path retained only for non-wavefront attention helpers."""
    model_num_rows = tensors[0].shape[0]
    assert all(tensor.shape[0] == model_num_rows for tensor in tensors)
    assert 0 <= num_decode_tokens <= model_num_rows

    pcp_group = get_pcp_group()
    pcp_size = pcp_group.world_size
    if slot_mapping.numel() % pcp_size != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain equal collective slabs: "
            f"numel={slot_mapping.numel()}, pcp_size={pcp_size}"
        )
    collective_width = slot_mapping.numel() // pcp_size

    staged_inputs = tuple(
        _pad_to_rank_slab(tensor, collective_width) for tensor in tensors
    )
    gathered_inputs = tuple(
        pcp_group.all_gather(staged, dim=0) for staged in staged_inputs
    )
    return gathered_inputs, slot_mapping


def maybe_transfer_mla_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_rows = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_rows, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _transfer_mla_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


# Compatibility alias for existing MLA attention call sites.
maybe_gather_mla_latent_cache_inputs = maybe_transfer_mla_cache_inputs


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

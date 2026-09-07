# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.model_executor.layers.attention.pcp_wavefront_runtime import (
    PendingLayerReceive,
    post_layer_receive_into,
    post_layer_transfer,
    wait_layer_receive,
)


@dataclass
class PendingMLACacheTransfer:
    """State for a two-phase PCP MLA cache-input handoff.

    ``remote_inputs`` aliases the prefix of ``cache_inputs`` on rank1, so a
    future attention fast path can consume the received latent directly after
    ``finish_mla_cache_input_transfer`` without allocating another buffer.
    """

    cache_inputs: tuple[torch.Tensor, ...]
    cache_slot_mapping: torch.Tensor
    pending_receive: PendingLayerReceive | None = None
    remote_inputs: tuple[torch.Tensor, ...] | None = None


def _pad_to_rank_slab(
    tensor: torch.Tensor,
    rank_slab_width: int,
) -> torch.Tensor:
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


def begin_mla_cache_input_transfer(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
) -> PendingMLACacheTransfer:
    """Post the rank0->rank1 MLA handoff without waiting on rank1."""
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

    rank = pcp_group.rank_in_group
    if rank == 0:
        send_payload = tuple(
            _pad_to_rank_slab(tensor, rank_slab_width) for tensor in tensors
        )
        post_layer_transfer(send_payload)
        return PendingMLACacheTransfer(
            cache_inputs=tensors,
            cache_slot_mapping=slot_mapping[:model_num_rows],
        )
    if rank != 1:
        raise RuntimeError(f"Unexpected PCP rank for PCP=2 wavefront: {rank}")

    cache_inputs = tuple(
        tensor.new_empty(
            (rank_slab_width + model_num_rows, *tensor.shape[1:])
        )
        for tensor in tensors
    )
    remote_inputs = tuple(
        cache_input[:rank_slab_width] for cache_input in cache_inputs
    )
    pending_receive = post_layer_receive_into(remote_inputs)

    for cache_input, local_input in zip(cache_inputs, tensors):
        cache_input[rank_slab_width:].copy_(local_input)

    return PendingMLACacheTransfer(
        cache_inputs=cache_inputs,
        cache_slot_mapping=slot_mapping[: rank_slab_width + model_num_rows],
        pending_receive=pending_receive,
        remote_inputs=remote_inputs,
    )


def finish_mla_cache_input_transfer(
    pending: PendingMLACacheTransfer,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Complete a posted MLA cache-input handoff and expose its final views."""
    if pending.pending_receive is not None:
        wait_layer_receive(pending.pending_receive)
        pending.pending_receive = None
    return pending.cache_inputs, pending.cache_slot_mapping


def _transfer_mla_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Synchronous compatibility wrapper for the two-phase MLA handoff."""
    pending = begin_mla_cache_input_transfer(tensors, slot_mapping)
    return finish_mla_cache_input_transfer(pending)


def iter_tiled_mla_cache_inputs(*args, **kwargs):
    """Compatibility shim for the optional tiled PCP experiment.

    The production module does not import or retain tiled transport state.
    Tiled code is loaded only when an explicit experimental caller invokes it.
    """
    from vllm.model_executor.layers.attention.pcp_tiled import (
        iter_tiled_mla_cache_inputs as _iter_tiled_mla_cache_inputs,
    )

    return _iter_tiled_mla_cache_inputs(*args, **kwargs)


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
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
    k_pe_flat = k_pe.flatten(1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _transfer_mla_cache_inputs(
        (kv_c_normed, k_pe_flat), slot_mapping
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

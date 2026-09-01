# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.model_executor.layers.attention.pcp_wavefront_runtime import (
    post_layer_transfer,
    post_tile_transfer,
    recv_layer_payload_into,
    recv_tile_payload_into,
)


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


def _slice_or_pad_transport_tile(
    tensor: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    width = stop - start
    if width <= 0:
        raise ValueError(f"Invalid PCP transport tile [{start}, {stop})")
    num_rows = tensor.shape[0]
    if start >= num_rows:
        return tensor.new_zeros((width, *tensor.shape[1:]))
    real_stop = min(stop, num_rows)
    tile = tensor[start:real_stop]
    if tile.shape[0] == width:
        return tile.contiguous()
    padded = tensor.new_zeros((width, *tensor.shape[1:]))
    padded[: tile.shape[0]].copy_(tile)
    return padded


def _transfer_mla_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Baseline full-layer rank0->rank1 MLA cache handoff.

    Rank1 allocates the final cache-input slabs up front, receives rank0's
    payload directly into their prefix, and copies only its local rows into the
    suffix. This avoids both the duplicate receive allocation and the full
    remote/local ``torch.cat`` allocation while preserving one cache update.
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
        return tensors, rank_slot_mappings[0, :model_num_rows]
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
    recv_layer_payload_into(remote_inputs)

    for cache_input, local_input in zip(cache_inputs, tensors):
        cache_input[rank_slab_width:].copy_(local_input)

    local_slots = rank_slot_mappings[1, :model_num_rows]
    cache_slot_mapping = slot_mapping.new_empty(rank_slab_width + model_num_rows)
    cache_slot_mapping[:rank_slab_width].copy_(rank_slot_mappings[0])
    cache_slot_mapping[rank_slab_width:].copy_(local_slots)
    return cache_inputs, cache_slot_mapping


def iter_tiled_mla_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
    tile_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Yield bounded cache-update tiles and perform tiled PCP transport.

    Both ranks iterate the shared rank-slab width so weighted partitions with
    unequal local row counts still post/receive the same tile sequence.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    num_rows = kv_c_normed.shape[0]
    if k_pe.shape[0] != num_rows:
        raise RuntimeError("MLA cache inputs disagree on row count")

    if not use_pcp or num_decode_tokens is None:
        for start in range(0, num_rows, tile_size):
            stop = min(start + tile_size, num_rows)
            slots = None if slot_mapping is None else slot_mapping[start:stop]
            yield kv_c_normed[start:stop], k_pe[start:stop], slots
        return

    assert slot_mapping is not None
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2:
        raise NotImplementedError("Tiled Wavefront MLA transfer requires PCP=2.")
    if slot_mapping.numel() % 2 != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain two equal rank slabs: "
            f"numel={slot_mapping.numel()}"
        )

    rank_slab_width = slot_mapping.numel() // 2
    if num_rows > rank_slab_width:
        raise RuntimeError(
            "PCP local model rows exceed rank slab: "
            f"{num_rows} > {rank_slab_width}"
        )
    rank_slots = slot_mapping.view(2, rank_slab_width)
    rank = pcp_group.rank_in_group
    k_pe_flat = k_pe.flatten(1)

    if rank == 0:
        for start in range(0, rank_slab_width, tile_size):
            stop = min(start + tile_size, rank_slab_width)
            send_kv = _slice_or_pad_transport_tile(kv_c_normed, start, stop)
            send_kpe = _slice_or_pad_transport_tile(k_pe_flat, start, stop)
            post_tile_transfer((send_kv, send_kpe))
            if start < num_rows:
                local_stop = min(stop, num_rows)
                yield (
                    kv_c_normed[start:local_stop],
                    k_pe[start:local_stop],
                    rank_slots[0, start:local_stop],
                )
        return

    if rank != 1:
        raise RuntimeError(f"Unexpected PCP rank for PCP=2 wavefront: {rank}")

    for start in range(0, rank_slab_width, tile_size):
        stop = min(start + tile_size, rank_slab_width)
        width = stop - start
        recv_kv = kv_c_normed.new_empty((width, *kv_c_normed.shape[1:]))
        recv_kpe_flat = k_pe_flat.new_empty((width, k_pe_flat.shape[1]))
        recv_tile_payload_into((recv_kv, recv_kpe_flat))
        recv_kpe = recv_kpe_flat.view(width, *k_pe.shape[1:])
        yield recv_kv, recv_kpe, rank_slots[0, start:stop]

        if start < num_rows:
            local_stop = min(stop, num_rows)
            yield (
                kv_c_normed[start:local_stop],
                k_pe[start:local_stop],
                rank_slots[1, start:local_stop],
            )


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
    num_rows = kv_c_normed.shape[0]
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

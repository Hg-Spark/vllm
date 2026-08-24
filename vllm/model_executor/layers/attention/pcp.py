# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)

_PCP_CROSS_LAYER_OVERLAP = os.getenv("PCP_CROSS_LAYER_OVERLAP", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_PCP_CACHE_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_PCP_CACHE_WRITE_EVENTS: dict[int, torch.cuda.Event] = {}
_PCP_PENDING_CACHE_WRITES: set[int] = set()
# Padding exists only at the collective ABI. Reuse scratch per execution stream
# so uneven weighted partitions do not allocate/copy a fresh slab every layer.
_PCP_COLLECTIVE_SCRATCH: dict[
    tuple[int, int, torch.dtype, tuple[int, ...]], torch.Tensor
] = {}


def _collective_scratch(
    tensor: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    if tensor.device.type == "cuda":
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream_id = int(torch.cuda.current_stream(tensor.device).cuda_stream)
    else:
        device_index = -1
        stream_id = 0
    key = (device_index, stream_id, tensor.dtype, tuple(tensor.shape[1:]))
    scratch = _PCP_COLLECTIVE_SCRATCH.get(key)
    if scratch is None or scratch.shape[0] < num_tokens:
        scratch = tensor.new_empty((num_tokens, *tensor.shape[1:]))
        _PCP_COLLECTIVE_SCRATCH[key] = scratch
    return scratch[:num_tokens]


def _pad_prefill_for_collective(
    tensor: torch.Tensor,
    num_decode_tokens: int,
    collective_num_tokens: int,
) -> torch.Tensor:
    """Pad only the collective payload using reusable stream-local storage."""
    local_prefill = tensor[num_decode_tokens:]
    collective_prefill_tokens = collective_num_tokens - num_decode_tokens
    if collective_prefill_tokens < local_prefill.shape[0]:
        raise RuntimeError(
            "PCP collective slab is smaller than the local prefill payload: "
            f"collective={collective_prefill_tokens}, local={local_prefill.shape[0]}"
        )
    if collective_prefill_tokens == local_prefill.shape[0]:
        return local_prefill.contiguous()

    padded = _collective_scratch(tensor, collective_prefill_tokens)
    local_prefill_tokens = local_prefill.shape[0]
    if local_prefill_tokens:
        padded[:local_prefill_tokens].copy_(local_prefill)
    padded[local_prefill_tokens:].zero_()
    return padded


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills.

    Weighted PCP ranks may execute different token counts. The model consumes
    only each rank's real tokens; equal-width padding is introduced only at the
    collective boundary. PAD_SLOT_ID entries prevent padded values from being
    committed to KV cache.
    """
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    pcp_size = pcp_group.world_size
    flat_slot_mapping = slot_mapping.reshape(-1)
    if flat_slot_mapping.numel() % pcp_size != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain equal collective slabs: "
            f"slots={flat_slot_mapping.numel()}, world_size={pcp_size}"
        )
    collective_num_tokens = flat_slot_mapping.numel() // pcp_size
    if local_num_tokens > collective_num_tokens:
        raise RuntimeError(
            "PCP local token count exceeds the collective slab: "
            f"local={local_num_tokens}, collective={collective_num_tokens}"
        )

    gathered_prefills = tuple(
        pcp_group.all_gather(
            _pad_prefill_for_collective(
                tensor,
                num_decode_tokens,
                collective_num_tokens,
            ),
            dim=0,
        )
        for tensor in tensors
    )
    gathered_slot_mapping = flat_slot_mapping[: pcp_size * collective_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, collective_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


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


def _pcp_cache_comm_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _PCP_CACHE_COMM_STREAMS.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream(device=device_index)
        _PCP_CACHE_COMM_STREAMS[device_index] = stream
    return stream


def _wait_previous_mla_pcp_cache_write(
    attn_layer: Any,
    device: torch.device,
) -> None:
    layer_key = id(attn_layer)
    if layer_key not in _PCP_PENDING_CACHE_WRITES:
        return
    torch.cuda.current_stream(device).wait_event(_PCP_CACHE_WRITE_EVENTS[layer_key])
    _PCP_PENDING_CACHE_WRITES.remove(layer_key)


def maybe_launch_mla_pcp_cache_update(
    attn_layer: Any,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> bool:
    """Launch pure-prefill PCP cache replication on a dedicated CUDA stream.

    Returns True only when this function owns an asynchronous pure-prefill cache
    update. Decode and mixed batches intentionally return False so the caller
    executes their required synchronous cache update. Configurations that are
    incompatible with cross-layer overlap raise instead of silently reverting
    to the synchronous PCP path.

    A pending write from the previous invocation of this same layer is fenced on
    the current compute stream before handling decode/mixed batches or launching
    the next asynchronous pure-prefill update.
    """
    if not _PCP_CROSS_LAYER_OVERLAP:
        return False

    if kv_c_normed.device.type != "cuda":
        raise RuntimeError("PCP cross-layer overlap requires CUDA tensors")

    _wait_previous_mla_pcp_cache_write(attn_layer, kv_c_normed.device)

    if num_decode_tokens != 0:
        return False

    if not attn_layer.use_pcp or slot_mapping is None:
        return False

    if getattr(attn_layer.impl, "is_sparse", False):
        raise RuntimeError("PCP cross-layer overlap does not support sparse MLA")

    vllm_config = getattr(attn_layer, "_vllm_config", None)
    if vllm_config is not None:
        if getattr(vllm_config, "kv_transfer_config", None) is not None:
            raise RuntimeError(
                "PCP cross-layer overlap does not support KV transfer"
            )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if parallel_config is not None and getattr(
            parallel_config, "enable_dbo", False
        ):
            raise RuntimeError("PCP cross-layer overlap does not support DBO")

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "PCP cross-layer overlap does not support CUDA graph capture"
        )

    compute_stream = torch.cuda.current_stream(kv_c_normed.device)
    comm_stream = _pcp_cache_comm_stream(kv_c_normed.device)
    comm_stream.wait_stream(compute_stream)

    kv_c_normed.record_stream(comm_stream)
    k_pe.record_stream(comm_stream)
    if slot_mapping.is_cuda:
        slot_mapping.record_stream(comm_stream)

    layer_key = id(attn_layer)
    event = _PCP_CACHE_WRITE_EVENTS.get(layer_key)
    if event is None:
        device_index = kv_c_normed.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            event = torch.cuda.Event(blocking=False, interprocess=False)
        _PCP_CACHE_WRITE_EVENTS[layer_key] = event

    with torch.cuda.stream(comm_stream):
        cache_kv_c, cache_k_pe, cache_slot_mapping = (
            maybe_gather_mla_latent_cache_inputs(
                kv_c_normed,
                k_pe,
                slot_mapping,
                num_decode_tokens,
                True,
            )
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            cache_kv_c,
            cache_k_pe,
            kv_cache,
            cache_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )
        event.record(comm_stream)

    _PCP_PENDING_CACHE_WRITES.add(layer_key)
    return True


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

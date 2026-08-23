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


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    gathered_prefills = tuple(
        pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
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

    Returns True when this function owns the cache update. The caller must keep
    the original synchronous gather/write path when False.

    A pending write from the previous invocation of the same layer is fenced on
    the current compute stream before deciding whether the new invocation can be
    asynchronous. This keeps decode/mixed batches and cache reuse ordered while
    allowing layer-L cache replication to overlap layer-L/later computation.
    """
    if not _PCP_CROSS_LAYER_OVERLAP or kv_c_normed.device.type != "cuda":
        return False

    _wait_previous_mla_pcp_cache_write(attn_layer, kv_c_normed.device)

    if (
        not attn_layer.use_pcp
        or num_decode_tokens != 0
        or slot_mapping is None
        or getattr(attn_layer.impl, "is_sparse", False)
    ):
        return False

    # KV-transfer hooks may consume the just-written cache at the layer
    # boundary. Keep that configuration synchronous until a transfer-side fence
    # is wired explicitly.
    vllm_config = getattr(attn_layer, "_vllm_config", None)
    if vllm_config is not None and vllm_config.kv_transfer_config is not None:
        return False

    # Cross-stream work must not be introduced while this execution is being
    # captured into a CUDA graph.
    if torch.cuda.is_current_stream_capturing():
        return False

    compute_stream = torch.cuda.current_stream(kv_c_normed.device)
    comm_stream = _pcp_cache_comm_stream(kv_c_normed.device)
    comm_stream.wait_stream(compute_stream)

    # These tensors are produced/owned by the compute stream but remain inputs
    # to communication after the Python cache-update op returns.
    kv_c_normed.record_stream(comm_stream)
    k_pe.record_stream(comm_stream)
    if slot_mapping.is_cuda:
        slot_mapping.record_stream(comm_stream)

    layer_key = id(attn_layer)
    event = _PCP_CACHE_WRITE_EVENTS.get(layer_key)
    if event is None:
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

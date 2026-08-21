# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard MHA/GQA/MQA KV-cache transport for PCP."""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


def _write_local_kv_cache_for_page_pull(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> None:
    """Write rank-local K/V before publishing page READY.

    The normal FlashAttention ``reshape_and_cache_flash`` call still executes
    after this helper returns. This early write is intentionally duplicated so
    page-pull can expose registered cache pages without changing the large
    FlashAttention backend. The duplicate write is restricted to the local
    segment(s); remote page pulls target disjoint blocks.

    The experimental page-pull path currently targets unquantized FP16/BF16 KV
    cache. Quantized cache needs the layer's scale tensors and should be wired at
    the backend hook rather than reimplemented here.
    """
    if key.dtype != value.dtype or kv_cache.dtype != key.dtype:
        raise RuntimeError(
            "PCP page_pull currently requires unquantized KV cache with matching "
            f"dtype: key={key.dtype}, value={value.dtype}, cache={kv_cache.dtype}"
        )
    if kv_cache.ndim != 4:
        raise RuntimeError(
            "PCP page_pull expects FlashAttention block-major KV cache "
            f"[B,H,N,2D], got shape={tuple(kv_cache.shape)}"
        )
    if key.shape != value.shape:
        raise RuntimeError(
            f"PCP page_pull key/value shapes differ: {key.shape} vs {value.shape}"
        )
    if slot_mapping.shape[0] != key.shape[0]:
        raise RuntimeError(
            "PCP page_pull local slot count does not match local K/V rows: "
            f"slots={slot_mapping.shape[0]}, rows={key.shape[0]}"
        )

    valid = slot_mapping >= 0
    if not bool(valid.all()):
        slots = slot_mapping[valid]
        key_rows = key[valid]
        value_rows = value[valid]
    else:
        slots = slot_mapping
        key_rows = key
        value_rows = value
    if slots.numel() == 0:
        return

    block_size = int(kv_cache.shape[2])
    block_ids = torch.div(slots, block_size, rounding_mode="floor").long()
    block_offsets = torch.remainder(slots, block_size).long()
    head_size = int(key.shape[-1])
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    with pcp_nvtx_range("pcp.page_pull_local_cache_write"):
        key_cache[block_ids, block_offsets] = key_rows
        value_cache[block_ids, block_offsets] = value_rows


def prepare_standard_pcp_kv_cache_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FlashAttention cache-write inputs using PCP's active policy."""
    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        if runtime.transport == "prefix_p2p":
            with pcp_nvtx_range("pcp.prefix_exchange"):
                (key, value), slot_mapping = runtime.exchange_prefix(
                    (key, value), slot_mapping
                )
            return key, value, slot_mapping

        if runtime.transport == "direct_p2p":
            with pcp_nvtx_range("pcp.direct_exchange"):
                (key, value), slot_mapping = runtime.exchange_direct(
                    (key, value), slot_mapping
                )
            return key, value, slot_mapping

        if runtime.transport == "page_pull":
            if key.shape[0] != runtime.local_rows or value.shape[0] != runtime.local_rows:
                raise RuntimeError(
                    "PCP page_pull expects configured rank-local K/V rows: "
                    f"key={key.shape[0]}, value={value.shape[0]}, "
                    f"expected={runtime.local_rows}"
                )
            local_slot_mapping = runtime.rank_local_slot_mapping(slot_mapping)
            layer_ordinal = runtime.page_pull_register_layer(kv_cache)
            _write_local_kv_cache_for_page_pull(
                key, value, local_slot_mapping, kv_cache
            )
            with pcp_nvtx_range("pcp.page_pull_exchange"):
                runtime.page_pull_publish_and_wait(layer_ordinal)
            # The normal FlashAttention cache-update path writes local K/V once
            # more after this function returns. It must use only local slots;
            # remote blocks have already been filled by one-sided READs.
            return key, value, local_slot_mapping

        if runtime.transport == "full_kv_collective":
            if (
                key.shape[0] != runtime.local_rows
                or value.shape[0] != runtime.local_rows
            ):
                raise RuntimeError(
                    "PCP compact full-KV collective expects configured local rows: "
                    f"key={key.shape[0]}, value={value.shape[0]}, "
                    f"expected={runtime.local_rows}"
                )
            if slot_mapping.shape[0] < runtime.total_rows:
                raise RuntimeError(
                    "PCP compact slot mapping is shorter than full gathered rows: "
                    f"slots={slot_mapping.shape[0]}, rows={runtime.total_rows}"
                )
            pcp_group = get_pcp_group()
            with pcp_nvtx_range("pcp.full_kv_allgatherv"):
                key, value = pcp_group.all_gatherv(
                    [key.contiguous(), value.contiguous()],
                    dim=0,
                    sizes=list(runtime.rows_per_rank),
                )
            key = runtime.rank_major_to_segment_major(key)
            value = runtime.rank_major_to_segment_major(value)
            slot_mapping = runtime.rank_major_to_segment_major(slot_mapping)
            return key, value, slot_mapping

        raise RuntimeError(f"unsupported active PCP transport: {runtime.transport!r}")

    pcp_group = get_pcp_group()
    if pcp_group.world_size <= 1 or slot_mapping.shape[0] <= key.shape[0]:
        return key, value, slot_mapping

    pcp_size = pcp_group.world_size
    if slot_mapping.shape[0] % pcp_size != 0:
        raise RuntimeError(
            "PCP gathered slot mapping is not divisible by PCP size: "
            f"slots={slot_mapping.shape[0]}, pcp={pcp_size}"
        )
    local_rows = slot_mapping.shape[0] // pcp_size
    if key.shape[0] < local_rows or value.shape[0] < local_rows:
        raise RuntimeError(
            "PCP standard-attention K/V rows are smaller than the rank-local "
            f"slab: key={key.shape[0]}, value={value.shape[0]}, rows={local_rows}"
        )

    with pcp_nvtx_range("pcp.baseline_kv_allgather"):
        key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
        value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)
    return key, value, slot_mapping


__all__ = [
    "prepare_standard_pcp_kv_cache_inputs",
    "_write_local_kv_cache_for_page_pull",
]
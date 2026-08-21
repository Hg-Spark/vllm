# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard MHA/GQA/MQA KV-cache transport for PCP."""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


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
            runtime.page_pull_prepare_layer(kv_cache)
            return key, value, local_slot_mapping

        if runtime.transport == "full_kv_collective":
            if key.shape[0] != runtime.local_rows or value.shape[0] != runtime.local_rows:
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
            group = runtime._group()
            sizes = list(runtime.rows_per_rank)
            if len(set(sizes)) == 1:
                with pcp_nvtx_range("pcp.full_kv_allgather"):
                    key = group.all_gather(key.contiguous(), dim=0)
                    value = group.all_gather(value.contiguous(), dim=0)
            else:
                with pcp_nvtx_range("pcp.full_kv_allgatherv"):
                    key, value = group.all_gatherv(
                        [key.contiguous(), value.contiguous()],
                        dim=0,
                        sizes=sizes,
                    )
            return key, value, slot_mapping[: runtime.total_rows]

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
            "PCP standard-attention K/V rows are smaller than the rank-local slab: "
            f"key={key.shape[0]}, value={value.shape[0]}, rows={local_rows}"
        )

    with pcp_nvtx_range("pcp.baseline_kv_allgather"):
        key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
        value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)
    return key, value, slot_mapping


def finish_standard_pcp_kv_cache_update(kv_cache: torch.Tensor) -> None:
    """Publish page-pull READY after FlashAttention's native cache write."""
    runtime = get_pcp_runahead_runtime()
    if runtime is not None and runtime.transport == "page_pull":
        runtime.page_pull_after_cache_write(kv_cache)


__all__ = [
    "finish_standard_pcp_kv_cache_update",
    "prepare_standard_pcp_kv_cache_inputs",
]

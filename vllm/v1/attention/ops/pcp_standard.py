# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard MHA/GQA/MQA KV-cache transport for PCP."""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


_TENSOR_TRANSPORT_RANGES = {
    "full_kv_collective": "pcp.full_kv_exchange",
    "prefix_p2p": "pcp.prefix_exchange",
    "direct_p2p": "pcp.direct_exchange",
}


def prepare_standard_pcp_kv_cache_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FlashAttention cache-write inputs using PCP's active policy."""
    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
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

        range_name = _TENSOR_TRANSPORT_RANGES.get(runtime.transport)
        if range_name is None:
            raise RuntimeError(f"unsupported active PCP transport: {runtime.transport!r}")
        with pcp_nvtx_range(
            range_name,
            e=runtime.epoch,
            rank=runtime.rank,
            rows=runtime.local_rows,
        ):
            (key, value), slot_mapping = runtime.exchange_cache_inputs(
                (key, value), slot_mapping
            )
        return key, value, slot_mapping

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

    with pcp_nvtx_range(
        "pcp.baseline_kv_allgather",
        rank=pcp_group.rank_in_group,
        rows=local_rows,
    ):
        key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
        value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)
    return key, value, slot_mapping


__all__ = ["prepare_standard_pcp_kv_cache_inputs"]

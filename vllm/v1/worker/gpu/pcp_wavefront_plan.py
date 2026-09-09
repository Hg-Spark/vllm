# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.pcp_execution import PCPBatchPlan


@dataclass(frozen=True)
class PCPWavefrontPlan:
    """Step-level one-way PCP wavefront layout for PCP=2.

    Rank 0 produces the causal prefix and rank 1 consumes it. Slot mappings are
    kept in cache-group-major form so the same step plan can be reused by every
    Transformer layer that belongs to a cache group.
    """

    producer_rank: int
    consumer_rank: int
    num_remote_tokens: int
    remote_slot_mappings: torch.Tensor

    def slot_mapping_for_group(self, cache_group_idx: int) -> torch.Tensor:
        return self.remote_slot_mappings[cache_group_idx]


def build_pcp_wavefront_plan(
    batch_plan: PCPBatchPlan,
    slab_slot_mappings: torch.Tensor,
) -> PCPWavefrontPlan:
    """Build rank0-prefix -> rank1 slot placement from the current step layout."""
    if len(batch_plan.per_rank_num_tokens) != 2:
        raise NotImplementedError("PCP wavefront currently requires PCP=2.")
    if slab_slot_mappings.ndim != 2:
        raise ValueError("PCP slab slot mappings must be cache-group-major 2D tensor.")

    expected_width = batch_plan.rank_slab_width * 2
    if slab_slot_mappings.shape[1] != expected_width:
        raise ValueError(
            "PCP slab slot width does not match the batch plan: "
            f"{slab_slot_mappings.shape[1]} != {expected_width}"
        )

    # Wavefront decode ownership is rank1-only, so every semantic row owned by
    # rank0 is a prefill row. Rank-local rows occupy the beginning of rank0's
    # fixed-width rank slab in exactly the order emitted by rank0.
    num_remote_tokens = batch_plan.per_rank_num_tokens[0]
    remote_slot_mappings = slab_slot_mappings[:, :num_remote_tokens]

    return PCPWavefrontPlan(
        producer_rank=0,
        consumer_rank=1,
        num_remote_tokens=num_remote_tokens,
        remote_slot_mappings=remote_slot_mappings,
    )

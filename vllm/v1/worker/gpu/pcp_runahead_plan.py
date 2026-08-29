# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.v1.worker.gpu.pcp_execution import PCPBatchPlan


@dataclass(frozen=True)
class PCPRunaheadPlan:
    """Step-level one-way PCP runahead layout for PCP=2.

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


def build_pcp_runahead_plan(
    batch_plan: PCPBatchPlan,
    gathered_slot_mappings: torch.Tensor,
) -> PCPRunaheadPlan:
    """Build rank0-prefix -> rank1 slot placement from the current step layout."""
    if len(batch_plan.per_rank_num_tokens) != 2:
        raise NotImplementedError("Wavefront runahead currently requires PCP=2.")
    if gathered_slot_mappings.ndim != 2:
        raise ValueError(
            "PCP gathered slot mappings must be cache-group-major 2D tensor."
        )

    expected_width = batch_plan.collective_width * 2
    if gathered_slot_mappings.shape[1] != expected_width:
        raise ValueError(
            "PCP gathered slot width does not match the batch plan: "
            f"{gathered_slot_mappings.shape[1]} != {expected_width}"
        )

    # Wavefront decode ownership is rank1-only, so every semantic row owned by
    # rank0 is a prefill row. Rank-local rows occupy the beginning of rank0's
    # fixed-width collective slab in exactly the order emitted by rank0.
    num_remote_tokens = batch_plan.per_rank_num_tokens[0]
    remote_slot_mappings = gathered_slot_mappings[:, :num_remote_tokens]

    return PCPRunaheadPlan(
        producer_rank=0,
        consumer_rank=1,
        num_remote_tokens=num_remote_tokens,
        remote_slot_mappings=remote_slot_mappings,
    )

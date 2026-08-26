# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small execution-policy helpers for optional PCP implementations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_MOE_ISOLATED_PARTITION_IMPLS = frozenset(
    {
        "weighted_contiguous",
        "weighted_dual_chunk",
    }
)


def is_moe_isolated_pcp(vllm_config: "VllmConfig") -> bool:
    """Whether the active PCP policy treats PCP as a pure context axis."""
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return False
    partition = additional_config.get("pcp_partition")
    if not isinstance(partition, dict):
        return False
    return partition.get("impl") in _MOE_ISOLATED_PARTITION_IMPLS


def resolve_pcp_moe_size(vllm_config: "VllmConfig", default_size: int) -> int:
    """Return the PCP dimension that FusedMoE should see.

    Canonical PCP keeps the upstream behavior. Experimental isolated PCP
    policies execute complete MoE layers per PCP member, so PCP must not be
    folded into MoE TP/EP topology and is exposed to FusedMoE as size 1.
    """
    if not is_moe_isolated_pcp(vllm_config):
        return default_size

    if vllm_config.parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "isolated PCP does not compose with expert parallel yet: the upstream "
            "EP process group still folds the PCP dimension into EP topology"
        )

    return 1

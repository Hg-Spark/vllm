# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small execution-policy helpers for PCP partition weights."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def is_moe_isolated_pcp(vllm_config: "VllmConfig") -> bool:
    """Whether PCP uses the isolated weighted partition execution path."""
    additional_config = vllm_config.additional_config
    return isinstance(additional_config, dict) and "pcp_partition_weights" in additional_config


def resolve_pcp_moe_size(vllm_config: "VllmConfig", default_size: int) -> int:
    """Return the PCP dimension that FusedMoE should see.

    The weighted PCP execution path treats PCP as a pure context axis, so PCP
    must not be folded into MoE TP/EP topology and is exposed to FusedMoE as
    size 1.
    """
    if not is_moe_isolated_pcp(vllm_config):
        return default_size

    if vllm_config.parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "isolated PCP does not compose with expert parallel yet: the upstream "
            "EP process group still folds the PCP dimension into EP topology"
        )

    return 1

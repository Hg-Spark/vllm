# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-specific startup profiling helpers."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def resolve_pcp_profile_num_tokens(worker: "Worker") -> int | None:
    """Return this rank's weighted-PCP profile width, or None for baseline PCP."""
    parallel_config = worker.vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    additional_config = worker.vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return None
    if "pcp_partition_weights" not in additional_config:
        return None

    from vllm.distributed.parallel_state import get_pcp_group
    from vllm.v1.worker.gpu.pcp_weighted_partition import (
        parse_pcp_partition_weights,
        weighted_partition_lengths,
    )

    global_num_tokens = worker.model_runner.max_num_tokens
    pcp_rank = get_pcp_group().rank_in_group
    weights = parse_pcp_partition_weights(additional_config, pcp_size)
    per_rank_tokens = weighted_partition_lengths(global_num_tokens, weights)
    profile_num_tokens = max(1, per_rank_tokens[pcp_rank])
    logger.info(
        "Resolved PCP startup profile width: rank=%d global_tokens=%d "
        "profile_tokens=%d per_rank_tokens=%s",
        pcp_rank,
        global_num_tokens,
        profile_num_tokens,
        per_rank_tokens,
    )
    return profile_num_tokens


def maybe_prepare_pcp_profile_run(worker: "Worker") -> None:
    """Make exactly one startup profile call use the rank-local PCP width."""
    profile_num_tokens = resolve_pcp_profile_num_tokens(worker)
    if profile_num_tokens is None:
        return

    runner = worker.model_runner
    original_profile_run = runner.profile_run

    def profile_run_once() -> None:
        previous_max_num_tokens = runner.max_num_tokens
        runner.max_num_tokens = profile_num_tokens
        try:
            original_profile_run()
        finally:
            runner.max_num_tokens = previous_max_num_tokens
            runner.profile_run = original_profile_run

    runner.profile_run = profile_run_once

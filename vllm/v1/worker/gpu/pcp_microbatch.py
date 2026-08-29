# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PCP memory microbatching for token-wise feed-forward sublayers.

Wavefront microbatching exists only to cap transient GPU memory. It does not
change the rank0->rank1 causal dependency or the full-layer KV handoff. The MLA
attention path keeps its existing layer-level execution and context-workspace
chunking; this module bounds the token-wise MLP/MoE temporaries that otherwise
scale with all rank-local prefill rows.
"""

from collections.abc import Callable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_KEY = "pcp_microbatch_size"
_PATCHED = False


def parse_pcp_microbatch_size(additional_config: object) -> int:
    """Return the configured rank-local memory microbatch size.

    ``0`` means disabled. The value is intentionally PCP-specific instead of
    reusing DBO/``--ubatch-size``: DBO changes model-forward scheduling, while
    this knob only bounds token-wise transient allocations inside each layer.
    """
    if not isinstance(additional_config, dict):
        return 0
    raw = additional_config.get(_CONFIG_KEY, 0)
    if raw is None:
        return 0
    if isinstance(raw, bool):
        raise ValueError(f"{_CONFIG_KEY} must be a non-negative integer, got {raw}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_CONFIG_KEY} must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0 or value != raw:
        raise ValueError(
            f"{_CONFIG_KEY} must be a non-negative integer, got {raw!r}"
        )
    return value


def _validate_config(vllm_config: VllmConfig, microbatch_size: int) -> None:
    if microbatch_size == 0:
        return

    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    if parallel_config.prefill_context_parallel_size != 2:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires prefill_context_parallel_size=2."
        )
    if parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires tensor_parallel_size=1."
        )
    if parallel_config.data_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires data_parallel_size=1."
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires decode_context_parallel_size=1."
        )
    if parallel_config.use_sequence_parallel_moe:
        raise NotImplementedError(
            "pcp_microbatch_size does not support sequence-parallel MoE yet."
        )
    if scheduler_config.max_num_seqs != 1:
        raise NotImplementedError(
            "pcp_microbatch_size currently requires max_num_seqs=1 so the long "
            "prefill memory bound is unambiguous."
        )


def microbatch_slices(num_tokens: int, microbatch_size: int) -> tuple[slice, ...]:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if microbatch_size <= 0 or num_tokens <= microbatch_size:
        return (slice(0, num_tokens),) if num_tokens > 0 else ()
    return tuple(
        slice(start, min(start + microbatch_size, num_tokens))
        for start in range(0, num_tokens, microbatch_size)
    )


def run_tokenwise_microbatches(
    forward_fn: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    microbatch_size: int,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run a token-wise sublayer in bounded chunks and restore row order."""
    slices = microbatch_slices(hidden_states.shape[0], microbatch_size)
    if len(slices) <= 1:
        return forward_fn(hidden_states, *args, **kwargs)

    output: torch.Tensor | None = None
    for token_slice in slices:
        chunk_output = forward_fn(hidden_states[token_slice], *args, **kwargs)
        if output is None:
            output = chunk_output.new_empty(
                (hidden_states.shape[0], *chunk_output.shape[1:])
            )
        output[token_slice].copy_(chunk_output)
    assert output is not None
    return output


def configure_pcp_memory_microbatching(vllm_config: VllmConfig) -> int:
    """Validate the knob and install the DeepSeek/GLM-DSA feed-forward wrappers."""
    global _PATCHED

    microbatch_size = parse_pcp_microbatch_size(vllm_config.additional_config)
    _validate_config(vllm_config, microbatch_size)
    if microbatch_size == 0 or _PATCHED:
        return microbatch_size

    # GLM DSA models are registered through deepseek_v2, so this covers the
    # current GLM-4.7-Flash Wavefront target as well as DeepSeek V2/V3.
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2MLP,
        DeepseekV2MoE,
    )

    original_mlp_forward = DeepseekV2MLP.forward
    original_moe_forward = DeepseekV2MoE.forward

    def mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return run_tokenwise_microbatches(
            lambda chunk: original_mlp_forward(self, chunk),
            hidden_states,
            microbatch_size,
        )

    def moe_forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if already_sequence_parallel:
            # Rejected by config validation for the current POC, but preserve
            # the original path if a caller reaches this method during setup.
            return original_moe_forward(
                self,
                hidden_states,
                already_sequence_parallel=already_sequence_parallel,
            )
        return run_tokenwise_microbatches(
            lambda chunk: original_moe_forward(
                self,
                chunk,
                already_sequence_parallel=False,
            ),
            hidden_states,
            microbatch_size,
        )

    DeepseekV2MLP.forward = mlp_forward
    DeepseekV2MoE.forward = moe_forward
    _PATCHED = True
    logger.info(
        "Enabled PCP memory microbatching for token-wise MLP/MoE rows: size=%d",
        microbatch_size,
    )
    return microbatch_size

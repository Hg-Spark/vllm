# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .communication_op import *
from .parallel_state import *
from .utils import *


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """Initialize model-parallel groups with runahead PCP ordering at startup."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.config.pcp_runahead import parse_pcp_runahead_config

    from . import parallel_state as ps

    pcp_group_order: tuple[int, ...] | None = None
    config = get_current_vllm_config_or_none()
    if config is not None and prefill_context_model_parallel_size > 1:
        runahead = parse_pcp_runahead_config(
            config.additional_config, prefill_context_model_parallel_size
        )
        if runahead is not None:
            if config.kv_transfer_config is not None:
                raise NotImplementedError(
                    "PCP runahead does not support request-level KV transfer connectors"
                )

            parallel = config.parallel_config
            supported_layout = (
                tensor_model_parallel_size == 1
                and pipeline_model_parallel_size == 1
                and parallel.data_parallel_size == 1
                and (decode_context_model_parallel_size or 1) == 1
            )
            if supported_layout:
                pcp_group_order = runahead.pcp_group_order

    ps.ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        decode_context_model_parallel_size,
        backend,
        pcp_group_order=pcp_group_order,
    )

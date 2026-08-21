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
    """Initialize model-parallel groups, applying PCP runahead rank binding once.

    A one-to-one runahead segment permutation changes the member order of the
    primary PCP group during startup. No second PCP communicator is created.
    """
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.worker.gpu.pcp_runahead_config import get_pcp_process_group_order

    from . import parallel_state as _parallel_state

    config = get_current_vllm_config_or_none()
    if config is None or prefill_context_model_parallel_size <= 1:
        return _parallel_state.ensure_model_parallel_initialized(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )

    parallel = config.parallel_config
    supported_layout = (
        tensor_model_parallel_size == 1
        and pipeline_model_parallel_size == 1
        and parallel.data_parallel_size == 1
        and (decode_context_model_parallel_size or 1) == 1
    )
    order = (
        get_pcp_process_group_order(
            config.additional_config, prefill_context_model_parallel_size
        )
        if supported_layout
        else tuple(range(prefill_context_model_parallel_size))
    )
    if order == tuple(range(prefill_context_model_parallel_size)):
        return _parallel_state.ensure_model_parallel_initialized(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )

    original_init_group = _parallel_state.init_model_parallel_group

    def _init_group(*args, **kwargs):
        group_name = kwargs.get("group_name")
        if group_name == "pcp":
            if args:
                group_ranks = args[0]
                group_ranks = [
                    [ranks[physical_rank] for physical_rank in order]
                    for ranks in group_ranks
                ]
                args = (group_ranks, *args[1:])
            else:
                group_ranks = kwargs["group_ranks"]
                kwargs["group_ranks"] = [
                    [ranks[physical_rank] for physical_rank in order]
                    for ranks in group_ranks
                ]
        return original_init_group(*args, **kwargs)

    _parallel_state.init_model_parallel_group = _init_group
    try:
        _parallel_state.ensure_model_parallel_initialized(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )
    finally:
        _parallel_state.init_model_parallel_group = original_init_group

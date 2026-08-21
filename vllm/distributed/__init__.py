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

    def initialize() -> None:
        ps.ensure_model_parallel_initialized(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )

    config = get_current_vllm_config_or_none()
    if config is None or prefill_context_model_parallel_size <= 1:
        initialize()
        return

    runahead = parse_pcp_runahead_config(
        config.additional_config, prefill_context_model_parallel_size
    )
    if runahead is None:
        initialize()
        return
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
    order = (
        runahead.pcp_group_order
        if supported_layout
        else tuple(range(prefill_context_model_parallel_size))
    )
    if order == tuple(range(prefill_context_model_parallel_size)):
        initialize()
        return

    original_init_group = ps.init_model_parallel_group

    def init_group(*args, **kwargs):
        if kwargs.get("group_name") == "pcp":
            group_ranks = args[0] if args else kwargs["group_ranks"]
            group_ranks = [
                [ranks[physical_rank] for physical_rank in order]
                for ranks in group_ranks
            ]
            if args:
                args = (group_ranks, *args[1:])
            else:
                kwargs["group_ranks"] = group_ranks
        return original_init_group(*args, **kwargs)

    ps.init_model_parallel_group = init_group
    try:
        initialize()
    finally:
        ps.init_model_parallel_group = original_init_group

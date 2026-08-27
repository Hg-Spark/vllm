# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .communication_op import *
from .parallel_state import *
from .utils import *


_base_get_ep_group = get_ep_group


def get_ep_group():
    """Keep PCP ranks model-local for the current TP=DP=1, EP-off path."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is not None:
        parallel_config = config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            and parallel_config.tensor_parallel_size == 1
            and parallel_config.data_parallel_size == 1
            and not parallel_config.enable_expert_parallel
        ):
            return get_tp_group()
    return _base_get_ep_group()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.config import (
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)
from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime


def test_page_pull_restores_captured_vllm_config_for_cache_discovery() -> None:
    vllm_config = MagicMock()
    group = MagicMock()
    page_pull = MagicMock()

    def assert_configured_context(*args, **kwargs):
        assert get_current_vllm_config() is vllm_config

    page_pull.configure_step.side_effect = assert_configured_context
    page_pull.register_current_layer.side_effect = lambda kv_cache: (
        assert_configured_context() or 0
    )

    with set_current_vllm_config(vllm_config):
        runtime = PCPRunaheadRuntime(
            pcp_world_size=2,
            pcp_rank=0,
            device=torch.device("cpu"),
            pcp_group=group,
        )

    assert get_current_vllm_config_or_none() is None

    plan = MagicMock(world_size=2)
    with patch(
        "vllm.v1.attention.ops.pcp_runahead.PCPPagePullTransport",
        return_value=page_pull,
    ):
        runtime.begin_step((1, 1), transport="page_pull")
        runtime.configure_page_plan(plan)

    assert get_current_vllm_config_or_none() is None
    page_pull.configure_step.assert_called_once_with(epoch=1, plan=plan)

    kv_cache = torch.empty(1)
    runtime.page_pull_prepare_layer(kv_cache)
    assert get_current_vllm_config_or_none() is None
    page_pull.register_current_layer.assert_called_once_with(kv_cache)

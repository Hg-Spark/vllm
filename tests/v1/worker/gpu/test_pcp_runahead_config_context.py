# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime


def test_page_pull_uses_captured_static_forward_context_outside_config_context() -> None:
    kv_cache = torch.empty(1)
    layer = SimpleNamespace(
        kv_cache=kv_cache,
        kv_sharing_target_layer_name=None,
    )
    static_forward_context = {"model.layers.0.self_attn": layer}

    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = static_forward_context
    group = MagicMock()
    page_pull = MagicMock()
    page_pull.register_current_layer.return_value = 0

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
    ) as transport_cls:
        runtime.begin_step((1, 1), transport="page_pull")
        runtime.configure_page_plan(plan)

    assert get_current_vllm_config_or_none() is None
    assert transport_cls.call_args.kwargs["static_forward_context"] is static_forward_context
    page_pull.configure_step.assert_called_once_with(epoch=1, plan=plan)

    runtime.page_pull_prepare_layer(kv_cache)
    assert get_current_vllm_config_or_none() is None
    page_pull.register_current_layer.assert_called_once_with(kv_cache)

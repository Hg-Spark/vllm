# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch

from vllm import envs
from vllm.v1.attention.ops import pcp_profile


def test_pcp_nvtx_env_is_registered(monkeypatch) -> None:
    assert "VLLM_PCP_NVTX" in envs.environment_variables
    monkeypatch.setenv("VLLM_PCP_NVTX", "yes")
    assert envs.environment_variables["VLLM_PCP_NVTX"]()
    monkeypatch.setenv("VLLM_PCP_NVTX", "0")
    assert not envs.environment_variables["VLLM_PCP_NVTX"]()


def test_pcp_nvtx_range_formats_fields() -> None:
    with (
        patch.object(pcp_profile, "_PCP_NVTX_ENABLED", True),
        patch.object(torch.cuda.nvtx, "range_push") as push,
        patch.object(torch.cuda.nvtx, "range_pop") as pop,
    ):
        with pcp_profile.pcp_nvtx_range(
            "pcp.page_pull_wait", e=3, l="model.layers.7", dst=2, sources=2
        ):
            pass
    push.assert_called_once_with(
        "pcp.page_pull_wait[e=3,l=model.layers.7,dst=2,sources=2]"
    )
    pop.assert_called_once_with()


def test_pcp_nvtx_mark_formats_fields() -> None:
    with (
        patch.object(pcp_profile, "_PCP_NVTX_ENABLED", True),
        patch.object(torch.cuda.nvtx, "mark") as mark,
    ):
        pcp_profile.pcp_nvtx_mark(
            "pcp.direct_send", e=4, src=1, dst=3, rows=128
        )
    mark.assert_called_once_with("pcp.direct_send[e=4,src=1,dst=3,rows=128]")

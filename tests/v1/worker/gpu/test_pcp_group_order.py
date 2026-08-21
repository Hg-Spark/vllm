# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import vllm.distributed as distributed
from vllm.distributed import parallel_state


def _config(additional_config: dict, *, kv_transfer_config=None) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config=additional_config,
        kv_transfer_config=kv_transfer_config,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )


def test_primary_pcp_group_forwards_runahead_permutation() -> None:
    config = _config(
        {
            "pcp_runahead": {
                "transport": "direct_p2p",
                "partition": {
                    "segments": [
                        {"pcp_rank": 1},
                        {"pcp_rank": 0},
                    ]
                },
            }
        }
    )
    ensure = MagicMock()

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch.object(parallel_state, "ensure_model_parallel_initialized", ensure),
    ):
        distributed.ensure_model_parallel_initialized(1, 1, 2, 1)

    ensure.assert_called_once_with(
        1,
        1,
        2,
        1,
        None,
        pcp_group_order=(1, 0),
    )


def test_repeated_page_pull_binding_forwards_identity_group_order() -> None:
    config = _config(
        {
            "pcp_runahead": {
                "transport": "page_pull",
                "partition": {
                    "segments": [
                        {"pcp_rank": 1},
                        {"pcp_rank": 0},
                        {"pcp_rank": 1},
                    ]
                },
            }
        }
    )
    ensure = MagicMock()

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch.object(parallel_state, "ensure_model_parallel_initialized", ensure),
    ):
        distributed.ensure_model_parallel_initialized(1, 1, 2, 1)

    ensure.assert_called_once_with(
        1,
        1,
        2,
        1,
        None,
        pcp_group_order=(0, 1),
    )


def test_parallel_state_applies_explicit_pcp_group_order() -> None:
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        enable_elastic_ep=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(is_moe=False),
    )
    world = SimpleNamespace(local_rank=0)
    group = SimpleNamespace(rank_in_group=0)
    init_group = MagicMock(return_value=group)

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=0),
        patch("vllm.config.get_current_vllm_config", return_value=config),
        patch.object(parallel_state, "get_world_group", return_value=world),
        patch.object(parallel_state, "init_model_parallel_group", init_group),
        patch.multiple(
            parallel_state,
            _TP=None,
            _DCP=None,
            _PCP=None,
            _PP=None,
            _DP=None,
            _EP=None,
            _EPLB=None,
        ),
    ):
        parallel_state.initialize_model_parallel(
            1,
            1,
            2,
            1,
            "nccl",
            pcp_group_order=(1, 0),
        )

    pcp_calls = [
        call
        for call in init_group.call_args_list
        if call.kwargs.get("group_name") == "pcp"
    ]
    assert len(pcp_calls) == 1
    assert pcp_calls[0].args[0] == [[1, 0]]


def test_runahead_rejects_request_level_kv_connector() -> None:
    config = _config(
        {"pcp_runahead": {"transport": "prefix_p2p"}},
        kv_transfer_config=object(),
    )
    ensure = MagicMock()
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch.object(parallel_state, "ensure_model_parallel_initialized", ensure),
        pytest.raises(NotImplementedError, match="KV transfer connectors"),
    ):
        distributed.ensure_model_parallel_initialized(1, 1, 2, 1)

    ensure.assert_not_called()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import vllm.distributed as distributed
from vllm.distributed import parallel_state


def test_primary_pcp_group_uses_runahead_permutation() -> None:
    config = SimpleNamespace(
        additional_config={
            "pcp_runahead": {
                "transport": "direct_p2p",
                "partition": {
                    "segments": [
                        {"pcp_rank": 1},
                        {"pcp_rank": 0},
                    ]
                },
            }
        },
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    group_init = MagicMock()

    def fake_ensure(*args, **kwargs) -> None:
        parallel_state.init_model_parallel_group(
            [[10, 11]],
            0,
            "nccl",
            group_name="pcp",
        )

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch.object(
            parallel_state,
            "ensure_model_parallel_initialized",
            side_effect=fake_ensure,
        ),
        patch.object(parallel_state, "init_model_parallel_group", group_init),
    ):
        distributed.ensure_model_parallel_initialized(1, 1, 2, 1)
        assert parallel_state.init_model_parallel_group is group_init

    group_init.assert_called_once()
    assert group_init.call_args.args[0] == [[11, 10]]


def test_repeated_page_pull_binding_does_not_reorder_primary_group() -> None:
    config = SimpleNamespace(
        additional_config={
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
        },
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    group_init = MagicMock()

    def fake_ensure(*args, **kwargs) -> None:
        parallel_state.init_model_parallel_group(
            [[10, 11]],
            0,
            "nccl",
            group_name="pcp",
        )

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        patch.object(
            parallel_state,
            "ensure_model_parallel_initialized",
            side_effect=fake_ensure,
        ),
        patch.object(parallel_state, "init_model_parallel_group", group_init),
    ):
        distributed.ensure_model_parallel_initialized(1, 1, 2, 1)

    group_init.assert_called_once()
    assert group_init.call_args.args[0] == [[10, 11]]

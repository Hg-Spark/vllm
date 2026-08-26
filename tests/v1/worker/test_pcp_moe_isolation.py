# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.pcp_policy import is_moe_isolated_pcp, resolve_pcp_moe_size


def _config(
    additional_config: object,
    *,
    enable_expert_parallel: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=enable_expert_parallel,
        ),
    )


def test_default_pcp_keeps_upstream_moe_pcp_size() -> None:
    config = _config({})

    assert not is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 4


def test_unrelated_additional_config_does_not_change_moe_topology() -> None:
    config = _config({"other_option": True})

    assert not is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 4


def test_partition_weights_remove_pcp_from_moe_topology() -> None:
    config = _config({"pcp_partition_weights": [1.0, 1.0]})

    assert is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 1


def test_partition_weights_reject_expert_parallel() -> None:
    config = _config(
        {"pcp_partition_weights": [1.0, 1.0]},
        enable_expert_parallel=True,
    )

    with pytest.raises(NotImplementedError, match="expert parallel"):
        resolve_pcp_moe_size(config, 4)

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


def test_canonical_pcp_keeps_upstream_moe_pcp_size() -> None:
    config = _config({})

    assert not is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 4


def test_unknown_partition_does_not_change_moe_topology() -> None:
    config = _config({"pcp_partition": {"impl": "other"}})

    assert not is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 4


@pytest.mark.parametrize(
    "impl",
    ["weighted_contiguous", "weighted_dual_chunk"],
)
def test_experimental_partition_removes_pcp_from_moe_topology(impl: str) -> None:
    config = _config({"pcp_partition": {"impl": impl}})

    assert is_moe_isolated_pcp(config)
    assert resolve_pcp_moe_size(config, 4) == 1


@pytest.mark.parametrize(
    "impl",
    ["weighted_contiguous", "weighted_dual_chunk"],
)
def test_experimental_partition_rejects_expert_parallel(impl: str) -> None:
    config = _config(
        {"pcp_partition": {"impl": impl}},
        enable_expert_parallel=True,
    )

    with pytest.raises(NotImplementedError, match="expert parallel"):
        resolve_pcp_moe_size(config, 4)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe.layer import make_parallel_config


def _parallel_config() -> SimpleNamespace:
    return SimpleNamespace(
        enable_expert_parallel=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def test_runahead_replicates_moe_across_pcp() -> None:
    config = make_parallel_config(
        tp_size=1,
        dp_size=1,
        pcp_size=4,
        is_sequence_parallel=False,
        parallel_config=_parallel_config(),
        replicate_across_pcp=True,
    )

    assert config.tp_size == 1
    assert config.tp_rank == 0
    assert config.pcp_size == 1
    assert config.pcp_rank == 0
    assert config.ep_size == 1
    assert config.ep_rank == 0
    assert not config.use_ep


def test_standard_pcp_moe_keeps_existing_flattening() -> None:
    pcp_group = MagicMock()
    pcp_group.rank_in_group = 2

    with patch(
        "vllm.model_executor.layers.fused_moe.config.get_pcp_group",
        return_value=pcp_group,
    ):
        config = make_parallel_config(
            tp_size=1,
            dp_size=1,
            pcp_size=4,
            is_sequence_parallel=False,
            parallel_config=_parallel_config(),
        )

    assert config.tp_size == 4
    assert config.tp_rank == 2
    assert config.pcp_size == 4
    assert config.pcp_rank == 2
    assert config.ep_size == 1
    assert not config.use_ep

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe.layer import make_parallel_config


def _parallel_config(*, enable_expert_parallel: bool = False):
    return SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def test_pcp_does_not_shard_local_moe_weights() -> None:
    config = make_parallel_config(
        tp_size=1,
        dp_size=1,
        pcp_size=2,
        is_sequence_parallel=False,
        parallel_config=_parallel_config(),
    )

    assert config.tp_size == 1
    assert config.tp_rank == 0
    assert config.pcp_size == 1
    assert config.pcp_rank == 0
    assert config.dp_size == 1
    assert config.ep_size == 1
    assert not config.use_ep

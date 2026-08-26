# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

import vllm.v1.worker.gpu.pcp_execution as pcp_execution
from vllm.v1.worker.gpu.pcp_execution import PCPExecutionManager
from vllm.v1.worker.gpu.pcp_weighted_partition import WeightedPCPManager


def _block_tables() -> SimpleNamespace:
    return SimpleNamespace(
        kernel_block_sizes=[128],
        num_kv_cache_groups=0,
    )


def _layout_inputs(num_tokens: int) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([True], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def _copy_to_cpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    del device
    value = torch.as_tensor(x).clone()
    if out is None:
        return value
    out.copy_(value)
    return out


def test_weighted_manager_uses_pcp_execution_contract() -> None:
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    assert isinstance(manager, PCPExecutionManager)


def test_batch_plan_separates_actual_model_and_collective_width(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
        partition_weights=(2.0, 1.0),
    )

    plan = manager._build_batch_plan(*_layout_inputs(4096))

    assert plan.per_rank_num_tokens == (2688, 1408)
    assert plan.actual_num_tokens == 1408
    assert plan.model_num_tokens == 1408
    assert plan.collective_width == 2688
    assert not plan.uses_dummy_execution_row
    assert plan.collective_global_idx.numel() == 2 * 2688


def test_empty_owner_gets_exactly_one_dummy_model_row(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    manager = WeightedPCPManager(
        pcp_world_size=4,
        pcp_rank=3,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )

    plan = manager._build_batch_plan(*_layout_inputs(1))

    assert plan.per_rank_num_tokens == (1, 0, 0, 0)
    assert plan.actual_num_tokens == 0
    assert plan.model_num_tokens == 1
    assert plan.collective_width == 1
    assert plan.uses_dummy_execution_row
    assert not bool(plan.kv_write_mask[3].item())

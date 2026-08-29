# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_execution as pcp_execution
from vllm.v1.worker.gpu.pcp_wavefront_plan import build_pcp_wavefront_plan
from vllm.v1.worker.gpu.pcp_weighted_partition import WeightedPCPManager


def _block_tables() -> SimpleNamespace:
    return SimpleNamespace(kernel_block_sizes=[1], num_kv_cache_groups=0)


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


def _layout_inputs(
    num_tokens: int,
    *,
    is_prefilling: bool,
) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([is_prefilling], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def test_wavefront_plan_selects_rank0_valid_prefix_slots(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    batch_plan = manager._build_batch_plan(*_layout_inputs(5, is_prefilling=True))
    assert batch_plan.per_rank_num_tokens == (3, 2)
    assert batch_plan.rank_slab_width == 3

    # Two cache groups, rank-major fixed-width layout:
    # rank0 [10, 11, 12] | rank1 [20, 21, PAD]
    slab_slots = torch.tensor(
        [[10, 11, 12, 20, 21, -1], [30, 31, 32, 40, 41, -1]],
        dtype=torch.int64,
    )

    plan = build_pcp_wavefront_plan(batch_plan, slab_slots)

    assert plan.producer_rank == 0
    assert plan.consumer_rank == 1
    assert plan.num_remote_tokens == 3
    assert torch.equal(plan.slot_mapping_for_group(0), torch.tensor([10, 11, 12]))
    assert torch.equal(plan.slot_mapping_for_group(1), torch.tensor([30, 31, 32]))


def test_decode_only_wavefront_plan_has_no_remote_prefix(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    manager = WeightedPCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    batch_plan = manager._build_batch_plan(*_layout_inputs(2, is_prefilling=False))
    assert batch_plan.per_rank_num_tokens == (0, 2)

    slab_slots = torch.tensor([[-1, -1, 50, 51]], dtype=torch.int64)
    plan = build_pcp_wavefront_plan(batch_plan, slab_slots)

    assert plan.num_remote_tokens == 0
    assert plan.remote_slot_mappings.shape == (1, 0)


def test_wavefront_plan_rejects_non_pcp2(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    manager = WeightedPCPManager(
        pcp_world_size=4,
        pcp_rank=3,
        device=torch.device("cpu"),
        block_tables=_block_tables(),
    )
    batch_plan = manager._build_batch_plan(*_layout_inputs(4, is_prefilling=True))
    slab_slots = torch.arange(4, dtype=torch.int64).view(1, 4)

    with pytest.raises(NotImplementedError, match="PCP=2"):
        build_pcp_wavefront_plan(batch_plan, slab_slots)

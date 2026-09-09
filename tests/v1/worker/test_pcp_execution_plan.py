# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.v1.worker.gpu.pcp_execution as pcp_execution
from vllm.v1.worker.gpu.pcp_execution import PCPExecutionPlanner
from vllm.v1.worker.gpu.pcp_manager import RankSegment


class _Planner(PCPExecutionPlanner):
    def __init__(self, rank: int, segments_by_rank):
        self.pcp_world_size = len(segments_by_rank)
        self.pcp_rank = rank
        self.device = torch.device("cpu")
        self._segments = tuple(tuple(x) for x in segments_by_rank)
        self._batch_plan = None
        self._input_buffers = None
        self._layout_token_capacity = 0

    def _get_segments_by_rank(self, *args):
        del args
        return self._segments


def _segment(start: int, stop: int, local_start: int = 0) -> RankSegment:
    return RankSegment(
        global_batch_req_idx=0,
        global_batch_slice=slice(start, stop),
        rank_local_batch_slice=slice(local_start, local_start + stop - start),
    )


def _inputs(num_tokens: int):
    return (
        np.asarray([num_tokens], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([True], dtype=np.bool_),
        np.asarray([0, num_tokens], dtype=np.int32),
    )


def _copy_to_cpu(x, out=None, device=None):
    del device
    value = torch.as_tensor(x).clone()
    if out is None:
        return value
    out.copy_(value)
    return out


def test_plan_separates_owned_rows_from_rank_slab_width(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    planner = _Planner(1, ((_segment(0, 6),), (_segment(6, 9),)))
    plan = planner._build_batch_plan(*_inputs(9))

    assert plan.per_rank_num_tokens == (6, 3)
    assert plan.owned_num_tokens == 3
    assert plan.model_num_rows == 3
    assert plan.rank_slab_width == 6
    assert not plan.uses_dummy_execution_row
    assert plan.slab_global_idx.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0]
    assert plan.kv_write_mask.tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert plan.hidden_restore_idx.tolist() == list(range(9))


def test_empty_owner_gets_one_compatibility_row(monkeypatch) -> None:
    monkeypatch.setattr(pcp_execution, "async_copy_to_gpu", _copy_to_cpu)
    planner = _Planner(1, ((_segment(0, 1),), ()))
    plan = planner._build_batch_plan(*_inputs(1))

    assert plan.per_rank_num_tokens == (1, 0)
    assert plan.owned_num_tokens == 0
    assert plan.model_num_rows == 1
    assert plan.rank_slab_width == 1
    assert plan.uses_dummy_execution_row
    assert plan.kv_write_mask.tolist() == [True, False]

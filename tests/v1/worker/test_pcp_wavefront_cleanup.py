# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.pcp_execution import _model_num_rows
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pcp_weighted_partition import effective_partition_alignment


def test_model_num_rows_preserves_existing_dummy_row_rule() -> None:
    assert _model_num_rows(7, 11) == 7
    assert _model_num_rows(0, 11) == 1
    assert _model_num_rows(0, 0) == 0


def test_base_pcp_model_rows_preserve_padded_rank_width() -> None:
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    assert manager._get_model_num_rows([3, 5]) == 5
    assert manager._get_model_num_rows([0, 0]) == 0


def test_effective_partition_alignment_preserves_threshold_rule() -> None:
    assert effective_partition_alignment(127, 2, 64) == 1
    assert effective_partition_alignment(128, 2, 64) == 64
    assert effective_partition_alignment(256, 2, 64) == 64

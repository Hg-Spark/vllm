# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.pcp_execution import _model_num_rows, _segment_start_pos
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.pcp_weighted_partition import effective_partition_alignment


def test_model_num_rows_preserves_existing_dummy_row_rule() -> None:
    assert _model_num_rows(7, 11) == 7
    assert _model_num_rows(0, 11) == 1
    assert _model_num_rows(0, 0) == 0


def test_segment_start_pos_matches_existing_formula() -> None:
    segment = RankSegment(
        global_batch_req_idx=1,
        global_batch_slice=slice(8, 12),
        rank_local_batch_slice=slice(0, 4),
    )
    num_computed_tokens = np.asarray([3, 20], dtype=np.int32)
    query_start_loc = np.asarray([0, 5, 13], dtype=np.int32)

    expected = (
        num_computed_tokens[1]
        + segment.global_batch_slice.start
        - query_start_loc[1]
    )
    assert _segment_start_pos(segment, num_computed_tokens, query_start_loc) == expected


def test_effective_partition_alignment_preserves_threshold_rule() -> None:
    assert effective_partition_alignment(127, 2, 64) == 1
    assert effective_partition_alignment(128, 2, 64) == 64
    assert effective_partition_alignment(256, 2, 64) == 64

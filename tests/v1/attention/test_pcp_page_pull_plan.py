# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan


def test_page_plan_precompiles_repeated_owner_routes() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(1, 0, 1),
        blocks_by_segment=((0, 1), (2, 3), (4, 5)),
        block_size=16,
    )

    assert plan.required_source_ranks(0) == (1,)
    assert plan.required_source_ranks(1) == (0,)
    assert plan.consumer_ranks(0) == (1,)
    assert plan.consumer_ranks(1) == (0,)
    assert plan.requires_source(1, 0)
    assert not plan.requires_source(1, 1)
    assert plan.transfer_block_ids(1, 0) == ((2, 3), (2, 3))


def test_page_plan_reuses_precompiled_transfer_arrays() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(0, 1, 2, 3),
        blocks_by_segment=((0,), (1, 2), (3,), (4, 5)),
        block_size=16,
    )

    destination0, source0, max0 = plan.transfer_block_arrays(3, 0)
    destination1, source1, max1 = plan.transfer_block_arrays(3, 0)

    assert destination0 is destination1
    assert source0 is source1
    assert destination0 is source0
    assert destination0.dtype == np.int64
    assert destination0.tolist() == [0]
    assert max0 == max1 == 0


def test_page_plan_rejects_missing_physical_rank() -> None:
    with pytest.raises(ValueError, match="cover every PCP rank"):
        PCPPagePlan(
            segment_to_rank=(0, 2),
            blocks_by_segment=((0,), (1,)),
            block_size=16,
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan


def test_page_route_supports_distinct_source_and_destination_blocks() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(0, 1),
        blocks_by_segment=((7, 8), (20,)),
        destination_blocks_by_segment=((1, 2), (3,)),
        block_size=16,
    )

    route = plan.transfer_route(destination_rank=1, source_rank=0)

    assert route.source_block_ids == (7, 8)
    assert route.destination_block_ids == (1, 2)
    assert route.source_block_array.dtype == np.int64
    assert route.destination_block_array.dtype == np.int64
    assert route.source_max_block_id == 8
    assert route.destination_max_block_id == 2
    assert route.num_pages == 2
    assert plan.transfer_block_ids(1, 0) == ((1, 2), (7, 8))


def test_page_route_defaults_to_identical_block_ids() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(0, 1),
        blocks_by_segment=((4, 5), (6,)),
        block_size=16,
    )

    route = plan.transfer_route(destination_rank=1, source_rank=0)
    assert route.destination_block_ids == route.source_block_ids == (4, 5)

    destination, source, maximum = plan.transfer_block_arrays(1, 0)
    assert destination is source
    assert destination.tolist() == source.tolist() == [4, 5]
    assert maximum == 5


def test_page_plan_rejects_mismatched_destination_page_count() -> None:
    with pytest.raises(ValueError, match="page counts differ"):
        PCPPagePlan(
            segment_to_rank=(0, 1),
            blocks_by_segment=((7, 8), (20,)),
            destination_blocks_by_segment=((1,), (3,)),
            block_size=16,
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan


def test_page_plan_precompiles_repeated_owner_routes() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(1, 0, 1),
        blocks_by_segment=((0, 1), (2, 3), (4, 5)),
        block_size=16,
    )

    assert plan.current_source_ranks(0) == (1,)
    assert plan.current_source_ranks(1) == (0,)
    assert plan.consumer_ranks(0) == (1,)
    assert plan.consumer_ranks(1) == (0,)
    assert plan.requires_current_source(1, 0)
    assert not plan.requires_current_source(1, 1)

    route = plan.current_transfer_route(1, 0)
    assert route.destination_block_ids == route.source_block_ids == (2, 3)


def test_page_route_supports_distinct_source_and_destination_blocks() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(0, 1),
        blocks_by_segment=((7, 8), (20,)),
        destination_blocks_by_segment=((1, 2), (3,)),
        block_size=16,
    )

    route = plan.current_transfer_route(destination_rank=1, source_rank=0)

    assert route.source_block_ids == (7, 8)
    assert route.destination_block_ids == (1, 2)
    assert route.source_block_array.dtype == np.int64
    assert route.destination_block_array.dtype == np.int64
    assert route.source_max_block_id == 8
    assert route.destination_max_block_id == 2
    assert route.num_pages == 2


def test_page_route_reuses_precompiled_arrays() -> None:
    plan = PCPPagePlan(
        segment_to_rank=(0, 1, 2, 3),
        blocks_by_segment=((0,), (1, 2), (3,), (4, 5)),
        block_size=16,
    )

    first = plan.current_transfer_route(3, 0)
    second = plan.current_transfer_route(3, 0)

    assert first is second
    assert first.destination_block_array is first.source_block_array
    assert first.source_block_array.tolist() == [0]
    assert first.source_max_block_id == first.destination_max_block_id == 0


def test_page_plan_rejects_missing_physical_rank() -> None:
    with pytest.raises(ValueError, match="cover every PCP rank"):
        PCPPagePlan(
            segment_to_rank=(0, 2),
            blocks_by_segment=((0,), (1,)),
            block_size=16,
        )


def test_page_plan_rejects_mismatched_destination_page_count() -> None:
    with pytest.raises(ValueError, match="page counts differ"):
        PCPPagePlan(
            segment_to_rank=(0, 1),
            blocks_by_segment=((7, 8), (20,)),
            destination_blocks_by_segment=((1,), (3,)),
            block_size=16,
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan, PCPPagePullTransport
from vllm.v1.attention.ops.pcp_page_plan import PCPPageRoute


def _transport() -> PCPPagePullTransport:
    return PCPPagePullTransport(
        world_size=2,
        rank=1,
        device=torch.device("cpu"),
        max_inflight_reads=2,
    )


def test_ready_scheduler_prioritizes_demanded_layer() -> None:
    transport = _transport()
    transport._ready_waiting.extend(
        ((9, 0, "history"), (4, 0, "current"), (5, 0, "history"))
    )
    transport._demand_layer = 4

    assert transport._pop_ready_locked() == (4, 0, "current")
    assert tuple(transport._ready_waiting) == (
        (9, 0, "history"),
        (5, 0, "history"),
    )


def test_ready_scheduler_prefers_demanded_history_before_current() -> None:
    transport = _transport()
    transport._ready_waiting.extend(
        ((4, 0, "current"), (4, 0, "history"), (5, 0, "history"))
    )
    transport._demand_layer = 4

    assert transport._pop_ready_locked() == (4, 0, "history")


def test_ready_scheduler_keeps_fifo_without_demand() -> None:
    transport = _transport()
    transport._ready_waiting.extend(((9, 0, "history"), (4, 0, "current")))

    assert transport._pop_ready_locked() == (9, 0, "history")
    assert transport._pop_ready_locked() == (4, 0, "current")


def test_explicit_plan_only_sends_ready_for_current_routes() -> None:
    history = (
        (None, None),
        (
            PCPPageRoute(
                destination_rank=1,
                source_rank=0,
                destination_block_ids=(3,),
                source_block_ids=(3,),
            ),
            None,
        ),
    )
    current = (
        (None, None),
        (None, None),
    )
    plan = PCPPagePlan(
        segment_to_rank=(),
        blocks_by_segment=(),
        block_size=16,
        history_routes_by_rank=history,
        current_routes_by_rank=current,
        explicit_world_size=2,
    )

    assert plan.historical_source_ranks(1) == (0,)
    assert plan.current_source_ranks(1) == ()
    assert plan.consumer_ranks(0) == ()


def test_configure_step_pre_registers_bound_caches() -> None:
    transport = _transport()
    cache = torch.empty(8, 16, 4)
    plan = PCPPagePlan(
        segment_to_rank=(0, 1),
        blocks_by_segment=((0,), (1,)),
        block_size=16,
    )

    with (
        patch.object(
            transport,
            "_discover_bound_layer_caches",
            return_value={"model.layers.0.self_attn": cache},
        ),
        patch.object(transport, "register_layer_caches") as register,
    ):
        transport.configure_step(epoch=1, plan=plan)

    register.assert_called_once_with({"model.layers.0.self_attn": cache})


def test_static_cache_discovery_ignores_shared_layers() -> None:
    owned = torch.empty(8, 16, 4)
    shared = torch.empty(8, 16, 4)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "owned": SimpleNamespace(
                    kv_cache=owned,
                    kv_sharing_target_layer_name=None,
                ),
                "shared": SimpleNamespace(
                    kv_cache=shared,
                    kv_sharing_target_layer_name="owned",
                ),
            }
        )
    )

    with patch(
        "vllm.v1.attention.ops.pcp_page_pull_impl.get_current_vllm_config",
        return_value=config,
    ):
        discovered = PCPPagePullTransport._discover_bound_layer_caches()

    assert discovered == {"owned": owned}

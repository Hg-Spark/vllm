# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion
from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan, PCPPageRoute
from vllm.v1.attention.ops.pcp_page_push_impl import PCPPagePullTransport


def _empty_matrix(world_size: int):
    return tuple(tuple(None for _ in range(world_size)) for _ in range(world_size))


def _plan_with_routes(*, current=None, history=None, block_size: int = 4):
    world_size = 2
    current_matrix = [list(row) for row in _empty_matrix(world_size)]
    history_matrix = [list(row) for row in _empty_matrix(world_size)]
    if current is not None:
        current_matrix[current.destination_rank][current.source_rank] = current
    if history is not None:
        history_matrix[history.destination_rank][history.source_rank] = history
    return PCPPagePlan(
        segment_to_rank=(),
        blocks_by_segment=(),
        block_size=block_size,
        history_routes_by_rank=tuple(tuple(row) for row in history_matrix),
        current_routes_by_rank=tuple(tuple(row) for row in current_matrix),
        explicit_world_size=world_size,
    )


def test_replica_plan_fills_non_current_destination() -> None:
    # Rank 0's finalized page is already pushed to rank 1 by CURRENT. Rank 1's
    # finalized page is not needed by rank 0 in this chunk, so it must become a
    # background REPLICA route for the next chunk.
    current = PCPPageRoute(
        destination_rank=1,
        source_rank=0,
        destination_block_ids=(0,),
        source_block_ids=(0,),
    )
    plan = _plan_with_routes(current=current)
    transport = PCPPagePullTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._plan = plan
    transport._epoch = 1

    # block_size=4. Each rank writes one complete physical page.
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    transport.configure_slot_mapping(slots, (0, 4, 8))

    assert (1, 0) not in transport._replica_routes
    replica = transport._replica_routes[(0, 1)]
    assert replica.source_block_ids == (1,)
    assert replica.destination_block_ids == (1,)


def test_history_route_is_assertion_without_read_fallback() -> None:
    history = PCPPageRoute(
        destination_rank=0,
        source_rank=1,
        destination_block_ids=(7,),
        source_block_ids=(7,),
    )
    plan = _plan_with_routes(history=history)
    transport = PCPPagePullTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._plan = plan
    transport._epoch = 2
    transport._layer_names = ("model.layers.0",)
    # Only len() is used by historical validation; NIXL registration is not
    # required for this pure state-machine test.
    transport._layer_memory = [object()]  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        transport._validate_history_replicas_locked()

    transport._persistent_visible_blocks[(0, 1)] = {7}
    transport._validate_history_replicas_locked()


def test_current_visibility_requires_all_causal_sources() -> None:
    current = PCPPageRoute(
        destination_rank=1,
        source_rank=0,
        destination_block_ids=(3,),
        source_block_ids=(3,),
    )
    plan = _plan_with_routes(current=current)
    transport = PCPPagePullTransport(
        world_size=2,
        rank=1,
        device=torch.device("cpu"),
    )
    transport._plan = plan

    assert not transport._layer_visible_locked(0)
    transport._incoming_done.add((0, 0, "current"))
    assert transport._layer_visible_locked(0)


class _FakeNixlPeer:
    def register_tensor(self, tensor: torch.Tensor) -> NixlMemoryRegion:
        return NixlMemoryRegion(
            base_addr=tensor.data_ptr(),
            block_bytes=tensor.element_size(),
            num_blocks=tensor.numel(),
            device_id=0,
            local_xfer_handle=0,
        )

    def exchange_regions(self, _regions) -> None:
        pass


def test_layer_registration_preserves_execution_insertion_order() -> None:
    transport = PCPPagePullTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._peer = _FakeNixlPeer()  # type: ignore[assignment]
    caches = {
        "model.layers.0": torch.empty(1),
        "model.layers.1": torch.empty(1),
        "model.layers.10": torch.empty(1),
        "model.layers.2": torch.empty(1),
    }

    transport.register_layer_caches(caches)

    assert transport.registered_layer_names == tuple(caches)

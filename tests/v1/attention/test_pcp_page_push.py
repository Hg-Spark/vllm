# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import pytest
import torch

from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion
from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan, PCPPageRoute
from vllm.v1.attention.ops.pcp_page_push_impl import PCPPagePushTransport


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
    current = PCPPageRoute(
        destination_rank=1,
        source_rank=0,
        destination_block_ids=(0,),
        source_block_ids=(0,),
    )
    transport = PCPPagePushTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._plan = _plan_with_routes(current=current)
    transport._epoch = 1

    # block_size=4. Each rank finalizes one physical page.
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    transport.configure_slot_mapping(slots, (0, 4, 8))

    # Rank 0 -> rank 1 is already covered by CURRENT. Rank 1 -> rank 0 needs a
    # background REPLICA for the next chunk.
    assert (1, 0) not in transport._replica_routes
    replica = transport._replica_routes[(0, 1)]
    assert replica.source_block_ids == replica.destination_block_ids == (1,)


def test_history_route_is_strict_assertion_without_read_fallback() -> None:
    history = PCPPageRoute(
        destination_rank=0,
        source_rank=1,
        destination_block_ids=(7,),
        source_block_ids=(7,),
    )
    transport = PCPPagePushTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._plan = _plan_with_routes(history=history)
    transport._epoch = 2
    transport._layer_names = ("model.layers.0",)
    transport._layer_memory = [object()]  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="READ fallback is disabled"):
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
    transport = PCPPagePushTransport(
        world_size=2,
        rank=1,
        device=torch.device("cpu"),
    )
    transport._plan = _plan_with_routes(current=current)

    assert not transport._layer_visible_locked(0)
    transport._incoming_done.add((0, 0, "current"))
    assert transport._layer_visible_locked(0)


def test_ready_write_batch_groups_same_destination_without_waiting() -> None:
    transport = PCPPagePushTransport(
        world_size=3,
        rank=0,
        device=torch.device("cpu"),
        max_batch_layers=2,
    )
    queue = deque(
        (
            (0, 1, "current"),
            (0, 2, "current"),
            (1, 1, "current"),
            (2, 1, "current"),
        )
    )

    batch = transport._take_batch_locked(queue)  # type: ignore[arg-type]

    assert batch == ((0, 1, "current"), (1, 1, "current"))
    assert tuple(queue) == ((0, 2, "current"), (2, 1, "current"))


class _FakeNixlPeer:
    def register_tensor(self, tensor: torch.Tensor) -> NixlMemoryRegion:
        return NixlMemoryRegion(
            base_addr=tensor.data_ptr(),
            block_bytes=tensor.element_size(),
            num_blocks=tensor.numel(),
            device_id=0,
        )

    def exchange_regions(self, _regions) -> None:
        pass


def test_layer_registration_preserves_execution_insertion_order() -> None:
    transport = PCPPagePushTransport(
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

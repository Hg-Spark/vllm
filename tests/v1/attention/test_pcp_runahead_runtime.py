# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime


def _group(rank: int) -> MagicMock:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = rank
    group.ranks = [0, 1, 2, 3]
    group.device_group = MagicMock()
    return group


def test_exchange_prefix_uses_variable_rank_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))
    group = _group(2)
    recv_work = MagicMock()
    send_work = MagicMock()
    send_work.is_completed.return_value = False

    def p2p_side_effect(tensors, *, peer: int, recv: bool):
        if recv:
            assert peer == 1
            assert [tensor.shape[0] for tensor in tensors] == [7, 7]
            for tensor in tensors:
                tensor.zero_()
            return [recv_work]
        assert peer == 3
        assert [tensor.shape[0] for tensor in tensors] == [9, 9]
        return [send_work]

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch.object(runtime, "_p2p", side_effect=p2p_side_effect) as p2p,
    ):
        visible, visible_slots = runtime.exchange_prefix(
            (torch.empty(2, 3), torch.empty(2, 3)),
            torch.arange(10),
        )

    assert p2p.call_count == 2
    recv_work.wait.assert_called_once_with()
    assert [tensor.shape[0] for tensor in visible] == [9, 9]
    assert visible_slots.tolist() == list(range(9))


def test_step_level_paged_repair_runs_only_at_flush() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))
    group = _group(2)
    kv_cache = torch.zeros((8, 2, 4), dtype=torch.uint8)

    def broadcast_side_effect(payload: torch.Tensor, src: int) -> torch.Tensor:
        assert src == 3
        payload.fill_(7)
        return payload

    group.broadcast.side_effect = broadcast_side_effect

    with patch(
        "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
        return_value=group,
    ):
        runtime.set_repair_block_ids(torch.arange(5))
        runtime.register_kv_cache(kv_cache)

        assert runtime.num_cache_block_views == 1
        group.broadcast.assert_not_called()
        group.barrier.assert_not_called()

        runtime.flush()

    group.barrier.assert_called_once_with()
    group.broadcast.assert_called_once()
    assert group.broadcast.call_args.kwargs == {"src": 3}
    assert torch.all(kv_cache[:5] == 7)
    assert torch.all(kv_cache[5:] == 0)
    assert runtime.num_cache_block_views == 0


def test_packed_backing_storage_is_registered_once_per_step() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    runtime.begin_step((2, 2, 2, 2))
    backing = torch.zeros((8, 16), dtype=torch.uint8)
    layer0_view = backing.view(8, 2, 8)
    layer1_view = backing.view(8, 4, 4)

    runtime.register_kv_cache(layer0_view)
    runtime.register_kv_cache(layer1_view)

    assert runtime.num_cache_block_views == 1


def test_repair_union_ignores_ids_outside_a_storage() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=1,
        device=torch.device("cpu"),
    )
    runtime.begin_step((2, 2, 2, 2))
    group = _group(1)
    kv_cache = torch.zeros((4, 8), dtype=torch.uint8)

    def broadcast_side_effect(payload: torch.Tensor, src: int) -> torch.Tensor:
        assert src == 3
        payload.fill_(5)
        return payload

    group.broadcast.side_effect = broadcast_side_effect

    with patch(
        "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
        return_value=group,
    ):
        runtime.register_kv_cache(kv_cache)
        runtime.set_repair_block_ids(torch.tensor([1, 3, 9]))
        runtime.flush()

    assert torch.all(kv_cache[1] == 5)
    assert torch.all(kv_cache[3] == 5)
    assert torch.all(kv_cache[0] == 0)
    assert torch.all(kv_cache[2] == 0)

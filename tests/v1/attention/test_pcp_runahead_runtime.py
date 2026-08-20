# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime


def _group(rank: int) -> MagicMock:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = rank
    group.cpu_group = MagicMock()
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

    def recv_side_effect(
        _group: MagicMock,
        recv_tensors: tuple[torch.Tensor, ...],
        _src: int,
    ) -> list[MagicMock]:
        assert [tensor.shape[0] for tensor in recv_tensors] == [7, 7]
        return [recv_work]

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_runahead.batch_irecv_tensors",
            side_effect=recv_side_effect,
        ) as recv,
        patch(
            "vllm.v1.attention.ops.pcp_runahead.batch_isend_tensors",
            return_value=[send_work],
        ) as send,
    ):
        visible, visible_slots = runtime.exchange_prefix(
            (torch.empty(2, 3), torch.empty(2, 3)),
            torch.arange(10),
        )

    recv.assert_called_once()
    assert recv.call_args.args[2] == 1
    assert [tensor.shape[0] for tensor in visible] == [9, 9]
    assert visible_slots.tolist() == list(range(9))
    send.assert_called_once()
    assert send.call_args.args[2] == 3
    assert [tensor.shape[0] for tensor in send.call_args.args[1]] == [9, 9]


def test_paged_repair_is_deferred_until_flush() -> None:
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
        runtime.defer_paged_repair(
            kv_cache,
            torch.arange(10),
            cache_block_size=2,
        )

        assert runtime.num_deferred_repair_buffers == 1
        group.broadcast.assert_not_called()
        group.barrier.assert_not_called()

        runtime.flush()

    group.barrier.assert_called_once_with()
    group.broadcast.assert_called_once()
    assert group.broadcast.call_args.kwargs == {"src": 3}
    assert torch.all(kv_cache[:5] == 7)
    assert torch.all(kv_cache[5:] == 0)
    assert runtime.num_deferred_repair_buffers == 0


def test_layer_update_keeps_repair_off_critical_path() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    runtime.begin_step((2, 2, 2, 2))
    group = _group(0)
    send_work = MagicMock()
    send_work.is_completed.return_value = False
    apply = MagicMock()
    kv_cache = torch.zeros((8, 2, 4), dtype=torch.uint8)

    def broadcast_side_effect(payload: torch.Tensor, src: int) -> torch.Tensor:
        assert src == 3
        payload.fill_(9)
        return payload

    group.broadcast.side_effect = broadcast_side_effect

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_runahead.batch_isend_tensors",
            return_value=[send_work],
        ),
    ):
        runtime.update_visible_and_defer_repair(
            (torch.empty(2, 3),),
            torch.arange(8),
            apply,
            kv_cache,
            cache_block_size=2,
        )

        group.broadcast.assert_not_called()
        group.barrier.assert_not_called()
        apply.assert_called_once()
        visible_tensors, visible_slots = apply.call_args.args
        assert visible_tensors[0].shape == (2, 3)
        assert visible_slots.tolist() == [0, 1]
        assert runtime.num_deferred_repair_buffers == 1

        runtime.flush()

    send_work.wait.assert_called_once_with()
    group.barrier.assert_called_once_with()
    group.broadcast.assert_called_once()
    # Repair copies raw persistent pages. It never replays the layer cache writer
    # with retained activation tensors.
    assert apply.call_count == 1
    assert torch.all(kv_cache[:4] == 9)
    assert torch.all(kv_cache[4:] == 0)


def test_packed_backing_storage_is_registered_once() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    runtime.begin_step((2, 2, 2, 2))
    group = _group(0)
    backing = torch.zeros((8, 16), dtype=torch.uint8)
    layer0_view = backing.view(8, 2, 8)
    layer1_view = backing.view(8, 4, 4)

    with patch(
        "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
        return_value=group,
    ):
        runtime.defer_paged_repair(
            layer0_view,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            cache_block_size=2,
        )
        runtime.defer_paged_repair(
            layer1_view,
            torch.tensor([2, 3, 4, 5, 6, 7, 8, 9]),
            cache_block_size=2,
        )

    assert runtime.num_deferred_repair_buffers == 1

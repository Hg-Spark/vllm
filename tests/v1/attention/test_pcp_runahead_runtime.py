# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime


def _group(rank: int) -> MagicMock:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = rank
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


def test_replication_is_deferred_until_flush() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))
    group = _group(2)
    handles = [MagicMock(), MagicMock()]
    apply = MagicMock()

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_runahead.all_gather_variable_into_tensor_async",
            side_effect=handles,
        ) as gather,
    ):
        runtime.defer_replication(
            (torch.empty(2, 3), torch.empty(2, 5)),
            torch.arange(10),
            apply,
        )

        assert gather.call_count == 0
        assert runtime.num_deferred_replica_layers == 1

        runtime.flush()

        assert gather.call_count == 2
        assert gather.call_args_list[0].args[1].shape == (10, 3)
        assert gather.call_args_list[0].args[2].shape == (2, 3)
        assert gather.call_args_list[0].args[3] == (4, 3, 2, 1)
        assert gather.call_args_list[1].args[1].shape == (10, 5)

    apply.assert_called_once()
    gathered_tensors, gathered_slots = apply.call_args.args
    assert [tensor.shape[0] for tensor in gathered_tensors] == [10, 10]
    assert gathered_slots.tolist() == list(range(10))
    assert runtime.num_deferred_replica_layers == 0


def test_layer_update_does_not_launch_replica_allgather() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    runtime.begin_step((2, 2, 2, 2))
    group = _group(0)
    send_work = MagicMock()
    send_work.is_completed.return_value = False
    gather_work = MagicMock()
    apply = MagicMock()

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_runahead.batch_isend_tensors",
            return_value=[send_work],
        ),
        patch(
            "vllm.v1.attention.ops.pcp_runahead.all_gather_variable_into_tensor_async",
            return_value=gather_work,
        ) as gather,
    ):
        runtime.update_and_replicate(
            (torch.empty(2, 3),),
            torch.arange(8),
            apply,
        )

        gather.assert_not_called()
        apply.assert_called_once()
        visible_tensors, visible_slots = apply.call_args.args
        assert visible_tensors[0].shape == (2, 3)
        assert visible_slots.tolist() == [0, 1]
        assert runtime.num_deferred_replica_layers == 1

        runtime.flush()

    gather.assert_called_once()
    assert apply.call_count == 2

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

    local0 = torch.ones(2, 3)
    local1 = torch.ones(2, 3)
    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch.object(runtime, "_p2p", side_effect=p2p_side_effect) as p2p,
    ):
        visible, visible_slots = runtime.exchange_prefix(
            (local0, local1),
            torch.arange(10),
        )

    assert p2p.call_count == 2
    recv_work.wait.assert_called_once_with()
    assert [tensor.shape[0] for tensor in visible] == [9, 9]
    assert torch.all(visible[0][:7] == 0)
    assert torch.all(visible[0][7:] == 1)
    assert visible_slots.tolist() == list(range(9))


def test_flush_only_waits_outstanding_prefix_sends() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))
    group = _group(0)
    send_work = MagicMock()
    send_work.is_completed.return_value = False

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch.object(runtime, "_p2p", return_value=[send_work]),
    ):
        runtime.exchange_prefix((torch.ones(4, 2),), torch.arange(10))
        send_work.wait.assert_not_called()
        runtime.flush()

    send_work.wait.assert_called_once_with()
    group.broadcast.assert_not_called()
    group.barrier.assert_not_called()


def test_pending_prefix_sends_are_bounded() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_inflight_sends=1,
    )
    runtime.begin_step((4, 3, 2, 1))
    group = _group(0)
    first = MagicMock()
    second = MagicMock()
    first.is_completed.return_value = False
    second.is_completed.return_value = False

    with (
        patch(
            "vllm.v1.attention.ops.pcp_runahead.get_pcp_group",
            return_value=group,
        ),
        patch.object(runtime, "_p2p", side_effect=[[first], [second]]),
    ):
        runtime.exchange_prefix((torch.ones(4, 2),), torch.arange(10))
        first.wait.assert_not_called()
        runtime.exchange_prefix((torch.ones(4, 2),), torch.arange(10))
        first.wait.assert_called_once_with()
        second.wait.assert_not_called()
        runtime.flush()

    second.wait.assert_called_once_with()


def test_variable_width_runtime_builds_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))

    assert runtime.rows_per_rank == (4, 3, 2, 1)
    assert runtime.rank_offsets == (0, 4, 7, 9, 10)
    assert runtime.local_rows == 2
    assert runtime.prefix_rows == 7
    assert runtime.visible_rows == 9


def test_variable_width_runtime_rejects_empty_rank() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    try:
        runtime.begin_step((4, 3, 0, 1))
    except ValueError as exc:
        assert "positive rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")

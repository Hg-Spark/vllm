# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm.v1.attention.ops.pcp_transport import (
    all_gather_into_tensor_async,
    batch_irecv_tensors,
    batch_isend_tensors,
)


def _group() -> MagicMock:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = 1
    group.ranks = [8, 10, 12, 14]
    group.device_group = object()
    return group


def test_batch_isend_maps_peer_and_batches_all_tensors() -> None:
    group = _group()
    tensors = (torch.empty(2), torch.empty(3))
    handles = [MagicMock(), MagicMock()]

    with (
        patch(
            "vllm.v1.attention.ops.pcp_transport.dist.P2POp",
            side_effect=lambda *args, **kwargs: (args, kwargs),
        ) as p2p_op,
        patch(
            "vllm.v1.attention.ops.pcp_transport.dist.batch_isend_irecv",
            return_value=handles,
        ) as batch,
    ):
        assert batch_isend_tensors(group, tensors, 2) is handles

    assert p2p_op.call_args_list == [
        call(
            __import__("torch").distributed.isend,
            tensors[0],
            12,
            group=group.device_group,
        ),
        call(
            __import__("torch").distributed.isend,
            tensors[1],
            12,
            group=group.device_group,
        ),
    ]
    batch.assert_called_once()
    assert len(batch.call_args.args[0]) == 2


def test_batch_irecv_maps_peer_and_batches_all_tensors() -> None:
    group = _group()
    tensors = (torch.empty(2), torch.empty(3))
    handles = [MagicMock(), MagicMock()]

    with (
        patch(
            "vllm.v1.attention.ops.pcp_transport.dist.P2POp",
            side_effect=lambda *args, **kwargs: (args, kwargs),
        ) as p2p_op,
        patch(
            "vllm.v1.attention.ops.pcp_transport.dist.batch_isend_irecv",
            return_value=handles,
        ) as batch,
    ):
        assert batch_irecv_tensors(group, tensors, 0) is handles

    assert p2p_op.call_args_list == [
        call(
            __import__("torch").distributed.irecv,
            tensors[0],
            8,
            group=group.device_group,
        ),
        call(
            __import__("torch").distributed.irecv,
            tensors[1],
            8,
            group=group.device_group,
        ),
    ]
    batch.assert_called_once()
    assert len(batch.call_args.args[0]) == 2


@pytest.mark.parametrize("peer", [-1, 4])
def test_batch_p2p_rejects_invalid_group_rank(peer: int) -> None:
    group = _group()
    tensors = (torch.empty(2),)

    with pytest.raises(ValueError):
        batch_isend_tensors(group, tensors, peer)
    with pytest.raises(ValueError):
        batch_irecv_tensors(group, tensors, peer)


def test_batch_p2p_rejects_self_rank() -> None:
    group = _group()
    tensors = (torch.empty(2),)

    with pytest.raises(ValueError):
        batch_isend_tensors(group, tensors, group.rank_in_group)
    with pytest.raises(ValueError):
        batch_irecv_tensors(group, tensors, group.rank_in_group)


def test_all_gather_into_tensor_async_uses_device_group() -> None:
    group = _group()
    input_ = torch.empty(2)
    output = torch.empty(8)
    handle = MagicMock()

    with patch(
        "vllm.v1.attention.ops.pcp_transport.dist.all_gather_into_tensor",
        return_value=handle,
    ) as op:
        assert all_gather_into_tensor_async(group, output, input_) is handle

    op.assert_called_once_with(
        output,
        input_,
        group=group.device_group,
        async_op=True,
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.async_tensor_ops import (
    all_gather_into_tensor_async,
    irecv_tensor,
    isend_tensor,
)


def _group() -> MagicMock:
    group = MagicMock()
    group.world_size = 4
    group.rank_in_group = 1
    group.ranks = [8, 10, 12, 14]
    group.device_group = object()
    return group


def test_isend_tensor_maps_group_rank_to_global_rank() -> None:
    group = _group()
    tensor = torch.empty(2)
    handle = MagicMock()

    with patch("vllm.distributed.async_tensor_ops.dist.isend", return_value=handle) as op:
        assert isend_tensor(group, tensor, 2) is handle

    op.assert_called_once_with(tensor, dst=12, group=group.device_group)


def test_irecv_tensor_maps_group_rank_to_global_rank() -> None:
    group = _group()
    tensor = torch.empty(2)
    handle = MagicMock()

    with patch("vllm.distributed.async_tensor_ops.dist.irecv", return_value=handle) as op:
        assert irecv_tensor(group, tensor, 0) is handle

    op.assert_called_once_with(tensor, src=8, group=group.device_group)


@pytest.mark.parametrize("peer", [-1, 4])
def test_p2p_rejects_invalid_group_rank(peer: int) -> None:
    group = _group()
    tensor = torch.empty(2)

    with pytest.raises(ValueError):
        isend_tensor(group, tensor, peer)
    with pytest.raises(ValueError):
        irecv_tensor(group, tensor, peer)


def test_p2p_rejects_self_rank() -> None:
    group = _group()
    tensor = torch.empty(2)

    with pytest.raises(ValueError):
        isend_tensor(group, tensor, group.rank_in_group)
    with pytest.raises(ValueError):
        irecv_tensor(group, tensor, group.rank_in_group)


def test_all_gather_into_tensor_async_uses_device_group() -> None:
    group = _group()
    input_ = torch.empty(2)
    output = torch.empty(8)
    handle = MagicMock()

    with patch(
        "vllm.distributed.async_tensor_ops.dist.all_gather_into_tensor",
        return_value=handle,
    ) as op:
        assert all_gather_into_tensor_async(group, output, input_) is handle

    op.assert_called_once_with(
        output,
        input_,
        group=group.device_group,
        async_op=True,
    )

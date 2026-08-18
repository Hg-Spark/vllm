# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level asynchronous tensor communication helpers.

These helpers keep tensor-only communication inside the vLLM distributed
layer without paying the metadata exchange used by tensor-dict P2P APIs.
Peer ranks are expressed as ranks local to the supplied GroupCoordinator.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator, Handle


def isend_tensor(
    group: GroupCoordinator,
    tensor: torch.Tensor,
    dst: int,
) -> Handle:
    """Asynchronously send one tensor to a rank local to ``group``."""
    if not 0 <= dst < group.world_size:
        raise ValueError(f"Invalid destination rank ({dst})")
    if dst == group.rank_in_group:
        raise ValueError("Destination rank must differ from the current rank")

    handle = dist.isend(tensor, dst=group.ranks[dst], group=group.device_group)
    if tensor.is_cuda:
        tensor.record_stream(torch.cuda.current_stream(tensor.device))
    return handle


def irecv_tensor(
    group: GroupCoordinator,
    tensor: torch.Tensor,
    src: int,
) -> Handle:
    """Asynchronously receive one tensor from a rank local to ``group``."""
    if not 0 <= src < group.world_size:
        raise ValueError(f"Invalid source rank ({src})")
    if src == group.rank_in_group:
        raise ValueError("Source rank must differ from the current rank")

    return dist.irecv(tensor, src=group.ranks[src], group=group.device_group)


def all_gather_into_tensor_async(
    group: GroupCoordinator,
    output: torch.Tensor,
    input_: torch.Tensor,
) -> Handle:
    """Launch an asynchronous all-gather on ``group`` into ``output``."""
    return dist.all_gather_into_tensor(
        output,
        input_,
        group=group.device_group,
        async_op=True,
    )

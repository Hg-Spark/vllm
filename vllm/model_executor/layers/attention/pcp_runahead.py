# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group


# One outstanding layer is enough for the target wavefront:
# producer Layer L+1 || consumer Layer L. Waiting before posting the next layer
# also bounds source-tensor lifetime without a separate acknowledgement channel.
_MAX_OUTSTANDING_LAYERS = 1
_pending_sends: deque[tuple[list[Any], tuple[torch.Tensor, ...]]] = deque()


def _wait_oldest_send() -> None:
    works, tensors = _pending_sends.popleft()
    for work in works:
        work.wait()
    # Keep source storage alive through work.wait().
    del tensors


def post_layer_send(tensors: Sequence[torch.Tensor]) -> None:
    """Post one full-layer rank0->rank1 transfer and return immediately."""
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 0:
        raise RuntimeError("post_layer_send must run on PCP=2 rank0")

    while len(_pending_sends) >= _MAX_OUTSTANDING_LAYERS:
        _wait_oldest_send()

    retained = tuple(tensors)
    dst = pcp_group.ranks[1]
    works = [
        dist.isend(tensor, dst=dst, group=pcp_group.device_group)
        for tensor in retained
    ]
    _pending_sends.append((works, retained))


def recv_layer_like(templates: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Receive one full-layer rank0 payload into buffers shaped like templates."""
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 1:
        raise RuntimeError("recv_layer_like must run on PCP=2 rank1")

    recv_tensors = tuple(torch.empty_like(template) for template in templates)
    src = pcp_group.ranks[0]
    works = [
        dist.irecv(tensor, src=src, group=pcp_group.device_group)
        for tensor in recv_tensors
    ]
    for work in works:
        work.wait()
    return recv_tensors


def flush_pending_sends() -> None:
    """Complete retained producer sends, primarily for step/test cleanup."""
    while _pending_sends:
        _wait_oldest_send()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level tensor-only transport for PCP runahead.

The runahead critical path already knows tensor metadata, so this transport
avoids the CPU metadata exchange used by tensor-dict P2P APIs. Peer ranks are
expressed as ranks local to the PCP GroupCoordinator.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator, Handle
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range


def _validate_peer(group: GroupCoordinator, peer: int, kind: str) -> None:
    if not 0 <= peer < group.world_size:
        raise ValueError(f"Invalid {kind} rank ({peer})")
    if peer == group.rank_in_group:
        raise ValueError(f"{kind.capitalize()} rank must differ from the current rank")


def batch_irecv_tensors(
    group: GroupCoordinator,
    tensors: tuple[torch.Tensor, ...],
    src: int,
) -> list[Handle]:
    """Receive multiple tensors from one PCP-local peer in one P2P batch."""
    _validate_peer(group, src, "source")
    global_src = group.ranks[src]
    ops = [
        dist.P2POp(dist.irecv, tensor, global_src, group=group.device_group)
        for tensor in tensors
    ]
    with pcp_nvtx_range("pcp.transport.recv_enqueue"):
        return dist.batch_isend_irecv(ops)


def batch_isend_tensors(
    group: GroupCoordinator,
    tensors: tuple[torch.Tensor, ...],
    dst: int,
) -> list[Handle]:
    """Send multiple tensors to one PCP-local peer in one P2P batch."""
    _validate_peer(group, dst, "destination")
    global_dst = group.ranks[dst]
    ops = [
        dist.P2POp(dist.isend, tensor, global_dst, group=group.device_group)
        for tensor in tensors
    ]
    with pcp_nvtx_range("pcp.transport.send_enqueue"):
        works = dist.batch_isend_irecv(ops)
    for tensor in tensors:
        if tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
    return works


def all_gather_into_tensor_async(
    group: GroupCoordinator,
    output: torch.Tensor,
    input_: torch.Tensor,
) -> Handle:
    """Launch an asynchronous tensor all-gather on the PCP device group."""
    with pcp_nvtx_range("pcp.transport.allgather_enqueue"):
        return dist.all_gather_into_tensor(
            output,
            input_,
            group=group.device_group,
            async_op=True,
        )

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
    """Launch an asynchronous equal-width tensor all-gather."""
    with pcp_nvtx_range("pcp.transport.allgather_enqueue"):
        return dist.all_gather_into_tensor(
            output,
            input_,
            group=group.device_group,
            async_op=True,
        )


def all_gather_variable_into_tensor_async(
    group: GroupCoordinator,
    output: torch.Tensor,
    input_: torch.Tensor,
    rows_per_rank: tuple[int, ...],
) -> Handle:
    """Gather uneven dim-0 slabs into one compact rank-major output tensor."""
    if len(rows_per_rank) != group.world_size:
        raise ValueError(
            "PCP variable all-gather rows must match the group size: "
            f"rows={rows_per_rank}, world_size={group.world_size}"
        )
    if any(rows < 0 for rows in rows_per_rank):
        raise ValueError(
            f"PCP variable all-gather rows must be non-negative: {rows_per_rank}"
        )
    local_rows = rows_per_rank[group.rank_in_group]
    if input_.shape[0] != local_rows:
        raise ValueError(
            "PCP variable all-gather local input has the wrong row count: "
            f"rank={group.rank_in_group}, expected={local_rows}, got={input_.shape[0]}"
        )
    total_rows = sum(rows_per_rank)
    if output.shape[0] != total_rows:
        raise ValueError(
            "PCP variable all-gather output has the wrong row count: "
            f"expected={total_rows}, got={output.shape[0]}"
        )

    output_views: list[torch.Tensor] = []
    offset = 0
    for rows in rows_per_rank:
        output_views.append(output.narrow(0, offset, rows))
        offset += rows

    with pcp_nvtx_range("pcp.transport.variable_allgather_enqueue"):
        return dist.all_gather(
            output_views,
            input_,
            group=group.device_group,
            async_op=True,
        )

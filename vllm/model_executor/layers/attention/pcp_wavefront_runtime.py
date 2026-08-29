# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group


# One outstanding layer is enough for the target wavefront:
# producer Layer L+1 || consumer Layer L. Waiting before posting the next layer
# also bounds source-tensor lifetime without a separate acknowledgement channel.
_MAX_OUTSTANDING_LAYERS = 1
_pending_sends: deque[tuple[int, list[Any], tuple[torch.Tensor, ...]]] = deque()
_send_layer_seq = 0
_recv_layer_seq = 0
_NVTX_ENABLED = os.getenv("VLLM_PCP_WAVEFRONT_NVTX", "0") == "1"


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Mark wavefront runtime activity without changing synchronization."""
    enabled = _NVTX_ENABLED and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _nvtx_mark(name: str) -> None:
    """Emit a point marker for aligning producer and consumer layer progress."""
    if _NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.mark(name)


def _wait_oldest_send() -> None:
    _layer_seq, send_works, retained_tensors = _pending_sends.popleft()
    for work in send_works:
        work.wait()
    # Keep source storage alive through work.wait().
    del retained_tensors


def post_layer_transfer(payload: Sequence[torch.Tensor]) -> None:
    """Post one full-layer rank0->rank1 transfer and return immediately."""
    global _send_layer_seq

    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 0:
        raise RuntimeError("post_layer_transfer must run on PCP=2 rank0")

    while len(_pending_sends) >= _MAX_OUTSTANDING_LAYERS:
        pending_layer_seq = _pending_sends[0][0]
        with _nvtx_range(
            f"pcp_wavefront.send_credit_wait.layer_seq_{pending_layer_seq}"
        ):
            _wait_oldest_send()

    layer_seq = _send_layer_seq
    _send_layer_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank0.layer_seq_{layer_seq}.handoff")

    retained_tensors = tuple(payload)
    dst = pcp_group.ranks[1]
    with _nvtx_range(f"pcp_wavefront.send_post.layer_seq_{layer_seq}"):
        send_ops = [
            dist.P2POp(
                dist.isend,
                tensor,
                dst,
                group=pcp_group.device_group,
            )
            for tensor in retained_tensors
        ]
        send_works = dist.batch_isend_irecv(send_ops)
    _pending_sends.append((layer_seq, send_works, retained_tensors))


def recv_layer_payload(
    recv_templates: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Receive one full-layer rank0 payload into buffers shaped like templates."""
    global _recv_layer_seq

    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 1:
        raise RuntimeError("recv_layer_payload must run on PCP=2 rank1")

    layer_seq = _recv_layer_seq
    _recv_layer_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank1.layer_seq_{layer_seq}.recv_begin")

    recv_tensors = tuple(torch.empty_like(template) for template in recv_templates)
    src = pcp_group.ranks[0]
    with _nvtx_range(f"pcp_wavefront.recv_post.layer_seq_{layer_seq}"):
        recv_ops = [
            dist.P2POp(
                dist.irecv,
                tensor,
                src,
                group=pcp_group.device_group,
            )
            for tensor in recv_tensors
        ]
        recv_works = dist.batch_isend_irecv(recv_ops)
    with _nvtx_range(f"pcp_wavefront.recv_wait.layer_seq_{layer_seq}"):
        for work in recv_works:
            work.wait()
    _nvtx_mark(f"pcp_wavefront.rank1.layer_seq_{layer_seq}.recv_ready")
    return recv_tensors


def flush_pending_sends() -> None:
    """Complete retained producer sends, primarily for step/test cleanup."""
    with _nvtx_range("pcp_wavefront.final_flush"):
        while _pending_sends:
            layer_seq = _pending_sends[0][0]
            with _nvtx_range(f"pcp_wavefront.final_flush.layer_seq_{layer_seq}"):
                _wait_oldest_send()

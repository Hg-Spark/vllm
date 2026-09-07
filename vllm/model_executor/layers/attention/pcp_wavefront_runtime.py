# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production full-layer PCP Wavefront transport runtime."""

import os
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group

_MAX_OUTSTANDING_LAYERS_ENV = "VLLM_PCP_WAVEFRONT_MAX_OUTSTANDING_LAYERS"


def _read_max_outstanding_layers() -> int:
    raw = os.getenv(_MAX_OUTSTANDING_LAYERS_ENV, "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{_MAX_OUTSTANDING_LAYERS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise RuntimeError(
            f"{_MAX_OUTSTANDING_LAYERS_ENV} must be >= 1, got {value}"
        )
    return value


@dataclass
class PendingLayerReceive:
    layer_seq: int
    recv_tensors: tuple[torch.Tensor, ...]
    recv_works: list[Any]


_MAX_OUTSTANDING_LAYERS = _read_max_outstanding_layers()
_pending_sends: deque[tuple[int, list[Any], tuple[torch.Tensor, ...]]] = deque()
_send_layer_seq = 0
_recv_layer_seq = 0
_NVTX_ENABLED = os.getenv("VLLM_PCP_WAVEFRONT_NVTX", "0") == "1"


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    enabled = _NVTX_ENABLED and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _nvtx_mark(name: str) -> None:
    if _NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.mark(name)


def _wait_oldest_send() -> None:
    _seq, send_works, retained_tensors = _pending_sends.popleft()
    for work in send_works:
        work.wait()
    del retained_tensors


def post_layer_transfer(payload: Sequence[torch.Tensor]) -> None:
    global _send_layer_seq
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 0:
        raise RuntimeError("PCP layer transfer must run on PCP=2 rank0")

    while len(_pending_sends) >= _MAX_OUTSTANDING_LAYERS:
        pending_seq = _pending_sends[0][0]
        with _nvtx_range(f"pcp_wavefront.layer.send_credit_wait.seq_{pending_seq}"):
            _wait_oldest_send()

    layer_seq = _send_layer_seq
    _send_layer_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank0.layer_seq_{layer_seq}.handoff")

    retained_tensors = tuple(payload)
    dst = pcp_group.ranks[1]
    with _nvtx_range(f"pcp_wavefront.layer.send_post.seq_{layer_seq}"):
        send_ops = [
            dist.P2POp(dist.isend, tensor, dst, group=pcp_group.device_group)
            for tensor in retained_tensors
        ]
        send_works = dist.batch_isend_irecv(send_ops)
    _pending_sends.append((layer_seq, send_works, retained_tensors))


def post_layer_receive_into(
    recv_buffers: Sequence[torch.Tensor],
) -> PendingLayerReceive:
    """Post one full-layer receive into caller-owned buffers without waiting."""
    global _recv_layer_seq
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 1:
        raise RuntimeError("PCP layer receive must run on PCP=2 rank1")

    layer_seq = _recv_layer_seq
    _recv_layer_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank1.layer_seq_{layer_seq}.recv_begin")

    recv_tensors = tuple(recv_buffers)
    src = pcp_group.ranks[0]
    with _nvtx_range(f"pcp_wavefront.layer.recv_post.seq_{layer_seq}"):
        recv_ops = [
            dist.P2POp(dist.irecv, tensor, src, group=pcp_group.device_group)
            for tensor in recv_tensors
        ]
        recv_works = dist.batch_isend_irecv(recv_ops)
    return PendingLayerReceive(layer_seq, recv_tensors, recv_works)


def wait_layer_receive(
    pending: PendingLayerReceive,
) -> tuple[torch.Tensor, ...]:
    """Wait for a previously posted full-layer receive to become usable."""
    with _nvtx_range(f"pcp_wavefront.layer.recv_wait.seq_{pending.layer_seq}"):
        for work in pending.recv_works:
            work.wait()

    _nvtx_mark(f"pcp_wavefront.rank1.layer_seq_{pending.layer_seq}.recv_ready")
    return pending.recv_tensors


def recv_layer_payload_into(
    recv_buffers: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Receive one full-layer payload directly into caller-owned buffers."""
    return wait_layer_receive(post_layer_receive_into(recv_buffers))


def recv_layer_payload(
    recv_templates: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    recv_tensors = tuple(torch.empty_like(template) for template in recv_templates)
    return recv_layer_payload_into(recv_tensors)


def flush_pending_sends() -> None:
    with _nvtx_range("pcp_wavefront.final_flush"):
        while _pending_sends:
            layer_seq = _pending_sends[0][0]
            with _nvtx_range(f"pcp_wavefront.final_flush.layer_seq_{layer_seq}"):
                _wait_oldest_send()

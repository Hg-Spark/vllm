# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental fine-grained PCP tile transport.

This module is intentionally separate from the production full-layer
Wavefront runtime. It is imported only by the optional PCP tiled execution
path so tile credits and pending buffers do not participate in the default PCP
lifecycle.
"""

import os
from collections import deque
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group

_MAX_OUTSTANDING_TILES_ENV = "VLLM_PCP_WAVEFRONT_MAX_OUTSTANDING_TILES"


def _read_max_outstanding_tiles() -> int:
    raw = os.getenv(_MAX_OUTSTANDING_TILES_ENV, "2")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{_MAX_OUTSTANDING_TILES_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise RuntimeError(f"{_MAX_OUTSTANDING_TILES_ENV} must be >= 1, got {value}")
    return value


_MAX_OUTSTANDING_TILES = _read_max_outstanding_tiles()
_pending_tile_sends: deque[
    tuple[list[Any], tuple[torch.Tensor, ...]]
] = deque()


def _wait_oldest_send() -> None:
    send_works, retained_tensors = _pending_tile_sends.popleft()
    for work in send_works:
        work.wait()
    del retained_tensors


def post_tile_transfer(payload: Sequence[torch.Tensor]) -> None:
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 0:
        raise RuntimeError("PCP tile transfer must run on PCP=2 rank0")

    while len(_pending_tile_sends) >= _MAX_OUTSTANDING_TILES:
        _wait_oldest_send()

    retained_tensors = tuple(payload)
    dst = pcp_group.ranks[1]
    send_ops = [
        dist.P2POp(dist.isend, tensor, dst, group=pcp_group.device_group)
        for tensor in retained_tensors
    ]
    send_works = dist.batch_isend_irecv(send_ops)
    _pending_tile_sends.append((send_works, retained_tensors))


def recv_tile_payload_into(
    recv_buffers: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 1:
        raise RuntimeError("PCP tile receive must run on PCP=2 rank1")

    recv_tensors = tuple(recv_buffers)
    src = pcp_group.ranks[0]
    recv_ops = [
        dist.P2POp(dist.irecv, tensor, src, group=pcp_group.device_group)
        for tensor in recv_tensors
    ]
    recv_works = dist.batch_isend_irecv(recv_ops)
    for work in recv_works:
        work.wait()
    return recv_tensors


def flush_pending_tile_sends() -> None:
    while _pending_tile_sends:
        _wait_oldest_send()

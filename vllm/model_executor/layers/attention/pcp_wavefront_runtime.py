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


_MAX_OUTSTANDING_TILES_ENV = "VLLM_PCP_WAVEFRONT_MAX_OUTSTANDING_TILES"


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")
    return value


_MAX_OUTSTANDING_TILES = _read_positive_int_env(_MAX_OUTSTANDING_TILES_ENV, 2)

_pending_tile_sends: deque[
    tuple[int, list[Any], tuple[torch.Tensor, ...]]
] = deque()
_send_tile_seq = 0
_recv_tile_seq = 0
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
    _seq, send_works, retained_tensors = _pending_tile_sends.popleft()
    for work in send_works:
        work.wait()
    del retained_tensors


def post_tile_transfer(payload: Sequence[torch.Tensor]) -> None:
    """Post one Wavefront transport tile from PCP rank0 to rank1.

    A short prefill or a non-streaming fallback is represented as a single
    tile, so tile credits are the only transport backpressure mechanism.
    """
    global _send_tile_seq
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 0:
        raise RuntimeError("PCP tile transfer must run on PCP=2 rank0")

    while len(_pending_tile_sends) >= _MAX_OUTSTANDING_TILES:
        pending_seq = _pending_tile_sends[0][0]
        with _nvtx_range(
            f"pcp_wavefront.tile.send_credit_wait.seq_{pending_seq}"
        ):
            _wait_oldest_send()

    tile_seq = _send_tile_seq
    _send_tile_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank0.tile_seq_{tile_seq}.handoff")

    retained_tensors = tuple(payload)
    dst = pcp_group.ranks[1]
    with _nvtx_range(f"pcp_wavefront.tile.send_post.seq_{tile_seq}"):
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
    _pending_tile_sends.append((tile_seq, send_works, retained_tensors))


def recv_tile_payload_into(
    recv_buffers: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Receive one Wavefront transport tile into caller-owned buffers."""
    global _recv_tile_seq
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2 or pcp_group.rank_in_group != 1:
        raise RuntimeError("PCP tile receive must run on PCP=2 rank1")

    tile_seq = _recv_tile_seq
    _recv_tile_seq += 1
    _nvtx_mark(f"pcp_wavefront.rank1.tile_seq_{tile_seq}.recv_begin")

    recv_tensors = tuple(recv_buffers)
    src = pcp_group.ranks[0]
    with _nvtx_range(f"pcp_wavefront.tile.recv_post.seq_{tile_seq}"):
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
    with _nvtx_range(f"pcp_wavefront.tile.recv_wait.seq_{tile_seq}"):
        for work in recv_works:
            work.wait()

    _nvtx_mark(f"pcp_wavefront.rank1.tile_seq_{tile_seq}.recv_ready")
    return recv_tensors


def flush_pending_sends() -> None:
    with _nvtx_range("pcp_wavefront.final_flush"):
        while _pending_tile_sends:
            tile_seq = _pending_tile_sends[0][0]
            with _nvtx_range(f"pcp_wavefront.final_flush.tile_seq_{tile_seq}"):
                _wait_oldest_send()

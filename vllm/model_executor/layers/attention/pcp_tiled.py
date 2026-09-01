# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental tiled MLA cache handoff for PCP.

Production PCP uses full-layer handoff from ``attention.pcp``. This module is
loaded only by the optional tiled execution path.
"""

from collections.abc import Iterator

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.worker.gpu.pcp_tile_transport import (
    flush_pending_tile_sends,
    post_tile_transfer,
    recv_tile_payload_into,
)


def _slice_or_pad_transport_tile(
    tensor: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    width = stop - start
    if width <= 0:
        raise ValueError(f"Invalid PCP transport tile [{start}, {stop})")
    num_rows = tensor.shape[0]
    if start >= num_rows:
        return tensor.new_zeros((width, *tensor.shape[1:]))
    real_stop = min(stop, num_rows)
    tile = tensor[start:real_stop]
    if tile.shape[0] == width:
        return tile.contiguous()
    padded = tensor.new_zeros((width, *tensor.shape[1:]))
    padded[: tile.shape[0]].copy_(tile)
    return padded


def iter_tiled_mla_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
    tile_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    num_rows = kv_c_normed.shape[0]
    if k_pe.shape[0] != num_rows:
        raise RuntimeError("MLA cache inputs disagree on row count")

    if not use_pcp or num_decode_tokens is None:
        for start in range(0, num_rows, tile_size):
            stop = min(start + tile_size, num_rows)
            slots = None if slot_mapping is None else slot_mapping[start:stop]
            yield kv_c_normed[start:stop], k_pe[start:stop], slots
        return

    assert slot_mapping is not None
    pcp_group = get_pcp_group()
    if pcp_group.world_size != 2:
        raise NotImplementedError("Tiled Wavefront MLA transfer requires PCP=2.")
    if slot_mapping.numel() % 2 != 0:
        raise RuntimeError(
            "PCP slot mapping does not contain two equal rank slabs: "
            f"numel={slot_mapping.numel()}"
        )

    rank_slab_width = slot_mapping.numel() // 2
    if num_rows > rank_slab_width:
        raise RuntimeError(
            "PCP local model rows exceed rank slab: "
            f"{num_rows} > {rank_slab_width}"
        )
    rank_slots = slot_mapping.view(2, rank_slab_width)
    rank = pcp_group.rank_in_group
    k_pe_flat = k_pe.flatten(1)

    if rank == 0:
        try:
            for start in range(0, rank_slab_width, tile_size):
                stop = min(start + tile_size, rank_slab_width)
                send_kv = _slice_or_pad_transport_tile(kv_c_normed, start, stop)
                send_kpe = _slice_or_pad_transport_tile(k_pe_flat, start, stop)
                post_tile_transfer((send_kv, send_kpe))
                if start < num_rows:
                    local_stop = min(stop, num_rows)
                    yield (
                        kv_c_normed[start:local_stop],
                        k_pe[start:local_stop],
                        rank_slots[0, start:local_stop],
                    )
        finally:
            # Keep experimental tile-buffer lifetimes out of the production
            # step-finalization path. This intentionally favors isolation over
            # cross-layer tile pipelining.
            flush_pending_tile_sends()
        return

    if rank != 1:
        raise RuntimeError(f"Unexpected PCP rank for PCP=2 wavefront: {rank}")

    for start in range(0, rank_slab_width, tile_size):
        stop = min(start + tile_size, rank_slab_width)
        width = stop - start
        recv_kv = kv_c_normed.new_empty((width, *kv_c_normed.shape[1:]))
        recv_kpe_flat = k_pe_flat.new_empty((width, k_pe_flat.shape[1]))
        recv_tile_payload_into((recv_kv, recv_kpe_flat))
        recv_kpe = recv_kpe_flat.view(width, *k_pe.shape[1:])
        yield recv_kv, recv_kpe, rank_slots[0, start:stop]

        if start < num_rows:
            local_stop = min(stop, num_rows)
            yield (
                kv_c_normed[start:local_stop],
                k_pe[start:local_stop],
                rank_slots[1, start:local_stop],
            )

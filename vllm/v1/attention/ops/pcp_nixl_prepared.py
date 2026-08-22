# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared NIXL fast-path helpers for PCP page-pull.

The stable memory/peer lifecycle remains in ``pcp_nixl``. This adapter only
adds allocation-free notification iteration and READ submission from
precompiled NumPy descriptor-index arrays.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion, PCPNixlPeerTransport


class PCPNixlPreparedTransport(PCPNixlPeerTransport):
    """PCP NIXL transport with precompiled descriptor-index submission."""

    def iter_notifications(self) -> Iterator[bytes]:
        self._ensure_wrapper()
        assert self._wrapper is not None
        for notifications in self._wrapper.get_new_notifs().values():
            yield from notifications

    def submit_prepared_read(
        self,
        *,
        local_region: NixlMemoryRegion,
        local_block_ids: np.ndarray,
        local_max_block_id: int,
        source_rank: int,
        remote_region_id: int,
        remote_block_ids: np.ndarray,
        remote_max_block_id: int,
    ) -> int:
        if local_block_ids.dtype != np.int64 or remote_block_ids.dtype != np.int64:
            raise TypeError("PCP prepared READ block IDs must be int64 NumPy arrays")
        if local_block_ids.ndim != 1 or remote_block_ids.ndim != 1:
            raise ValueError("PCP prepared READ block IDs must be one-dimensional")
        if local_block_ids.size != remote_block_ids.size:
            raise ValueError("PCP NIXL READ source/destination page counts differ")
        if local_block_ids.size == 0:
            raise ValueError("PCP NIXL READ requires at least one page")
        if local_max_block_id >= local_region.num_blocks:
            raise RuntimeError(
                "PCP page-pull destination block id exceeds local cache: "
                f"max={local_max_block_id}, num_blocks={local_region.num_blocks}"
            )
        remote_num_blocks = self._remote_num_blocks[(source_rank, remote_region_id)]
        if remote_max_block_id >= remote_num_blocks:
            raise RuntimeError(
                "PCP page-pull source block id exceeds remote cache: "
                f"max={remote_max_block_id}, num_blocks={remote_num_blocks}"
            )
        assert self._wrapper is not None
        handle = self._wrapper.make_prepped_xfer(
            "READ",
            local_region.local_xfer_handle,
            local_block_ids,
            self._remote_region_handles[(source_rank, remote_region_id)],
            remote_block_ids,
        )
        self._wrapper.transfer(handle)
        return handle


__all__ = ["PCPNixlPreparedTransport"]

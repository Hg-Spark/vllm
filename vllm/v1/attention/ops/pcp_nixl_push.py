# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-push extensions for the PCP-local NIXL transport."""

from __future__ import annotations

import numpy as np

from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion, PCPNixlPeerTransport


class PCPNixlPushTransport(PCPNixlPeerTransport):
    """Add one-sided WRITE submission to the stable PCP NIXL transport.

    The base transport owns memory registration, peer metadata exchange and
    prepared descriptor lists. Producer-push PCP uses the same prepared lists
    in the opposite direction, so no additional registration lifecycle is
    required here.
    """

    def submit_prepared_write(
        self,
        *,
        local_region: NixlMemoryRegion,
        local_block_ids: np.ndarray,
        local_max_block_id: int,
        destination_rank: int,
        remote_region_id: int,
        remote_block_ids: np.ndarray,
        remote_max_block_id: int,
    ) -> int:
        if local_block_ids.dtype != np.int64 or remote_block_ids.dtype != np.int64:
            raise TypeError("PCP prepared WRITE block IDs must be int64 NumPy arrays")
        if local_block_ids.ndim != 1 or remote_block_ids.ndim != 1:
            raise ValueError("PCP prepared WRITE block IDs must be one-dimensional")
        if local_block_ids.size != remote_block_ids.size:
            raise ValueError("PCP NIXL WRITE source/destination page counts differ")
        if local_block_ids.size == 0:
            raise ValueError("PCP NIXL WRITE requires at least one page")
        if local_max_block_id >= local_region.num_blocks:
            raise RuntimeError(
                "PCP page-push source block id exceeds local cache: "
                f"max={local_max_block_id}, num_blocks={local_region.num_blocks}"
            )
        remote_num_blocks = self._remote_num_blocks[(destination_rank, remote_region_id)]
        if remote_max_block_id >= remote_num_blocks:
            raise RuntimeError(
                "PCP page-push destination block id exceeds remote cache: "
                f"max={remote_max_block_id}, num_blocks={remote_num_blocks}"
            )
        assert self._wrapper is not None
        handle = self._wrapper.make_prepped_xfer(
            "WRITE",
            local_region.local_xfer_handle,
            local_block_ids,
            self._remote_region_handles[(destination_rank, remote_region_id)],
            remote_block_ids,
        )
        self._wrapper.transfer(handle)
        return handle

    def check_write(self, handle: int) -> str:
        # READ and WRITE share the same NIXL transfer-state lifecycle.
        return self.check_read(handle)


__all__ = ["PCPNixlPushTransport"]

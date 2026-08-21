# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-local NIXL peer transport primitives.

This module deliberately stays below PCP scheduling semantics. It owns NIXL
agent setup, stable memory registration, peer-region metadata, notifications,
and asynchronous READ submission while callers own causal dependencies and
step/layer scheduling.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pcp_group
from vllm.platforms import current_platform


@dataclass(frozen=True)
class NixlMemoryRegion:
    """Stable block-major memory region registered with NIXL."""

    base_addr: int
    block_bytes: int
    num_blocks: int
    device_id: int
    local_xfer_handle: int


class PCPNixlPeerTransport:
    """Minimal same-engine peer-memory transport for PCP.

    The abstraction intentionally does not depend on KVConnector request,
    scheduler, lease, or producer/consumer state. PCP page-pull only needs
    stable peer regions, notifications, and one-sided READs between ranks in
    the existing PCP process group.
    """

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
        nixl_backends: tuple[str, ...] = ("UCX",),
        pcp_group: Any | None = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.nixl_backends = nixl_backends
        self.group = pcp_group

        self._wrapper: Any | None = None
        self._memory_type: str | None = None
        self._remote_agents: dict[int, str] = {}
        self._registered_descs: list[Any] = []
        self._memory_by_ptr: dict[int, NixlMemoryRegion] = {}
        self._remote_region_handles: dict[tuple[int, int], int] = {}
        self._remote_num_blocks: dict[tuple[int, int], int] = {}
        self._registered_region_signature: tuple[tuple[int, int, int, int], ...] = ()
        self._metadata_exchanged = False

    def _group(self):
        return self.group if self.group is not None else get_pcp_group()

    def _ensure_wrapper(self) -> None:
        if self._wrapper is not None:
            return
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise RuntimeError(
                "PCP transport=page_pull requires NIXL, but NIXL is unavailable"
            )
        non_ucx = [backend for backend in self.nixl_backends if backend != "UCX"]
        if nixl_agent_config is None:
            config = None
        elif non_ucx:
            config = nixl_agent_config(
                backends=list(self.nixl_backends), capture_telemetry=True
            )
        else:
            config = nixl_agent_config(num_threads=4, capture_telemetry=True)
        self._wrapper = NixlWrapper(
            f"pcp-page-pull-r{self.rank}-{uuid.uuid4()}", config
        )
        memory_type = current_platform.get_nixl_memory_type()
        if memory_type is None:
            memory_type = "VRAM" if self.device.type in ("cuda", "xpu") else "DRAM"
        self._memory_type = memory_type

    def ensure_peers(self) -> None:
        if self._remote_agents or self.world_size <= 1:
            return
        self._ensure_wrapper()
        assert self._wrapper is not None
        group = self._group()
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            self._wrapper.get_agent_metadata(),
            group=group.cpu_group,
        )
        for source_rank, metadata in enumerate(gathered):
            if source_rank != self.rank:
                self._remote_agents[source_rank] = self._wrapper.add_remote_agent(metadata)

    @staticmethod
    def physical_page_geometry(kv_cache: torch.Tensor) -> tuple[int, int]:
        if kv_cache.ndim < 2:
            raise RuntimeError(
                f"PCP page-pull expects block-major KV cache, got {kv_cache.shape}"
            )
        num_blocks = int(kv_cache.shape[0])
        page_elements = math.prod(kv_cache.shape[1:])
        # NHD and HND are different logical views of the same dense physical
        # page. Inner logical contiguity is irrelevant for whole-page NIXL I/O.
        if kv_cache.stride(0) != page_elements:
            raise NotImplementedError(
                "PCP page_pull requires dense block-major KV pages; "
                f"shape={tuple(kv_cache.shape)}, stride={tuple(kv_cache.stride())}"
            )
        return num_blocks, int(page_elements * kv_cache.element_size())

    @staticmethod
    def _block_descriptors(
        *, base_addr: int, block_bytes: int, num_blocks: int, device_id: int
    ) -> np.ndarray:
        blocks = np.arange(num_blocks, dtype=np.uint64)
        result = np.empty((num_blocks, 3), dtype=np.uint64)
        result[:, 0] = np.uint64(base_addr) + blocks * np.uint64(block_bytes)
        result[:, 1] = np.uint64(block_bytes)
        result[:, 2] = np.uint64(device_id)
        return result

    def register_tensor(self, kv_cache: torch.Tensor) -> NixlMemoryRegion:
        self._ensure_wrapper()
        assert self._wrapper is not None and self._memory_type is not None
        ptr = kv_cache.data_ptr()
        existing = self._memory_by_ptr.get(ptr)
        if existing is not None:
            return existing
        if kv_cache.device.type != self.device.type:
            raise RuntimeError(
                "PCP page-pull KV cache device changed unexpectedly: "
                f"runtime={self.device}, cache={kv_cache.device}"
            )
        num_blocks, block_bytes = self.physical_page_geometry(kv_cache)
        if num_blocks <= 0 or block_bytes <= 0:
            raise RuntimeError("PCP page-pull cannot register an empty KV cache")
        device_id = max(kv_cache.get_device(), 0)
        reg_descs = self._wrapper.get_reg_descs(
            [(ptr, num_blocks * block_bytes, device_id, "")], self._memory_type
        )
        self._wrapper.register_memory(reg_descs, backends=list(self.nixl_backends))
        block_data = self._block_descriptors(
            base_addr=ptr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
        )
        xfer_descs = self._wrapper.get_xfer_descs(block_data, self._memory_type)
        local_handle = self._wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
        region = NixlMemoryRegion(
            base_addr=ptr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
            local_xfer_handle=local_handle,
        )
        self._registered_descs.append(reg_descs)
        self._memory_by_ptr[ptr] = region
        return region

    @staticmethod
    def _wire_meta(region: NixlMemoryRegion) -> tuple[int, int, int, int]:
        return (
            region.base_addr,
            region.block_bytes,
            region.num_blocks,
            region.device_id,
        )

    def exchange_regions(self, regions: Sequence[NixlMemoryRegion]) -> None:
        """Exchange stable peer geometry once and prepare remote descriptor lists."""
        signature = tuple(self._wire_meta(region) for region in regions)
        if self._metadata_exchanged:
            if signature != self._registered_region_signature:
                raise RuntimeError("PCP page-pull registered region geometry changed")
            return
        if not regions:
            return
        self.ensure_peers()
        assert self._wrapper is not None and self._memory_type is not None
        group = self._group()
        local_meta = tuple(self._wire_meta(region) for region in regions)
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, local_meta, group=group.cpu_group)
        for source_rank, remote_regions in enumerate(gathered):
            if len(remote_regions) != len(regions):
                raise RuntimeError(
                    "PCP page-pull region registration differs across ranks: "
                    f"rank={source_rank}, regions={len(remote_regions)}, "
                    f"local={len(regions)}"
                )
            if source_rank == self.rank:
                continue
            for region_id, meta in enumerate(remote_regions):
                base_addr, block_bytes, num_blocks, device_id = map(int, meta)
                local = regions[region_id]
                if block_bytes != local.block_bytes:
                    raise RuntimeError(
                        "PCP page-pull requires homogeneous KV page bytes: "
                        f"local={local.block_bytes}, remote={block_bytes}, "
                        f"source_rank={source_rank}, region_id={region_id}"
                    )
                block_data = self._block_descriptors(
                    base_addr=base_addr,
                    block_bytes=block_bytes,
                    num_blocks=num_blocks,
                    device_id=device_id,
                )
                descs = self._wrapper.get_xfer_descs(block_data, self._memory_type)
                self._remote_region_handles[(source_rank, region_id)] = (
                    self._wrapper.prep_xfer_dlist(
                        self._remote_agents[source_rank], descs
                    )
                )
                self._remote_num_blocks[(source_rank, region_id)] = num_blocks
        self._registered_region_signature = signature
        self._metadata_exchanged = True

    def send_notification(self, destination_ranks: Sequence[int], payload: bytes) -> None:
        self.ensure_peers()
        assert self._wrapper is not None
        for destination_rank in destination_ranks:
            self._wrapper.send_notif(
                self._remote_agents[destination_rank], notif_msg=payload
            )

    def get_notifications(self) -> list[bytes]:
        self._ensure_wrapper()
        assert self._wrapper is not None
        return [
            raw
            for notifications in self._wrapper.get_new_notifs().values()
            for raw in notifications
        ]

    def submit_read(
        self,
        *,
        local_region: NixlMemoryRegion,
        local_block_ids: Sequence[int],
        source_rank: int,
        remote_region_id: int,
        remote_block_ids: Sequence[int],
    ) -> int:
        if len(local_block_ids) != len(remote_block_ids):
            raise ValueError("PCP NIXL READ source/destination page counts differ")
        if not local_block_ids:
            raise ValueError("PCP NIXL READ requires at least one page")
        if max(local_block_ids) >= local_region.num_blocks:
            raise RuntimeError(
                "PCP page-pull destination block id exceeds local cache: "
                f"max={max(local_block_ids)}, num_blocks={local_region.num_blocks}"
            )
        remote_num_blocks = self._remote_num_blocks[(source_rank, remote_region_id)]
        if max(remote_block_ids) >= remote_num_blocks:
            raise RuntimeError(
                "PCP page-pull source block id exceeds remote cache: "
                f"max={max(remote_block_ids)}, num_blocks={remote_num_blocks}"
            )
        assert self._wrapper is not None
        handle = self._wrapper.make_prepped_xfer(
            "READ",
            local_region.local_xfer_handle,
            np.asarray(local_block_ids, dtype=np.int64),
            self._remote_region_handles[(source_rank, remote_region_id)],
            np.asarray(remote_block_ids, dtype=np.int64),
        )
        self._wrapper.transfer(handle)
        return handle

    def check_read(self, handle: int) -> str:
        assert self._wrapper is not None
        state = self._wrapper.check_xfer_state(handle)
        if state != "PROC":
            self._wrapper.release_xfer_handle(handle)
        return state


__all__ = ["NixlMemoryRegion", "PCPNixlPeerTransport"]

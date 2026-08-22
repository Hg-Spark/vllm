# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-local NIXL peer-memory transport primitives.

This module stays below PCP scheduling semantics. It owns NIXL agent setup,
stable memory registration, aggregate per-peer descriptor lists, notifications,
and asynchronous producer-driven WRITE submission.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator, Sequence
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


@dataclass(frozen=True)
class NixlWrite:
    """One region-local page mapping in an aggregate WRITE."""

    local_region_id: int
    remote_region_id: int
    local_block_ids: np.ndarray
    remote_block_ids: np.ndarray


class PCPNixlPeerTransport:
    """Minimal same-engine peer-memory WRITE transport for PCP.

    All registered layer caches are flattened into one prepared descriptor list
    per process. A single NIXL handle can therefore cover pages from multiple
    layers while scheduling and causal dependency tracking remain above this
    transport boundary.
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

        self._registered_region_signature: tuple[tuple[int, int, int, int], ...] = ()
        self._metadata_exchanged = False
        self._local_xfer_handle: int | None = None
        self._local_region_offsets: tuple[int, ...] = ()
        self._local_num_blocks: tuple[int, ...] = ()
        self._remote_xfer_handles: dict[int, int] = {}
        self._remote_region_offsets: dict[int, tuple[int, ...]] = {}
        self._remote_num_blocks: dict[tuple[int, int], int] = {}

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
            f"pcp-page-push-r{self.rank}-{uuid.uuid4()}", config
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
        for peer_rank, metadata in enumerate(gathered):
            if peer_rank != self.rank:
                self._remote_agents[peer_rank] = self._wrapper.add_remote_agent(metadata)

    @staticmethod
    def physical_page_geometry(kv_cache: torch.Tensor) -> tuple[int, int]:
        if kv_cache.ndim < 2:
            raise RuntimeError(
                f"PCP page-push expects block-major KV cache, got {kv_cache.shape}"
            )
        num_blocks = int(kv_cache.shape[0])
        page_elements = math.prod(kv_cache.shape[1:])
        if kv_cache.stride(0) != page_elements:
            raise NotImplementedError(
                "PCP page-push requires dense block-major KV pages; "
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

    @staticmethod
    def _region_offsets(num_blocks: Sequence[int]) -> tuple[int, ...]:
        offsets: list[int] = []
        total = 0
        for count in num_blocks:
            offsets.append(total)
            total += int(count)
        return tuple(offsets)

    @classmethod
    def _aggregate_block_descriptors(
        cls, metadata: Sequence[tuple[int, int, int, int]]
    ) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
        counts = tuple(int(item[2]) for item in metadata)
        offsets = cls._region_offsets(counts)
        blocks = [
            cls._block_descriptors(
                base_addr=int(base_addr),
                block_bytes=int(block_bytes),
                num_blocks=int(num_blocks),
                device_id=int(device_id),
            )
            for base_addr, block_bytes, num_blocks, device_id in metadata
        ]
        if not blocks:
            raise RuntimeError("PCP page-push cannot prepare an empty region set")
        return np.concatenate(blocks, axis=0), offsets, counts

    def register_tensor(self, kv_cache: torch.Tensor) -> NixlMemoryRegion:
        self._ensure_wrapper()
        assert self._wrapper is not None and self._memory_type is not None
        ptr = kv_cache.data_ptr()
        existing = self._memory_by_ptr.get(ptr)
        if existing is not None:
            return existing
        if kv_cache.device.type != self.device.type:
            raise RuntimeError(
                "PCP page-push KV cache device changed unexpectedly: "
                f"runtime={self.device}, cache={kv_cache.device}"
            )
        num_blocks, block_bytes = self.physical_page_geometry(kv_cache)
        if num_blocks <= 0 or block_bytes <= 0:
            raise RuntimeError("PCP page-push cannot register an empty KV cache")
        device_id = max(kv_cache.get_device(), 0)
        reg_descs = self._wrapper.get_reg_descs(
            [(ptr, num_blocks * block_bytes, device_id, "")], self._memory_type
        )
        self._wrapper.register_memory(reg_descs, backends=list(self.nixl_backends))
        region = NixlMemoryRegion(
            base_addr=ptr,
            block_bytes=block_bytes,
            num_blocks=num_blocks,
            device_id=device_id,
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
        """Exchange stable geometry and prepare aggregate descriptor lists once."""
        signature = tuple(self._wire_meta(region) for region in regions)
        if self._metadata_exchanged:
            if signature != self._registered_region_signature:
                raise RuntimeError("PCP page-push registered region geometry changed")
            return
        if not regions:
            return
        self.ensure_peers()
        assert self._wrapper is not None and self._memory_type is not None

        group = self._group()
        local_meta = tuple(self._wire_meta(region) for region in regions)
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, local_meta, group=group.cpu_group)

        local_blocks, local_offsets, local_counts = self._aggregate_block_descriptors(
            local_meta
        )
        local_descs = self._wrapper.get_xfer_descs(local_blocks, self._memory_type)
        self._local_xfer_handle = self._wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_descs
        )
        self._local_region_offsets = local_offsets
        self._local_num_blocks = local_counts

        for peer_rank, remote_regions in enumerate(gathered):
            if len(remote_regions) != len(regions):
                raise RuntimeError(
                    "PCP page-push region registration differs across ranks: "
                    f"rank={peer_rank}, regions={len(remote_regions)}, "
                    f"local={len(regions)}"
                )
            for region_id, meta in enumerate(remote_regions):
                _, block_bytes, num_blocks, _ = map(int, meta)
                if block_bytes != regions[region_id].block_bytes:
                    raise RuntimeError(
                        "PCP page-push requires homogeneous KV page bytes: "
                        f"local={regions[region_id].block_bytes}, remote={block_bytes}, "
                        f"peer_rank={peer_rank}, region_id={region_id}"
                    )
                self._remote_num_blocks[(peer_rank, region_id)] = num_blocks
            if peer_rank == self.rank:
                continue
            remote_blocks, remote_offsets, _ = self._aggregate_block_descriptors(
                remote_regions
            )
            remote_descs = self._wrapper.get_xfer_descs(
                remote_blocks, self._memory_type
            )
            self._remote_xfer_handles[peer_rank] = self._wrapper.prep_xfer_dlist(
                self._remote_agents[peer_rank], remote_descs
            )
            self._remote_region_offsets[peer_rank] = remote_offsets

        self._registered_region_signature = signature
        self._metadata_exchanged = True

    def send_notification(self, destination_rank: int, payload: bytes) -> None:
        self.ensure_peers()
        assert self._wrapper is not None
        self._wrapper.send_notif(
            self._remote_agents[destination_rank], notif_msg=payload
        )

    def iter_notifications(self) -> Iterator[bytes]:
        self._ensure_wrapper()
        assert self._wrapper is not None
        for notifications in self._wrapper.get_new_notifs().values():
            yield from notifications

    @staticmethod
    def _validate_block_ids(name: str, block_ids: np.ndarray, num_blocks: int) -> None:
        if block_ids.dtype != np.int64 or block_ids.ndim != 1:
            raise TypeError(f"PCP {name} block IDs must be one-dimensional int64 arrays")
        if block_ids.size == 0:
            raise ValueError(f"PCP {name} requires at least one page")
        if int(block_ids.min()) < 0 or int(block_ids.max()) >= num_blocks:
            raise RuntimeError(
                f"PCP {name} block id exceeds registered cache: "
                f"min={int(block_ids.min())}, max={int(block_ids.max())}, "
                f"num_blocks={num_blocks}"
            )

    def submit_write_batch(
        self,
        *,
        destination_rank: int,
        writes: Sequence[NixlWrite],
    ) -> int:
        """Submit one WRITE handle spanning one or more registered layer regions."""
        if not writes:
            raise ValueError("PCP NIXL WRITE batch requires at least one mapping")
        if self._local_xfer_handle is None:
            raise RuntimeError("PCP NIXL regions were not exchanged before WRITE")
        remote_handle = self._remote_xfer_handles.get(destination_rank)
        remote_offsets = self._remote_region_offsets.get(destination_rank)
        if remote_handle is None or remote_offsets is None:
            raise RuntimeError(
                f"PCP NIXL destination rank {destination_rank} has no prepared regions"
            )

        local_indices: list[np.ndarray] = []
        remote_indices: list[np.ndarray] = []
        for write in writes:
            local_region_id = write.local_region_id
            remote_region_id = write.remote_region_id
            if not 0 <= local_region_id < len(self._local_num_blocks):
                raise ValueError(f"invalid PCP local region id: {local_region_id}")
            if not 0 <= remote_region_id < len(remote_offsets):
                raise ValueError(f"invalid PCP remote region id: {remote_region_id}")
            remote_num_blocks = self._remote_num_blocks[
                (destination_rank, remote_region_id)
            ]
            self._validate_block_ids(
                "WRITE source",
                write.local_block_ids,
                self._local_num_blocks[local_region_id],
            )
            self._validate_block_ids(
                "WRITE destination",
                write.remote_block_ids,
                remote_num_blocks,
            )
            if write.local_block_ids.size != write.remote_block_ids.size:
                raise ValueError("PCP NIXL WRITE source/destination page counts differ")
            local_indices.append(
                write.local_block_ids + self._local_region_offsets[local_region_id]
            )
            remote_indices.append(
                write.remote_block_ids + remote_offsets[remote_region_id]
            )

        local_ids = (
            local_indices[0]
            if len(local_indices) == 1
            else np.concatenate(local_indices)
        )
        remote_ids = (
            remote_indices[0]
            if len(remote_indices) == 1
            else np.concatenate(remote_indices)
        )
        assert self._wrapper is not None
        handle = self._wrapper.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handle,
            local_ids,
            remote_handle,
            remote_ids,
        )
        self._wrapper.transfer(handle)
        return handle

    def check_transfer(self, handle: int) -> str:
        assert self._wrapper is not None
        state = self._wrapper.check_xfer_state(handle)
        if state != "PROC":
            self._wrapper.release_xfer_handle(handle)
        return state


__all__ = ["NixlMemoryRegion", "NixlWrite", "PCPNixlPeerTransport"]

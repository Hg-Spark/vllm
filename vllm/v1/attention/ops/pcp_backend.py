# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral contracts for PCP KV transport.

This module deliberately does not enable any new attention backend. It only
separates the transport-facing concepts that should not depend on a particular
attention kernel: transport capabilities and the physical byte layout of one
paged KV-cache block.

Current runahead execution remains FlashAttention-only. Future backends can
implement :class:`PCPKVCacheAdapter` without changing the PCP scheduler or
one-sided transport protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class PCPBackendCapabilities:
    """PCP features an attention backend/adapter explicitly supports."""

    tensor_transport: bool = False
    page_pull: bool = False
    split_kv_update: bool = False
    physical_page_access: bool = False


@dataclass(frozen=True)
class PCPPhysicalPageLayout:
    """Opaque physical layout of one backend KV-cache page.

    ``page_bytes`` is the byte extent that must be copied for one logical
    cache block. ``block_stride_bytes`` is the distance between consecutive
    physical block starts. Keeping the two values separate is important for
    padded, layer-packed, and quantized layouts where a page need not occupy
    exactly one stride.

    PCP treats page contents as opaque bytes; quantization formats, K/V packing,
    scales, and head ordering remain backend responsibilities.
    """

    num_blocks: int
    page_bytes: int
    block_stride_bytes: int
    block_dim: int = 0
    format_tag: str = "opaque"

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("PCP physical page layout requires num_blocks > 0")
        if self.page_bytes <= 0:
            raise ValueError("PCP physical page layout requires page_bytes > 0")
        if self.block_stride_bytes <= 0:
            raise ValueError(
                "PCP physical page layout requires block_stride_bytes > 0"
            )
        if self.block_stride_bytes < self.page_bytes:
            raise ValueError(
                "PCP page extent cannot exceed the physical block stride: "
                f"page_bytes={self.page_bytes}, "
                f"block_stride_bytes={self.block_stride_bytes}"
            )
        if self.block_dim < 0:
            raise ValueError("PCP physical page block_dim must be non-negative")
        if not self.format_tag:
            raise ValueError("PCP physical page format_tag must be non-empty")

    @property
    def registration_bytes(self) -> int:
        """Minimum byte span covering every block in this layout."""
        return (self.num_blocks - 1) * self.block_stride_bytes + self.page_bytes

    def block_offset_bytes(self, block_id: int) -> int:
        """Return one physical block's byte offset from the cache base pointer."""
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(
                f"PCP block id {block_id} outside [0, {self.num_blocks})"
            )
        return block_id * self.block_stride_bytes


@runtime_checkable
class PCPKVCacheAdapter(Protocol):
    """Minimal backend contract needed by future generic PCP orchestration."""

    @property
    def name(self) -> str:
        ...

    @property
    def capabilities(self) -> PCPBackendCapabilities:
        ...

    def describe_physical_pages(
        self, kv_cache: torch.Tensor
    ) -> PCPPhysicalPageLayout:
        """Describe transport-visible physical pages of ``kv_cache``."""
        ...


def contiguous_block_major_page_layout(
    kv_cache: torch.Tensor,
    *,
    block_dim: int = 0,
    format_tag: str = "contiguous_block_major",
) -> PCPPhysicalPageLayout:
    """Describe the layout currently required by PCP ``page_pull``.

    This helper captures existing behavior without assigning support to any new
    backend. A later backend adapter may replace it with a stride-aware or
    multi-span descriptor while leaving the page-pull scheduler unchanged.
    """
    if kv_cache.ndim < 2:
        raise ValueError(
            f"PCP block-major KV cache must have at least two dims: {kv_cache.shape}"
        )
    if block_dim != 0:
        raise NotImplementedError(
            "contiguous_block_major_page_layout currently requires block_dim=0"
        )

    num_blocks = int(kv_cache.shape[0])
    page_elements = math.prod(kv_cache.shape[1:])
    if num_blocks <= 0 or page_elements <= 0:
        raise ValueError(f"PCP cannot describe empty KV cache: {kv_cache.shape}")
    if kv_cache.stride(0) != page_elements or not kv_cache[0].is_contiguous():
        raise NotImplementedError(
            "PCP contiguous block-major adapter requires each page to be a "
            "contiguous slab: "
            f"shape={tuple(kv_cache.shape)}, stride={tuple(kv_cache.stride())}"
        )

    page_bytes = int(page_elements * kv_cache.element_size())
    block_stride_bytes = int(kv_cache.stride(0) * kv_cache.element_size())
    return PCPPhysicalPageLayout(
        num_blocks=num_blocks,
        page_bytes=page_bytes,
        block_stride_bytes=block_stride_bytes,
        block_dim=block_dim,
        format_tag=format_tag,
    )


__all__ = [
    "PCPBackendCapabilities",
    "PCPKVCacheAdapter",
    "PCPPhysicalPageLayout",
    "contiguous_block_major_page_layout",
]

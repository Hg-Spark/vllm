# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.pcp_backend import (
    PCPBackendCapabilities,
    PCPPhysicalPageLayout,
    contiguous_block_major_page_layout,
)
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_name


def test_pcp_backend_capabilities_default_closed() -> None:
    capabilities = PCPBackendCapabilities()
    assert not capabilities.tensor_transport
    assert not capabilities.page_pull
    assert not capabilities.split_kv_update
    assert not capabilities.physical_page_access


def test_physical_page_layout_tracks_extent_and_stride_separately() -> None:
    layout = PCPPhysicalPageLayout(
        num_blocks=4,
        page_bytes=128,
        block_stride_bytes=256,
        format_tag="test",
    )
    assert layout.block_offset_bytes(0) == 0
    assert layout.block_offset_bytes(3) == 768
    assert layout.registration_bytes == 896


@pytest.mark.parametrize("block_id", [-1, 4])
def test_physical_page_layout_rejects_invalid_block(block_id: int) -> None:
    layout = PCPPhysicalPageLayout(
        num_blocks=4,
        page_bytes=128,
        block_stride_bytes=128,
    )
    with pytest.raises(IndexError):
        layout.block_offset_bytes(block_id)


def test_physical_page_layout_rejects_page_larger_than_stride() -> None:
    with pytest.raises(ValueError, match="page extent"):
        PCPPhysicalPageLayout(
            num_blocks=4,
            page_bytes=256,
            block_stride_bytes=128,
        )


def test_contiguous_block_major_layout_matches_tensor_storage() -> None:
    cache = torch.empty((8, 4, 16, 256), dtype=torch.bfloat16)
    layout = contiguous_block_major_page_layout(cache, format_tag="flash_attn")
    expected_page_bytes = 4 * 16 * 256 * cache.element_size()
    assert layout.num_blocks == 8
    assert layout.page_bytes == expected_page_bytes
    assert layout.block_stride_bytes == expected_page_bytes
    assert layout.registration_bytes == 8 * expected_page_bytes
    assert layout.format_tag == "flash_attn"


def test_contiguous_block_major_layout_rejects_noncontiguous_pages() -> None:
    cache = torch.empty((8, 4, 16, 256), dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(NotImplementedError, match="contiguous slab"):
        contiguous_block_major_page_layout(cache)


def test_pcp_nvtx_name_is_structured_and_stable() -> None:
    assert (
        pcp_nvtx_name(
            "page_pull.read_submit",
            e=3,
            l="model.layers.7.self_attn",
            src=0,
            dst=2,
            pages=16,
        )
        == "pcp.page_pull.read_submit["
        "e=3,l=model.layers.7.self_attn,src=0,dst=2,pages=16]"
    )

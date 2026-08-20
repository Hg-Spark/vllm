# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.attention.pcp import update_standard_kv_cache


def _flash_kv_cache(num_blocks: int = 8, num_kv_heads: int = 2, block_size: int = 16):
    return torch.empty((num_blocks, num_kv_heads, block_size, 32))


def test_standard_runahead_preserves_gqa_kv_head_shape() -> None:
    # GQA-style K/V: fewer KV heads than query heads. The PCP cache policy must
    # treat the KV-head dimension as opaque and preserve it through runahead.
    key = torch.randn(6, 2, 16)
    value = torch.randn(6, 2, 16)
    slot_mapping = torch.arange(6, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    attn_layer = MagicMock()
    attn_layer.layer_name = "model.layers.0.self_attn"
    cache_writer = MagicMock()
    runtime = MagicMock()

    def update_visible_and_defer_repair(
        tensors, slots, apply, cache, cache_block_size
    ) -> None:
        cache_key, cache_value = tensors
        assert cache_key.shape == (6, 2, 16)
        assert cache_value.shape == (6, 2, 16)
        assert slots is slot_mapping
        assert cache is kv_cache
        assert cache_block_size == 16
        apply(tensors, slots)

    runtime.update_visible_and_defer_repair.side_effect = (
        update_visible_and_defer_repair
    )

    with patch(
        "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        update_standard_kv_cache(
            key,
            value,
            slot_mapping,
            attn_layer,
            cache_writer,
            kv_cache,
        )

    runtime.update_visible_and_defer_repair.assert_called_once()
    cache_writer.assert_called_once()
    call_args = cache_writer.call_args.args
    assert call_args[0] is attn_layer
    assert call_args[1] is key
    assert call_args[2] is value
    assert call_args[3] is kv_cache
    assert call_args[4] is slot_mapping


def test_standard_runahead_accepts_rank_major_batch_replica() -> None:
    # A homogeneous batch is packed rank-major by the PCP manager. Verify that
    # the standard cache policy forwards the causal-visible image unchanged;
    # request isolation is carried by the per-token slot mapping.
    local_key = torch.randn(3, 2, 8)
    local_value = torch.randn(3, 2, 8)
    visible_key = torch.randn(7, 2, 8)
    visible_value = torch.randn(7, 2, 8)
    visible_slots = torch.tensor([10, 11, 30, 12, 31, 13, 32], dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    attn_layer = MagicMock()
    attn_layer.layer_name = "model.layers.1.self_attn"
    cache_writer = MagicMock()
    runtime = MagicMock()

    def update_visible_and_defer_repair(
        _tensors, _slots, apply, cache, cache_block_size
    ) -> None:
        assert cache is kv_cache
        assert cache_block_size == 16
        apply((visible_key, visible_value), visible_slots)

    runtime.update_visible_and_defer_repair.side_effect = (
        update_visible_and_defer_repair
    )

    with patch(
        "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        update_standard_kv_cache(
            local_key,
            local_value,
            visible_slots,
            attn_layer,
            cache_writer,
            kv_cache,
        )

    cache_writer.assert_called_once()
    call_args = cache_writer.call_args.args
    assert call_args[0] is attn_layer
    assert call_args[1] is visible_key
    assert call_args[2] is visible_value
    assert call_args[3] is kv_cache
    assert call_args[4] is visible_slots

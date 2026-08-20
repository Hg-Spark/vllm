# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.attention.pcp import update_standard_kv_cache


def _flash_kv_cache(
    num_blocks: int = 8,
    num_kv_heads: int = 2,
    block_size: int = 16,
) -> torch.Tensor:
    return torch.empty((num_blocks, num_kv_heads, block_size, 32))


def test_standard_runahead_preserves_gqa_kv_head_shape() -> None:
    key = torch.randn(6, 2, 16)
    value = torch.randn(6, 2, 16)
    slot_mapping = torch.arange(6, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    attn_layer = MagicMock()
    attn_layer.layer_name = "model.layers.0.self_attn"
    cache_writer = MagicMock()
    runtime = MagicMock()
    runtime.exchange_prefix.return_value = ((key, value), slot_mapping)

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

    runtime.register_kv_cache.assert_called_once_with(kv_cache)
    runtime.exchange_prefix.assert_called_once()
    cache_writer.assert_called_once()
    args = cache_writer.call_args.args
    assert args[0] is attn_layer
    assert args[1] is key
    assert args[2] is value
    assert args[3] is kv_cache
    assert args[4] is slot_mapping


def test_standard_runahead_writes_causal_visible_image() -> None:
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
    runtime.exchange_prefix.return_value = (
        (visible_key, visible_value),
        visible_slots,
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

    args = cache_writer.call_args.args
    assert args[1] is visible_key
    assert args[2] is visible_value
    assert args[4] is visible_slots

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.attention.pcp import update_standard_kv_cache


def test_standard_runahead_preserves_gqa_kv_head_shape() -> None:
    # GQA-style K/V: fewer KV heads than query heads. The PCP cache policy must
    # treat the KV-head dimension as opaque and preserve it through runahead.
    key = torch.randn(6, 2, 16)
    value = torch.randn(6, 2, 16)
    slot_mapping = torch.arange(6, dtype=torch.int64)
    kv_cache = torch.empty(0)
    attn_layer = MagicMock()
    attn_layer.layer_name = "model.layers.0.self_attn"
    cache_writer = MagicMock()
    runtime = MagicMock()

    def update_and_replicate(tensors, slots, apply) -> None:
        cache_key, cache_value = tensors
        assert cache_key.shape == (6, 2, 16)
        assert cache_value.shape == (6, 2, 16)
        assert slots is slot_mapping
        apply(tensors, slots)

    runtime.update_and_replicate.side_effect = update_and_replicate

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

    runtime.update_and_replicate.assert_called_once()
    cache_writer.assert_called_once()
    call_args = cache_writer.call_args.args
    assert call_args[0] is attn_layer
    assert call_args[1] is key
    assert call_args[2] is value
    assert call_args[3] is kv_cache
    assert call_args[4] is slot_mapping


def test_standard_runahead_accepts_rank_major_batch_replica() -> None:
    # A homogeneous batch is packed rank-major by the PCP manager. Verify that
    # the standard cache policy forwards the compact gathered image unchanged;
    # request isolation is carried by the per-token slot mapping.
    local_key = torch.randn(3, 2, 8)
    local_value = torch.randn(3, 2, 8)
    gathered_key = torch.randn(7, 2, 8)
    gathered_value = torch.randn(7, 2, 8)
    gathered_slots = torch.tensor([10, 11, 30, 12, 31, 13, 32], dtype=torch.int64)
    kv_cache = torch.empty(0)
    attn_layer = MagicMock()
    attn_layer.layer_name = "model.layers.1.self_attn"
    cache_writer = MagicMock()
    runtime = MagicMock()

    def update_and_replicate(_tensors, _slots, apply) -> None:
        apply((gathered_key, gathered_value), gathered_slots)

    runtime.update_and_replicate.side_effect = update_and_replicate

    with patch(
        "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        update_standard_kv_cache(
            local_key,
            local_value,
            gathered_slots,
            attn_layer,
            cache_writer,
            kv_cache,
        )

    cache_writer.assert_called_once()
    call_args = cache_writer.call_args.args
    assert call_args[0] is attn_layer
    assert call_args[1] is gathered_key
    assert call_args[2] is gathered_value
    assert call_args[3] is kv_cache
    assert call_args[4] is gathered_slots

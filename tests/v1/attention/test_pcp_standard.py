# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.attention.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.v1.attention.ops.pcp_standard import prepare_standard_pcp_kv_cache_inputs


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
    runtime = MagicMock()
    runtime.transport = "prefix_p2p"
    runtime.exchange_prefix.return_value = ((key, value), slot_mapping)

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slot_mapping, kv_cache
        )

    runtime.exchange_prefix.assert_called_once()
    assert out_key is key
    assert out_value is value
    assert out_slots is slot_mapping
    assert out_key.shape[1:] == (2, 16)


def test_page_pull_prepare_does_not_duplicate_native_cache_write() -> None:
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.arange(4, dtype=torch.int64)
    kv_cache = torch.zeros((8, 2, 16, 16))
    runtime = MagicMock()
    runtime.transport = "page_pull"
    runtime.local_rows = 2
    runtime.rank_local_slot_mapping.return_value = slots[:2]

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slots, kv_cache
        )

    runtime.page_pull_prepare_layer.assert_called_once_with(kv_cache)
    runtime.page_pull_after_cache_write.assert_not_called()
    assert out_key is key
    assert out_value is value
    assert out_slots.tolist() == [0, 1]
    assert torch.count_nonzero(kv_cache) == 0


def test_page_pull_ready_is_published_before_attention() -> None:
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.transport = "page_pull"
    order: list[str] = []
    runtime.page_pull_after_cache_write.side_effect = lambda _: order.append("ready")

    def attention(layer_name: str) -> str:
        order.append("attention")
        return layer_name

    with (
        patch(
            "vllm.model_executor.layers.attention.kv_transfer_utils.get_pcp_runahead_runtime",
            return_value=runtime,
        ),
        patch(
            "vllm.model_executor.layers.attention.kv_transfer_utils.has_kv_transfer_group",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.attention.attention.get_attention_context",
            return_value=(None, MagicMock(), kv_cache, torch.tensor([0])),
        ),
    ):
        wrapped = maybe_transfer_kv_layer(attention)
        assert wrapped("layer") == "layer"

    runtime.page_pull_after_cache_write.assert_called_once_with(kv_cache)
    assert order == ["ready", "attention"]


def test_page_pull_skips_post_write_when_native_update_is_skipped() -> None:
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.transport = "page_pull"

    def attention(layer_name: str) -> str:
        return layer_name

    with (
        patch(
            "vllm.model_executor.layers.attention.kv_transfer_utils.get_pcp_runahead_runtime",
            return_value=runtime,
        ),
        patch(
            "vllm.model_executor.layers.attention.kv_transfer_utils.has_kv_transfer_group",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.attention.attention.get_attention_context",
            return_value=(None, MagicMock(), kv_cache, None),
        ),
    ):
        assert maybe_transfer_kv_layer(attention)("layer") == "layer"

    runtime.page_pull_after_cache_write.assert_not_called()


def test_standard_compact_full_kv_collective_uses_allgatherv_for_variable_width() -> None:
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.arange(5, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.transport = "full_kv_collective"
    runtime.local_rows = 2
    runtime.total_rows = 5
    runtime.rows_per_rank = (2, 3)
    group = MagicMock()
    runtime._group.return_value = group
    gathered_key = torch.randn(5, 2, 8)
    gathered_value = torch.randn(5, 2, 8)
    group.all_gatherv.return_value = [gathered_key, gathered_value]

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slots, kv_cache
        )

    group.all_gatherv.assert_called_once()
    group.all_gather.assert_not_called()
    assert out_key is gathered_key
    assert out_value is gathered_value
    assert out_slots.tolist() == list(range(5))


def test_standard_compact_full_kv_collective_keeps_equal_width_allgather_fast_path() -> None:
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.arange(4, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.transport = "full_kv_collective"
    runtime.local_rows = 2
    runtime.total_rows = 4
    runtime.rows_per_rank = (2, 2)
    group = MagicMock()
    runtime._group.return_value = group
    gathered_key = torch.randn(4, 2, 8)
    gathered_value = torch.randn(4, 2, 8)
    group.all_gather.side_effect = [gathered_key, gathered_value]

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, _ = prepare_standard_pcp_kv_cache_inputs(
            key, value, slots, kv_cache
        )

    assert group.all_gather.call_count == 2
    group.all_gatherv.assert_not_called()
    assert out_key is gathered_key
    assert out_value is gathered_value


def test_standard_fallback_reuses_baseline_allgather() -> None:
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.arange(4, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    group = MagicMock()
    group.world_size = 2
    group.all_gather.side_effect = lambda tensor, dim: torch.cat(
        (tensor, tensor), dim=dim
    )

    with (
        patch(
            "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
            return_value=None,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_standard.get_pcp_group",
            return_value=group,
        ),
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slots, kv_cache
        )

    assert out_key.shape[0] == 4
    assert out_value.shape[0] == 4
    assert out_slots is slots
    assert group.all_gather.call_count == 2

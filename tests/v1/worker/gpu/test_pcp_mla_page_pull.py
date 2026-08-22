# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention.pcp import (
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.v1.worker.gpu.pcp_runahead_manager import RunaheadPCPManager


def _page_pull_config(cache_dtype: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"pcp_runahead": {"transport": "page_pull"}},
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_dbo=False,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


def test_dense_mla_page_pull_config_allows_unquantized_cache() -> None:
    RunaheadPCPManager.validate_config(_page_pull_config("auto"), False)
    RunaheadPCPManager.validate_config(_page_pull_config("float16"), False)
    RunaheadPCPManager.validate_config(_page_pull_config("bfloat16"), False)


def test_dense_mla_page_pull_config_rejects_quantized_cache() -> None:
    with pytest.raises(NotImplementedError, match="unquantized FP16/BF16"):
        RunaheadPCPManager.validate_config(_page_pull_config("fp8"), False)


def test_mla_page_pull_uses_local_latents_and_native_page_lifecycle() -> None:
    runtime = MagicMock()
    runtime.transport = "page_pull"
    runtime.local_rows = 2
    local_slots = torch.tensor([12, 13], dtype=torch.int64)
    runtime.rank_local_slot_mapping.return_value = local_slots

    original_cache_update = MagicMock(return_value=None)
    impl = SimpleNamespace(do_kv_cache_update=original_cache_update)
    layer = SimpleNamespace(
        kv_lora_rank=4,
        kv_cache_dtype="auto",
        impl=impl,
    )
    forward_context = SimpleNamespace(no_compile_layers={"model.layers.0.self_attn": layer})

    kv_c = torch.arange(8, dtype=torch.float32).view(2, 4)
    k_pe = torch.arange(4, dtype=torch.float32).view(2, 1, 2)
    slots = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    kv_cache = torch.empty(8, 16, 6)
    k_scale = torch.tensor(1.0)

    with (
        patch(
            "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
            return_value=runtime,
        ),
        patch(
            "vllm.model_executor.layers.attention.pcp.get_forward_context",
            return_value=forward_context,
        ),
    ):
        cache_kv, cache_kpe, cache_slots = maybe_gather_mla_latent_cache_inputs(
            kv_c,
            k_pe,
            slots,
            num_decode_tokens=0,
            use_pcp=True,
        )
        impl.do_kv_cache_update(
            cache_kv,
            cache_kpe,
            kv_cache,
            cache_slots,
            "auto",
            k_scale,
        )

    runtime.rank_local_slot_mapping.assert_called_once_with(slots)
    runtime.exchange_cache_inputs.assert_not_called()
    runtime.page_pull_prepare_layer.assert_called_once_with(kv_cache)
    runtime.page_pull_after_cache_write.assert_called_once_with(kv_cache)
    original_cache_update.assert_called_once()
    call_args = original_cache_update.call_args.args
    assert call_args[0] is kv_c
    assert torch.equal(call_args[1], k_pe)
    assert call_args[2] is kv_cache
    assert call_args[3] is local_slots
    assert call_args[4] == "auto"
    assert call_args[5] is k_scale
    assert cache_kv is kv_c
    assert torch.equal(cache_kpe, k_pe)
    assert cache_slots is local_slots


def test_mla_page_pull_rejects_quantized_layer_cache() -> None:
    runtime = MagicMock()
    runtime.transport = "page_pull"
    runtime.local_rows = 2
    layer = SimpleNamespace(
        kv_lora_rank=4,
        kv_cache_dtype="fp8",
        impl=SimpleNamespace(do_kv_cache_update=MagicMock()),
    )
    forward_context = SimpleNamespace(no_compile_layers={"model.layers.0.self_attn": layer})

    with (
        patch(
            "vllm.model_executor.layers.attention.pcp.get_pcp_runahead_runtime",
            return_value=runtime,
        ),
        patch(
            "vllm.model_executor.layers.attention.pcp.get_forward_context",
            return_value=forward_context,
        ),
        pytest.raises(NotImplementedError, match="unquantized FP16/BF16"),
    ):
        maybe_gather_mla_latent_cache_inputs(
            torch.zeros(2, 4),
            torch.zeros(2, 1, 2),
            torch.arange(4),
            num_decode_tokens=0,
            use_pcp=True,
        )

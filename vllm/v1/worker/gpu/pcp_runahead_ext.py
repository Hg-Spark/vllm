# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped extensions for MRV2 PCP runahead experiments.

This keeps the experimental changes isolated from the baseline PCP manager:

* standard FlashAttention (MHA/GQA/MQA) can use PCP cache updates;
* pure decode keeps rank-local cache writes;
* standard-attention runahead accepts homogeneous prefill/extend batches,
  including requests with existing KV context;
* mixed decode+prefill batches retain the baseline PCP path;
* MLA runahead retains its existing single-fresh-prefill eligibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vllm.config import CUDAGraphMode
from vllm.v1.attention.ops.pcp_standard import (
    install_standard_attention_pcp_cache_updates,
)

RUNAHEAD_MIN_PREFILL_TOKENS = 1024


def runahead_batch_eligible(
    *,
    num_reqs: int,
    is_prefilling: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    pcp_world_size: int,
    min_prefill_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS,
) -> bool:
    """Return whether a standard-attention scheduler batch can use runahead.

    Existing context is intentionally absent from this predicate. Fresh prefill,
    chunked/continued prefill, and prefix-cache-hit extend all share the same
    transport as long as every scheduled request is still prefilling. Mixed
    decode+prefill batches fall back to baseline PCP.
    """
    if num_reqs <= 0:
        return False
    if not bool(is_prefilling[:num_reqs].all()):
        return False
    total_prefill_tokens = int(num_scheduled_tokens[:num_reqs].sum())
    return total_prefill_tokens >= max(pcp_world_size, min_prefill_tokens)


def install_pcp_extensions(pcp: Any) -> None:
    """Patch the experimental PCP classes once per process."""
    if getattr(pcp, "_runahead_ext_installed", False):
        return

    original_init = pcp.PCPManager.__init__
    original_partition = pcp.PCPManager.partition_batch
    original_runahead_partition = pcp.RunaheadPCPManager.partition_batch
    original_build = pcp.maybe_build_pcp_manager

    def manager_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._global_has_prefill = False
        self._standard_attention_pcp = False

    @staticmethod
    def validate_config(vllm_config: Any, supports_mm_inputs: bool) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        pcp_size = parallel_config.prefill_context_parallel_size
        if pcp_size <= 1:
            return

        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError(
                "MRV2 PCP does not support encoder-decoder models yet."
            )
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "MRV2 PCP does not support speculative decoding yet."
            )
        if not model_config.use_mla and parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "standard-attention PCP MVP currently requires TP=1"
            )
        is_sparse_mla = model_config.use_mla and hasattr(
            model_config.hf_text_config, "index_topk"
        )
        if (
            is_sparse_mla
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise NotImplementedError(
                "MRV2 sparse MLA PCP does not support CUDA graphs yet. "
                "Set -cc.cudagraph_mode=NONE."
            )
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise NotImplementedError("MRV2 PCP supports PIECEWISE CUDA graphs only.")

    def manager_partition(self: Any, input_batch: Any) -> Any:
        self._global_has_prefill = bool(
            input_batch.is_prefilling_np[: input_batch.num_reqs].any()
        )
        return original_partition(self, input_batch)

    def prepare_slot_mappings(self: Any) -> Any:
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert self._global_batch is not None
        global_batch = self._global_batch
        global_batch_slot_mappings = self._block_tables.compute_slot_mappings(
            global_batch.idx_mapping,
            global_batch.query_start_loc,
            global_batch.positions,
            global_batch.num_tokens,
            out=self._global_batch_slot_mappings,
        )
        if not self._global_has_prefill:
            return global_batch_slot_mappings
        return self._convert_to_gathered_slot_mappings(global_batch_slot_mappings)

    def runahead_partition(self: Any, input_batch: Any) -> Any:
        if not self._standard_attention_pcp:
            return original_runahead_partition(self, input_batch)

        use_runahead = runahead_batch_eligible(
            num_reqs=input_batch.num_reqs,
            is_prefilling=input_batch.is_prefilling_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            pcp_world_size=self.pcp_world_size,
        )
        self._use_runahead_partition = use_runahead
        local_batch = pcp.PCPManager.partition_batch(self, input_batch)
        if use_runahead:
            self._runahead_runtime.begin_step(local_batch.num_tokens_after_padding)
        else:
            self._runahead_runtime.disable_step()
        return local_batch

    def maybe_build_pcp_manager(*args: Any, **kwargs: Any) -> Any:
        manager = original_build(*args, **kwargs)
        if manager is None:
            return None

        vllm_config = args[0] if args else kwargs["vllm_config"]
        is_standard_attention = not vllm_config.model_config.use_mla
        manager._standard_attention_pcp = is_standard_attention
        if is_standard_attention:
            install_standard_attention_pcp_cache_updates(vllm_config)
        return manager

    pcp.PCPManager.__init__ = manager_init
    pcp.PCPManager.validate_config = validate_config
    pcp.PCPManager.partition_batch = manager_partition
    pcp.PCPManager.prepare_slot_mappings = prepare_slot_mappings
    pcp.RunaheadPCPManager.partition_batch = runahead_partition
    pcp.maybe_build_pcp_manager = maybe_build_pcp_manager
    pcp._runahead_ext_installed = True

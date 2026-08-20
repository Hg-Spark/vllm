# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility bridge for standard MHA/GQA/MQA PCP on MRV2."""

from __future__ import annotations

from typing import Any

_FLASH_ATTN_BACKEND = "FLASH_ATTN"
_PCP_UNSUPPORTED_FEATURE = "prefill context parallelism"
_PCP_LAYER_MARKER = "_standard_attention_pcp_enabled"


def enable_standard_attention_pcp_config_support(vllm_config_cls: type[Any]) -> None:
    """Allow standard-attention PCP to reach backend/runtime capability checks."""
    if getattr(vllm_config_cls, "_standard_attention_pcp_config_patched", False):
        return

    original_unsupported = vllm_config_cls._get_v2_model_runner_unsupported_features

    def patched_unsupported(self: Any) -> list[str]:
        unsupported = original_unsupported(self)
        model_config = self.model_config
        if (
            self.parallel_config.prefill_context_parallel_size > 1
            and model_config is not None
            and not model_config.use_mla
        ):
            unsupported = [
                feature
                for feature in unsupported
                if feature != _PCP_UNSUPPORTED_FEATURE
            ]
        return unsupported

    vllm_config_cls._get_v2_model_runner_unsupported_features = patched_unsupported
    vllm_config_cls._standard_attention_pcp_config_patched = True

    from vllm.v1.attention.backend import AttentionBackend

    if getattr(AttentionBackend, "_standard_attention_pcp_capability_patched", False):
        return

    original_supports_pcp = AttentionBackend.supports_pcp.__func__

    def patched_supports_pcp(cls: type[Any]) -> bool:
        if cls.get_name() == _FLASH_ATTN_BACKEND:
            return True
        return original_supports_pcp(cls)

    AttentionBackend.supports_pcp = classmethod(patched_supports_pcp)
    AttentionBackend._standard_attention_pcp_capability_patched = True


def install_standard_attention_pcp_cache_updates(vllm_config: Any) -> None:
    """Install one class-level hook and mark only this PCP engine's layers."""
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.attention.pcp import update_standard_kv_cache
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    for layer in vllm_config.compilation_config.static_forward_context.values():
        if isinstance(layer, Attention) and layer.get_attn_backend().get_name() == _FLASH_ATTN_BACKEND:
            setattr(layer, _PCP_LAYER_MARKER, True)

    if getattr(FlashAttentionImpl, "_standard_attention_pcp_cache_update_installed", False):
        return

    original_update = FlashAttentionImpl.do_kv_cache_update

    def wrapped_update(
        self_impl: Any,
        attn_layer: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        slot_mapping: Any,
    ) -> None:
        if not getattr(attn_layer, _PCP_LAYER_MARKER, False):
            original_update(
                self_impl,
                attn_layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )
            return

        def cache_writer(
            layer: Any,
            cache_key: Any,
            cache_value: Any,
            cache: Any,
            cache_slots: Any,
        ) -> None:
            original_update(
                self_impl,
                layer,
                cache_key,
                cache_value,
                cache,
                cache_slots,
            )

        update_standard_kv_cache(
            key,
            value,
            slot_mapping,
            attn_layer,
            cache_writer,
            kv_cache,
        )

    FlashAttentionImpl.do_kv_cache_update = wrapped_update
    FlashAttentionImpl.supports_pcp = True
    FlashAttentionImpl._standard_attention_pcp_cache_update_installed = True

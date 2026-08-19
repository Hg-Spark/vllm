# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility integration for standard MHA/GQA/MQA PCP.

Batch policy lives in ``pcp_manager.py`` and KV communication policy lives in
``model_executor/layers/attention/pcp.py``. This module only bridges the current
MRV2 MLA-only early config/backend gate and installs that policy at the standard
Attention cache-write entry point.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_FLASH_ATTN_BACKEND = "FLASH_ATTN"
_PCP_UNSUPPORTED_FEATURE = "prefill context parallelism"


def enable_standard_attention_pcp_config_support(vllm_config_cls: type[Any]) -> None:
    """Allow non-MLA PCP to reach FlashAttention's runtime capability checks."""
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

    attention_backend_cls: Any = AttentionBackend
    if not getattr(
        attention_backend_cls,
        "_standard_attention_pcp_capability_patched",
        False,
    ):
        original_supports_pcp = attention_backend_cls.supports_pcp.__func__

        def patched_supports_pcp(cls: type[Any]) -> bool:
            if cls.get_name() == _FLASH_ATTN_BACKEND:
                return True
            return original_supports_pcp(cls)

        attention_backend_cls.supports_pcp = classmethod(patched_supports_pcp)
        attention_backend_cls._standard_attention_pcp_capability_patched = True


def install_standard_attention_pcp_cache_updates(vllm_config: VllmConfig) -> None:
    """Install standard-attention PCP policy at the backend cache-write hook."""
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.attention.pcp import update_standard_kv_cache

    for layer in vllm_config.compilation_config.static_forward_context.values():
        if not isinstance(layer, Attention):
            continue

        backend = layer.get_attn_backend()
        if backend.get_name() != _FLASH_ATTN_BACKEND:
            raise NotImplementedError(
                "MRV2 PCP for standard MHA/GQA/MQA attention currently requires "
                f"FLASH_ATTN, got {backend.get_name()}."
            )

        impl: Any = layer.impl
        if getattr(impl, "_standard_attention_pcp_cache_update_installed", False):
            continue
        if not hasattr(impl, "do_kv_cache_update"):
            raise NotImplementedError(
                f"{impl.__class__.__name__} does not expose do_kv_cache_update"
            )

        original_update = impl.do_kv_cache_update

        def wrapped_update(
            self_impl: Any,
            attn_layer: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            slot_mapping: Any,
            *,
            _original_update: Any = original_update,
        ) -> None:
            update_standard_kv_cache(
                key,
                value,
                slot_mapping,
                attn_layer,
                _original_update,
                kv_cache,
            )

        impl.do_kv_cache_update = MethodType(wrapped_update, impl)
        impl._standard_attention_pcp_cache_update_installed = True
        impl_cls: Any = type(impl)
        impl_cls.supports_pcp = True

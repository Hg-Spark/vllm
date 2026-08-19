# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP integration for standard MHA/GQA/MQA attention.

The baseline path materializes partitioned prefill K/V with a PCP AllGather
before the backend cache write. When the experimental runahead runtime is
active, the same cache-write entry point uses causal-prefix P2P on the critical
path and asynchronously restores the replicated cache image.

This module deliberately keeps the integration outside the attention backend so
MLA and standard attention can share the same request-level PCP manager while
using their existing cache layouts.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_FLASH_ATTN_BACKEND = "FLASH_ATTN"
_PCP_UNSUPPORTED_FEATURE = "prefill context parallelism"


def enable_standard_attention_pcp_config_support(vllm_config_cls: type[Any]) -> None:
    """Allow non-MLA PCP to reach FlashAttention's runtime capability checks.

    MRV2 historically gated PCP on ``model_config.use_mla`` before attention
    backends were initialized. Standard attention now opts in through
    FlashAttention, so remove that early feature gate while retaining the normal
    backend capability check for every other backend.
    """
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

    # Backend selection and the later CP compatibility check both consult this
    # classmethod. Keep the opt-in narrow to FlashAttention; other standard
    # attention backends continue to reject PCP until they implement the same
    # cache-update contract.
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
    """Wrap standard FlashAttention KV writes with PCP transport.

    The slot-mapping width is the protocol between the PCP manager and this
    wrapper. A mapping wider than the rank-local K/V means baseline PCP prepared
    a rank-major gathered mapping, so K and V are AllGathered along the token
    dimension. Pure decode receives a rank-local mapping and therefore performs
    no PCP KV collective.

    When runahead is active, the runtime owns communication and calls the
    original backend cache writer first for the causal-visible prefix and later
    for the asynchronously gathered replicated image.
    """
    from vllm.distributed.parallel_state import get_pcp_group
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime

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
            runtime = get_pcp_runahead_runtime()
            if runtime is not None:

                def apply(tensors: tuple[Any, ...], cache_slot_mapping: Any) -> None:
                    cache_key, cache_value = tensors
                    _original_update(
                        attn_layer,
                        cache_key,
                        cache_value,
                        kv_cache,
                        cache_slot_mapping,
                    )

                runtime.update_and_replicate(
                    (key, value),
                    slot_mapping,
                    apply,
                )
                return

            # Baseline standard-attention PCP. The PCP manager uses a
            # pcp_size*rows mapping for partitioned prefills and a rows mapping
            # for replicated pure decode.
            if slot_mapping.shape[0] > key.shape[0]:
                pcp_group = get_pcp_group()
                pcp_size = pcp_group.world_size
                if slot_mapping.shape[0] % pcp_size != 0:
                    raise RuntimeError(
                        "PCP gathered slot mapping is not divisible by PCP size: "
                        f"slots={slot_mapping.shape[0]}, pcp={pcp_size}"
                    )
                local_rows = slot_mapping.shape[0] // pcp_size
                if key.shape[0] < local_rows or value.shape[0] < local_rows:
                    raise RuntimeError(
                        "PCP standard-attention K/V rows are smaller than the "
                        f"rank-local slab: key={key.shape[0]}, value={value.shape[0]}, "
                        f"rows={local_rows}"
                    )
                key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
                value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)

            _original_update(
                attn_layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )

        impl.do_kv_cache_update = MethodType(wrapped_update, impl)
        impl._standard_attention_pcp_cache_update_installed = True
        # Keep the concrete implementation consistent with the backend-level
        # capability exposed during early backend selection.
        impl_cls: Any = type(impl)
        impl_cls.supports_pcp = True

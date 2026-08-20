# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NVTX helpers for PCP profiling.

The helper deliberately emits ranges only. It never synchronizes CUDA, so
profiling does not serialize the communication/compute overlap that PCP
runahead is intended to expose.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

import torch

_PCP_NVTX_ENABLED = os.environ.get("VLLM_PCP_NVTX", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PCP_NVTX_HOOKS_INSTALLED = False
_PCP_NVTX_CURRENT_LAYER: str | None = None


def pcp_nvtx_enabled() -> bool:
    """Return whether PCP NVTX ranges are enabled for this process."""
    return _PCP_NVTX_ENABLED


@contextmanager
def pcp_nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range without adding CUDA synchronization."""
    if not _PCP_NVTX_ENABLED:
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _pcp_layer_name(layer: object) -> str:
    layer_name = getattr(layer, "layer_name", None)
    if layer_name:
        return str(layer_name)
    return layer.__class__.__name__


def install_pcp_nvtx_hooks() -> None:
    """Install detailed FlashAttention ranges when PCP NVTX is enabled.

    The hooks are process-global and installed once. They are never installed
    when ``VLLM_PCP_NVTX`` is disabled, so normal benchmark runs do not add
    per-layer Python wrappers to FlashAttention's hot path.
    """
    global _PCP_NVTX_HOOKS_INSTALLED
    if not _PCP_NVTX_ENABLED or _PCP_NVTX_HOOKS_INSTALLED:
        return

    from vllm.v1.attention.backends import flash_attn as flash_attn_module

    impl_cls = flash_attn_module.FlashAttentionImpl
    reshape_and_cache = getattr(flash_attn_module, "reshape_and_cache_flash", None)
    if reshape_and_cache is None:
        return

    original_kv_update = impl_cls.do_kv_cache_update
    original_forward = impl_cls.forward

    @wraps(original_kv_update)
    def profiled_kv_update(self, layer, *args, **kwargs):
        global _PCP_NVTX_CURRENT_LAYER
        previous_layer = _PCP_NVTX_CURRENT_LAYER
        layer_name = _pcp_layer_name(layer)
        _PCP_NVTX_CURRENT_LAYER = layer_name
        try:
            with pcp_nvtx_range(f"pcp.layer.{layer_name}.kv_update"):
                return original_kv_update(self, layer, *args, **kwargs)
        finally:
            _PCP_NVTX_CURRENT_LAYER = previous_layer

    @wraps(reshape_and_cache)
    def profiled_reshape_and_cache(*args, **kwargs):
        layer_name = _PCP_NVTX_CURRENT_LAYER
        range_name = (
            "pcp.kv_cache_write"
            if layer_name is None
            else f"pcp.layer.{layer_name}.kv_cache_write"
        )
        with pcp_nvtx_range(range_name):
            return reshape_and_cache(*args, **kwargs)

    @wraps(original_forward)
    def profiled_forward(self, layer, *args, **kwargs):
        layer_name = _pcp_layer_name(layer)
        with pcp_nvtx_range(f"pcp.layer.{layer_name}.attention"):
            return original_forward(self, layer, *args, **kwargs)

    impl_cls.do_kv_cache_update = profiled_kv_update
    impl_cls.forward = profiled_forward
    flash_attn_module.reshape_and_cache_flash = profiled_reshape_and_cache
    _PCP_NVTX_HOOKS_INSTALLED = True

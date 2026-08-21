# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NVTX helpers for PCP profiling.

The helper deliberately emits ranges and marks only. It never adds CUDA
synchronization, so profiling does not serialize the communication/compute
overlap that PCP runahead is intended to expose.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

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


def pcp_nvtx_name(event: str, **fields: object) -> str:
    """Build a stable structured NVTX label for PCP events.

    Field order follows call-site keyword order so Nsight traces remain easy to
    scan and grep, for example::

        pcp.page_pull.read_submit[e=3,l=model.layers.7.self_attn,src=0,dst=2,pages=16]
    """
    prefix = event if event.startswith("pcp.") else f"pcp.{event}"
    payload = ",".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    return prefix if not payload else f"{prefix}[{payload}]"


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


def pcp_nvtx_mark(name: str) -> None:
    """Emit an instantaneous NVTX mark without synchronizing CUDA."""
    if _PCP_NVTX_ENABLED:
        torch.cuda.nvtx.mark(name)


def _pcp_layer_name(layer: object) -> str:
    layer_name = getattr(layer, "layer_name", None)
    if layer_name:
        return str(layer_name)
    return layer.__class__.__name__


def _page_pull_layer_name(transport: Any, layer_id: int) -> str:
    names = transport.registered_layer_names
    if 0 <= layer_id < len(names):
        return names[layer_id]
    return str(layer_id)


def _page_pull_num_pages(transport: Any, layer_id: int, source_rank: int) -> int:
    del layer_id  # The current step plan is layer-invariant.
    plan = getattr(transport, "_plan", None)
    if plan is None:
        return 0
    destination_ids, _ = plan.transfer_block_ids(transport.rank, source_rank)
    return len(destination_ids)


def _install_flash_attention_hooks() -> None:
    """Install the current FlashAttention-specific compute/cache ranges."""
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
            with pcp_nvtx_range(pcp_nvtx_name("layer.kv_update", l=layer_name)):
                return original_kv_update(self, layer, *args, **kwargs)
        finally:
            _PCP_NVTX_CURRENT_LAYER = previous_layer

    @wraps(reshape_and_cache)
    def profiled_reshape_and_cache(*args, **kwargs):
        layer_name = _PCP_NVTX_CURRENT_LAYER
        with pcp_nvtx_range(
            pcp_nvtx_name("layer.kv_cache_write", l=layer_name)
        ):
            return reshape_and_cache(*args, **kwargs)

    @wraps(original_forward)
    def profiled_forward(self, layer, *args, **kwargs):
        layer_name = _pcp_layer_name(layer)
        with pcp_nvtx_range(pcp_nvtx_name("layer.attention", l=layer_name)):
            return original_forward(self, layer, *args, **kwargs)

    impl_cls.do_kv_cache_update = profiled_kv_update
    impl_cls.forward = profiled_forward
    flash_attn_module.reshape_and_cache_flash = profiled_reshape_and_cache


def _install_page_pull_hooks() -> None:
    """Trace the page-pull control/data path, including its progress thread."""
    from vllm.v1.attention.ops.pcp_page_pull import PCPPagePullTransport

    transport_cls = PCPPagePullTransport
    original_configure_step = transport_cls.configure_step
    original_publish_ready = transport_cls.publish_ready
    original_start_read = transport_cls._start_read
    original_progress_once = transport_cls._progress_once
    original_progress_loop = transport_cls._progress_loop
    original_wait_layer = transport_cls.wait_layer
    original_finish_step = transport_cls.finish_step

    @wraps(original_configure_step)
    def profiled_configure_step(self, *, epoch, plan):
        with pcp_nvtx_range(
            pcp_nvtx_name("page_pull.step_config", e=epoch, rank=self.rank)
        ):
            result = original_configure_step(self, epoch=epoch, plan=plan)
        pcp_nvtx_mark(pcp_nvtx_name("page_pull.step_begin", e=epoch, rank=self.rank))
        return result

    @wraps(original_publish_ready)
    def profiled_publish_ready(self, layer_id):
        layer_name = _page_pull_layer_name(self, layer_id)
        consumers = () if self._plan is None else self._plan.consumer_ranks(self.rank)
        with pcp_nvtx_range(
            pcp_nvtx_name(
                "page_pull.ready_publish",
                e=self._epoch,
                l=layer_name,
                src=self.rank,
                consumers=len(consumers),
            )
        ):
            result = original_publish_ready(self, layer_id)
        for destination_rank in consumers:
            pcp_nvtx_mark(
                pcp_nvtx_name(
                    "page_pull.ready_send",
                    e=self._epoch,
                    l=layer_name,
                    src=self.rank,
                    dst=destination_rank,
                )
            )
        return result

    @wraps(original_start_read)
    def profiled_start_read(self, layer_id, source_rank, meta):
        layer_name = _page_pull_layer_name(self, layer_id)
        pages = _page_pull_num_pages(self, layer_id, source_rank)
        with pcp_nvtx_range(
            pcp_nvtx_name(
                "page_pull.read_submit",
                e=self._epoch,
                l=layer_name,
                src=source_rank,
                dst=self.rank,
                pages=pages,
            )
        ):
            return original_start_read(self, layer_id, source_rank, meta)

    @wraps(original_progress_once)
    def profiled_progress_once(self):
        ready_before = set(self._ready_recvs)
        done_before = set(self._done_pairs)
        try:
            result = original_progress_once(self)
        except BaseException:
            pcp_nvtx_mark(
                pcp_nvtx_name("page_pull.progress_error", e=self._epoch, rank=self.rank)
            )
            raise

        for layer_id, source_rank in sorted(ready_before - set(self._ready_recvs)):
            pcp_nvtx_mark(
                pcp_nvtx_name(
                    "page_pull.ready_recv",
                    e=self._epoch,
                    l=_page_pull_layer_name(self, layer_id),
                    src=source_rank,
                    dst=self.rank,
                )
            )

        for layer_id, source_rank in sorted(self._done_pairs - done_before):
            pcp_nvtx_mark(
                pcp_nvtx_name(
                    "page_pull.read_done",
                    e=self._epoch,
                    l=_page_pull_layer_name(self, layer_id),
                    src=source_rank,
                    dst=self.rank,
                    pages=_page_pull_num_pages(self, layer_id, source_rank),
                )
            )
        return result

    @wraps(original_progress_loop)
    def profiled_progress_loop(self):
        with pcp_nvtx_range(
            pcp_nvtx_name("page_pull.progress_thread", e=self._epoch, rank=self.rank)
        ):
            return original_progress_loop(self)

    @wraps(original_wait_layer)
    def profiled_wait_layer(self, layer_id):
        layer_name = _page_pull_layer_name(self, layer_id)
        sources = () if self._plan is None else self._plan.required_source_ranks(self.rank)
        with pcp_nvtx_range(
            pcp_nvtx_name(
                "page_pull.wait",
                e=self._epoch,
                l=layer_name,
                dst=self.rank,
                sources=len(sources),
            )
        ):
            return original_wait_layer(self, layer_id)

    @wraps(original_finish_step)
    def profiled_finish_step(self):
        epoch = self._epoch
        with pcp_nvtx_range(
            pcp_nvtx_name("page_pull.step_finish", e=epoch, rank=self.rank)
        ):
            result = original_finish_step(self)
        if epoch:
            pcp_nvtx_mark(
                pcp_nvtx_name("page_pull.step_end", e=epoch, rank=self.rank)
            )
        return result

    transport_cls.configure_step = profiled_configure_step
    transport_cls.publish_ready = profiled_publish_ready
    transport_cls._start_read = profiled_start_read
    transport_cls._progress_once = profiled_progress_once
    transport_cls._progress_loop = profiled_progress_loop
    transport_cls.wait_layer = profiled_wait_layer
    transport_cls.finish_step = profiled_finish_step


def install_pcp_nvtx_hooks() -> None:
    """Install detailed PCP ranges when ``VLLM_PCP_NVTX`` is enabled.

    The hooks are process-global and installed once. They remain opt-in so
    normal benchmark runs do not add per-layer Python wrappers or progress
    bookkeeping to PCP's hot path.
    """
    global _PCP_NVTX_HOOKS_INSTALLED
    if not _PCP_NVTX_ENABLED or _PCP_NVTX_HOOKS_INSTALLED:
        return

    _install_flash_attention_hooks()
    _install_page_pull_hooks()
    _PCP_NVTX_HOOKS_INSTALLED = True


__all__ = [
    "install_pcp_nvtx_hooks",
    "pcp_nvtx_enabled",
    "pcp_nvtx_mark",
    "pcp_nvtx_name",
    "pcp_nvtx_range",
]

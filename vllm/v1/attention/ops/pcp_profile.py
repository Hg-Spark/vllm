# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small opt-in NVTX helpers for PCP execution paths.

PCP profiling is instrumented at the relevant call sites. Keeping this module
free of monkey patches avoids process-global wrappers and keeps profiling from
changing transport scheduling or adding synchronization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import torch

from vllm import envs

_PCP_NVTX_ENABLED = envs.VLLM_PCP_NVTX
_P = ParamSpec("_P")
_R = TypeVar("_R")


def pcp_nvtx_enabled() -> bool:
    return _PCP_NVTX_ENABLED


def pcp_nvtx_name(event: str, **fields: object) -> str:
    prefix = event if event.startswith("pcp.") else f"pcp.{event}"
    payload = ",".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    return prefix if not payload else f"{prefix}[{payload}]"


@contextmanager
def pcp_nvtx_range(name: str, **fields: object) -> Iterator[None]:
    if not _PCP_NVTX_ENABLED:
        yield
        return
    if fields:
        name = pcp_nvtx_name(name, **fields)
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def pcp_nvtx_mark(name: str, **fields: object) -> None:
    if not _PCP_NVTX_ENABLED:
        return
    if fields:
        name = pcp_nvtx_name(name, **fields)
    torch.cuda.nvtx.mark(name)


def pcp_nvtx_layer_forward(
    func: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Wrap a decoder-layer forward call with PCP-aware NVTX metadata.

    The decorator is a no-op when PCP NVTX profiling is disabled, so normal
    execution does not pay a per-layer wrapper cost. Runtime imports are lazy
    to avoid introducing model/attention import cycles.
    """
    if not _PCP_NVTX_ENABLED:
        return func

    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        layer = args[0] if args else None
        fields: dict[str, Any] = {
            "l": getattr(layer, "layer_idx", None),
            "transport": "baseline",
        }

        from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime

        runtime = get_pcp_runahead_runtime()
        if runtime is not None:
            fields.update(
                e=runtime.epoch,
                rank=runtime.rank,
                transport=runtime.transport,
            )
        else:
            try:
                from vllm.distributed.parallel_state import get_pcp_group

                fields["rank"] = get_pcp_group().rank_in_group
            except (AssertionError, RuntimeError):
                pass

        with pcp_nvtx_range("pcp.layer", **fields):
            return func(*args, **kwargs)

    return cast(Callable[_P, _R], wrapped)


__all__ = [
    "pcp_nvtx_enabled",
    "pcp_nvtx_layer_forward",
    "pcp_nvtx_mark",
    "pcp_nvtx_name",
    "pcp_nvtx_range",
]

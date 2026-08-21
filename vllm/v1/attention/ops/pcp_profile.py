# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small opt-in NVTX helpers for PCP execution paths.

PCP profiling is instrumented at the relevant call sites. Keeping this module
free of monkey patches avoids process-global wrappers and keeps profiling from
changing transport scheduling or adding synchronization.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm import envs

_PCP_NVTX_ENABLED = envs.VLLM_PCP_NVTX


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


__all__ = [
    "pcp_nvtx_enabled",
    "pcp_nvtx_mark",
    "pcp_nvtx_name",
    "pcp_nvtx_range",
]

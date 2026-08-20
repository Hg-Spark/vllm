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

import torch

_PCP_NVTX_ENABLED = os.environ.get("VLLM_PCP_NVTX", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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

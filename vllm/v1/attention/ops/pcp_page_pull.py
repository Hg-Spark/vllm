# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility boundary for the historical ``page_pull`` transport name."""

from typing import Any

import torch

from vllm.v1.attention.ops.pcp_page_plan import PCPPagePlan
from vllm.v1.attention.ops.pcp_page_push_impl import PCPPagePushTransport


class PCPPagePullTransport(PCPPagePushTransport):
    """Map the legacy READ-era constructor name onto producer-push semantics."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
        max_inflight_reads: int = 4,
        nixl_backends: tuple[str, ...] = ("UCX",),
        pcp_group: Any | None = None,
        static_forward_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            world_size=world_size,
            rank=rank,
            device=device,
            max_inflight_writes=max_inflight_reads,
            nixl_backends=nixl_backends,
            pcp_group=pcp_group,
            static_forward_context=static_forward_context,
        )


__all__ = ["PCPPagePlan", "PCPPagePullTransport"]

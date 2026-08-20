# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration parsing for the experimental PCP runahead path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TransportPolicy = Literal["full_kv_collective", "prefix_p2p"]
PartitionPolicy = Literal["stock", "equal_contiguous", "weighted_contiguous"]
LayoutPolicy = Literal["padded", "compact"]

RUNAHEAD_CONFIG_KEY = "pcp_runahead"
RUNAHEAD_WEIGHTS_KEY = "pcp_runahead_weights"
RUNAHEAD_MIN_PREFILL_TOKENS = 1024


@dataclass(frozen=True)
class PCPRunaheadConfig:
    transport: TransportPolicy = "prefix_p2p"
    partition_policy: PartitionPolicy = "equal_contiguous"
    weights: tuple[float, ...] | None = None
    page_align: bool = True
    layout: LayoutPolicy = "compact"
    require_full_prefill: bool = True
    min_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS
    max_inflight_sends: int = 4


def parse_runahead_weights(
    raw: object,
    pcp_world_size: int,
) -> tuple[float, ...] | None:
    """Parse manually supplied positive per-rank load weights."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"weights must be a JSON list with {pcp_world_size} positive numbers"
        )
    if len(raw) != pcp_world_size:
        raise ValueError(
            f"weights requires {pcp_world_size} values, got {len(raw)}: {raw}"
        )
    try:
        weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weights values must be numeric: {raw}") from exc
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError(f"weights values must be finite and positive: {weights}")
    return weights


def _mapping(raw: object, name: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object")
    return raw


def parse_pcp_runahead_config(
    additional_config: object,
    pcp_world_size: int,
) -> PCPRunaheadConfig | None:
    """Parse the nested ``pcp_runahead`` experiment configuration."""
    if not isinstance(additional_config, dict):
        return None

    if RUNAHEAD_WEIGHTS_KEY in additional_config:
        raise ValueError(
            f"{RUNAHEAD_WEIGHTS_KEY} is not supported; use "
            f"{RUNAHEAD_CONFIG_KEY}.partition.weights"
        )

    raw = additional_config.get(RUNAHEAD_CONFIG_KEY, False)
    if raw in (False, None):
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{RUNAHEAD_CONFIG_KEY} must be a JSON object")
    if not raw:
        raise ValueError(
            f"{RUNAHEAD_CONFIG_KEY} must be non-empty when used as an object"
        )
    if "enabled" in raw:
        raise ValueError(
            f"{RUNAHEAD_CONFIG_KEY}.enabled is not used; omit the object to disable"
        )

    partition = _mapping(raw.get("partition"), f"{RUNAHEAD_CONFIG_KEY}.partition")
    eligibility = _mapping(
        raw.get("eligibility"), f"{RUNAHEAD_CONFIG_KEY}.eligibility"
    )
    runtime = _mapping(raw.get("runtime"), f"{RUNAHEAD_CONFIG_KEY}.runtime")

    transport = raw.get("transport", "prefix_p2p")
    layout = raw.get("layout", "compact")
    weights = parse_runahead_weights(partition.get("weights"), pcp_world_size)
    partition_policy = partition.get(
        "policy", "weighted_contiguous" if weights is not None else "equal_contiguous"
    )
    page_align = partition.get("page_align", True)
    require_full_prefill = eligibility.get("require_full_prefill", True)
    min_tokens = eligibility.get("min_tokens", RUNAHEAD_MIN_PREFILL_TOKENS)
    max_inflight_sends = runtime.get("max_inflight_sends", 4)

    if transport not in ("full_kv_collective", "prefix_p2p"):
        raise ValueError(f"unsupported PCP runahead transport: {transport!r}")
    if partition_policy not in (
        "stock",
        "equal_contiguous",
        "weighted_contiguous",
    ):
        raise ValueError(f"unsupported PCP partition policy: {partition_policy!r}")
    if layout not in ("padded", "compact"):
        raise ValueError(f"unsupported PCP layout: {layout!r}")
    if not isinstance(page_align, bool):
        raise ValueError("partition.page_align must be boolean")
    if not isinstance(require_full_prefill, bool):
        raise ValueError("eligibility.require_full_prefill must be boolean")
    if (
        not isinstance(min_tokens, int)
        or isinstance(min_tokens, bool)
        or min_tokens < 0
    ):
        raise ValueError("eligibility.min_tokens must be a non-negative integer")
    if (
        not isinstance(max_inflight_sends, int)
        or isinstance(max_inflight_sends, bool)
        or max_inflight_sends <= 0
    ):
        raise ValueError("runtime.max_inflight_sends must be a positive integer")

    if partition_policy == "weighted_contiguous" and weights is None:
        raise ValueError("weighted_contiguous partition requires weights")
    if partition_policy != "weighted_contiguous" and weights is not None:
        raise ValueError(
            f"weights are only valid with weighted_contiguous, got {partition_policy}"
        )
    if partition_policy == "stock":
        if transport != "full_kv_collective" or layout != "padded":
            raise ValueError(
                "stock partition is only supported with "
                "transport=full_kv_collective and layout=padded"
            )
    if transport == "prefix_p2p":
        if partition_policy == "stock":
            raise ValueError("prefix_p2p requires a contiguous partition")
        if layout != "compact":
            raise ValueError("prefix_p2p requires layout=compact")
        if not require_full_prefill:
            raise ValueError(
                "prefix_p2p currently requires eligibility.require_full_prefill=true"
            )

    return PCPRunaheadConfig(
        transport=transport,
        partition_policy=partition_policy,
        weights=weights,
        page_align=page_align,
        layout=layout,
        require_full_prefill=require_full_prefill,
        min_tokens=min_tokens,
        max_inflight_sends=max_inflight_sends,
    )

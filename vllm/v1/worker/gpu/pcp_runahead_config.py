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
    segment_to_rank: tuple[int, ...] = ()
    page_align: bool = True
    layout: LayoutPolicy = "compact"
    require_full_prefill: bool = True
    min_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS
    max_inflight_sends: int = 4


def parse_runahead_weights(
    raw: object,
    pcp_world_size: int,
) -> tuple[float, ...] | None:
    """Parse manually supplied positive per-segment load weights."""
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


def parse_runahead_segments(
    raw: object,
    pcp_world_size: int,
) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    """Parse logical segments and their physical PCP-rank bindings."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"segments must be a JSON list with {pcp_world_size} objects"
        )
    if len(raw) != pcp_world_size:
        raise ValueError(
            f"segments requires {pcp_world_size} entries, got {len(raw)}: {raw}"
        )

    weights: list[float] = []
    segment_to_rank: list[int] = []
    for segment_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"segments[{segment_idx}] must be a JSON object")
        unknown = set(item) - {"weight", "pcp_rank"}
        if unknown:
            raise ValueError(
                f"segments[{segment_idx}] has unsupported keys: {sorted(unknown)}"
            )
        if "pcp_rank" not in item:
            raise ValueError(f"segments[{segment_idx}].pcp_rank is required")

        weight_raw = item.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"segments[{segment_idx}].weight must be numeric: {weight_raw!r}"
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"segments[{segment_idx}].weight must be finite and positive: "
                f"{weight!r}"
            )

        pcp_rank = item["pcp_rank"]
        if not isinstance(pcp_rank, int) or isinstance(pcp_rank, bool):
            raise ValueError(
                f"segments[{segment_idx}].pcp_rank must be an integer: {pcp_rank!r}"
            )
        if not 0 <= pcp_rank < pcp_world_size:
            raise ValueError(
                f"segments[{segment_idx}].pcp_rank must be in "
                f"[0, {pcp_world_size}): {pcp_rank}"
            )

        weights.append(weight)
        segment_to_rank.append(pcp_rank)

    expected_ranks = list(range(pcp_world_size))
    if sorted(segment_to_rank) != expected_ranks:
        raise ValueError(
            "segments[].pcp_rank must be a permutation of PCP ranks "
            f"{expected_ranks}, got {segment_to_rank}"
        )
    return tuple(weights), tuple(segment_to_rank)


def invert_segment_to_rank(segment_to_rank: tuple[int, ...]) -> tuple[int, ...]:
    """Return physical-rank -> logical-segment mapping for a rank permutation."""
    rank_to_segment = [0] * len(segment_to_rank)
    for segment_idx, rank in enumerate(segment_to_rank):
        rank_to_segment[rank] = segment_idx
    return tuple(rank_to_segment)


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

    raw_weights = partition.get("weights")
    parsed_segments = parse_runahead_segments(
        partition.get("segments"), pcp_world_size
    )
    if raw_weights is not None and parsed_segments is not None:
        raise ValueError("partition.weights and partition.segments are mutually exclusive")

    if parsed_segments is not None:
        weights, segment_to_rank = parsed_segments
    else:
        weights = parse_runahead_weights(raw_weights, pcp_world_size)
        segment_to_rank = tuple(range(pcp_world_size))

    partition_policy = partition.get(
        "policy",
        "weighted_contiguous" if weights is not None else "equal_contiguous",
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
        raise ValueError("weighted_contiguous partition requires weights or segments")
    if partition_policy != "weighted_contiguous" and weights is not None:
        raise ValueError(
            "weights/segments are only valid with weighted_contiguous, got "
            f"{partition_policy}"
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
        segment_to_rank=segment_to_rank,
        page_align=page_align,
        layout=layout,
        require_full_prefill=require_full_prefill,
        min_tokens=min_tokens,
        max_inflight_sends=max_inflight_sends,
    )

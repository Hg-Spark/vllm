# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration parsing for the experimental PCP runahead path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

TransportPolicy = Literal[
    "full_kv_collective",
    "prefix_p2p",
    "direct_p2p",
    "page_pull",
]

RUNAHEAD_CONFIG_KEY = "pcp_runahead"
RUNAHEAD_WEIGHTS_KEY = "pcp_runahead_weights"
RUNAHEAD_MIN_PREFILL_TOKENS = 1024


@dataclass(frozen=True)
class PCPRunaheadConfig:
    """Canonical runahead configuration.

    Runahead always uses a compact contiguous partition of a fresh full
    prefill. One-segment-per-rank permutations are compiled into PCP process
    group ordering; repeated ownership is retained only for ``page_pull``.
    """

    pcp_world_size: int = 0
    transport: TransportPolicy = "prefix_p2p"
    weights: tuple[float, ...] = ()
    segment_to_rank: tuple[int, ...] = ()
    min_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS
    max_inflight_sends: int = 4
    max_inflight_reads: int = 4
    nixl_backends: tuple[str, ...] = ("UCX",)

    @property
    def num_segments(self) -> int:
        return len(self.segment_to_rank)

    @property
    def mapping_is_permutation(self) -> bool:
        return is_rank_permutation(self.segment_to_rank, self.pcp_world_size)

    @property
    def process_group_order(self) -> tuple[int, ...]:
        """Logical PCP rank -> original physical PCP rank for PG creation."""
        if self.mapping_is_permutation:
            return self.segment_to_rank
        return tuple(range(self.pcp_world_size))

    @property
    def runtime_segment_to_rank(self) -> tuple[int, ...]:
        """Mapping visible after process-group ordering has been applied."""
        if self.mapping_is_permutation:
            return tuple(range(self.pcp_world_size))
        return self.segment_to_rank


def parse_runahead_weights(
    raw: object,
    pcp_world_size: int,
) -> tuple[float, ...] | None:
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
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("segments must be a JSON list of objects")
    if len(raw) < pcp_world_size:
        raise ValueError(
            "segments must contain at least one logical segment per PCP rank: "
            f"segments={len(raw)}, pcp_world_size={pcp_world_size}"
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

    expected = set(range(pcp_world_size))
    present = set(segment_to_rank)
    if present != expected:
        raise ValueError(
            "segments[].pcp_rank must cover every PCP rank at least once; "
            f"missing={sorted(expected - present)}, mapping={segment_to_rank}"
        )
    return tuple(weights), tuple(segment_to_rank)


def is_rank_permutation(
    segment_to_rank: tuple[int, ...], pcp_world_size: int
) -> bool:
    return len(segment_to_rank) == pcp_world_size and sorted(segment_to_rank) == list(
        range(pcp_world_size)
    )


def invert_segment_to_rank(segment_to_rank: tuple[int, ...]) -> tuple[int, ...]:
    if len(set(segment_to_rank)) != len(segment_to_rank):
        raise ValueError("segment_to_rank is not invertible because ranks repeat")
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


def _string_tuple(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{name} must be a non-empty JSON list of strings")
    values = tuple(raw)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain only non-empty strings: {raw}")
    return values


def parse_pcp_runahead_config(
    additional_config: object,
    pcp_world_size: int,
) -> PCPRunaheadConfig | None:
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

    unknown = set(raw) - {"transport", "partition", "eligibility", "runtime", "layout"}
    if unknown:
        raise ValueError(f"unsupported {RUNAHEAD_CONFIG_KEY} keys: {sorted(unknown)}")

    transport = raw.get("transport", "prefix_p2p")
    if transport not in (
        "full_kv_collective",
        "prefix_p2p",
        "direct_p2p",
        "page_pull",
    ):
        raise ValueError(f"unsupported PCP runahead transport: {transport!r}")

    partition = _mapping(raw.get("partition"), f"{RUNAHEAD_CONFIG_KEY}.partition")
    unknown_partition = set(partition) - {"segments", "weights", "policy", "page_align"}
    if unknown_partition:
        raise ValueError(
            f"unsupported {RUNAHEAD_CONFIG_KEY}.partition keys: "
            f"{sorted(unknown_partition)}"
        )

    raw_weights = partition.get("weights")
    parsed_segments = parse_runahead_segments(partition.get("segments"), pcp_world_size)
    if raw_weights is not None and parsed_segments is not None:
        raise ValueError("partition.weights and partition.segments are mutually exclusive")
    if parsed_segments is not None:
        weights, segment_to_rank = parsed_segments
    else:
        parsed_weights = parse_runahead_weights(raw_weights, pcp_world_size)
        weights = parsed_weights or (1.0,) * pcp_world_size
        segment_to_rank = tuple(range(pcp_world_size))

    # Compatibility validation for knobs that are now derived invariants.
    policy = partition.get("policy")
    if policy is not None and policy not in ("equal_contiguous", "weighted_contiguous"):
        raise ValueError(
            "runahead uses a compact contiguous partition; "
            f"unsupported partition.policy={policy!r}"
        )
    if policy == "equal_contiguous" and len(set(weights)) != 1:
        raise ValueError("equal_contiguous is incompatible with non-uniform weights")
    page_align = partition.get("page_align", True)
    if not isinstance(page_align, bool):
        raise ValueError("partition.page_align must be boolean")
    if transport == "page_pull" and not page_align:
        raise ValueError("page_pull requires partition.page_align=true")

    layout = raw.get("layout", "compact")
    if layout != "compact":
        raise ValueError("runahead always uses layout=compact")

    eligibility = _mapping(
        raw.get("eligibility"), f"{RUNAHEAD_CONFIG_KEY}.eligibility"
    )
    unknown_eligibility = set(eligibility) - {"require_full_prefill", "min_tokens"}
    if unknown_eligibility:
        raise ValueError(
            f"unsupported {RUNAHEAD_CONFIG_KEY}.eligibility keys: "
            f"{sorted(unknown_eligibility)}"
        )
    if eligibility.get("require_full_prefill", True) is not True:
        raise ValueError("runahead requires eligibility.require_full_prefill=true")
    min_tokens = eligibility.get("min_tokens", RUNAHEAD_MIN_PREFILL_TOKENS)
    if (
        not isinstance(min_tokens, int)
        or isinstance(min_tokens, bool)
        or min_tokens < 0
    ):
        raise ValueError("eligibility.min_tokens must be a non-negative integer")

    runtime = _mapping(raw.get("runtime"), f"{RUNAHEAD_CONFIG_KEY}.runtime")
    unknown_runtime = set(runtime) - {
        "max_inflight_sends",
        "max_inflight_reads",
        "nixl_backends",
        "page_pull_backend",
    }
    if unknown_runtime:
        raise ValueError(
            f"unsupported {RUNAHEAD_CONFIG_KEY}.runtime keys: {sorted(unknown_runtime)}"
        )
    if runtime.get("page_pull_backend", "nixl") != "nixl":
        raise ValueError("page_pull uses the NIXL backend")
    max_inflight_sends = runtime.get("max_inflight_sends", 4)
    max_inflight_reads = runtime.get("max_inflight_reads", 4)
    for name, value in (
        ("runtime.max_inflight_sends", max_inflight_sends),
        ("runtime.max_inflight_reads", max_inflight_reads),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    nixl_backends = _string_tuple(
        runtime.get("nixl_backends", ["UCX"]), "runtime.nixl_backends"
    )

    if transport != "page_pull" and not is_rank_permutation(
        segment_to_rank, pcp_world_size
    ):
        raise ValueError(
            f"transport={transport} requires one logical segment per PCP rank; "
            "repeated segment bindings require transport=page_pull"
        )

    return PCPRunaheadConfig(
        pcp_world_size=pcp_world_size,
        transport=transport,
        weights=weights,
        segment_to_rank=segment_to_rank,
        min_tokens=min_tokens,
        max_inflight_sends=max_inflight_sends,
        max_inflight_reads=max_inflight_reads,
        nixl_backends=nixl_backends,
    )

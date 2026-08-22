# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and rank binding for the experimental PCP runahead path."""

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


@dataclass(frozen=True)
class PCPBindingPlan:
    """Compiled rank coordinates for one PCP runahead configuration.

    ``segments[].pcp_rank`` is always a physical PCP rank. Tensor transports
    run in primary-PCP-group rank coordinates. For one-segment-per-rank
    permutations the primary group is reordered once at startup, so logical
    segment order becomes group-rank order. Repeated ownership keeps the
    primary group in physical order and remains explicit in the segment map.
    """

    segment_to_physical_rank: tuple[int, ...]
    group_rank_to_physical_rank: tuple[int, ...]
    physical_rank_to_group_rank: tuple[int, ...]
    segment_to_group_rank: tuple[int, ...]

    @property
    def pcp_world_size(self) -> int:
        return len(self.group_rank_to_physical_rank)

    @property
    def is_permutation(self) -> bool:
        return len(self.segment_to_physical_rank) == self.pcp_world_size

    @property
    def reorders_primary_group(self) -> bool:
        return self.group_rank_to_physical_rank != tuple(range(self.pcp_world_size))


def is_rank_permutation(
    segment_to_physical_rank: tuple[int, ...], pcp_world_size: int
) -> bool:
    return len(segment_to_physical_rank) == pcp_world_size and sorted(
        segment_to_physical_rank
    ) == list(range(pcp_world_size))


def compile_pcp_binding(
    segment_to_physical_rank: tuple[int, ...],
    pcp_world_size: int,
) -> PCPBindingPlan:
    """Compile physical ownership into stable PCP-group coordinates."""
    if pcp_world_size <= 0:
        raise ValueError("pcp_world_size must be positive")
    if not segment_to_physical_rank:
        raise ValueError("PCP binding requires at least one logical segment")
    if any(
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < pcp_world_size
        for rank in segment_to_physical_rank
    ):
        raise ValueError(
            "segment physical ranks must be integers in "
            f"[0, {pcp_world_size}): {segment_to_physical_rank}"
        )

    missing = set(range(pcp_world_size)) - set(segment_to_physical_rank)
    if missing:
        raise ValueError(
            "segment physical ranks must cover every PCP rank; "
            f"missing={sorted(missing)}"
        )

    identity = tuple(range(pcp_world_size))
    if is_rank_permutation(segment_to_physical_rank, pcp_world_size):
        group_rank_to_physical_rank = segment_to_physical_rank
        physical_rank_to_group_rank = [0] * pcp_world_size
        for group_rank, physical_rank in enumerate(group_rank_to_physical_rank):
            physical_rank_to_group_rank[physical_rank] = group_rank
        segment_to_group_rank = identity
    else:
        group_rank_to_physical_rank = identity
        physical_rank_to_group_rank = list(identity)
        segment_to_group_rank = segment_to_physical_rank

    return PCPBindingPlan(
        segment_to_physical_rank=segment_to_physical_rank,
        group_rank_to_physical_rank=group_rank_to_physical_rank,
        physical_rank_to_group_rank=tuple(physical_rank_to_group_rank),
        segment_to_group_rank=segment_to_group_rank,
    )


@dataclass(frozen=True)
class PCPRunaheadConfig:
    """Runtime axes that remain variable in PCP runahead."""

    transport: TransportPolicy
    weights: tuple[float, ...]
    binding: PCPBindingPlan
    max_inflight_sends: int = 4
    max_inflight_reads: int = 4
    nixl_backends: tuple[str, ...] = ("UCX",)

    @property
    def pcp_world_size(self) -> int:
        return self.binding.pcp_world_size

    @property
    def segment_to_physical_rank(self) -> tuple[int, ...]:
        return self.binding.segment_to_physical_rank

    @property
    def segment_to_group_rank(self) -> tuple[int, ...]:
        return self.binding.segment_to_group_rank

    @property
    def pcp_group_order(self) -> tuple[int, ...]:
        return self.binding.group_rank_to_physical_rank


def parse_runahead_weights(
    raw: object,
    pcp_world_size: int,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != pcp_world_size:
        got = len(raw) if isinstance(raw, (list, tuple)) else type(raw).__name__
        raise ValueError(
            f"weights requires {pcp_world_size} positive values, got {got}: {raw}"
        )
    try:
        weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weights values must be numeric: {raw}") from exc
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(f"weights values must be finite and positive: {weights}")
    return weights


def parse_runahead_segments(
    raw: object,
    pcp_world_size: int,
) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < pcp_world_size:
        raise ValueError(
            "segments must be a JSON list with at least one segment per PCP rank"
        )

    weights: list[float] = []
    owners: list[int] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) - {"weight", "pcp_rank"}:
            raise ValueError(
                f"segments[{index}] must contain only weight and pcp_rank"
            )
        if "pcp_rank" not in item:
            raise ValueError(f"segments[{index}].pcp_rank is required")
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"segments[{index}].weight must be numeric") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"segments[{index}].weight must be finite and positive")
        rank = item["pcp_rank"]
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < pcp_world_size
        ):
            raise ValueError(
                f"segments[{index}].pcp_rank must be in [0, {pcp_world_size})"
            )
        weights.append(weight)
        owners.append(rank)

    return tuple(weights), tuple(owners)


def _object(raw: object, name: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object")
    return raw


def _positive_int(raw: object, name: str, default: int) -> int:
    value = default if raw is None else raw
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_pcp_runahead_config(
    additional_config: object,
    pcp_world_size: int,
) -> PCPRunaheadConfig | None:
    if not isinstance(additional_config, dict):
        return None
    if "pcp_runahead_weights" in additional_config:
        raise ValueError(
            "pcp_runahead_weights was removed; use pcp_runahead.partition.weights"
        )
    raw = additional_config.get(RUNAHEAD_CONFIG_KEY)
    if raw in (None, False):
        return None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{RUNAHEAD_CONFIG_KEY} must be a non-empty JSON object")
    unknown = set(raw) - {"transport", "partition", "runtime"}
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

    partition = _object(raw.get("partition"), "pcp_runahead.partition")
    unknown = set(partition) - {"weights", "segments"}
    if unknown:
        raise ValueError(f"unsupported partition keys: {sorted(unknown)}")
    parsed_segments = parse_runahead_segments(partition.get("segments"), pcp_world_size)
    if parsed_segments is not None and "weights" in partition:
        raise ValueError("partition.weights and partition.segments are mutually exclusive")
    if parsed_segments is None:
        weights = parse_runahead_weights(partition.get("weights"), pcp_world_size)
        weights = weights or (1.0,) * pcp_world_size
        segment_to_physical_rank = tuple(range(pcp_world_size))
    else:
        weights, segment_to_physical_rank = parsed_segments

    binding = compile_pcp_binding(segment_to_physical_rank, pcp_world_size)
    if transport != "page_pull" and not binding.is_permutation:
        raise ValueError(
            f"transport={transport} requires one logical segment per PCP rank; "
            "repeated segment bindings require transport=page_pull"
        )

    runtime = _object(raw.get("runtime"), "pcp_runahead.runtime")
    unknown = set(runtime) - {
        "max_inflight_sends",
        "max_inflight_reads",
        "nixl_backends",
    }
    if unknown:
        raise ValueError(f"unsupported runtime keys: {sorted(unknown)}")
    max_inflight_sends = _positive_int(
        runtime.get("max_inflight_sends"), "runtime.max_inflight_sends", 4
    )
    max_inflight_reads = _positive_int(
        runtime.get("max_inflight_reads"), "runtime.max_inflight_reads", 4
    )
    backends_raw = runtime.get("nixl_backends", ["UCX"])
    if (
        not isinstance(backends_raw, (list, tuple))
        or not backends_raw
        or any(not isinstance(item, str) or not item for item in backends_raw)
    ):
        raise ValueError("runtime.nixl_backends must be a non-empty string list")

    return PCPRunaheadConfig(
        transport=transport,
        weights=weights,
        binding=binding,
        max_inflight_sends=max_inflight_sends,
        max_inflight_reads=max_inflight_reads,
        nixl_backends=tuple(backends_raw),
    )


__all__ = [
    "PCPBindingPlan",
    "PCPRunaheadConfig",
    "RUNAHEAD_CONFIG_KEY",
    "TransportPolicy",
    "compile_pcp_binding",
    "is_rank_permutation",
    "parse_pcp_runahead_config",
    "parse_runahead_segments",
    "parse_runahead_weights",
]

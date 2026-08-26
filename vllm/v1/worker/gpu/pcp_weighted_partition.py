# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import torch

from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_execution import PCPExecutionManager
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState


def weighted_partition_lengths(
    num_tokens: int,
    pcp_partition_weights: tuple[float, ...],
    *,
    start_pos: int = 0,
    alignment: int = 1,
) -> tuple[int, ...]:
    """Partition tokens with cumulative weighted, optionally page-aligned cuts."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if not pcp_partition_weights:
        raise ValueError("weighted PCP partition requires at least one weight")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")

    total_weight = sum(pcp_partition_weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(f"invalid PCP load weights: {pcp_partition_weights}")
    num_segments = len(pcp_partition_weights)
    if num_tokens == 0:
        return (0,) * num_segments

    if alignment == 1:
        ideal = [
            num_tokens * weight / total_weight
            for weight in pcp_partition_weights
        ]
        lengths = [math.floor(value) for value in ideal]
        remainder = num_tokens - sum(lengths)
        order = sorted(
            range(num_segments),
            key=lambda segment: (-(ideal[segment] - lengths[segment]), segment),
        )
        for segment in order[:remainder]:
            lengths[segment] += 1
        return tuple(lengths)

    require_positive = (
        start_pos % alignment == 0
        and num_tokens >= (num_segments - 1) * alignment + 1
    )
    boundaries = [0]
    cumulative_weight = 0.0
    for segment in range(num_segments - 1):
        cumulative_weight += pcp_partition_weights[segment]
        ideal_rel = num_tokens * cumulative_weight / total_weight
        ideal_abs = start_pos + ideal_rel

        if require_positive:
            min_cut = (segment + 1) * alignment
            remaining_segments = num_segments - segment - 1
            max_cut = num_tokens - ((remaining_segments - 1) * alignment + 1)
            candidates: set[int] = set()
        else:
            min_cut = boundaries[-1]
            max_cut = num_tokens
            candidates = {boundaries[-1], num_tokens}

        lower_abs = math.floor(ideal_abs / alignment) * alignment
        upper_abs = math.ceil(ideal_abs / alignment) * alignment
        min_abs = math.ceil((start_pos + min_cut) / alignment) * alignment
        max_abs = math.floor((start_pos + max_cut) / alignment) * alignment
        for candidate_abs in (lower_abs, upper_abs, min_abs, max_abs):
            candidate_rel = int(candidate_abs - start_pos)
            if min_cut <= candidate_rel <= max_cut:
                candidates.add(candidate_rel)

        if not candidates:
            raise AssertionError(
                "PCP page-aligned partition has no legal cumulative boundary"
            )
        boundary = min(candidates, key=lambda cut: (abs(cut - ideal_rel), cut))
        boundaries.append(boundary)

    boundaries.append(num_tokens)
    return tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(num_segments)
    )


def parse_pcp_partition_weights(
    additional_config: object,
    pcp_world_size: int,
) -> tuple[float, ...]:
    """Read the optional top-level PCP rank-weight array.

    PCP owns only ``pcp_partition_weights``. Other additional-config keys are
    ignored so unrelated or future configuration does not become a PCP error.
    """
    default = (1.0,) * pcp_world_size
    if not isinstance(additional_config, dict):
        return default

    raw = additional_config.get("pcp_partition_weights")
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != pcp_world_size:
        got = len(raw) if isinstance(raw, (list, tuple)) else type(raw).__name__
        raise ValueError(
            f"pcp_partition_weights requires {pcp_world_size} positive values, "
            f"got {got}: {raw}"
        )
    try:
        pcp_partition_weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pcp_partition_weights must be numeric: {raw}") from exc
    if any(
        not math.isfinite(weight) or weight <= 0
        for weight in pcp_partition_weights
    ):
        raise ValueError(
            "pcp_partition_weights must contain finite positive values: "
            f"{pcp_partition_weights}"
        )
    return pcp_partition_weights


class WeightedPCPManager(PCPExecutionManager):
    """One causal contiguous prefill slice per PCP rank with optional weights."""

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
        pcp_partition_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        self._pcp_partition_weights = (
            (1.0,) * pcp_world_size
            if pcp_partition_weights is None
            else pcp_partition_weights
        )
        if len(self._pcp_partition_weights) != pcp_world_size:
            raise ValueError(
                "PCP partition weights must match PCP world size: "
                f"weights={self._pcp_partition_weights}, world_size={pcp_world_size}"
            )
        self._page_alignment = (
            math.lcm(*(int(size) for size in block_tables.kernel_block_sizes))
            if block_tables is not None and block_tables.kernel_block_sizes
            else 1
        )

    def _partition_lengths(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> tuple[int, ...]:
        alignment = self._page_alignment
        if query_len < self.pcp_world_size * alignment:
            alignment = 1
        return weighted_partition_lengths(
            query_len,
            self._pcp_partition_weights,
            start_pos=num_computed_tokens,
            alignment=alignment,
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        rank_segments = []
        rank_offset = 0
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            if bool(is_prefilling[global_batch_req_idx]):
                chunk_lengths = self._partition_lengths(
                    query_len,
                    int(num_computed_tokens[global_batch_req_idx]),
                )
                chunk_offsets = _chunk_offsets(chunk_lengths)
                chunk_indices = (rank,)
            else:
                chunk_lengths = (query_len,)
                chunk_offsets = (0,)
                chunk_indices = (0,)

            for chunk_idx in chunk_indices:
                chunk_offset = chunk_offsets[chunk_idx]
                chunk_len = chunk_lengths[chunk_idx]
                if chunk_len <= 0:
                    continue
                chunk_start = global_batch_start + chunk_offset
                rank_segments.append(
                    RankSegment(
                        global_batch_req_idx=global_batch_req_idx,
                        global_batch_slice=slice(chunk_start, chunk_start + chunk_len),
                        rank_local_batch_slice=slice(
                            rank_offset, rank_offset + chunk_len
                        ),
                    )
                )
                rank_offset += chunk_len
        return self._reorder_segments(
            rank_segments,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )


def _chunk_offsets(chunk_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    running = 0
    for chunk_len in chunk_lengths:
        offsets.append(running)
        running += chunk_len
    return tuple(offsets)

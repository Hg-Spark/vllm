# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_execution import PCPExecutionPlanner
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState


def weighted_partition_lengths(
    num_tokens: int,
    pcp_partition_weights: tuple[float, ...],
    *,
    start_pos: int = 0,
    alignment: int = 1,
) -> tuple[int, ...]:
    """Partition tokens with cumulative weighted, optionally aligned cuts."""
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
        ideal = [num_tokens * weight / total_weight for weight in pcp_partition_weights]
        lengths = [math.floor(value) for value in ideal]
        remainder = num_tokens - sum(lengths)
        order = sorted(
            range(num_segments),
            key=lambda index: (-(ideal[index] - lengths[index]), index),
        )
        for index in order[:remainder]:
            lengths[index] += 1
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
            raise AssertionError("PCP page-aligned partition has no legal boundary")
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
        weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pcp_partition_weights must be numeric: {raw}") from exc
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(
            "pcp_partition_weights must contain finite positive values: "
            f"{weights}"
        )
    return weights


class WeightedPCPManager(PCPExecutionPlanner):
    """Causal contiguous prefill slices with decode owned by the last PCP rank."""

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
        self._selected_candidate_buffer: torch.Tensor | None = None

    def restore_selected_hidden_states(
        self,
        hidden_states: torch.Tensor,
        global_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Restore only selected global rows from the weighted slab layout."""
        plan = self.batch_plan
        if plan is None:
            return hidden_states[global_indices]
        if global_indices.numel() == 0 or plan.rank_slab_width == 0:
            return hidden_states[:0]
        if hidden_states.shape[0] < plan.owned_num_tokens:
            raise RuntimeError(
                "PCP hidden-state rows are smaller than owned token count: "
                f"{hidden_states.shape[0]} < {plan.owned_num_tokens}"
            )

        selected_gather_idx = plan.hidden_restore_idx[global_indices]
        owner_rank = torch.div(
            selected_gather_idx,
            plan.rank_slab_width,
            rounding_mode="floor",
        )
        local_idx = torch.remainder(selected_gather_idx, plan.rank_slab_width)
        num_selected = global_indices.numel()
        required_shape = (num_selected, *hidden_states.shape[1:])

        buffer = self._selected_candidate_buffer
        if (
            buffer is None
            or buffer.device != hidden_states.device
            or buffer.dtype != hidden_states.dtype
            or buffer.shape[1:] != hidden_states.shape[1:]
            or buffer.shape[0] < num_selected
        ):
            buffer = hidden_states.new_empty(required_shape)
            self._selected_candidate_buffer = buffer
        local_candidates = buffer[:num_selected]
        local_candidates.zero_()

        owner_mask = owner_rank == self.pcp_rank
        local_candidates[owner_mask] = hidden_states[local_idx[owner_mask]]

        gathered_candidates = get_pcp_group().all_gather(local_candidates, dim=0)
        selected_rows = owner_rank * num_selected + torch.arange(
            num_selected,
            dtype=owner_rank.dtype,
            device=owner_rank.device,
        )
        return gathered_candidates[selected_rows]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
        force_full: bool = False,
    ):
        if self.pcp_rank == 0:
            from vllm.model_executor.layers.attention.pcp_wavefront_runtime import (
                flush_pending_sends,
            )

            flush_pending_sends()
        if self.batch_plan is None:
            force_full = True
        return super().restore_for_sampling(hidden_states, force_full=force_full)

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

    def _get_segments_by_rank(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[tuple[RankSegment, ...], ...]:
        segments_by_rank: list[list[RankSegment]] = [
            [] for _ in range(self.pcp_world_size)
        ]
        rank_offsets = [0] * self.pcp_world_size

        for global_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_start = int(query_start_loc_np[global_req_idx])

            if bool(is_prefilling[global_req_idx]):
                lengths = self._partition_lengths(
                    query_len, int(num_computed_tokens[global_req_idx])
                )
                query_offset = 0
                for rank, length in enumerate(lengths):
                    if length > 0:
                        segments_by_rank[rank].append(
                            RankSegment(
                                global_batch_req_idx=global_req_idx,
                                global_batch_slice=slice(
                                    global_start + query_offset,
                                    global_start + query_offset + length,
                                ),
                                rank_local_batch_slice=slice(
                                    rank_offsets[rank], rank_offsets[rank] + length
                                ),
                            )
                        )
                        rank_offsets[rank] += length
                    query_offset += length
            else:
                rank = self.pcp_world_size - 1
                segments_by_rank[rank].append(
                    RankSegment(
                        global_batch_req_idx=global_req_idx,
                        global_batch_slice=slice(global_start, global_start + query_len),
                        rank_local_batch_slice=slice(
                            rank_offsets[rank], rank_offsets[rank] + query_len
                        ),
                    )
                )
                rank_offsets[rank] += query_len

        return tuple(
            tuple(
                self._reorder_segments(
                    segments,
                    num_computed_tokens,
                    is_prefilling,
                    query_start_loc_np,
                )
            )
            for segments in segments_by_rank
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        return list(
            self._get_segments_by_rank(
                num_scheduled_tokens,
                num_computed_tokens,
                is_prefilling,
                query_start_loc_np,
            )[rank]
        )

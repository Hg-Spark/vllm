# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import torch

from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_execution import PCPExecutionPlanner
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState


def weighted_partition_lengths(
    num_tokens: int,
    pcp_partition_weights: tuple[float, ...],
) -> tuple[int, ...]:
    """Split tokens by positive weights with deterministic largest remainders."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if not pcp_partition_weights:
        raise ValueError("weighted PCP partition requires at least one weight")

    total_weight = sum(pcp_partition_weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(f"invalid PCP load weights: {pcp_partition_weights}")
    if num_tokens == 0:
        return (0,) * len(pcp_partition_weights)

    ideal = [num_tokens * weight / total_weight for weight in pcp_partition_weights]
    lengths = [math.floor(value) for value in ideal]
    remainder = num_tokens - sum(lengths)
    order = sorted(
        range(len(lengths)),
        key=lambda index: (-(ideal[index] - lengths[index]), index),
    )
    for index in order[:remainder]:
        lengths[index] += 1
    return tuple(lengths)


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

    def _partition_lengths(self, query_len: int) -> tuple[int, ...]:
        return weighted_partition_lengths(query_len, self._pcp_partition_weights)

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
                lengths = self._partition_lengths(query_len)
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

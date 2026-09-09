# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execution layout contract for rank-local PCP policies."""

from dataclasses import dataclass

import numpy as np

from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment


@dataclass(frozen=True)
class PCPBatchPlan:
    """One-step semantic and communication widths for rank-local PCP."""

    segments_by_rank: tuple[tuple[RankSegment, ...], ...]
    per_rank_num_tokens: tuple[int, ...]
    local_segments: tuple[RankSegment, ...]
    owned_num_tokens: int
    model_num_rows: int
    rank_slab_width: int

    @property
    def uses_dummy_execution_row(self) -> bool:
        return self.owned_num_tokens == 0 and self.model_num_rows == 1


class PCPExecutionPlanner(PCPManager):
    """Materialize PCP execution from a rank-partition policy.

    Partition policies may override ``_get_segments_by_rank`` to compute all
    rank ownership in one pass. The default implementation preserves the
    existing ``_get_rank_segments`` interface used by canonical PCP.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_plan: PCPBatchPlan | None = None

    @property
    def batch_plan(self) -> PCPBatchPlan | None:
        return self._batch_plan

    def _get_segments_by_rank(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[tuple[RankSegment, ...], ...]:
        return tuple(
            tuple(
                self._get_rank_segments(
                    rank,
                    num_scheduled_tokens,
                    num_computed_tokens,
                    is_prefilling,
                    query_start_loc_np,
                )
            )
            for rank in range(self.pcp_world_size)
        )

    def _build_batch_plan(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> PCPBatchPlan:
        segments_by_rank = self._get_segments_by_rank(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )
        per_rank_num_tokens = tuple(
            sum(segment.num_tokens for segment in segments)
            for segments in segments_by_rank
        )
        rank_slab_width = max(per_rank_num_tokens, default=0)
        owned_num_tokens = per_rank_num_tokens[self.pcp_rank]
        model_num_rows = (
            owned_num_tokens
            if owned_num_tokens > 0
            else (1 if rank_slab_width > 0 else 0)
        )

        plan = PCPBatchPlan(
            segments_by_rank=segments_by_rank,
            per_rank_num_tokens=per_rank_num_tokens,
            local_segments=segments_by_rank[self.pcp_rank],
            owned_num_tokens=owned_num_tokens,
            model_num_rows=model_num_rows,
            rank_slab_width=rank_slab_width,
        )
        self._batch_plan = plan
        return plan

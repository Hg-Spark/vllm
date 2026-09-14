# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execution planning and slab mapping for rank-local PCP."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment


def _model_num_rows(owned_num_tokens: int, rank_slab_width: int) -> int:
    """Return the existing rank-local execution width, including dummy-row use."""
    return (
        owned_num_tokens
        if owned_num_tokens > 0
        else (1 if rank_slab_width > 0 else 0)
    )


@dataclass(frozen=True)
class PCPBatchPlan:
    """One-step execution and communication-slab layout for rank-local PCP."""

    segments_by_rank: tuple[tuple[RankSegment, ...], ...]
    per_rank_num_tokens: tuple[int, ...]
    local_segments: tuple[RankSegment, ...]
    owned_num_tokens: int
    model_num_rows: int
    rank_slab_width: int
    slab_global_idx: torch.Tensor
    kv_write_mask: torch.Tensor
    hidden_restore_idx: torch.Tensor

    @property
    def uses_dummy_execution_row(self) -> bool:
        return self.owned_num_tokens == 0 and self.model_num_rows == 1


class PCPExecutionPlanner(PCPManager):
    """Plan rank-local execution while reusing ``PCPManager`` materialization.

    Partition policies may override ``_get_segments_by_rank`` to compute all
    rank ownership in one pass. The base ``PCPManager.partition_batch`` owns
    InputBatch materialization; this class only supplies Wavefront's execution
    width, local input indices, communication slab, and restore mappings.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_plan: PCPBatchPlan | None = None
        self._layout_token_capacity = 0

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

    def _ensure_layout_scratch(self, global_num_tokens: int) -> None:
        input_buffers = getattr(self, "_input_buffers", None)
        configured_tokens = (
            input_buffers.max_num_tokens if input_buffers is not None else 0
        )
        token_capacity = max(global_num_tokens, configured_tokens)
        if token_capacity <= self._layout_token_capacity:
            return

        self._layout_token_capacity = token_capacity
        slab_capacity = token_capacity * self.pcp_world_size
        self._slab_global_idx_np = np.empty(slab_capacity, dtype=np.int64)
        self._kv_write_mask_np = np.empty(slab_capacity, dtype=np.bool_)
        self._hidden_restore_idx_np = np.empty(token_capacity, dtype=np.int64)

        device = getattr(self, "device", torch.device("cpu"))
        self._slab_global_idx_gpu = torch.empty(
            slab_capacity, dtype=torch.int64, device=device
        )
        self._kv_write_mask_gpu = torch.empty(
            slab_capacity, dtype=torch.bool, device=device
        )
        self._hidden_restore_idx_gpu = torch.empty(
            token_capacity, dtype=torch.int64, device=device
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
        model_num_rows = _model_num_rows(owned_num_tokens, rank_slab_width)

        global_num_tokens = int(query_start_loc_np[-1])
        self._ensure_layout_scratch(global_num_tokens)
        num_slab_rows = rank_slab_width * self.pcp_world_size
        slab_global_idx_np = self._slab_global_idx_np[:num_slab_rows]
        kv_write_mask_np = self._kv_write_mask_np[:num_slab_rows]
        hidden_restore_idx_np = self._hidden_restore_idx_np[:global_num_tokens]
        slab_global_idx_np.fill(0)
        kv_write_mask_np.fill(False)

        for rank, segments in enumerate(segments_by_rank):
            rank_offset = rank * rank_slab_width
            for segment in segments:
                slab_slice = slice(
                    rank_offset + segment.rank_local_batch_slice.start,
                    rank_offset + segment.rank_local_batch_slice.stop,
                )
                global_slice = segment.global_batch_slice
                slab_global_idx_np[slab_slice] = np.arange(
                    global_slice.start, global_slice.stop, dtype=np.int64
                )
                kv_write_mask_np[slab_slice] = True
                hidden_restore_idx_np[global_slice] = np.arange(
                    slab_slice.start, slab_slice.stop, dtype=np.int64
                )

        slab_global_idx = self._slab_global_idx_gpu[:num_slab_rows]
        kv_write_mask = self._kv_write_mask_gpu[:num_slab_rows]
        hidden_restore_idx = self._hidden_restore_idx_gpu[:global_num_tokens]
        async_copy_to_gpu(slab_global_idx_np, out=slab_global_idx)
        async_copy_to_gpu(kv_write_mask_np, out=kv_write_mask)
        async_copy_to_gpu(hidden_restore_idx_np, out=hidden_restore_idx)

        plan = PCPBatchPlan(
            segments_by_rank=segments_by_rank,
            per_rank_num_tokens=per_rank_num_tokens,
            local_segments=segments_by_rank[self.pcp_rank],
            owned_num_tokens=owned_num_tokens,
            model_num_rows=model_num_rows,
            rank_slab_width=rank_slab_width,
            slab_global_idx=slab_global_idx,
            kv_write_mask=kv_write_mask,
            hidden_restore_idx=hidden_restore_idx,
        )
        self._batch_plan = plan
        return plan

    def _build_batch_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[list[list[RankSegment]], list[int]]:
        plan = self._build_batch_plan(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )
        return (
            [list(segments) for segments in plan.segments_by_rank],
            list(plan.per_rank_num_tokens),
        )

    def _get_model_num_rows(self, per_rank_num_tokens: list[int]) -> int:
        del per_rank_num_tokens
        plan = self._batch_plan
        if plan is None:
            raise RuntimeError("PCP execution width requested without a batch plan")
        return plan.model_num_rows

    def _get_local_input_idx(self, model_num_rows: int) -> torch.Tensor:
        plan = self._batch_plan
        if plan is None:
            raise RuntimeError("PCP input indices requested without a batch plan")
        if model_num_rows != plan.model_num_rows:
            raise RuntimeError(
                "PCP model row count does not match the current batch plan: "
                f"{model_num_rows} != {plan.model_num_rows}"
            )
        if model_num_rows == 0:
            return plan.slab_global_idx[:0]
        rank_start = self.pcp_rank * plan.rank_slab_width
        return plan.slab_global_idx[rank_start : rank_start + model_num_rows]

    def _convert_to_gathered_slot_mappings(
        self,
        global_batch_slot_mappings: torch.Tensor,
    ) -> torch.Tensor:
        plan = self._batch_plan
        if plan is None:
            raise RuntimeError("PCP slot mapping requested without a batch plan")
        num_slab_rows = plan.slab_global_idx.shape[0]
        if self._gathered_kv_slot_mappings is None:
            self._gathered_kv_slot_mappings = global_batch_slot_mappings.new_empty(
                global_batch_slot_mappings.shape[0], num_slab_rows
            )
        gathered_kv_slot_mappings = self._gathered_kv_slot_mappings[:, :num_slab_rows]
        if num_slab_rows == 0:
            return gathered_kv_slot_mappings
        torch.index_select(
            global_batch_slot_mappings,
            1,
            plan.slab_global_idx,
            out=gathered_kv_slot_mappings,
        )
        torch.where(
            plan.kv_write_mask.unsqueeze(0),
            gathered_kv_slot_mappings,
            self._pad_slot_id,
            out=gathered_kv_slot_mappings,
        )
        return gathered_kv_slot_mappings

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        plan = self._batch_plan
        if plan is None:
            return hidden_states
        if plan.rank_slab_width == 0:
            return hidden_states[:0]
        if hidden_states.shape[0] < plan.owned_num_tokens:
            raise RuntimeError(
                "PCP hidden-state rows are smaller than owned token count: "
                f"{hidden_states.shape[0]} < {plan.owned_num_tokens}"
            )

        if (
            plan.owned_num_tokens == plan.rank_slab_width
            and hidden_states.shape[0] == plan.rank_slab_width
        ):
            slab_hidden_states = hidden_states
        else:
            slab_hidden_states = hidden_states.new_zeros(
                (plan.rank_slab_width, *hidden_states.shape[1:])
            )
            if plan.owned_num_tokens > 0:
                slab_hidden_states[: plan.owned_num_tokens].copy_(
                    hidden_states[: plan.owned_num_tokens]
                )

        gathered = get_pcp_group().all_gather(slab_hidden_states, dim=0)
        return gathered[plan.hidden_restore_idx]

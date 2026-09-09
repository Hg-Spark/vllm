# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execution layout, batch materialization, and slab mapping for rank-local PCP."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment

logger = init_logger(__name__)


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
    """Materialize PCP execution from a rank-partition policy.

    Partition policies may override ``_get_segments_by_rank`` to compute all
    rank ownership in one pass. The default implementation preserves the
    existing ``_get_rank_segments`` interface used by canonical PCP.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_plan: PCPBatchPlan | None = None
        self._scratch_max_reqs = 0
        self._scratch_max_tokens = 0
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
        model_num_rows = (
            owned_num_tokens
            if owned_num_tokens > 0
            else (1 if rank_slab_width > 0 else 0)
        )

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

    def _ensure_batch_scratch(self) -> None:
        assert self._input_buffers is not None
        max_reqs = self._input_buffers.max_num_reqs
        max_tokens = self._input_buffers.max_num_tokens
        if max_reqs <= self._scratch_max_reqs and max_tokens <= self._scratch_max_tokens:
            return

        self._scratch_max_reqs = max_reqs
        self._scratch_max_tokens = max_tokens
        self._local_to_global_batch_req_idx_np = np.empty(max_reqs, dtype=np.int32)
        self._local_start_pos_np = np.empty(max_reqs, dtype=np.int32)
        self._local_num_scheduled_tokens_np = np.empty(max_reqs, dtype=np.int32)
        self._local_query_start_loc_np = np.empty(max_reqs + 1, dtype=np.int32)
        self._local_prefill_len_np = np.empty(max_reqs, dtype=np.int32)
        self._local_num_computed_prefill_tokens_np = np.empty(max_reqs, dtype=np.int32)
        self._local_is_prefilling_np = np.empty(max_reqs, dtype=np.bool_)
        self._seq_lens_cpu_upper_bound_np = np.empty(max_reqs, dtype=np.int32)
        self._local_input_idx_np = np.empty(max_tokens, dtype=np.int64)
        self._local_to_global_req_idx_np = np.empty(max_reqs, dtype=np.int32)
        self._req_range_np = np.arange(max_reqs + 1, dtype=np.int32)
        self._zero_req_range_np = np.zeros(max_reqs + 1, dtype=np.int32)

        self._local_to_global_req_idx_gpu = torch.empty(
            max_reqs, dtype=torch.int32, device=self.device
        )
        self._local_start_pos_gpu = torch.empty(
            max_reqs, dtype=torch.int32, device=self.device
        )
        self._local_input_idx_gpu = torch.empty(
            max_tokens, dtype=torch.int64, device=self.device
        )
        self._expanded_local_pos_gpu = torch.zeros(
            max_reqs, dtype=torch.int32, device=self.device
        )
        self._req_range_gpu = torch.arange(
            max_reqs + 1, dtype=torch.int32, device=self.device
        )
        self._zero_req_range_gpu = torch.zeros(
            max_reqs + 1, dtype=torch.int32, device=self.device
        )

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        assert self._req_states is not None
        assert self._input_buffers is not None
        if input_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")

        self._ensure_batch_scratch()
        req_states = self._req_states
        input_buffers = self._input_buffers
        global_batch = input_batch
        self._global_batch = global_batch

        num_scheduled_tokens = global_batch.num_scheduled_tokens
        num_computed_tokens = global_batch.num_computed_tokens_np
        is_prefilling = global_batch.is_prefilling_np
        plan = self._build_batch_plan(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            global_batch.query_start_loc_np,
        )

        local_segments = list(plan.local_segments)
        if not local_segments:
            local_segments = [
                RankSegment(
                    global_batch_req_idx=0,
                    global_batch_slice=slice(0, 0),
                    rank_local_batch_slice=slice(0, 0),
                )
            ]

        num_local_reqs = len(local_segments)
        if num_local_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP local request count exceeds input buffer capacity: "
                f"{num_local_reqs} > {input_buffers.max_num_reqs}."
            )
        if plan.model_num_rows > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP local model row count exceeds input buffer capacity: "
                f"{plan.model_num_rows} > {input_buffers.max_num_tokens}."
            )

        local_to_global_batch_req_idx_np = self._local_to_global_batch_req_idx_np[
            :num_local_reqs
        ]
        local_start_pos_np = self._local_start_pos_np[:num_local_reqs]
        local_num_scheduled_tokens = self._local_num_scheduled_tokens_np[
            :num_local_reqs
        ]

        for local_req_idx, segment in enumerate(local_segments):
            global_batch_req_idx = segment.global_batch_req_idx
            local_to_global_batch_req_idx_np[local_req_idx] = global_batch_req_idx
            local_start_pos_np[local_req_idx] = (
                num_computed_tokens[global_batch_req_idx]
                + segment.global_batch_slice.start
                - global_batch.query_start_loc_np[global_batch_req_idx]
            )
            local_num_scheduled_tokens[local_req_idx] = segment.num_tokens

        local_to_global_req_idx_np = self._local_to_global_req_idx_np[:num_local_reqs]
        np.take(
            global_batch.idx_mapping_np,
            local_to_global_batch_req_idx_np,
            out=local_to_global_req_idx_np,
        )
        local_req_ids = [
            global_batch.req_ids[global_batch_req_idx]
            for global_batch_req_idx in local_to_global_batch_req_idx_np
        ]

        if plan.owned_num_tokens > 0:
            rank_start = self.pcp_rank * plan.rank_slab_width
            local_input_idx = plan.slab_global_idx[
                rank_start : rank_start + plan.owned_num_tokens
            ]
            torch.index_select(
                global_batch.input_ids,
                0,
                local_input_idx,
                out=input_buffers.input_ids[: plan.owned_num_tokens],
            )
        elif plan.model_num_rows == 1:
            input_buffers.input_ids[:1].zero_()

        local_query_start_loc_np = self._local_query_start_loc_np
        local_query_start_loc_np[0] = 0
        np.cumsum(
            local_num_scheduled_tokens,
            out=local_query_start_loc_np[1 : num_local_reqs + 1],
        )
        local_query_start_loc_np[num_local_reqs + 1 :] = plan.owned_num_tokens
        async_copy_to_gpu(local_query_start_loc_np, out=input_buffers.query_start_loc)
        local_query_start_loc = input_buffers.query_start_loc[: num_local_reqs + 1]

        local_to_global_req_idx = self._local_to_global_req_idx_gpu[:num_local_reqs]
        local_start_pos = self._local_start_pos_gpu[:num_local_reqs]
        async_copy_to_gpu(local_to_global_req_idx_np, out=local_to_global_req_idx)
        async_copy_to_gpu(local_start_pos_np, out=local_start_pos)

        assert self._local_req_idx is not None
        prepare_pos_seq_lens(
            self._local_req_idx[:num_local_reqs],
            local_query_start_loc,
            local_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_local_reqs],
        )
        seq_lens = input_buffers.seq_lens[:num_local_reqs]

        is_padding = input_buffers.is_padding[: plan.model_num_rows]
        if plan.owned_num_tokens > 0:
            is_padding.fill_(False)
        elif plan.model_num_rows == 1:
            is_padding.fill_(True)
            input_buffers.positions[:1].zero_()

        total_num_logits = num_local_reqs if plan.owned_num_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = self._req_range_np[: num_local_reqs + 1]
            cu_num_logits = self._req_range_gpu[: num_local_reqs + 1]
        else:
            cu_num_logits_np = self._zero_req_range_np[: num_local_reqs + 1]
            cu_num_logits = self._zero_req_range_gpu[: num_local_reqs + 1]
        logits_indices = combine_sampled_and_draft_tokens(
            input_buffers.input_ids,
            local_to_global_req_idx,
            req_states.last_sampled_tokens,
            local_query_start_loc,
            seq_lens,
            req_states.prefill_len.gpu,
            req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            1,
        )

        local_prefill_len_np = self._local_prefill_len_np[:num_local_reqs]
        np.take(
            global_batch.prefill_len_np,
            local_to_global_batch_req_idx_np,
            out=local_prefill_len_np,
        )
        local_num_computed_prefill_tokens_np = (
            self._local_num_computed_prefill_tokens_np[:num_local_reqs]
        )
        np.minimum(
            local_start_pos_np,
            local_prefill_len_np,
            out=local_num_computed_prefill_tokens_np,
        )
        local_is_prefilling_np = self._local_is_prefilling_np[:num_local_reqs]
        np.less(
            local_num_computed_prefill_tokens_np,
            local_prefill_len_np,
            out=local_is_prefilling_np,
        )
        seq_lens_cpu_upper_bound_np = self._seq_lens_cpu_upper_bound_np[
            :num_local_reqs
        ]
        np.add(
            local_start_pos_np,
            local_num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np,
        )

        dcp_local_seq_lens = None
        if self.dcp_world_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                seq_lens,
                num_local_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_local_reqs]

        logger.debug(
            "Rank-local PCP batch: rank=%d owned_tokens=%d model_rows=%d "
            "rank_slab_width=%d dummy_row=%s per_rank_tokens=%s",
            self.pcp_rank,
            plan.owned_num_tokens,
            plan.model_num_rows,
            plan.rank_slab_width,
            plan.uses_dummy_execution_row,
            plan.per_rank_num_tokens,
        )

        return replace(
            input_batch,
            req_ids=local_req_ids,
            num_reqs=num_local_reqs,
            num_reqs_after_padding=num_local_reqs,
            idx_mapping=local_to_global_req_idx,
            idx_mapping_np=local_to_global_req_idx_np,
            expanded_idx_mapping=local_to_global_req_idx,
            expanded_local_pos=self._expanded_local_pos_gpu[:num_local_reqs],
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=plan.owned_num_tokens,
            num_tokens_after_padding=plan.model_num_rows,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=local_query_start_loc,
            query_start_loc_np=local_query_start_loc_np[: num_local_reqs + 1],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=local_prefill_len_np,
            num_computed_prefill_tokens_np=local_num_computed_prefill_tokens_np,
            is_prefilling_np=local_is_prefilling_np,
            max_seq_len_np=(
                global_batch.max_seq_len_np[local_to_global_batch_req_idx_np]
                if global_batch.max_seq_len_np is not None
                else None
            ),
            input_ids=input_buffers.input_ids[: plan.model_num_rows],
            positions=input_buffers.positions[: plan.model_num_rows],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
        )

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

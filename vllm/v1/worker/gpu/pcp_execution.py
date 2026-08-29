# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execution contract for weighted PCP partitioning.

Weighted PCP separates three widths that canonical PCP historically folds into
one value:

* owned_num_tokens: semantic tokens owned by this PCP rank;
* model_num_rows: rows passed through model forward (one dummy row only when
  owned_num_tokens is zero);
* rank_slab_width: fixed-width per-rank communication slab.

This keeps imbalance padding out of Transformer compute while preserving the
fixed-shape slab ABI used by MLA cache transfer and final hidden-state restore.
"""

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
    """One-step execution and communication-slab layout for weighted PCP."""

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
    """Plan execution and communication layout for weighted contiguous PCP.

    Partition subclasses only decide token ownership through
    ``_get_rank_segments``. This class materializes the local InputBatch and
    owns the communication layout used by the weighted PCP path.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_plan: PCPBatchPlan | None = None

    @property
    def batch_plan(self) -> PCPBatchPlan | None:
        return self._batch_plan

    def _build_batch_plan(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> PCPBatchPlan:
        segments_by_rank: list[tuple[RankSegment, ...]] = []
        per_rank_num_tokens: list[int] = []
        for pcp_rank in range(self.pcp_world_size):
            segments = tuple(
                self._get_rank_segments(
                    pcp_rank,
                    num_scheduled_tokens,
                    num_computed_tokens,
                    is_prefilling,
                    query_start_loc_np,
                )
            )
            segments_by_rank.append(segments)
            per_rank_num_tokens.append(sum(segment.num_tokens for segment in segments))

        rank_slab_width = max(per_rank_num_tokens, default=0)
        owned_num_tokens = per_rank_num_tokens[self.pcp_rank]
        # Keep exactly one compatibility row only for a truly empty owner. It is
        # marked padding and never enters KV writes, hidden restore, logits, or
        # RequestState accounting.
        model_num_rows = (
            owned_num_tokens
            if owned_num_tokens > 0
            else (1 if rank_slab_width > 0 else 0)
        )

        global_num_tokens = int(query_start_loc_np[-1])
        hidden_restore_idx = np.empty(global_num_tokens, dtype=np.int64)
        num_slab_rows = rank_slab_width * self.pcp_world_size
        slab_global_idx = np.zeros(num_slab_rows, dtype=np.int64)
        kv_write_mask = np.zeros(num_slab_rows, dtype=np.bool_)

        for pcp_rank, segments in enumerate(segments_by_rank):
            rank_offset = pcp_rank * rank_slab_width
            for segment in segments:
                slab_slice = slice(
                    rank_offset + segment.rank_local_batch_slice.start,
                    rank_offset + segment.rank_local_batch_slice.stop,
                )
                slab_global_idx[slab_slice] = np.arange(
                    segment.global_batch_slice.start,
                    segment.global_batch_slice.stop,
                    dtype=np.int64,
                )
                kv_write_mask[slab_slice] = True
                hidden_restore_idx[segment.global_batch_slice] = np.arange(
                    slab_slice.start,
                    slab_slice.stop,
                    dtype=np.int64,
                )

        plan = PCPBatchPlan(
            segments_by_rank=tuple(segments_by_rank),
            per_rank_num_tokens=tuple(per_rank_num_tokens),
            local_segments=segments_by_rank[self.pcp_rank],
            owned_num_tokens=owned_num_tokens,
            model_num_rows=model_num_rows,
            rank_slab_width=rank_slab_width,
            slab_global_idx=async_copy_to_gpu(
                slab_global_idx,
                device=self.device,
            ),
            kv_write_mask=async_copy_to_gpu(kv_write_mask, device=self.device),
            hidden_restore_idx=async_copy_to_gpu(
                hidden_restore_idx,
                device=self.device,
            ),
        )
        self._batch_plan = plan
        return plan

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        assert self._req_states is not None
        assert self._input_buffers is not None
        req_states = self._req_states
        input_buffers = self._input_buffers
        if input_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")

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
            # Metadata compatibility only. This zero-length segment does not own
            # a token; the single model row is marked as padding below.
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
                "PCP local request count exceeds the MRV2 input buffer size: "
                f"{num_local_reqs} > {input_buffers.max_num_reqs}."
            )
        if plan.model_num_rows > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP local model row count exceeds the MRV2 input buffer size: "
                f"{plan.model_num_rows} > {input_buffers.max_num_tokens}."
            )

        local_to_global_batch_req_idx_np = np.fromiter(
            (segment.global_batch_req_idx for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_start_pos_np = np.fromiter(
            (
                num_computed_tokens[segment.global_batch_req_idx]
                + segment.global_batch_slice.start
                - global_batch.query_start_loc_np[segment.global_batch_req_idx]
                for segment in local_segments
            ),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_num_scheduled_tokens = np.fromiter(
            (segment.num_tokens for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_to_global_req_idx_np = global_batch.idx_mapping_np[
            local_to_global_batch_req_idx_np
        ]
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

        local_query_start_loc_np = np.empty(
            input_buffers.max_num_reqs + 1,
            dtype=np.int32,
        )
        local_query_start_loc_np[0] = 0
        local_query_start_loc_out = local_query_start_loc_np[1 : num_local_reqs + 1]
        np.cumsum(local_num_scheduled_tokens, out=local_query_start_loc_out)
        local_query_start_loc_np[num_local_reqs + 1 :] = plan.owned_num_tokens
        async_copy_to_gpu(local_query_start_loc_np, out=input_buffers.query_start_loc)
        local_query_start_loc = input_buffers.query_start_loc[: num_local_reqs + 1]

        local_to_global_req_idx = async_copy_to_gpu(
            local_to_global_req_idx_np,
            device=self.device,
        )
        local_start_pos = async_copy_to_gpu(local_start_pos_np, device=self.device)

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
            input_buffers.input_ids[:1].zero_()
            input_buffers.positions[:1].zero_()

        total_num_logits = num_local_reqs if plan.owned_num_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = np.arange(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_local_reqs + 1,
                device=self.device,
                dtype=torch.int32,
            )
        else:
            cu_num_logits_np = np.zeros(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.zeros(
                num_local_reqs + 1,
                device=self.device,
                dtype=torch.int32,
            )
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

        local_prefill_len_np = global_batch.prefill_len_np[
            local_to_global_batch_req_idx_np
        ]
        local_num_computed_prefill_tokens_np = np.minimum(
            local_start_pos_np,
            local_prefill_len_np,
        )
        local_is_prefilling_np = (
            local_num_computed_prefill_tokens_np < local_prefill_len_np
        )
        seq_lens_cpu_upper_bound_np = np.zeros(num_local_reqs, dtype=np.int32)
        seq_lens_cpu_upper_bound_np[:] = local_start_pos_np + local_num_scheduled_tokens

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
            "Weighted PCP batch: rank=%d owned_tokens=%d model_rows=%d "
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
            expanded_local_pos=torch.zeros(
                num_local_reqs,
                dtype=torch.int32,
                device=self.device,
            ),
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
            max_seq_len_np=global_batch.max_seq_len_np[local_to_global_batch_req_idx_np]
            if global_batch.max_seq_len_np is not None
            else None,
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
                global_batch_slot_mappings.shape[0],
                num_slab_rows,
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

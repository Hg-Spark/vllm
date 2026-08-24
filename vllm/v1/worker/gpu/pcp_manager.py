# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


def weighted_partition_lengths(
    num_tokens: int,
    weights: tuple[float, ...],
    *,
    start_pos: int = 0,
    alignment: int = 1,
) -> tuple[int, ...]:
    """Partition tokens using cumulative weighted, optionally page-aligned cuts.

    Alignment is applied to cumulative absolute boundaries instead of rounding
    each segment independently. When there are at least as many tokens as PCP
    ranks, every positive-weight rank receives at least one real token.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if not weights:
        raise ValueError("weighted PCP partition requires at least one weight")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")

    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(f"invalid PCP load weights: {weights}")
    num_segments = len(weights)
    if num_tokens == 0:
        return (0,) * num_segments

    if alignment == 1:
        ideal = [num_tokens * weight / total_weight for weight in weights]
        lengths = [math.floor(value) for value in ideal]
        remainder = num_tokens - sum(lengths)
        order = sorted(
            range(num_segments),
            key=lambda segment: (-(ideal[segment] - lengths[segment]), segment),
        )
        for segment in order[:remainder]:
            lengths[segment] += 1

        # Extremely skewed weights can otherwise round a rank to zero even when
        # the chunk has enough real tokens for every PCP rank. Repair by moving
        # one token from the largest donor; this affects only the rounding tail.
        if num_tokens >= num_segments:
            for empty in [i for i, length in enumerate(lengths) if length == 0]:
                donor = max(range(num_segments), key=lambda i: (lengths[i], -i))
                if lengths[donor] <= 1:
                    raise AssertionError("PCP positive partition has no donor")
                lengths[donor] -= 1
                lengths[empty] += 1
        return tuple(lengths)

    require_positive = (
        start_pos % alignment == 0
        and num_tokens >= (num_segments - 1) * alignment + 1
    )
    boundaries = [0]
    cumulative_weight = 0.0
    for segment in range(num_segments - 1):
        cumulative_weight += weights[segment]
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


def _parse_partition_weights(
    additional_config: object,
    pcp_world_size: int,
) -> tuple[float, ...]:
    default = (1.0,) * pcp_world_size
    if not isinstance(additional_config, dict):
        return default
    partition = additional_config.get("pcp_partition")
    if partition is None:
        return default
    if not isinstance(partition, dict):
        raise ValueError("pcp_partition must be a JSON object")
    unknown = set(partition) - {"weights"}
    if unknown:
        raise ValueError(f"unsupported pcp_partition keys: {sorted(unknown)}")
    raw = partition.get("weights")
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != pcp_world_size:
        got = len(raw) if isinstance(raw, (list, tuple)) else type(raw).__name__
        raise ValueError(
            f"pcp_partition.weights requires {pcp_world_size} positive values, "
            f"got {got}: {raw}"
        )
    try:
        weights = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pcp_partition.weights must be numeric: {raw}") from exc
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(
            "pcp_partition.weights must contain finite positive values: "
            f"{weights}"
        )
    return weights


@dataclass(frozen=True)
class RankSegment:
    global_batch_req_idx: int
    global_batch_slice: slice
    rank_local_batch_slice: slice

    @property
    def num_tokens(self) -> int:
        return self.global_batch_slice.stop - self.global_batch_slice.start


class PCPManager:
    """MRV2 PC batch manager with per-step contiguous PCP ownership."""

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
        partition_weights: tuple[float, ...] | None = None,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.device = device
        self.dcp_world_size = dcp_world_size
        self.dcp_rank = dcp_rank
        self.cp_interleave = cp_interleave
        self._partition_weights = partition_weights or (1.0,) * pcp_world_size
        if len(self._partition_weights) != pcp_world_size:
            raise ValueError(
                "PCP partition weights must match PCP world size: "
                f"weights={self._partition_weights}, world_size={pcp_world_size}"
            )
        self._page_alignment = (
            math.lcm(*(int(size) for size in block_tables.kernel_block_sizes))
            if block_tables is not None and block_tables.kernel_block_sizes
            else 1
        )

        self._global_batch: InputBatch | None = None
        self._req_states = req_states
        self._block_tables = block_tables
        self._hidden_restore_idx: torch.Tensor | None = None
        self._padded_gather_idx: torch.Tensor | None = None
        self._gathered_kv_write_mask: torch.Tensor | None = None
        self._collective_num_tokens = 0
        self._hidden_collective_scratch: torch.Tensor | None = None
        self._pad_slot_id = torch.tensor(PAD_SLOT_ID, dtype=torch.int64, device=device)

        max_num_local_reqs = 2 * max_num_reqs if max_num_reqs is not None else None
        self._input_buffers = (
            InputBuffers(max_num_local_reqs, max_num_tokens, device)
            if max_num_local_reqs is not None and max_num_tokens is not None
            else None
        )
        self._local_req_idx = (
            torch.arange(max_num_local_reqs, dtype=torch.int32, device=device)
            if max_num_local_reqs is not None
            else None
        )
        self._local_block_tables: tuple[torch.Tensor, ...] | None
        self._local_block_table_ptrs: torch.Tensor | None
        if block_tables is not None and max_num_local_reqs is not None:
            self._local_block_tables = tuple(
                table.new_zeros((max_num_local_reqs, table.shape[1]))
                for table in block_tables.input_block_tables
            )
            self._local_block_table_ptrs = torch.tensor(
                [table.data_ptr() for table in self._local_block_tables],
                dtype=torch.uint64,
                device=device,
            )
        else:
            self._local_block_tables = None
            self._local_block_table_ptrs = None
        num_kv_cache_groups = (
            block_tables.num_kv_cache_groups if block_tables is not None else 0
        )
        self._global_batch_slot_mappings = (
            torch.empty(
                num_kv_cache_groups,
                max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if max_num_tokens is not None and num_kv_cache_groups > 0
            else None
        )
        self._gathered_kv_slot_mappings = (
            torch.empty(
                num_kv_cache_groups,
                max_num_tokens * pcp_world_size,
                dtype=torch.int64,
                device=device,
            )
            if max_num_tokens is not None and num_kv_cache_groups > 0
            else None
        )

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        pcp_size = parallel_config.prefill_context_parallel_size
        if pcp_size <= 1:
            return

        if not model_config.use_mla:
            raise NotImplementedError("MRV2 PCP currently supports MLA models only.")
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError(
                "MRV2 PCP does not support encoder-decoder models yet."
            )
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "MRV2 PCP does not support speculative decoding yet."
            )
        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        if (
            is_sparse_mla
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise NotImplementedError(
                "MRV2 sparse MLA PCP does not support CUDA graphs yet. "
                "Set -cc.cudagraph_mode=NONE."
            )
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise NotImplementedError("MRV2 PCP supports PIECEWISE CUDA graphs only.")

    @staticmethod
    def _reorder_segments(
        segments: list[RankSegment],
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        def is_pure_prefill(segment: RankSegment) -> bool:
            req_idx = segment.global_batch_req_idx
            start_pos = (
                num_computed_tokens[req_idx]
                + segment.global_batch_slice.start
                - query_start_loc_np[req_idx]
            )
            return is_prefilling[req_idx] and start_pos == 0

        segments.sort(key=is_pure_prefill)
        rank_offset = 0
        for index, segment in enumerate(segments):
            segments[index] = replace(
                segment,
                rank_local_batch_slice=slice(
                    rank_offset, rank_offset + segment.num_tokens
                ),
            )
            rank_offset += segment.num_tokens
        return segments

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        """Build one contiguous prefill row per request for one PCP rank.

        Normal scheduler chunks are partitioned across all ranks. A tail shorter
        than PCP world size is replicated because there are not enough distinct
        tokens to keep every rank non-empty; this removes the old slice(0, 0)
        compatibility request without introducing an idle-owner execution mode.
        """
        rank_segments = []
        rank_offset = 0
        num_chunks = self.pcp_world_size
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            chunk_indices: tuple[int, ...]
            if bool(is_prefilling[global_batch_req_idx]):
                if query_len < num_chunks:
                    # Replicate a tiny tail on every rank. The work is bounded by
                    # PCP size-1 tokens and preserves the non-empty model contract.
                    chunk_lengths = (query_len,)
                    chunk_offsets = (0,)
                    chunk_indices = (0,)
                else:
                    alignment = self._page_alignment
                    start_pos = int(num_computed_tokens[global_batch_req_idx])
                    if query_len < num_chunks * alignment or start_pos % alignment:
                        alignment = 1
                    chunk_lengths = weighted_partition_lengths(
                        query_len,
                        self._partition_weights,
                        start_pos=start_pos,
                        alignment=alignment,
                    )
                    chunk_offsets = [0] * num_chunks
                    running = 0
                    for chunk_idx, chunk_len in enumerate(chunk_lengths):
                        chunk_offsets[chunk_idx] = running
                        running += chunk_len
                    assert running == query_len
                    if any(length <= 0 for length in chunk_lengths):
                        raise AssertionError(
                            "PCP non-tiny prefill partition produced an empty rank: "
                            f"lengths={chunk_lengths}, query_len={query_len}"
                        )
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

    def _build_batch_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[list[list[RankSegment]], list[int]]:
        segments_by_rank = []
        per_rank_num_tokens = []
        for rank in range(self.pcp_world_size):
            segments = self._get_rank_segments(
                rank,
                num_scheduled_tokens,
                num_computed_tokens,
                is_prefilling,
                query_start_loc_np,
            )
            num_rank_tokens = sum(segment.num_tokens for segment in segments)
            segments_by_rank.append(segments)
            per_rank_num_tokens.append(num_rank_tokens)

        hidden_restore_idx = np.empty(int(query_start_loc_np[-1]), dtype=np.int64)
        collective_num_tokens = max(per_rank_num_tokens, default=0)
        self._collective_num_tokens = collective_num_tokens
        num_expanded_tokens = collective_num_tokens * self.pcp_world_size
        padded_gather_idx = np.zeros(num_expanded_tokens, dtype=np.int64)
        gathered_kv_write_mask = np.zeros(num_expanded_tokens, dtype=np.bool_)
        for rank, segments in enumerate(segments_by_rank):
            expanded_rank_offset = rank * collective_num_tokens
            for segment in segments:
                gathered_slice = slice(
                    expanded_rank_offset + segment.rank_local_batch_slice.start,
                    expanded_rank_offset + segment.rank_local_batch_slice.stop,
                )
                padded_gather_idx[gathered_slice] = np.arange(
                    segment.global_batch_slice.start,
                    segment.global_batch_slice.stop,
                    dtype=np.int64,
                )
                if not bool(is_prefilling[segment.global_batch_req_idx]) and rank != 0:
                    continue
                gathered_kv_write_mask[gathered_slice] = True
                hidden_restore_idx[segment.global_batch_slice] = np.arange(
                    gathered_slice.start,
                    gathered_slice.stop,
                    dtype=np.int64,
                )

        self._hidden_restore_idx = async_copy_to_gpu(
            hidden_restore_idx, device=self.device
        )
        self._padded_gather_idx = async_copy_to_gpu(
            padded_gather_idx, device=self.device
        )
        self._gathered_kv_write_mask = async_copy_to_gpu(
            gathered_kv_write_mask, device=self.device
        )
        return segments_by_rank, per_rank_num_tokens

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

        segments_by_rank, per_rank_num_tokens = self._build_batch_layout(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            global_batch.query_start_loc_np,
        )

        local_segments = segments_by_rank[self.pcp_rank]
        if not local_segments:
            raise RuntimeError(
                "PCP produced an empty local batch. Non-tiny prefills must give "
                "every rank real tokens and tiny tails are replicated."
            )

        num_local_reqs = len(local_segments)
        if num_local_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP local request count exceeds the MRV2 input buffer size: "
                f"{num_local_reqs} > {input_buffers.max_num_reqs}."
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

        num_local_tokens = int(local_num_scheduled_tokens.sum())
        fresh_prefills = int(
            np.count_nonzero(is_prefilling & (num_computed_tokens == 0))
        )
        continued_prefills = int(
            np.count_nonzero(is_prefilling & (num_computed_tokens > 0))
        )
        logger.debug(
            "PCP batch: rank=%d global_batch_reqs=%d fresh_prefills=%d "
            "continued_prefills=%d decodes=%d local_reqs=%d "
            "local_tokens=%d collective_width=%d per_rank_tokens=%s",
            self.pcp_rank,
            global_batch.num_reqs,
            fresh_prefills,
            continued_prefills,
            global_batch.num_reqs - fresh_prefills - continued_prefills,
            num_local_reqs,
            num_local_tokens,
            self._collective_num_tokens,
            per_rank_num_tokens,
        )
        if num_local_tokens > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP local token count exceeds the MRV2 input buffer size: "
                f"{num_local_tokens} > {input_buffers.max_num_tokens}."
            )
        rank_token_start = self.pcp_rank * self._collective_num_tokens
        assert self._padded_gather_idx is not None
        local_gather_idx = self._padded_gather_idx[
            rank_token_start : rank_token_start + num_local_tokens
        ]
        torch.index_select(
            global_batch.input_ids,
            0,
            local_gather_idx,
            out=input_buffers.input_ids[:num_local_tokens],
        )

        local_query_start_loc_np = np.empty(
            input_buffers.max_num_reqs + 1, dtype=np.int32
        )
        local_query_start_loc_np[0] = 0
        local_query_start_loc_out = local_query_start_loc_np[1 : num_local_reqs + 1]
        np.cumsum(local_num_scheduled_tokens, out=local_query_start_loc_out)
        local_query_start_loc_np[num_local_reqs + 1 :] = num_local_tokens
        async_copy_to_gpu(local_query_start_loc_np, out=input_buffers.query_start_loc)
        local_query_start_loc = input_buffers.query_start_loc[: num_local_reqs + 1]

        local_to_global_req_idx = async_copy_to_gpu(
            local_to_global_req_idx_np, device=self.device
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
        is_padding = input_buffers.is_padding[:num_local_tokens]
        is_padding.fill_(False)

        total_num_logits = num_local_reqs if num_local_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = np.arange(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_local_reqs + 1, device=self.device, dtype=torch.int32
            )
        else:
            cu_num_logits_np = np.zeros(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.zeros(
                num_local_reqs + 1, device=self.device, dtype=torch.int32
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
            local_start_pos_np, local_prefill_len_np
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

        return replace(
            input_batch,
            req_ids=local_req_ids,
            num_reqs=num_local_reqs,
            num_reqs_after_padding=num_local_reqs,
            idx_mapping=local_to_global_req_idx,
            idx_mapping_np=local_to_global_req_idx_np,
            expanded_idx_mapping=local_to_global_req_idx,
            expanded_local_pos=torch.zeros(
                num_local_reqs, dtype=torch.int32, device=self.device
            ),
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=num_local_tokens,
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
            input_ids=input_buffers.input_ids[:num_local_tokens],
            positions=input_buffers.positions[:num_local_tokens],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        assert self._block_tables is not None
        assert self._local_block_tables is not None
        assert self._local_block_table_ptrs is not None
        block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            input_batch.num_reqs_after_padding,
            out=self._local_block_tables,
            out_ptrs=self._local_block_table_ptrs,
        )
        slot_mappings = self.prepare_slot_mappings()
        return block_tables, slot_mappings

    def prepare_slot_mappings(self) -> torch.Tensor:
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert self._global_batch is not None
        global_batch = self._global_batch
        global_batch_slot_mappings = self._block_tables.compute_slot_mappings(
            global_batch.idx_mapping,
            global_batch.query_start_loc,
            global_batch.positions,
            global_batch.num_tokens,
            out=self._global_batch_slot_mappings,
        )
        return self._convert_to_gathered_slot_mappings(global_batch_slot_mappings)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        assert self._gathered_kv_slot_mappings is not None
        self._gathered_kv_slot_mappings.fill_(PAD_SLOT_ID)
        return self._gathered_kv_slot_mappings[:, : num_tokens * self.pcp_world_size]

    def _convert_to_gathered_slot_mappings(
        self,
        global_batch_slot_mappings: torch.Tensor,
    ) -> torch.Tensor:
        assert self._padded_gather_idx is not None
        assert self._gathered_kv_write_mask is not None
        padded_gather_idx = self._padded_gather_idx
        num_expanded_tokens = padded_gather_idx.shape[0]
        if self._gathered_kv_slot_mappings is None:
            self._gathered_kv_slot_mappings = global_batch_slot_mappings.new_empty(
                global_batch_slot_mappings.shape[0], num_expanded_tokens
            )
        gathered_kv_slot_mappings = self._gathered_kv_slot_mappings[
            :, :num_expanded_tokens
        ]
        torch.index_select(
            global_batch_slot_mappings,
            1,
            padded_gather_idx,
            out=gathered_kv_slot_mappings,
        )
        torch.where(
            self._gathered_kv_write_mask.unsqueeze(0),
            gathered_kv_slot_mappings,
            self._pad_slot_id,
            out=gathered_kv_slot_mappings,
        )
        return gathered_kv_slot_mappings

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._hidden_restore_idx is None:
            return hidden_states
        collective_num_tokens = self._collective_num_tokens
        if hidden_states.shape[0] > collective_num_tokens:
            raise RuntimeError(
                "PCP hidden-state width exceeds the collective slab: "
                f"{hidden_states.shape[0]} > {collective_num_tokens}"
            )
        if hidden_states.shape[0] < collective_num_tokens:
            wanted_shape = (collective_num_tokens, *hidden_states.shape[1:])
            scratch = self._hidden_collective_scratch
            if (
                scratch is None
                or scratch.dtype != hidden_states.dtype
                or scratch.device != hidden_states.device
                or scratch.shape[1:] != hidden_states.shape[1:]
                or scratch.shape[0] < collective_num_tokens
            ):
                scratch = hidden_states.new_empty(wanted_shape)
                self._hidden_collective_scratch = scratch
            collective_hidden_states = scratch[:collective_num_tokens]
            local_tokens = hidden_states.shape[0]
            if local_tokens:
                collective_hidden_states[:local_tokens].copy_(hidden_states)
            collective_hidden_states[local_tokens:].zero_()
        else:
            collective_hidden_states = hidden_states
        gathered = get_pcp_group().all_gather(collective_hidden_states, dim=0)
        return gathered[self._hidden_restore_idx]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        assert self._global_batch is not None
        return self.restore_hidden_states(hidden_states), self._global_batch


def maybe_partition_pcp_batch(
    manager: PCPManager | None,
    input_batch: InputBatch,
) -> InputBatch:
    if manager is None:
        return input_batch
    return manager.partition_batch(input_batch)


def maybe_get_pcp_dummy_slot_mappings(
    manager: PCPManager | None,
    block_tables: BlockTables,
    num_tokens: int,
) -> torch.Tensor:
    if manager is None:
        return block_tables.get_dummy_slot_mappings(num_tokens)
    return manager.get_dummy_slot_mappings(num_tokens)


def maybe_restore_pcp_for_sampling(
    manager: PCPManager | None,
    hidden_states: torch.Tensor | None,
    input_batch: InputBatch,
) -> tuple[torch.Tensor, InputBatch]:
    assert hidden_states is not None
    if manager is None:
        return hidden_states, input_batch
    return manager.restore_for_sampling(hidden_states)


def maybe_build_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
) -> PCPManager | None:
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    PCPManager.validate_config(vllm_config, supports_mm_inputs)

    pcp_rank = get_pcp_group().rank_in_group
    dcp_size = parallel_config.decode_context_parallel_size
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0

    return PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=pcp_rank,
        device=device,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
        partition_weights=_parse_partition_weights(
            vllm_config.additional_config, pcp_size
        ),
    )

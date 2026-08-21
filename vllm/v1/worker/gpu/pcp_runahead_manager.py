# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP manager extension for causal-prefix runahead."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.config.pcp_runahead import (
    RUNAHEAD_MIN_PREFILL_TOKENS,
    TransportPolicy,
    parse_pcp_runahead_config,
    parse_runahead_weights,
)
from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import (
    PCPRunaheadRuntime,
    register_pcp_runahead_runtime,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@dataclass(frozen=True)
class LogicalSegment:
    global_batch_req_idx: int
    start_pos: int
    end_pos: int
    owner_group_rank: int


@dataclass(frozen=True)
class SegmentLayout:
    """One compilation of logical segments for the current global batch."""

    segments_by_rank: tuple[tuple[RankSegment, ...], ...]
    rows_per_rank: tuple[int, ...]
    logical_segments: tuple[tuple[LogicalSegment, ...], ...]


@dataclass(frozen=True)
class RunaheadStep:
    """All per-step runahead state consumed by batch and KV paths."""

    layout: SegmentLayout
    transport: TransportPolicy


def weighted_partition_lengths(
    num_tokens: int,
    weights: tuple[float, ...],
    *,
    start_pos: int = 0,
    alignment: int = 1,
) -> tuple[int, ...]:
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

    if alignment == 1 or num_tokens < num_segments * alignment:
        ideal = [num_tokens * weight / total_weight for weight in weights]
        lengths = [math.floor(value) for value in ideal]
        remainder = num_tokens - sum(lengths)
        order = sorted(
            range(num_segments),
            key=lambda segment: (-(ideal[segment] - lengths[segment]), segment),
        )
        for segment in order[:remainder]:
            lengths[segment] += 1
        return tuple(lengths)

    boundaries = [0]
    cumulative_weight = 0.0
    for segment in range(num_segments - 1):
        cumulative_weight += weights[segment]
        ideal_abs = start_pos + num_tokens * cumulative_weight / total_weight
        min_cut = boundaries[-1] + 1
        max_cut = num_tokens - (num_segments - segment - 1)
        min_abs = start_pos + min_cut
        max_abs = start_pos + max_cut
        candidate_abs = int(round(ideal_abs / alignment)) * alignment
        if candidate_abs < min_abs:
            candidate_abs = math.ceil(min_abs / alignment) * alignment
        if candidate_abs > max_abs:
            candidate_abs = math.floor(max_abs / alignment) * alignment
        if candidate_abs < min_abs or candidate_abs > max_abs:
            candidate_abs = min(max(int(round(ideal_abs)), min_abs), max_abs)
        boundaries.append(candidate_abs - start_pos)
    boundaries.append(num_tokens)
    return tuple(
        boundaries[index + 1] - boundaries[index] for index in range(num_segments)
    )


def runahead_batch_eligible(
    *,
    num_reqs: int,
    is_prefilling: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    prefill_len: np.ndarray,
    pcp_world_size: int,
    require_full_prefill: bool = True,
    min_prefill_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS,
) -> bool:
    if num_reqs <= 0 or not bool(is_prefilling[:num_reqs].all()):
        return False
    if require_full_prefill:
        if not bool((num_computed_tokens[:num_reqs] == 0).all()):
            return False
        if not bool(
            (num_scheduled_tokens[:num_reqs] == prefill_len[:num_reqs]).all()
        ):
            return False
    total_prefill_tokens = int(num_scheduled_tokens[:num_reqs].sum())
    return total_prefill_tokens >= max(pcp_world_size, min_prefill_tokens)


class RunaheadPCPManager(PCPManager):
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
    ) -> None:
        vllm_config = get_current_vllm_config()
        config = parse_pcp_runahead_config(
            vllm_config.additional_config, pcp_world_size
        )
        if config is None:
            raise RuntimeError("PCP runahead manager requires an enabled config")
        self._config = config

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
        self._standard_attention_pcp = False
        self._active_step: RunaheadStep | None = None
        self._page_alignment = (
            math.lcm(*block_tables.kernel_block_sizes)
            if config.transport == "page_pull"
            and block_tables is not None
            and block_tables.kernel_block_sizes
            else 1
        )
        if (
            config.transport == "page_pull"
            and block_tables is not None
            and block_tables.num_kv_cache_groups != 1
        ):
            raise NotImplementedError(
                "PCP page_pull currently supports one standard-attention KV cache "
                f"group, got {block_tables.num_kv_cache_groups}"
            )

        self._sharded_kv_history = False
        self._runahead_runtime = PCPRunaheadRuntime(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
            max_inflight_sends=config.max_inflight_sends,
            max_inflight_reads=config.max_inflight_reads,
            nixl_backends=config.nixl_backends,
            pcp_group=get_pcp_group(),
        )
        register_pcp_runahead_runtime(self._runahead_runtime)
        self._resize_local_request_buffers_if_needed(
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            device=device,
        )

    def _compact_layout_enabled(self) -> bool:
        return self._active_step is not None

    def _resize_local_request_buffers_if_needed(
        self,
        *,
        max_num_reqs: int | None,
        max_num_tokens: int | None,
        block_tables: BlockTables | None,
        device: torch.device,
    ) -> None:
        mapping = self._config.segment_to_group_rank
        max_segments_per_rank = max(
            mapping.count(rank) for rank in range(self.pcp_world_size)
        )
        if max_segments_per_rank <= 2 or max_num_reqs is None or max_num_tokens is None:
            return
        max_num_local_reqs = max_segments_per_rank * max_num_reqs
        self._input_buffers = InputBuffers(max_num_local_reqs, max_num_tokens, device)
        self._local_req_idx = torch.arange(
            max_num_local_reqs, dtype=torch.int32, device=device
        )
        if block_tables is not None:
            self._local_block_tables = tuple(
                table.new_zeros((max_num_local_reqs, table.shape[1]))
                for table in block_tables.input_block_tables
            )
            self._local_block_table_ptrs = torch.tensor(
                [table.data_ptr() for table in self._local_block_tables],
                dtype=torch.uint64,
                device=device,
            )

    def set_standard_attention(self, enabled: bool) -> None:
        self._standard_attention_pcp = enabled

    @classmethod
    def validate_config(
        cls,
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        assert model is not None
        config = parse_pcp_runahead_config(
            vllm_config.additional_config,
            parallel.prefill_context_parallel_size,
        )
        if config is None:
            raise ValueError("pcp_runahead manager requires an enabled config")

        if model.is_encoder_decoder:
            raise NotImplementedError("PCP runahead does not support encoder-decoder")
        if supports_mm_inputs:
            raise NotImplementedError("PCP runahead does not support MM inputs")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("PCP runahead does not support LoRA")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("PCP runahead does not support speculative decoding")
        if parallel.tensor_parallel_size != 1:
            raise NotImplementedError("PCP runahead requires TP=1")
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("PCP runahead requires PP=1")
        if parallel.data_parallel_size != 1:
            raise NotImplementedError("PCP runahead requires DP=1")
        if parallel.decode_context_parallel_size != 1:
            raise NotImplementedError("PCP runahead requires DCP=1")
        if parallel.enable_expert_parallel:
            raise NotImplementedError("PCP runahead does not support expert parallelism")
        if parallel.enable_dbo:
            raise NotImplementedError("PCP runahead does not support DBO")
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise NotImplementedError("PCP runahead requires --enforce-eager")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError("PCP runahead does not support async scheduling")

        if model.use_mla:
            if hasattr(model.hf_text_config, "index_topk"):
                raise NotImplementedError(
                    "PCP runahead does not support sparse MLA/indexer yet"
                )
            if config.transport == "page_pull":
                raise NotImplementedError(
                    "PCP runahead MLA does not support page_pull yet; use "
                    "full_kv_collective, prefix_p2p, or direct_p2p"
                )

        if config.transport == "page_pull":
            cache_dtype = str(vllm_config.cache_config.cache_dtype)
            if cache_dtype not in ("auto", "float16", "bfloat16"):
                raise NotImplementedError(
                    "PCP page_pull requires unquantized FP16/BF16 KV cache, "
                    f"got cache_dtype={cache_dtype}"
                )

    def _partition_lengths(self, query_len: int, start_pos: int = 0) -> tuple[int, ...]:
        return weighted_partition_lengths(
            query_len,
            self._config.weights,
            start_pos=start_pos,
            alignment=self._page_alignment,
        )

    def _compile_segment_layout(self, input_batch: InputBatch) -> SegmentLayout | None:
        mapping = self._config.segment_to_group_rank
        num_segments = len(mapping)
        segments_by_rank: list[list[RankSegment]] = [
            [] for _ in range(self.pcp_world_size)
        ]
        logical_segments: list[list[LogicalSegment]] = [
            [] for _ in range(num_segments)
        ]
        rank_rows = [0] * self.pcp_world_size

        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            query_len = int(num_tokens)
            if query_len <= 0:
                continue
            start_pos = int(input_batch.num_computed_tokens_np[req_idx])
            global_start = int(input_batch.query_start_loc_np[req_idx])
            lengths = self._partition_lengths(query_len, start_pos)
            offset = 0
            absolute = start_pos
            for segment_idx, length in enumerate(lengths):
                next_offset = offset + length
                end_pos = absolute + length
                owner_group_rank = mapping[segment_idx]
                if length > 0:
                    global_slice = slice(
                        global_start + offset, global_start + next_offset
                    )
                    local_start = rank_rows[owner_group_rank]
                    segments_by_rank[owner_group_rank].append(
                        RankSegment(
                            global_batch_req_idx=req_idx,
                            global_batch_slice=global_slice,
                            rank_local_batch_slice=slice(
                                local_start, local_start + length
                            ),
                        )
                    )
                    logical_segments[segment_idx].append(
                        LogicalSegment(
                            global_batch_req_idx=req_idx,
                            start_pos=absolute,
                            end_pos=end_pos,
                            owner_group_rank=owner_group_rank,
                        )
                    )
                    rank_rows[owner_group_rank] += length
                if (
                    segment_idx + 1 < num_segments
                    and self._page_alignment > 1
                    and end_pos % self._page_alignment != 0
                ):
                    return None
                offset = next_offset
                absolute = end_pos

        return SegmentLayout(
            segments_by_rank=tuple(tuple(items) for items in segments_by_rank),
            rows_per_rank=tuple(rank_rows),
            logical_segments=tuple(tuple(items) for items in logical_segments),
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        step = self._active_step
        if step is not None:
            return list(step.layout.segments_by_rank[rank])
        return super()._get_rank_segments(
            rank,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        if self._sharded_kv_history:
            raise RuntimeError(
                "PCP runahead left persistent causal-prefix KV sharded across ranks; "
                "continued/decode execution is not implemented"
            )

        eligible = runahead_batch_eligible(
            num_reqs=input_batch.num_reqs,
            is_prefilling=input_batch.is_prefilling_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            num_computed_tokens=input_batch.num_computed_tokens_np,
            prefill_len=input_batch.prefill_len_np,
            pcp_world_size=self.pcp_world_size,
            require_full_prefill=True,
            min_prefill_tokens=self._config.min_tokens,
        )
        layout = self._compile_segment_layout(input_batch) if eligible else None
        if eligible and layout is None:
            logger.debug(
                "PCP page_pull falling back because a segment boundary is off-page"
            )
            eligible = False
        if layout is not None and any(rows <= 0 for rows in layout.rows_per_rank):
            logger.debug(
                "PCP runahead partition produced an empty rank; falling back: rows=%s",
                layout.rows_per_rank,
            )
            eligible = False
            layout = None

        self._active_step = (
            RunaheadStep(layout=layout, transport=self._config.transport)
            if eligible and layout is not None
            else None
        )
        local_batch = super().partition_batch(input_batch)
        step = self._active_step
        if step is None:
            self._runahead_runtime.disable_step()
            return local_batch

        self._runahead_runtime.begin_step(
            step.layout.rows_per_rank,
            transport=step.transport,
        )
        return local_batch

    def _configure_page_pull_plan(self) -> None:
        step = self._active_step
        global_batch = self._global_batch
        block_tables = self._block_tables
        if step is None or step.transport != "page_pull" or global_batch is None:
            raise RuntimeError("PCP page_pull has no active page-pull step")
        if block_tables is None:
            raise RuntimeError("PCP page_pull requires block tables")
        if block_tables.num_kv_cache_groups != 1:
            raise RuntimeError("PCP page_pull currently requires one KV cache group")

        block_size = int(block_tables.kernel_block_sizes[0])
        blocks_by_segment: list[tuple[int, ...]] = []
        for pieces in step.layout.logical_segments:
            blocks: list[int] = []
            for piece in pieces:
                req_state_idx = int(
                    global_batch.idx_mapping_np[piece.global_batch_req_idx]
                )
                start_block = piece.start_pos // block_size
                end_block = (piece.end_pos + block_size - 1) // block_size
                blocks.extend(
                    block_tables.get_block_ids_cpu(
                        0, req_state_idx, start_block, end_block
                    )
                )
            blocks_by_segment.append(tuple(blocks))

        self._runahead_runtime.configure_page_plan(
            PCPPagePlan(
                segment_to_rank=self._config.segment_to_group_rank,
                blocks_by_segment=tuple(blocks_by_segment),
                block_size=block_size,
            )
        )

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        step = self._active_step
        if step is not None and step.transport == "page_pull":
            with pcp_nvtx_range("pcp.page_pull_plan"):
                self._configure_page_pull_plan()
        return slot_mappings

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        step = self._active_step
        self._runahead_runtime.flush()
        result = super().restore_for_sampling(hidden_states)
        if step is not None and step.transport in (
            "prefix_p2p",
            "direct_p2p",
            "page_pull",
        ):
            self._sharded_kv_history = True
        self._runahead_runtime.disable_step()
        self._active_step = None
        return result


__all__ = [
    "LogicalSegment",
    "RUNAHEAD_MIN_PREFILL_TOKENS",
    "RunaheadPCPManager",
    "RunaheadStep",
    "SegmentLayout",
    "parse_runahead_weights",
    "runahead_batch_eligible",
    "weighted_partition_lengths",
]

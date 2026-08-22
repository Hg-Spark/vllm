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
from vllm.v1.attention.ops.pcp_page_plan import PCPPageRoute
from vllm.v1.attention.ops.pcp_page_state import PCPPageStateTracker
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
class PageOwnerUpdate:
    req_state_idx: int
    request_id: str
    page_idx: int
    owner_rank: int


@dataclass(frozen=True)
class SegmentLayout:
    """One compilation of logical segments for the current global batch."""

    segments_by_rank: tuple[tuple[RankSegment, ...], ...]
    rows_per_rank: tuple[int, ...]
    logical_segments: tuple[tuple[LogicalSegment, ...], ...]
    causal_segments_by_request: tuple[tuple[LogicalSegment, ...], ...] = ()
    page_owner_updates: tuple[PageOwnerUpdate, ...] = ()


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
                "PCP page_pull currently supports one KV cache group, "
                f"got {block_tables.num_kv_cache_groups}"
            )

        self._tensor_sharded_kv_history = False
        self._page_state = (
            PCPPageStateTracker(
                rank=pcp_rank,
                block_size=int(block_tables.kernel_block_sizes[0]),
                max_model_len=req_states.max_model_len,
            )
            if config.transport == "page_pull"
            and block_tables is not None
            and block_tables.kernel_block_sizes
            and req_states is not None
            else None
        )
        self._pending_page_valid_updates: list[tuple[int, str, int, int]] = []
        self._pending_page_advances: list[tuple[int, str, int]] = []

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
        # A chunk that starts inside a mutable page can add one forced segment
        # on its existing owner in addition to the configured logical segments.
        if self._config.transport == "page_pull":
            max_segments_per_rank += 1
        if (
            max_segments_per_rank <= 2
            or max_num_reqs is None
            or max_num_tokens is None
        ):
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

        if model.use_mla and hasattr(model.hf_text_config, "index_topk"):
            raise NotImplementedError(
                "PCP runahead does not support sparse MLA/indexer yet"
            )

        if config.transport == "page_pull":
            cache_dtype = str(vllm_config.cache_config.cache_dtype)
            if cache_dtype not in ("auto", "float16", "bfloat16"):
                raise NotImplementedError(
                    "PCP page_pull requires unquantized FP16/BF16 KV cache, "
                    f"got cache_dtype={cache_dtype}"
                )

    def _partition_lengths(
        self, query_len: int, start_pos: int = 0
    ) -> tuple[int, ...]:
        return weighted_partition_lengths(
            query_len,
            self._config.weights,
            start_pos=start_pos,
            alignment=self._page_alignment,
        )

    def _request_identity(
        self, input_batch: InputBatch, global_req_idx: int
    ) -> tuple[int, str] | None:
        req_states = getattr(self, "_req_states", None)
        if req_states is None or not hasattr(input_batch, "idx_mapping_np"):
            return None
        req_state_idx = int(input_batch.idx_mapping_np[global_req_idx])
        request_id = req_states.index_to_req_id.get(req_state_idx)
        if request_id is None:
            return None
        return req_state_idx, request_id

    def _compile_legacy_segment_layout(
        self, input_batch: InputBatch
    ) -> SegmentLayout | None:
        mapping = self._config.segment_to_group_rank
        num_segments = len(mapping)
        segments_by_rank: list[list[RankSegment]] = [
            [] for _ in range(self.pcp_world_size)
        ]
        logical_segments: list[list[LogicalSegment]] = [
            [] for _ in range(num_segments)
        ]
        causal_by_request: list[list[LogicalSegment]] = [
            [] for _ in range(input_batch.num_reqs)
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
                    piece = LogicalSegment(
                        global_batch_req_idx=req_idx,
                        start_pos=absolute,
                        end_pos=end_pos,
                        owner_group_rank=owner_group_rank,
                    )
                    logical_segments[segment_idx].append(piece)
                    causal_by_request[req_idx].append(piece)
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
            causal_segments_by_request=tuple(
                tuple(items) for items in causal_by_request
            ),
        )

    def _compile_page_pull_segment_layout(
        self, input_batch: InputBatch
    ) -> SegmentLayout | None:
        tracker = self._page_state
        if tracker is None:
            return self._compile_legacy_segment_layout(input_batch)

        mapping = self._config.segment_to_group_rank
        num_segments = len(mapping)
        block_size = tracker.block_size
        segments_by_rank: list[list[RankSegment]] = [
            [] for _ in range(self.pcp_world_size)
        ]
        logical_segments: list[list[LogicalSegment]] = [
            [] for _ in range(num_segments)
        ]
        causal_by_request: list[list[LogicalSegment]] = [
            [] for _ in range(input_batch.num_reqs)
        ]
        rank_rows = [0] * self.pcp_world_size
        owner_updates: dict[tuple[int, int], PageOwnerUpdate] = {}

        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            query_len = int(num_tokens)
            if query_len <= 0:
                continue
            identity = self._request_identity(input_batch, req_idx)
            if identity is None:
                return None
            req_state_idx, request_id = identity
            start_pos = int(input_batch.num_computed_tokens_np[req_idx])
            state = tracker.prepare_request(req_state_idx, request_id, start_pos)
            if start_pos > 0 and not tracker.has_known_prefix(state, start_pos):
                # A nonzero prefix without PCP page state is typically a new
                # request with an APC/prefix-cache hit. Leave it on baseline PCP.
                return None

            global_start = int(input_batch.query_start_loc_np[req_idx])
            offset = 0
            absolute = start_pos
            end_of_query = start_pos + query_len

            def append_piece(
                length: int,
                owner_rank: int,
                legacy_segment_idx: int | None,
                global_batch_req_idx: int = req_idx,
            ) -> bool:
                nonlocal offset, absolute
                if length <= 0:
                    return True
                next_offset = offset + length
                end_pos = absolute + length
                local_start = rank_rows[owner_rank]
                segments_by_rank[owner_rank].append(
                    RankSegment(
                        global_batch_req_idx=global_batch_req_idx,
                        global_batch_slice=slice(
                            global_start + offset, global_start + next_offset
                        ),
                        rank_local_batch_slice=slice(
                            local_start, local_start + length
                        ),
                    )
                )
                piece = LogicalSegment(
                    global_batch_req_idx=global_batch_req_idx,
                    start_pos=absolute,
                    end_pos=end_pos,
                    owner_group_rank=owner_rank,
                )
                causal_by_request[global_batch_req_idx].append(piece)
                if legacy_segment_idx is not None:
                    logical_segments[legacy_segment_idx].append(piece)
                rank_rows[owner_rank] += length

                first_page = absolute // block_size
                last_page = (end_pos + block_size - 1) // block_size
                for page_idx in range(first_page, last_page):
                    known_owner = tracker.owner(state, page_idx)
                    pending = owner_updates.get((req_state_idx, page_idx))
                    pending_owner = pending.owner_rank if pending is not None else -1
                    if known_owner >= 0 and known_owner != owner_rank:
                        return False
                    if pending_owner >= 0 and pending_owner != owner_rank:
                        return False
                    if known_owner < 0 and pending is None:
                        owner_updates[(req_state_idx, page_idx)] = PageOwnerUpdate(
                            req_state_idx=req_state_idx,
                            request_id=request_id,
                            page_idx=page_idx,
                            owner_rank=owner_rank,
                        )
                offset = next_offset
                absolute = end_pos
                return True

            # A partial page already contains historical KV. Keep its existing
            # authoritative owner so native whole-page writes never require an
            # ownership migration or overwrite an old prefix after a remote READ.
            if absolute % block_size:
                tail_page = absolute // block_size
                tail_owner = tracker.owner(state, tail_page)
                if tail_owner < 0:
                    return None
                tail_len = min(query_len, block_size - (absolute % block_size))
                if not append_piece(tail_len, tail_owner, None):
                    return None

            remaining = end_of_query - absolute
            if remaining > 0:
                lengths = weighted_partition_lengths(
                    remaining,
                    self._config.weights,
                    start_pos=absolute,
                    alignment=block_size,
                )
                for segment_idx, length in enumerate(lengths):
                    if length <= 0:
                        continue
                    end_pos = absolute + length
                    if end_pos < end_of_query and end_pos % block_size != 0:
                        return None
                    if not append_piece(
                        length, mapping[segment_idx], segment_idx
                    ):
                        return None

            if absolute != end_of_query:
                raise AssertionError("PCP chunk segment compilation lost tokens")

        return SegmentLayout(
            segments_by_rank=tuple(tuple(items) for items in segments_by_rank),
            rows_per_rank=tuple(rank_rows),
            logical_segments=tuple(tuple(items) for items in logical_segments),
            causal_segments_by_request=tuple(
                tuple(items) for items in causal_by_request
            ),
            page_owner_updates=tuple(owner_updates.values()),
        )

    def _compile_segment_layout(
        self, input_batch: InputBatch
    ) -> SegmentLayout | None:
        if self._config.transport == "page_pull" and getattr(
            self, "_page_state", None
        ) is not None:
            return self._compile_page_pull_segment_layout(input_batch)
        return self._compile_legacy_segment_layout(input_batch)

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

    def _batch_has_known_page_history(self, input_batch: InputBatch) -> bool:
        tracker = getattr(self, "_page_state", None)
        if tracker is None:
            return False
        for req_idx in range(input_batch.num_reqs):
            computed = int(input_batch.num_computed_tokens_np[req_idx])
            if computed <= 0:
                continue
            identity = self._request_identity(input_batch, req_idx)
            if identity is None:
                continue
            req_state_idx, request_id = identity
            state = tracker.existing_request(req_state_idx, request_id)
            if state is not None and tracker.has_known_prefix(state, computed):
                return True
        return False

    def _commit_page_owner_updates(
        self, layout: SegmentLayout, input_batch: InputBatch
    ) -> None:
        tracker = self._page_state
        if tracker is None:
            return
        for update in layout.page_owner_updates:
            state = tracker.existing_request(update.req_state_idx, update.request_id)
            if state is None:
                raise RuntimeError("PCP page-state request disappeared during planning")
            tracker.assign_owner(state, update.page_idx, update.owner_rank)
        for req_idx in range(input_batch.num_reqs):
            identity = self._request_identity(input_batch, req_idx)
            if identity is None:
                continue
            req_state_idx, request_id = identity
            state = tracker.existing_request(req_state_idx, request_id)
            if state is None:
                continue
            tracker.invalidate_mutable_tail(
                state, int(input_batch.num_computed_tokens_np[req_idx])
            )

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        if getattr(self, "_tensor_sharded_kv_history", False):
            raise RuntimeError(
                "PCP tensor runahead left persistent causal-prefix KV sharded across "
                "ranks; continued/decode execution is not implemented"
            )

        page_pull = self._config.transport == "page_pull"
        known_page_history = page_pull and self._batch_has_known_page_history(input_batch)
        all_prefilling = bool(
            input_batch.is_prefilling_np[: input_batch.num_reqs].all()
        )
        if known_page_history and not all_prefilling:
            raise RuntimeError(
                "PCP page_pull chunk history is sharded across ranks; decode after "
                "runahead prefill is not implemented"
            )

        eligible = runahead_batch_eligible(
            num_reqs=input_batch.num_reqs,
            is_prefilling=input_batch.is_prefilling_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            num_computed_tokens=input_batch.num_computed_tokens_np,
            prefill_len=input_batch.prefill_len_np,
            pcp_world_size=self.pcp_world_size,
            require_full_prefill=not page_pull,
            # Once a request has PCP-sharded history it must remain on page-pull
            # for subsequent chunks. The fresh-step threshold is only a policy gate.
            min_prefill_tokens=0 if known_page_history else self._config.min_tokens,
        )
        if known_page_history and not eligible:
            raise RuntimeError(
                "PCP page_pull continuation cannot safely fall back to baseline; "
                "the scheduled chunk is too small for the active PCP world size"
            )

        layout = self._compile_segment_layout(input_batch) if eligible else None
        if eligible and layout is None:
            if known_page_history:
                raise RuntimeError(
                    "PCP page_pull continuation could not compile a safe chunk plan; "
                    "refusing baseline fallback with sharded history"
                )
            logger.debug(
                "PCP page_pull falling back because the chunk/page state is unsupported"
            )
            eligible = False
        if layout is not None and any(rows <= 0 for rows in layout.rows_per_rank):
            if known_page_history:
                raise RuntimeError(
                    "PCP page_pull continuation produced an empty physical rank; "
                    "inactive-rank execution is not implemented"
                )
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
            self._pending_page_valid_updates.clear()
            self._pending_page_advances.clear()
            self._runahead_runtime.disable_step()
            return local_batch

        if step.transport == "page_pull":
            self._commit_page_owner_updates(step.layout, input_batch)
        self._runahead_runtime.begin_step(
            step.layout.rows_per_rank,
            transport=step.transport,
        )
        return local_batch

    @staticmethod
    def _build_route_matrix(
        pages: list[list[list[int]]], world_size: int
    ) -> tuple[tuple[PCPPageRoute | None, ...], ...]:
        rows: list[tuple[PCPPageRoute | None, ...]] = []
        for destination_rank in range(world_size):
            row: list[PCPPageRoute | None] = []
            for source_rank in range(world_size):
                block_ids = tuple(dict.fromkeys(pages[destination_rank][source_rank]))
                row.append(
                    PCPPageRoute(
                        destination_rank=destination_rank,
                        source_rank=source_rank,
                        destination_block_ids=block_ids,
                        source_block_ids=block_ids,
                    )
                    if block_ids
                    else None
                )
            rows.append(tuple(row))
        return tuple(rows)

    def _configure_chunked_page_pull_plan(self) -> None:
        step = self._active_step
        global_batch = self._global_batch
        block_tables = self._block_tables
        tracker = self._page_state
        if (
            step is None
            or global_batch is None
            or block_tables is None
            or tracker is None
        ):
            raise RuntimeError("PCP chunked page_pull is missing planner state")

        world_size = self.pcp_world_size
        block_size = tracker.block_size
        history_pages = [
            [[] for _ in range(world_size)] for _ in range(world_size)
        ]
        current_pages = [
            [[] for _ in range(world_size)] for _ in range(world_size)
        ]
        self._pending_page_valid_updates.clear()
        self._pending_page_advances.clear()

        for req_idx, segments in enumerate(step.layout.causal_segments_by_request):
            if not segments:
                continue
            identity = self._request_identity(global_batch, req_idx)
            if identity is None:
                raise RuntimeError("PCP page_pull request identity is unavailable")
            req_state_idx, request_id = identity
            start_pos = int(global_batch.num_computed_tokens_np[req_idx])
            end_pos = start_pos + int(global_batch.num_scheduled_tokens[req_idx])
            state = tracker.existing_request(req_state_idx, request_id)
            if state is None or not tracker.has_known_prefix(state, start_pos):
                raise RuntimeError("PCP page_pull lost persistent request page state")

            end_page = (end_pos + block_size - 1) // block_size
            request_blocks = block_tables.get_block_ids_cpu(
                0, req_state_idx, 0, end_page
            )
            if len(request_blocks) < end_page:
                raise RuntimeError(
                    "PCP page_pull block table is shorter than chunk plan"
                )

            owned_indices_by_rank = [
                [
                    index
                    for index, piece in enumerate(segments)
                    if piece.owner_group_rank == rank
                ]
                for rank in range(world_size)
            ]

            # Historical full pages are already complete on their authoritative
            # producer. Only this process' missing pages need explicit routes.
            local_owned_indices = owned_indices_by_rank[self.pcp_rank]
            history_page_limit = start_pos // block_size
            if local_owned_indices:
                for page_idx in range(history_page_limit):
                    block_id = int(request_blocks[page_idx])
                    owner_rank = tracker.owner(state, page_idx)
                    if owner_rank < 0:
                        raise RuntimeError(
                            "PCP history page has no authoritative owner"
                        )
                    if owner_rank == self.pcp_rank:
                        # Local allocation/COW is managed by vLLM. Treat the
                        # authoritative owner's current physical block as valid.
                        self._pending_page_valid_updates.append(
                            (req_state_idx, request_id, page_idx, block_id)
                        )
                    elif not tracker.local_is_valid(state, page_idx, block_id):
                        history_pages[self.pcp_rank][owner_rank].append(block_id)
                        self._pending_page_valid_updates.append(
                            (req_state_idx, request_id, page_idx, block_id)
                        )

            # Current-chunk routes are deterministic across ranks and therefore
            # compiled for every destination. READY fanout uses these rows.
            for destination_rank, owned_indices in enumerate(
                owned_indices_by_rank
            ):
                if not owned_indices:
                    continue
                max_owned_index = owned_indices[-1]
                for piece in segments[:max_owned_index]:
                    source_rank = piece.owner_group_rank
                    if source_rank == destination_rank:
                        continue
                    first_page = piece.start_pos // block_size
                    last_page = (piece.end_pos + block_size - 1) // block_size
                    current_pages[destination_rank][source_rank].extend(
                        int(request_blocks[page_idx])
                        for page_idx in range(first_page, last_page)
                    )

            # After a successful forward, this rank owns a valid copy of every
            # current page up to its last causal segment: locally produced pages
            # plus predecessor pages pulled before attention.
            if local_owned_indices:
                max_local_index = local_owned_indices[-1]
                seen_pages: set[int] = set()
                for piece in segments[: max_local_index + 1]:
                    first_page = piece.start_pos // block_size
                    last_page = (piece.end_pos + block_size - 1) // block_size
                    for page_idx in range(first_page, last_page):
                        if page_idx in seen_pages:
                            continue
                        seen_pages.add(page_idx)
                        self._pending_page_valid_updates.append(
                            (
                                req_state_idx,
                                request_id,
                                page_idx,
                                int(request_blocks[page_idx]),
                            )
                        )
            self._pending_page_advances.append(
                (req_state_idx, request_id, end_pos)
            )

        self._runahead_runtime.configure_page_plan(
            PCPPagePlan(
                segment_to_rank=(),
                blocks_by_segment=(),
                block_size=block_size,
                history_routes_by_rank=self._build_route_matrix(
                    history_pages, world_size
                ),
                current_routes_by_rank=self._build_route_matrix(
                    current_pages, world_size
                ),
                explicit_world_size=world_size,
            )
        )

    def _configure_legacy_page_pull_plan(self) -> None:
        step = self._active_step
        global_batch = self._global_batch
        block_tables = self._block_tables
        if step is None or global_batch is None or block_tables is None:
            raise RuntimeError("PCP page_pull has no active planner state")
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

    def _configure_page_pull_plan(self) -> None:
        step = self._active_step
        block_tables = self._block_tables
        if step is None or step.transport != "page_pull":
            raise RuntimeError("PCP page_pull has no active page-pull step")
        if block_tables is None:
            raise RuntimeError("PCP page_pull requires block tables")
        if block_tables.num_kv_cache_groups != 1:
            raise RuntimeError("PCP page_pull currently requires one KV cache group")
        if self._page_state is not None and step.layout.causal_segments_by_request:
            self._configure_chunked_page_pull_plan()
        else:
            self._configure_legacy_page_pull_plan()

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        step = self._active_step
        if step is not None and step.transport == "page_pull":
            with pcp_nvtx_range("pcp.page_pull_plan"):
                self._configure_page_pull_plan()
        return slot_mappings

    def _commit_page_step_completion(self) -> None:
        tracker = self._page_state
        if tracker is None:
            return
        for req_state_idx, request_id, page_idx, block_id in (
            self._pending_page_valid_updates
        ):
            state = tracker.existing_request(req_state_idx, request_id)
            if state is None:
                raise RuntimeError("PCP page-state request disappeared after forward")
            tracker.mark_local_valid(state, page_idx, block_id)
        for req_state_idx, request_id, computed_tokens in self._pending_page_advances:
            state = tracker.existing_request(req_state_idx, request_id)
            if state is None:
                raise RuntimeError("PCP page-state request disappeared after forward")
            tracker.advance(state, computed_tokens)
        self._pending_page_valid_updates.clear()
        self._pending_page_advances.clear()

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        step = self._active_step
        self._runahead_runtime.flush()
        result = super().restore_for_sampling(hidden_states)
        if step is not None and step.transport == "page_pull":
            self._commit_page_step_completion()
        elif step is not None and step.transport in ("prefix_p2p", "direct_p2p"):
            self._tensor_sharded_kv_history = True
        self._runahead_runtime.disable_step()
        self._active_step = None
        return result


__all__ = [
    "LogicalSegment",
    "PageOwnerUpdate",
    "RUNAHEAD_MIN_PREFILL_TOKENS",
    "RunaheadPCPManager",
    "RunaheadStep",
    "SegmentLayout",
    "parse_runahead_weights",
    "runahead_batch_eligible",
    "weighted_partition_lengths",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP manager extension for configurable causal-prefix experiments."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan
from vllm.v1.attention.ops.pcp_profile import (
    install_pcp_nvtx_hooks,
    pcp_nvtx_range,
)
from vllm.v1.attention.ops.pcp_runahead import (
    PCPRunaheadRuntime,
    register_pcp_runahead_runtime,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.pcp_runahead_config import (
    RUNAHEAD_MIN_PREFILL_TOKENS,
    RUNAHEAD_WEIGHTS_KEY,
    PCPRunaheadConfig,
    parse_pcp_runahead_config,
    parse_runahead_weights,
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
    """Split one request by weights and align internal absolute cuts."""
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
        boundaries[index + 1] - boundaries[index]
        for index in range(num_segments)
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
    """Select batches eligible for the configured PCP experiment step."""
    if num_reqs <= 0:
        return False
    if not bool(is_prefilling[:num_reqs].all()):
        return False
    if require_full_prefill:
        if not bool((num_computed_tokens[:num_reqs] == 0).all()):
            return False
        if not bool(
            (
                num_scheduled_tokens[:num_reqs]
                == prefill_len[:num_reqs]
            ).all()
        ):
            return False
    total_prefill_tokens = int(num_scheduled_tokens[:num_reqs].sum())
    return total_prefill_tokens >= max(pcp_world_size, min_prefill_tokens)


def compact_hidden_restore_idx(
    padded_restore_idx: torch.Tensor,
    *,
    padded_rows: int,
    rows_per_rank: tuple[int, ...],
) -> torch.Tensor:
    """Map vLLM's padded rank-major restore index into compact rank-major space."""
    if padded_rows <= 0:
        raise ValueError(f"padded_rows must be positive, got {padded_rows}")
    offsets = [0]
    for rows in rows_per_rank[:-1]:
        offsets.append(offsets[-1] + int(rows))
    offset_tensor = torch.tensor(
        offsets,
        dtype=padded_restore_idx.dtype,
        device=padded_restore_idx.device,
    )
    ranks = torch.div(padded_restore_idx, padded_rows, rounding_mode="floor")
    local = padded_restore_idx - ranks * padded_rows
    return offset_tensor[ranks] + local


class RunaheadPCPManager(PCPManager):
    """Reuse PCP batch machinery while replacing partition/transport policies."""

    _validated_config: ClassVar[PCPRunaheadConfig | None] = None

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
        config = type(self)._validated_config
        if config is None:
            raise RuntimeError(
                "PCP runahead manager was built without validated config"
            )
        self._config = config
        install_pcp_nvtx_hooks()
        self._standard_attention_pcp = False
        self._use_custom_partition = False
        self._use_compact_layout = False
        self._step_transport: str | None = None
        self._page_alignment = (
            math.lcm(*block_tables.kernel_block_sizes)
            if config.page_align
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

        self._rows_per_rank: tuple[int, ...] = ()
        self._segment_rows: tuple[int, ...] = ()
        self._logical_segment_slices: tuple[tuple[slice, ...], ...] = ()
        self._compact_hidden_restore_idx: torch.Tensor | None = None
        self._sharded_kv_history = False
        self._runahead_runtime = PCPRunaheadRuntime(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
            max_inflight_sends=config.max_inflight_sends,
            max_inflight_reads=config.max_inflight_reads,
            nixl_backends=config.nixl_backends,
        )
        register_pcp_runahead_runtime(self._runahead_runtime)
        self._resize_local_request_buffers_if_needed(
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            device=device,
        )

    def _resize_local_request_buffers_if_needed(
        self,
        *,
        max_num_reqs: int | None,
        max_num_tokens: int | None,
        block_tables: BlockTables | None,
        device: torch.device,
    ) -> None:
        """Allow more than the stock two logical chunks on one physical rank."""
        max_segments_per_rank = max(
            self._segment_to_rank().count(rank) for rank in range(self.pcp_world_size)
        )
        if max_segments_per_rank <= 2 or max_num_reqs is None or max_num_tokens is None:
            return
        max_num_local_reqs = max_segments_per_rank * max_num_reqs
        self._input_buffers = InputBuffers(
            max_num_local_reqs, max_num_tokens, device
        )
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

    def _segment_to_rank(self) -> tuple[int, ...]:
        mapping = self._config.segment_to_rank
        if mapping:
            return mapping
        return tuple(range(self.pcp_world_size))

    def _segments_for_rank(self, rank: int) -> tuple[int, ...]:
        return tuple(
            segment_idx
            for segment_idx, owner in enumerate(self._segment_to_rank())
            if owner == rank
        )

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
        cls._validated_config = config

        if model.use_mla:
            raise NotImplementedError(
                "experimental PCP runahead currently supports standard attention only"
            )
        if model.is_encoder_decoder:
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

        if parallel.tensor_parallel_size != 1:
            raise NotImplementedError("runahead PCP MVP requires TP=1")
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("runahead PCP MVP requires PP=1")
        if parallel.data_parallel_size != 1:
            raise NotImplementedError("runahead PCP MVP requires DP=1")
        if parallel.decode_context_parallel_size != 1:
            raise NotImplementedError("runahead PCP MVP requires DCP=1")
        if parallel.enable_expert_parallel:
            raise NotImplementedError("runahead PCP MVP does not support EP")
        if parallel.enable_dbo:
            raise NotImplementedError("runahead PCP MVP does not support DBO")
        if model.is_moe:
            raise NotImplementedError(
                "runahead PCP MVP currently rejects MoE layer collectives"
            )
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise NotImplementedError("runahead PCP MVP requires --enforce-eager")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "runahead PCP MVP does not support async scheduling"
            )
        if config.transport == "page_pull":
            cache_dtype = str(vllm_config.cache_config.cache_dtype)
            if cache_dtype not in ("auto", "float16", "bfloat16"):
                raise NotImplementedError(
                    "PCP page_pull currently requires unquantized FP16/BF16 KV "
                    f"cache, got cache_dtype={cache_dtype}"
                )

    def _partition_lengths(
        self,
        query_len: int,
        start_pos: int = 0,
    ) -> tuple[int, ...]:
        if self._config.partition_policy == "weighted_contiguous":
            assert self._config.weights is not None
            weights = self._config.weights
        else:
            weights = (1.0,) * len(self._segment_to_rank())
        return weighted_partition_lengths(
            query_len,
            weights,
            start_pos=start_pos,
            alignment=self._page_alignment,
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        if not self._use_custom_partition:
            return super()._get_rank_segments(
                rank,
                num_scheduled_tokens,
                num_computed_tokens,
                is_prefilling,
                query_start_loc_np,
            )

        segment_indices = self._segments_for_rank(rank)
        rank_segments: list[RankSegment] = []
        rank_offset = 0
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            start_pos = int(num_computed_tokens[global_batch_req_idx])
            lengths = self._partition_lengths(query_len, start_pos)
            offsets = np.cumsum((0, *lengths))
            for segment_idx in segment_indices:
                chunk_len = lengths[segment_idx]
                if chunk_len <= 0:
                    continue
                chunk_start = global_batch_start + int(offsets[segment_idx])
                rank_segments.append(
                    RankSegment(
                        global_batch_req_idx=global_batch_req_idx,
                        global_batch_slice=slice(
                            chunk_start, chunk_start + chunk_len
                        ),
                        rank_local_batch_slice=slice(
                            rank_offset, rank_offset + chunk_len
                        ),
                    )
                )
                rank_offset += chunk_len
        return rank_segments

    def _custom_rows_per_rank(self, input_batch: InputBatch) -> tuple[int, ...]:
        rows = [0] * self.pcp_world_size
        segment_to_rank = self._segment_to_rank()
        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            lengths = self._partition_lengths(
                int(num_tokens),
                int(input_batch.num_computed_tokens_np[req_idx]),
            )
            for segment_idx, length in enumerate(lengths):
                rows[segment_to_rank[segment_idx]] += length
        return tuple(rows)

    def _custom_rows_per_segment(self, input_batch: InputBatch) -> tuple[int, ...]:
        rows = [0] * len(self._segment_to_rank())
        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            lengths = self._partition_lengths(
                int(num_tokens),
                int(input_batch.num_computed_tokens_np[req_idx]),
            )
            for segment_idx, length in enumerate(lengths):
                rows[segment_idx] += length
        return tuple(rows)

    def _build_logical_segment_slices(
        self, input_batch: InputBatch
    ) -> tuple[tuple[slice, ...], ...]:
        result: list[list[slice]] = [
            [] for _ in range(len(self._segment_to_rank()))
        ]
        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            query_len = int(num_tokens)
            if query_len <= 0:
                continue
            start_pos = int(input_batch.num_computed_tokens_np[req_idx])
            lengths = self._partition_lengths(query_len, start_pos)
            offsets = np.cumsum((0, *lengths))
            global_start = int(input_batch.query_start_loc_np[req_idx])
            for segment_idx, length in enumerate(lengths):
                if length <= 0:
                    continue
                start = global_start + int(offsets[segment_idx])
                result[segment_idx].append(slice(start, start + length))
        return tuple(tuple(slices) for slices in result)

    def _page_pull_boundaries_are_aligned(self, input_batch: InputBatch) -> bool:
        if self._page_alignment <= 1:
            return True
        num_segments = len(self._segment_to_rank())
        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            start_pos = int(input_batch.num_computed_tokens_np[req_idx])
            lengths = self._partition_lengths(int(num_tokens), start_pos)
            absolute = start_pos
            for segment_idx, length in enumerate(lengths[:-1]):
                absolute += length
                if absolute % self._page_alignment != 0:
                    logger.debug(
                        "PCP page_pull falling back because request %d segment %d "
                        "ends off page boundary: absolute=%d alignment=%d",
                        req_idx,
                        segment_idx,
                        absolute,
                        self._page_alignment,
                    )
                    return False
            if len(lengths) != num_segments:
                return False
        return True

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        if self._sharded_kv_history:
            raise RuntimeError(
                "PCP runahead left persistent KV causal-prefix sharded across ranks; "
                "another model step requires a sharded-KV decode/continue path, "
                "which is not implemented by this experimental branch."
            )

        eligible = self._standard_attention_pcp and runahead_batch_eligible(
            num_reqs=input_batch.num_reqs,
            is_prefilling=input_batch.is_prefilling_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            num_computed_tokens=input_batch.num_computed_tokens_np,
            prefill_len=input_batch.prefill_len_np,
            pcp_world_size=self.pcp_world_size,
            require_full_prefill=self._config.require_full_prefill,
            min_prefill_tokens=self._config.min_tokens,
        )
        if (
            eligible
            and self._config.transport == "page_pull"
            and not self._page_pull_boundaries_are_aligned(input_batch)
        ):
            eligible = False

        self._use_custom_partition = (
            eligible and self._config.partition_policy != "stock"
        )
        rows_per_rank: tuple[int, ...] = ()
        segment_rows: tuple[int, ...] = ()
        logical_segment_slices: tuple[tuple[slice, ...], ...] = ()
        if self._use_custom_partition:
            rows_per_rank = self._custom_rows_per_rank(input_batch)
            segment_rows = self._custom_rows_per_segment(input_batch)
            logical_segment_slices = self._build_logical_segment_slices(input_batch)
            if any(rows <= 0 for rows in rows_per_rank):
                logger.debug(
                    "PCP custom partition produced an empty rank; "
                    "falling back: rows=%s",
                    rows_per_rank,
                )
                eligible = False
                self._use_custom_partition = False
                rows_per_rank = ()
                segment_rows = ()
                logical_segment_slices = ()

        local_batch = super().partition_batch(input_batch)

        compact = (
            eligible
            and self._use_custom_partition
            and self._config.layout == "compact"
        )
        self._use_compact_layout = compact
        self._step_transport = self._config.transport if eligible else None

        if compact:
            padded_rows = int(local_batch.num_tokens_after_padding)
            local_rows = rows_per_rank[self.pcp_rank]
            assert self._hidden_restore_idx is not None
            self._compact_hidden_restore_idx = compact_hidden_restore_idx(
                self._hidden_restore_idx,
                padded_rows=padded_rows,
                rows_per_rank=rows_per_rank,
            )
            self._rows_per_rank = rows_per_rank
            self._segment_rows = segment_rows
            self._logical_segment_slices = logical_segment_slices
            local_batch = replace(
                local_batch,
                num_tokens_after_padding=local_rows,
                input_ids=local_batch.input_ids[:local_rows],
                positions=local_batch.positions[:local_rows],
                is_padding=local_batch.is_padding[:local_rows],
            )
            self._runahead_runtime.begin_step(
                rows_per_rank,
                transport=self._config.transport,
                segment_to_rank=self._segment_to_rank(),
                segment_rows=segment_rows,
            )
        else:
            self._rows_per_rank = ()
            self._segment_rows = ()
            self._logical_segment_slices = ()
            self._compact_hidden_restore_idx = None
            self._runahead_runtime.disable_step()

        return local_batch

    @staticmethod
    def _blocks_from_segment_slots(
        slots: torch.Tensor,
        *,
        block_size: int,
    ) -> tuple[int, ...]:
        if slots.numel() == 0:
            return ()
        if bool((slots < 0).any()):
            raise RuntimeError("PCP page_pull encountered PAD slot inside a segment")
        block_ids = torch.div(slots, block_size, rounding_mode="floor")
        return tuple(int(value) for value in torch.unique_consecutive(block_ids).cpu())

    def _configure_page_pull_plan(self) -> None:
        if self._config.transport != "page_pull":
            return
        if not self._logical_segment_slices:
            raise RuntimeError("PCP page_pull has no logical segment slices")
        if self._block_tables is None or self._global_batch_slot_mappings is None:
            raise RuntimeError("PCP page_pull requires block tables and global slots")
        if self._block_tables.num_kv_cache_groups != 1:
            raise RuntimeError("PCP page_pull currently requires one KV cache group")

        block_size = int(self._block_tables.kernel_block_sizes[0])
        global_slots = self._global_batch_slot_mappings[0]
        destination_blocks: list[tuple[int, ...]] = []
        for segment_slices in self._logical_segment_slices:
            blocks: list[int] = []
            for segment_slice in segment_slices:
                blocks.extend(
                    self._blocks_from_segment_slots(
                        global_slots[segment_slice], block_size=block_size
                    )
                )
            destination_blocks.append(tuple(blocks))

        mapping = self._segment_to_rank()
        locally_owned = {
            segment_idx: destination_blocks[segment_idx]
            for segment_idx, owner in enumerate(mapping)
            if owner == self.pcp_rank
        }
        group = get_pcp_group()
        gathered: list[dict[int, tuple[int, ...]] | None] = [
            None for _ in range(self.pcp_world_size)
        ]
        dist.all_gather_object(gathered, locally_owned, group=group.cpu_group)

        source_blocks: list[tuple[int, ...]] = []
        for segment_idx, owner in enumerate(mapping):
            owner_map = gathered[owner]
            if owner_map is None or segment_idx not in owner_map:
                raise RuntimeError(
                    "PCP page_pull did not receive source block metadata for "
                    f"segment={segment_idx}, owner={owner}"
                )
            source_blocks.append(tuple(owner_map[segment_idx]))

        plan = PCPPagePlan(
            segment_to_rank=mapping,
            source_blocks_by_segment=tuple(source_blocks),
            destination_blocks_by_segment=tuple(destination_blocks),
            block_size=block_size,
        )
        self._runahead_runtime.configure_page_plan(plan)

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        if not self._use_compact_layout:
            return slot_mappings
        assert self._gathered_kv_write_mask is not None
        with pcp_nvtx_range("pcp.compact_slot_mapping"):
            compact = slot_mappings[:, self._gathered_kv_write_mask].contiguous()
        if self._config.transport == "page_pull":
            with pcp_nvtx_range("pcp.page_pull_plan"):
                self._configure_page_pull_plan()
        return compact

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._compact_hidden_restore_idx is None:
            return super().restore_hidden_states(hidden_states)
        with pcp_nvtx_range("pcp.restore_hidden_variable_allgather"):
            gathered = get_pcp_group().all_gatherv(
                hidden_states.contiguous(),
                dim=0,
                sizes=list(self._rows_per_rank),
            )
        return gathered[self._compact_hidden_restore_idx]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        self._runahead_runtime.flush()
        result = super().restore_for_sampling(hidden_states)
        if self._step_transport in ("prefix_p2p", "direct_p2p", "page_pull"):
            self._sharded_kv_history = True
        self._runahead_runtime.disable_step()
        self._step_transport = None
        return result


__all__ = [
    "RUNAHEAD_MIN_PREFILL_TOKENS",
    "RUNAHEAD_WEIGHTS_KEY",
    "RunaheadPCPManager",
    "compact_hidden_restore_idx",
    "parse_runahead_weights",
    "runahead_batch_eligible",
    "weighted_partition_lengths",
]
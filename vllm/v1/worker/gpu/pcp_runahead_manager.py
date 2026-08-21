# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP manager extension for causal-prefix runahead."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_world_group,
    init_model_parallel_group,
)
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
from vllm.v1.worker.gpu.pcp_runahead_config import (
    RUNAHEAD_MIN_PREFILL_TOKENS,
    RUNAHEAD_WEIGHTS_KEY,
    PCPRunaheadConfig,
    parse_pcp_runahead_config,
    parse_runahead_weights,
)
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

_RUNAHEAD_PCP_GROUP: Any | None = None
_RUNAHEAD_PCP_GROUP_ORDER: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SegmentLayout:
    """One compilation of logical segments for the current global batch."""

    segments_by_rank: tuple[tuple[RankSegment, ...], ...]
    rows_per_rank: tuple[int, ...]
    rows_per_segment: tuple[int, ...]
    logical_segment_slices: tuple[tuple[slice, ...], ...]
    boundaries_aligned: bool


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


def compact_hidden_restore_idx(
    padded_restore_idx: torch.Tensor,
    *,
    padded_rows: int,
    rows_per_rank: tuple[int, ...],
) -> torch.Tensor:
    """Map the base manager's padded restore index into compact rank-major space."""
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


def _build_logical_pcp_group(config: PCPRunaheadConfig):
    """Build a communicator whose rank order is logical segment order."""
    base = get_pcp_group()
    if not config.mapping_is_permutation or config.process_group_order == tuple(
        range(config.pcp_world_size)
    ):
        return base

    global _RUNAHEAD_PCP_GROUP, _RUNAHEAD_PCP_GROUP_ORDER
    order = config.process_group_order
    if _RUNAHEAD_PCP_GROUP is not None:
        if _RUNAHEAD_PCP_GROUP_ORDER != order:
            raise RuntimeError(
                "PCP runahead process-group order changed in one process: "
                f"old={_RUNAHEAD_PCP_GROUP_ORDER}, new={order}"
            )
        return _RUNAHEAD_PCP_GROUP

    world = get_world_group()
    if world.world_size % config.pcp_world_size != 0:
        raise RuntimeError(
            "world size is not divisible by PCP size for runahead group binding"
        )
    groups = []
    for first in range(0, world.world_size, config.pcp_world_size):
        natural = list(range(first, first + config.pcp_world_size))
        groups.append([natural[physical_rank] for physical_rank in order])

    backend = torch.distributed.get_backend(base.device_group)
    _RUNAHEAD_PCP_GROUP = init_model_parallel_group(
        groups,
        world.local_rank,
        backend,
        group_name="pcp_runahead",
    )
    _RUNAHEAD_PCP_GROUP_ORDER = order
    return _RUNAHEAD_PCP_GROUP


class RunaheadPCPManager(PCPManager):
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
        config = type(self)._validated_config
        if config is None:
            raise RuntimeError("PCP runahead manager was built without validated config")
        self._config = config
        self._physical_pcp_rank = pcp_rank
        self._pcp_group = _build_logical_pcp_group(config)
        self._logical_pcp_rank = self._pcp_group.rank_in_group

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
        self._use_custom_partition = False
        self._use_compact_layout = False
        self._active_layout: SegmentLayout | None = None
        self._step_transport: str | None = None
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

        self._rows_per_rank: tuple[int, ...] = ()
        self._logical_segment_slices: tuple[tuple[slice, ...], ...] = ()
        self._compact_hidden_restore_idx: torch.Tensor | None = None
        self._sharded_kv_history = False
        runtime_rank = (
            self._logical_pcp_rank if config.mapping_is_permutation else pcp_rank
        )
        runtime_group = self._pcp_group if config.mapping_is_permutation else get_pcp_group()
        self._runahead_runtime = PCPRunaheadRuntime(
            pcp_world_size=pcp_world_size,
            pcp_rank=runtime_rank,
            device=device,
            max_inflight_sends=config.max_inflight_sends,
            max_inflight_reads=config.max_inflight_reads,
            nixl_backends=config.nixl_backends,
            pcp_group=runtime_group,
        )
        register_pcp_runahead_runtime(self._runahead_runtime)
        if config.transport == "page_pull":
            from vllm.v1.attention.ops.pcp_standard import (
                install_page_pull_cache_update_hook,
            )

            install_page_pull_cache_update_hook()
        self._resize_local_request_buffers_if_needed(
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            device=device,
        )

    def _effective_segment_to_rank(self) -> tuple[int, ...]:
        return self._config.runtime_segment_to_rank

    def _resize_local_request_buffers_if_needed(
        self,
        *,
        max_num_reqs: int | None,
        max_num_tokens: int | None,
        block_tables: BlockTables | None,
        device: torch.device,
    ) -> None:
        mapping = self._effective_segment_to_rank()
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
        cls._validated_config = config

        if model.use_mla:
            raise NotImplementedError("PCP runahead currently supports standard attention only")
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
        if parallel.enable_expert_parallel or model.is_moe:
            raise NotImplementedError("PCP runahead does not support expert/MoE parallelism")
        if parallel.enable_dbo:
            raise NotImplementedError("PCP runahead does not support DBO")
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise NotImplementedError("PCP runahead requires --enforce-eager")
        if vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError("PCP runahead does not support async scheduling")
        if config.transport == "page_pull":
            cache_dtype = str(vllm_config.cache_config.cache_dtype)
            if cache_dtype not in ("auto", "float16", "bfloat16"):
                raise NotImplementedError(
                    "PCP page_pull requires unquantized FP16/BF16 KV cache, "
                    f"got cache_dtype={cache_dtype}"
                )

    def _partition_lengths(self, query_len: int, start_pos: int = 0) -> tuple[int, ...]:
        weights = self._config.weights or (1.0,) * self.pcp_world_size
        return weighted_partition_lengths(
            query_len,
            weights,
            start_pos=start_pos,
            alignment=self._page_alignment,
        )

    def _compile_segment_layout(self, input_batch: InputBatch) -> SegmentLayout:
        mapping = self._effective_segment_to_rank()
        num_segments = len(mapping)
        segments_by_rank: list[list[RankSegment]] = [
            [] for _ in range(self.pcp_world_size)
        ]
        logical_slices: list[list[slice]] = [[] for _ in range(num_segments)]
        rank_rows = [0] * self.pcp_world_size
        segment_rows = [0] * num_segments
        aligned = True

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
                owner = mapping[segment_idx]
                if length > 0:
                    global_slice = slice(
                        global_start + offset, global_start + next_offset
                    )
                    local_start = rank_rows[owner]
                    segments_by_rank[owner].append(
                        RankSegment(
                            global_batch_req_idx=req_idx,
                            global_batch_slice=global_slice,
                            rank_local_batch_slice=slice(
                                local_start, local_start + length
                            ),
                        )
                    )
                    logical_slices[segment_idx].append(global_slice)
                    rank_rows[owner] += length
                    segment_rows[segment_idx] += length
                offset = next_offset
                absolute += length
                if segment_idx + 1 < num_segments and self._page_alignment > 1:
                    aligned = aligned and absolute % self._page_alignment == 0

        return SegmentLayout(
            segments_by_rank=tuple(tuple(items) for items in segments_by_rank),
            rows_per_rank=tuple(rank_rows),
            rows_per_segment=tuple(segment_rows),
            logical_segment_slices=tuple(tuple(items) for items in logical_slices),
            boundaries_aligned=aligned,
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        if self._use_custom_partition and self._active_layout is not None:
            return list(self._active_layout.segments_by_rank[rank])
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

        eligible = self._standard_attention_pcp and runahead_batch_eligible(
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
        if layout is not None and any(rows <= 0 for rows in layout.rows_per_rank):
            logger.debug(
                "PCP runahead partition produced an empty rank; falling back: rows=%s",
                layout.rows_per_rank,
            )
            eligible = False
            layout = None
        if (
            layout is not None
            and self._config.transport == "page_pull"
            and not layout.boundaries_aligned
        ):
            logger.debug("PCP page_pull falling back because a segment boundary is off-page")
            eligible = False
            layout = None

        self._use_custom_partition = eligible
        self._active_layout = layout
        self._use_compact_layout = eligible
        self._step_transport = self._config.transport if eligible else None
        self.pcp_rank = (
            self._logical_pcp_rank
            if eligible and self._config.mapping_is_permutation
            else self._physical_pcp_rank
        )

        local_batch = super().partition_batch(input_batch)
        if not eligible or layout is None:
            self._rows_per_rank = ()
            self._logical_segment_slices = ()
            self._compact_hidden_restore_idx = None
            self._runahead_runtime.disable_step()
            return local_batch

        padded_rows = int(local_batch.num_tokens_after_padding)
        local_rows = layout.rows_per_rank[self.pcp_rank]
        assert self._hidden_restore_idx is not None
        self._compact_hidden_restore_idx = compact_hidden_restore_idx(
            self._hidden_restore_idx,
            padded_rows=padded_rows,
            rows_per_rank=layout.rows_per_rank,
        )
        self._rows_per_rank = layout.rows_per_rank
        self._logical_segment_slices = layout.logical_segment_slices
        local_batch = replace(
            local_batch,
            num_tokens_after_padding=local_rows,
            input_ids=local_batch.input_ids[:local_rows],
            positions=local_batch.positions[:local_rows],
            is_padding=local_batch.is_padding[:local_rows],
        )
        self._runahead_runtime.begin_step(
            layout.rows_per_rank,
            transport=self._config.transport,
        )
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

    def _verify_page_mapping_invariant(
        self, blocks_by_segment: tuple[tuple[int, ...], ...]
    ) -> None:
        payload = repr(blocks_by_segment).encode("utf-8")
        fingerprint = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        ) & ((1 << 63) - 1)
        local = torch.tensor([fingerprint], dtype=torch.int64, device="cpu")
        gathered = [torch.empty_like(local) for _ in range(self.pcp_world_size)]
        dist.all_gather(gathered, local, group=self._pcp_group.cpu_group)
        values = [int(item.item()) for item in gathered]
        if len(set(values)) != 1:
            raise RuntimeError(
                "PCP page_pull requires identical global block allocation across ranks; "
                f"fingerprints={values}"
            )

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
        blocks_by_segment: list[tuple[int, ...]] = []
        for segment_slices in self._logical_segment_slices:
            blocks: list[int] = []
            for segment_slice in segment_slices:
                blocks.extend(
                    self._blocks_from_segment_slots(
                        global_slots[segment_slice], block_size=block_size
                    )
                )
            blocks_by_segment.append(tuple(blocks))
        blocks = tuple(blocks_by_segment)
        self._verify_page_mapping_invariant(blocks)
        self._runahead_runtime.configure_page_plan(
            PCPPagePlan(
                segment_to_rank=self._effective_segment_to_rank(),
                blocks_by_segment=blocks,
                block_size=block_size,
            )
        )

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
        sizes = list(self._rows_per_rank)
        if len(set(sizes)) == 1:
            with pcp_nvtx_range("pcp.restore_hidden_allgather"):
                gathered = self._pcp_group.all_gather(hidden_states.contiguous(), dim=0)
        else:
            with pcp_nvtx_range("pcp.restore_hidden_allgatherv"):
                gathered = self._pcp_group.all_gatherv(
                    hidden_states.contiguous(), dim=0, sizes=sizes
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
    "SegmentLayout",
    "compact_hidden_restore_idx",
    "parse_runahead_weights",
    "runahead_batch_eligible",
    "weighted_partition_lengths",
]

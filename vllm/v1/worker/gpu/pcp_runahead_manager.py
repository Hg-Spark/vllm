# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal PCP manager extension for configurable causal-prefix experiments."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.ops.pcp_profile import (
    install_pcp_nvtx_hooks,
    pcp_nvtx_range,
)
from vllm.v1.attention.ops.pcp_runahead import (
    PCPRunaheadRuntime,
    register_pcp_runahead_runtime,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
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

    world_size = len(weights)
    if num_tokens == 0:
        return (0,) * world_size

    if alignment == 1 or num_tokens < world_size * alignment:
        ideal = [num_tokens * weight / total_weight for weight in weights]
        lengths = [math.floor(value) for value in ideal]
        remainder = num_tokens - sum(lengths)
        order = sorted(
            range(world_size),
            key=lambda rank: (-(ideal[rank] - lengths[rank]), rank),
        )
        for rank in order[:remainder]:
            lengths[rank] += 1
        return tuple(lengths)

    boundaries = [0]
    cumulative_weight = 0.0
    for rank in range(world_size - 1):
        cumulative_weight += weights[rank]
        ideal_abs = start_pos + num_tokens * cumulative_weight / total_weight

        min_cut = boundaries[-1] + 1
        max_cut = num_tokens - (world_size - rank - 1)
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
        for index in range(world_size)
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
    """Reuse PCP mapping/layout machinery; replace only experiment policies."""

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
        self._rows_per_rank: tuple[int, ...] = ()
        self._compact_hidden_restore_idx: torch.Tensor | None = None
        self._sharded_kv_history = False
        self._runahead_runtime = PCPRunaheadRuntime(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
            max_inflight_sends=config.max_inflight_sends,
        )
        register_pcp_runahead_runtime(self._runahead_runtime)

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

    def _partition_lengths(
        self,
        query_len: int,
        start_pos: int = 0,
    ) -> tuple[int, ...]:
        if self._config.partition_policy == "weighted_contiguous":
            assert self._config.weights is not None
            weights = self._config.weights
        else:
            weights = (1.0,) * self.pcp_world_size
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

        rank_segments: list[RankSegment] = []
        rank_offset = 0
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            start_pos = int(num_computed_tokens[global_batch_req_idx])
            lengths = self._partition_lengths(query_len, start_pos)
            chunk_offset = sum(lengths[:rank])
            chunk_len = lengths[rank]
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
        return rank_segments

    def _custom_rows_per_rank(self, input_batch: InputBatch) -> tuple[int, ...]:
        rows = [0] * self.pcp_world_size
        for req_idx, num_tokens in enumerate(
            input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        ):
            lengths = self._partition_lengths(
                int(num_tokens),
                int(input_batch.num_computed_tokens_np[req_idx]),
            )
            for rank, length in enumerate(lengths):
                rows[rank] += length
        return tuple(rows)

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        if self._sharded_kv_history:
            raise RuntimeError(
                "PCP prefix_p2p left persistent KV sharded across ranks; "
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

        self._use_custom_partition = (
            eligible and self._config.partition_policy != "stock"
        )
        rows_per_rank: tuple[int, ...] = ()
        if self._use_custom_partition:
            rows_per_rank = self._custom_rows_per_rank(input_batch)
            if any(rows <= 0 for rows in rows_per_rank):
                logger.debug(
                    "PCP custom partition produced an empty rank; "
                    "falling back: rows=%s",
                    rows_per_rank,
                )
                eligible = False
                self._use_custom_partition = False
                rows_per_rank = ()

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
            )
        else:
            self._rows_per_rank = ()
            self._compact_hidden_restore_idx = None
            self._runahead_runtime.disable_step()

        return local_batch

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        if not self._use_compact_layout:
            return slot_mappings
        assert self._gathered_kv_write_mask is not None
        with pcp_nvtx_range("pcp.compact_slot_mapping"):
            return slot_mappings[:, self._gathered_kv_write_mask].contiguous()

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
        if self._step_transport == "prefix_p2p":
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

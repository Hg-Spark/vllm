# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState


def weighted_partition_lengths(
    num_tokens: int,
    weights: tuple[float, ...],
    *,
    start_pos: int = 0,
    alignment: int = 1,
) -> tuple[int, ...]:
    """Partition tokens with cumulative weighted, optionally page-aligned cuts."""
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


@dataclass(frozen=True)
class PCPLogicalTopology:
    """Logical PCP order mapped independently onto physical process ranks."""

    logical_to_physical: tuple[int, ...]
    physical_to_logical: tuple[int, ...]

    @classmethod
    def identity(cls, world_size: int) -> "PCPLogicalTopology":
        if world_size <= 0:
            raise ValueError(f"PCP world size must be positive, got {world_size}")
        order = tuple(range(world_size))
        return cls(order, order)

    @classmethod
    def from_logical_to_physical(
        cls,
        logical_to_physical: tuple[int, ...],
        world_size: int,
    ) -> "PCPLogicalTopology":
        if len(logical_to_physical) != world_size:
            raise ValueError(
                "pcp_partition.logical_to_physical must contain exactly "
                f"{world_size} ranks, got {logical_to_physical}"
            )
        if any(
            not isinstance(rank, int) or isinstance(rank, bool)
            for rank in logical_to_physical
        ):
            raise ValueError(
                "pcp_partition.logical_to_physical must contain integer ranks, "
                f"got {logical_to_physical}"
            )
        if set(logical_to_physical) != set(range(world_size)):
            raise ValueError(
                "pcp_partition.logical_to_physical must be a permutation of "
                f"0..{world_size - 1}, got {logical_to_physical}"
            )
        inverse = [0] * world_size
        for logical_rank, physical_rank in enumerate(logical_to_physical):
            inverse[physical_rank] = logical_rank
        return cls(logical_to_physical, tuple(inverse))

    @property
    def world_size(self) -> int:
        return len(self.logical_to_physical)

    def logical_rank(self, physical_rank: int) -> int:
        if not 0 <= physical_rank < self.world_size:
            raise ValueError(
                f"invalid physical PCP rank {physical_rank} for {self.world_size} ranks"
            )
        return self.physical_to_logical[physical_rank]

    def physical_rank(self, logical_rank: int) -> int:
        if not 0 <= logical_rank < self.world_size:
            raise ValueError(
                f"invalid logical PCP rank {logical_rank} for {self.world_size} ranks"
            )
        return self.logical_to_physical[logical_rank]


def _partition_config(additional_config: object) -> dict[str, object]:
    if not isinstance(additional_config, dict):
        raise ValueError("pcp_partition requires additional_config to be a JSON object")
    partition = additional_config.get("pcp_partition")
    if not isinstance(partition, dict):
        raise ValueError("pcp_partition must be a JSON object")
    return partition


def _parse_weights(
    partition: dict[str, object],
    pcp_world_size: int,
) -> tuple[float, ...]:
    raw = partition.get("weights")
    if raw is None:
        return (1.0,) * pcp_world_size
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


def parse_weighted_contiguous_partition(
    additional_config: object,
    pcp_world_size: int,
) -> tuple[float, ...]:
    partition = _partition_config(additional_config)
    unknown = set(partition) - {"impl", "weights"}
    if unknown:
        raise ValueError(f"unsupported pcp_partition keys: {sorted(unknown)}")
    impl = partition.get("impl")
    if impl != "weighted_contiguous":
        raise ValueError(
            "pcp_partition.impl must be 'weighted_contiguous' to enable the "
            f"experimental partition, got {impl!r}"
        )
    return _parse_weights(partition, pcp_world_size)


def parse_weighted_dual_chunk_partition(
    additional_config: object,
    pcp_world_size: int,
) -> tuple[tuple[float, ...], PCPLogicalTopology]:
    partition = _partition_config(additional_config)
    unknown = set(partition) - {"impl", "weights", "logical_to_physical"}
    if unknown:
        raise ValueError(f"unsupported pcp_partition keys: {sorted(unknown)}")
    impl = partition.get("impl")
    if impl != "weighted_dual_chunk":
        raise ValueError(
            "pcp_partition.impl must be 'weighted_dual_chunk' to enable the "
            f"experimental partition, got {impl!r}"
        )

    weights = _parse_weights(partition, pcp_world_size)
    raw_order = partition.get("logical_to_physical")
    if raw_order is None:
        topology = PCPLogicalTopology.identity(pcp_world_size)
    else:
        if not isinstance(raw_order, (list, tuple)):
            raise ValueError(
                "pcp_partition.logical_to_physical must be a list or tuple of ranks"
            )
        topology = PCPLogicalTopology.from_logical_to_physical(
            tuple(raw_order),
            pcp_world_size,
        )
    return weights, topology


class WeightedContiguousPCPManager(PCPManager):
    """Experimental PCP partition policy with one contiguous prefill slice per rank."""

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
        self._partition_weights = (
            (1.0,) * pcp_world_size
            if partition_weights is None
            else partition_weights
        )
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

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
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
                alignment = self._page_alignment
                if query_len < num_chunks * alignment:
                    alignment = 1
                chunk_lengths = weighted_partition_lengths(
                    query_len,
                    self._partition_weights,
                    start_pos=int(num_computed_tokens[global_batch_req_idx]),
                    alignment=alignment,
                )
                chunk_offsets = _chunk_offsets(chunk_lengths)
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


class WeightedDualChunkPCPManager(PCPManager):
    """Weighted DualChunkSwap with logical order decoupled from physical ranks."""

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
        topology: PCPLogicalTopology | None = None,
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
        self._partition_weights = (
            (1.0,) * pcp_world_size
            if partition_weights is None
            else partition_weights
        )
        if len(self._partition_weights) != pcp_world_size:
            raise ValueError(
                "PCP partition weights must match PCP world size: "
                f"weights={self._partition_weights}, world_size={pcp_world_size}"
            )
        self._topology = topology or PCPLogicalTopology.identity(pcp_world_size)
        if self._topology.world_size != pcp_world_size:
            raise ValueError(
                "PCP logical topology size must match PCP world size: "
                f"topology={self._topology.world_size}, world_size={pcp_world_size}"
            )
        self._logical_weights = tuple(
            self._partition_weights[self._topology.physical_rank(logical_rank)]
            for logical_rank in range(pcp_world_size)
        )
        segment_weights = [0.0] * (2 * pcp_world_size)
        for logical_rank, weight in enumerate(self._logical_weights):
            segment_weights[logical_rank] = weight
            segment_weights[-1 - logical_rank] = weight
        self._segment_weights = tuple(segment_weights)
        self._page_alignment = (
            math.lcm(*(int(size) for size in block_tables.kernel_block_sizes))
            if block_tables is not None and block_tables.kernel_block_sizes
            else 1
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        rank_segments = []
        rank_offset = 0
        num_chunks = 2 * self.pcp_world_size
        logical_rank = self._topology.logical_rank(rank)
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            chunk_indices: tuple[int, ...]
            if bool(is_prefilling[global_batch_req_idx]):
                alignment = self._page_alignment
                if query_len < num_chunks * alignment:
                    alignment = 1
                chunk_lengths = weighted_partition_lengths(
                    query_len,
                    self._segment_weights,
                    start_pos=int(num_computed_tokens[global_batch_req_idx]),
                    alignment=alignment,
                )
                chunk_offsets = _chunk_offsets(chunk_lengths)
                chunk_indices = (
                    logical_rank,
                    num_chunks - 1 - logical_rank,
                )
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


def _chunk_offsets(chunk_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    running = 0
    for chunk_len in chunk_lengths:
        offsets.append(running)
        running += chunk_len
    return tuple(offsets)


def build_experimental_pcp_manager(
    *,
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: RequestState,
    block_tables: BlockTables,
    pcp_rank: int,
    dcp_rank: int,
) -> PCPManager:
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    dcp_size = parallel_config.decode_context_parallel_size
    common = dict(
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
    )
    partition = _partition_config(vllm_config.additional_config)
    impl = partition.get("impl")

    if impl == "weighted_contiguous":
        return WeightedContiguousPCPManager(
            **common,
            partition_weights=parse_weighted_contiguous_partition(
                vllm_config.additional_config,
                pcp_size,
            ),
        )

    if impl == "weighted_dual_chunk":
        weights, topology = parse_weighted_dual_chunk_partition(
            vllm_config.additional_config,
            pcp_size,
        )
        return WeightedDualChunkPCPManager(
            **common,
            partition_weights=weights,
            topology=topology,
        )

    raise ValueError(
        "unsupported pcp_partition.impl: "
        f"{impl!r}; expected 'weighted_contiguous' or 'weighted_dual_chunk'"
    )


def build_weighted_contiguous_pcp_manager(
    *,
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: RequestState,
    block_tables: BlockTables,
    pcp_rank: int,
    dcp_rank: int,
) -> PCPManager:
    """Compatibility factory entry point; dispatches the explicit partition impl."""
    return build_experimental_pcp_manager(
        vllm_config=vllm_config,
        device=device,
        req_states=req_states,
        block_tables=block_tables,
        pcp_rank=pcp_rank,
        dcp_rank=dcp_rank,
    )

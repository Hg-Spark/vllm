from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "0ccafcd9d9cf57689023ef023c5c704d24fdc828"


def git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE}:{path}"], cwd=ROOT, text=True
    )


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1))


# Keep the transport/runtime and profiling implementation from the validated port.
for source_path in (
    "vllm/v1/attention/ops/pcp_runahead.py",
    "vllm/v1/attention/ops/pcp_profile.py",
    "tests/v1/attention/test_pcp_runahead_runtime.py",
):
    write(source_path, git_show(source_path))


write(
    "vllm/model_executor/layers/attention/pcp.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    with pcp_nvtx_range("pcp.baseline_prefill_allgather"):
        gathered_prefills = tuple(
            pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
            for tensor in tensors
        )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    with pcp_nvtx_range("pcp.baseline_cache_pack"):
        cache_inputs = tuple(
            torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
            for tensor, gathered_prefill in zip(tensors, gathered_prefills)
        )
        rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
        cache_slot_mapping = torch.cat(
            (
                rank_slot_mappings[0, :num_decode_tokens],
                rank_slot_mappings[:, num_decode_tokens:].flatten(),
            )
        )
    return cache_inputs, cache_slot_mapping


def maybe_register_pcp_runahead_cache(kv_cache: torch.Tensor) -> None:
    """Register persistent cache storage only while a runahead step is active."""
    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        runtime.register_kv_cache(kv_cache)


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)

    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        if num_decode_tokens != 0:
            raise RuntimeError("runahead PCP requires a prefill-only cache update")
        with pcp_nvtx_range("pcp.prefix_exchange"):
            (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = (
                runtime.exchange_prefix(
                    (kv_c_normed, k_pe_flat),
                    slot_mapping,
                )
            )
    else:
        (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = (
            _gather_prefill_cache_inputs(
                (kv_c_normed, k_pe_flat),
                slot_mapping,
                num_decode_tokens,
            )
        )

    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping

    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        if num_decode_tokens != 0:
            raise RuntimeError("runahead PCP requires a prefill-only indexer update")
        with pcp_nvtx_range("pcp.prefix_exchange"):
            (cache_k,), cache_slot_mapping = runtime.exchange_prefix(
                (k,), slot_mapping
            )
        return cache_k, cache_slot_mapping

    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        with pcp_nvtx_range("pcp.mla.decode_allgather"):
            output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
''',
)


write(
    "vllm/v1/attention/ops/pcp_standard.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard MHA/GQA/MQA KV-cache transport for PCP."""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


def prepare_standard_pcp_kv_cache_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FlashAttention cache-write inputs using PCP's active policy."""
    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        runtime.register_kv_cache(kv_cache)
        with pcp_nvtx_range("pcp.prefix_exchange"):
            (key, value), slot_mapping = runtime.exchange_prefix(
                (key, value), slot_mapping
            )
        return key, value, slot_mapping

    pcp_group = get_pcp_group()
    if pcp_group.world_size <= 1 or slot_mapping.shape[0] <= key.shape[0]:
        return key, value, slot_mapping

    pcp_size = pcp_group.world_size
    if slot_mapping.shape[0] % pcp_size != 0:
        raise RuntimeError(
            "PCP gathered slot mapping is not divisible by PCP size: "
            f"slots={slot_mapping.shape[0]}, pcp={pcp_size}"
        )
    local_rows = slot_mapping.shape[0] // pcp_size
    if key.shape[0] < local_rows or value.shape[0] < local_rows:
        raise RuntimeError(
            "PCP standard-attention K/V rows are smaller than the rank-local "
            f"slab: key={key.shape[0]}, value={value.shape[0]}, rows={local_rows}"
        )

    with pcp_nvtx_range("pcp.baseline_kv_allgather"):
        key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
        value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)
    return key, value, slot_mapping
''',
)


write(
    "vllm/v1/worker/gpu/pcp_runahead_manager.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal PCP manager extension for causal-prefix runahead."""

from __future__ import annotations

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.ops.pcp_runahead import (
    PCPRunaheadRuntime,
    register_pcp_runahead_runtime,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager, RankSegment
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

RUNAHEAD_MIN_PREFILL_TOKENS = 1024


def runahead_batch_eligible(
    *,
    num_reqs: int,
    is_prefilling: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    pcp_world_size: int,
    min_prefill_tokens: int = RUNAHEAD_MIN_PREFILL_TOKENS,
) -> bool:
    """Use runahead only for homogeneous, sufficiently large prefill batches."""
    if num_reqs <= 0:
        return False
    if not bool(is_prefilling[:num_reqs].all()):
        return False
    total_prefill_tokens = int(num_scheduled_tokens[:num_reqs].sum())
    return total_prefill_tokens >= max(pcp_world_size, min_prefill_tokens)


class RunaheadPCPManager(PCPManager):
    """Reuse vLLM PCP batch/layout machinery and replace only causal partitioning."""

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
        self._use_runahead_partition = False
        self._standard_attention_pcp = False
        self._runahead_runtime = PCPRunaheadRuntime(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
        )
        register_pcp_runahead_runtime(self._runahead_runtime)

    def set_standard_attention(self, enabled: bool) -> None:
        self._standard_attention_pcp = enabled

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        assert model is not None

        if model.use_mla:
            PCPManager.validate_config(vllm_config, supports_mm_inputs)
        else:
            if parallel.pipeline_parallel_size > 1:
                raise NotImplementedError("MRV2 PCP does not support PP yet.")
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

    def _partition_lengths(self, query_len: int) -> tuple[int, ...]:
        chunk_size = (query_len + self.pcp_world_size - 1) // self.pcp_world_size
        return tuple(
            max(0, min(chunk_size, query_len - rank * chunk_size))
            for rank in range(self.pcp_world_size)
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        if not self._use_runahead_partition:
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
            lengths = self._partition_lengths(query_len)
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

    def _runahead_rows_per_rank(self, input_batch: InputBatch) -> tuple[int, ...]:
        rows = [0] * self.pcp_world_size
        for num_tokens in input_batch.num_scheduled_tokens[: input_batch.num_reqs]:
            for rank, length in enumerate(self._partition_lengths(int(num_tokens))):
                rows[rank] += length
        return tuple(rows)

    def _plan_runahead_repair(self, slot_mappings: torch.Tensor) -> None:
        if not self._use_runahead_partition:
            self._runahead_runtime.set_repair_block_ids(None)
            return
        assert self._block_tables is not None

        parts: list[torch.Tensor] = []
        for group_id, block_size in enumerate(self._block_tables.kernel_block_sizes):
            slots = slot_mappings[group_id]
            valid_slots = slots[slots >= 0]
            if valid_slots.numel() == 0:
                continue
            parts.append(
                torch.div(valid_slots, block_size, rounding_mode="floor").to(
                    dtype=torch.long
                )
            )

        if parts:
            block_ids = torch.unique(torch.cat(parts, dim=0))
        else:
            block_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._runahead_runtime.set_repair_block_ids(block_ids)

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        if self._standard_attention_pcp:
            use_runahead = runahead_batch_eligible(
                num_reqs=input_batch.num_reqs,
                is_prefilling=input_batch.is_prefilling_np,
                num_scheduled_tokens=input_batch.num_scheduled_tokens,
                pcp_world_size=self.pcp_world_size,
            )
        else:
            use_runahead = (
                input_batch.num_reqs == 1
                and bool(input_batch.is_prefilling_np[0])
                and int(input_batch.num_computed_tokens_np[0]) == 0
                and int(input_batch.num_scheduled_tokens[0])
                == int(input_batch.prefill_len_np[0])
                and int(input_batch.num_scheduled_tokens[0]) >= self.pcp_world_size
            )

        if use_runahead:
            rows_per_rank = self._runahead_rows_per_rank(input_batch)
            if any(rows <= 0 for rows in rows_per_rank):
                logger.debug(
                    "PCP runahead produced an empty rank; falling back: rows=%s",
                    rows_per_rank,
                )
                use_runahead = False

        self._use_runahead_partition = use_runahead
        local_batch = super().partition_batch(input_batch)
        if use_runahead:
            padded_rows = int(local_batch.num_tokens_after_padding)
            self._runahead_runtime.begin_step(
                (padded_rows,) * self.pcp_world_size
            )
        else:
            self._runahead_runtime.disable_step()
        return local_batch

    def prepare_slot_mappings(self) -> torch.Tensor:
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert self._global_batch is not None
        global_batch = self._global_batch
        global_slot_mappings = self._block_tables.compute_slot_mappings(
            global_batch.idx_mapping,
            global_batch.query_start_loc,
            global_batch.positions,
            global_batch.num_tokens,
            out=self._global_batch_slot_mappings,
        )
        self._plan_runahead_repair(global_slot_mappings)
        has_prefill = bool(
            global_batch.is_prefilling_np[: global_batch.num_reqs].any()
        )
        if not has_prefill:
            return global_slot_mappings
        return self._convert_to_gathered_slot_mappings(global_slot_mappings)

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        self._runahead_runtime.flush()
        return super().restore_for_sampling(hidden_states)
''',
)


write(
    "tests/v1/attention/test_pcp_standard.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.v1.attention.ops.pcp_standard import (
    prepare_standard_pcp_kv_cache_inputs,
)


def _flash_kv_cache(
    num_blocks: int = 8,
    num_kv_heads: int = 2,
    block_size: int = 16,
) -> torch.Tensor:
    return torch.empty((num_blocks, num_kv_heads, block_size, 32))


def test_standard_runahead_preserves_gqa_kv_head_shape() -> None:
    key = torch.randn(6, 2, 16)
    value = torch.randn(6, 2, 16)
    slot_mapping = torch.arange(6, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.exchange_prefix.return_value = ((key, value), slot_mapping)

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slot_mapping, kv_cache
        )

    runtime.register_kv_cache.assert_called_once_with(kv_cache)
    runtime.exchange_prefix.assert_called_once()
    assert out_key is key
    assert out_value is value
    assert out_slots is slot_mapping
    assert out_key.shape[1:] == (2, 16)


def test_standard_runahead_returns_causal_visible_image() -> None:
    local_key = torch.randn(3, 2, 8)
    local_value = torch.randn(3, 2, 8)
    visible_key = torch.randn(7, 2, 8)
    visible_value = torch.randn(7, 2, 8)
    visible_slots = torch.arange(7, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    runtime = MagicMock()
    runtime.exchange_prefix.return_value = (
        (visible_key, visible_value), visible_slots
    )

    with patch(
        "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
        return_value=runtime,
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            local_key, local_value, visible_slots, kv_cache
        )

    assert out_key is visible_key
    assert out_value is visible_value
    assert out_slots is visible_slots


def test_standard_fallback_reuses_baseline_allgather() -> None:
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.arange(4, dtype=torch.int64)
    kv_cache = _flash_kv_cache()
    group = MagicMock()
    group.world_size = 2
    group.all_gather.side_effect = lambda tensor, dim: torch.cat(
        (tensor, tensor), dim=dim
    )

    with (
        patch(
            "vllm.v1.attention.ops.pcp_standard.get_pcp_runahead_runtime",
            return_value=None,
        ),
        patch(
            "vllm.v1.attention.ops.pcp_standard.get_pcp_group",
            return_value=group,
        ),
    ):
        out_key, out_value, out_slots = prepare_standard_pcp_kv_cache_inputs(
            key, value, slots, kv_cache
        )

    assert out_key.shape[0] == 4
    assert out_value.shape[0] == 4
    assert out_slots is slots
    assert group.all_gather.call_count == 2
''',
)


write(
    "tests/v1/worker/gpu/test_pcp_runahead.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from vllm.v1.attention.ops.pcp_runahead import PCPRunaheadRuntime
from vllm.v1.worker.gpu.pcp_runahead_manager import (
    RunaheadPCPManager,
    runahead_batch_eligible,
)


def _manager(world_size: int) -> RunaheadPCPManager:
    manager = object.__new__(RunaheadPCPManager)
    manager.pcp_world_size = world_size
    manager._standard_attention_pcp = True
    manager._use_runahead_partition = True
    return manager


def test_runahead_partition_is_contiguous_and_complete() -> None:
    manager = _manager(4)
    expected = [(0, 3), (3, 6), (6, 9), (9, 10)]
    actual = []

    for rank in range(4):
        (segment,) = manager._get_rank_segments(
            rank,
            np.asarray([10], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True]),
            np.asarray([0, 10], dtype=np.int32),
        )
        actual.append(
            (segment.global_batch_slice.start, segment.global_batch_slice.stop)
        )

    assert actual == expected


def test_runahead_partition_supports_multiple_prefill_requests() -> None:
    manager = _manager(4)
    scheduled = np.asarray([8, 4], dtype=np.int32)
    computed = np.asarray([0, 16384], dtype=np.int32)
    is_prefilling = np.asarray([True, True])
    query_start = np.asarray([0, 8, 12], dtype=np.int32)

    expected = [
        [(0, 2), (8, 9)],
        [(2, 4), (9, 10)],
        [(4, 6), (10, 11)],
        [(6, 8), (11, 12)],
    ]
    actual = []
    for rank in range(4):
        segments = manager._get_rank_segments(
            rank, scheduled, computed, is_prefilling, query_start
        )
        actual.append(
            [
                (segment.global_batch_slice.start, segment.global_batch_slice.stop)
                for segment in segments
            ]
        )

    assert actual == expected


def test_equal_partition_reuses_padded_pcp_layout() -> None:
    manager = _manager(4)
    assert manager._partition_lengths(10) == (3, 3, 3, 1)
    assert manager._partition_lengths(8) == (2, 2, 2, 2)


def test_step_level_repair_plan_uses_kernel_block_ids() -> None:
    manager = _manager(4)
    manager.device = torch.device("cpu")
    manager._block_tables = SimpleNamespace(kernel_block_sizes=[4, 2])
    manager._runahead_runtime = MagicMock()

    slot_mappings = torch.tensor(
        [
            [0, 1, 4, 7, 8, -1],
            [0, 1, 4, 5, 8, -1],
        ],
        dtype=torch.int64,
    )
    manager._plan_runahead_repair(slot_mappings)

    (block_ids,) = manager._runahead_runtime.set_repair_block_ids.call_args.args
    assert block_ids.tolist() == [0, 1, 2, 4]


def test_variable_width_runtime_builds_offsets() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=2,
        device=torch.device("cpu"),
    )
    runtime.begin_step((4, 3, 2, 1))

    assert runtime.rows_per_rank == (4, 3, 2, 1)
    assert runtime.rank_offsets == (0, 4, 7, 9, 10)
    assert runtime.local_rows == 2
    assert runtime.prefix_rows == 7
    assert runtime.visible_rows == 9


def test_variable_width_runtime_rejects_empty_rank() -> None:
    runtime = PCPRunaheadRuntime(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="positive rows"):
        runtime.begin_step((4, 3, 0, 1))


@pytest.mark.parametrize(
    ("is_prefilling", "scheduled", "expected"),
    [
        ([True, True], [2048, 4096], True),
        ([True], [2048], True),
        ([False, True], [1, 4096], False),
        ([True], [512], False),
    ],
)
def test_runahead_batch_eligibility(
    is_prefilling: list[bool],
    scheduled: list[int],
    expected: bool,
) -> None:
    assert (
        runahead_batch_eligible(
            num_reqs=len(is_prefilling),
            is_prefilling=np.asarray(is_prefilling),
            num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
            pcp_world_size=4,
        )
        is expected
    )
''',
)


write(
    "docs/design/pcp_runahead.md",
    '''# PCP causal-prefix runahead

This experimental path is based directly on the vLLM 0.27.1 V2 PCP lifecycle.
It reuses PCP batch construction, padded rank-major layout, slot mapping, block
tables, and hidden-state AllGather/restore. Runahead changes the prefill
partition and layer-critical KV transport, then repairs replicated persistent KV
pages before the existing sampling restore.

## Enable

```bash
--prefill-context-parallel-size 4 \\
--additional-config '{"pcp_runahead": true}' \\
--enforce-eager
```

`additional_config` is an existing vLLM configuration surface and participates
in `VllmConfig.compute_hash()`. No PCP-specific `ParallelConfig` or `EngineArgs`
field is required.

## Baseline PCP

For each attention layer, baseline PCP gathers every rank's newly generated
prefill KV before the cache write. This creates a layer-level collective on the
critical path.

## Runahead layer path

Runahead partitions each prefill into contiguous causal ranges. For PCP=4:

```text
rank0: K0 ───────────────►
rank1:    K0,K1 ─────────►
rank2:          K0,K1,K2 ─►
rank3:                K0,K1,K2,K3
```

Each layer executes:

```text
local K/V
   │
   ├─ receive prefix from rank-1
   ├─ append local K/V
   ├─ enqueue nonblocking send to rank+1
   ├─ write causal-visible KV to paged cache
   └─ attention
```

The transport runtime is isolated in `PCPRunaheadRuntime`.

## Reused PCP layout

The manager intentionally keeps vLLM 0.27.1's existing padded rank-major PCP
layout. Equal contiguous partitions are padded to the largest rank-local width,
so the existing hidden-state `AllGather` and `hidden_restore_idx` continue to
work unchanged. This isolates the performance effect of causal-prefix runahead
from compact-layout or weighted-partition optimizations.

## Persistent KV repair

Layer execution only writes the causal-visible cache image on each rank. Before
sampling, `RunaheadPCPManager.restore_for_sampling()` calls `runtime.flush()`:

1. wait outstanding prefix sends;
2. rendezvous on the PCP CPU/Gloo group;
3. broadcast touched raw paged-KV blocks from the final PCP rank;
4. update the same persistent backing pages on earlier ranks;
5. call the original `PCPManager.restore_for_sampling()`.

The final rank is the repair source because the causal-prefix chain gives it the
complete current-step KV image for every layer.

## Standard attention

Standard MHA/GQA/MQA uses FlashAttention with TP=1. FlashAttention declares PCP
capability through the existing `AttentionBackend.supports_pcp()` mechanism.
Its existing `do_kv_cache_update()` calls a small PCP input-preparation helper;
the original cache-write kernel remains unchanged.

When runahead is inactive, the helper falls back to the baseline PCP K/V
AllGather. Pure decode keeps replicated local cache writes.

## MLA and sparse indexer

The existing vLLM 0.27.1 `maybe_gather_mla_latent_cache_inputs()` and
`maybe_gather_indexer_k()` entry points are retained. They dispatch to causal
prefix exchange only while a runahead step is active. Their callers retain the
original cache-write functions; they only register persistent cache backing for
forward-boundary repair.

MLA keeps the narrower runahead eligibility of one fresh complete prefill.

## Eligibility and fallback

Standard attention runahead requires a homogeneous prefill/extend batch with at
least 1024 scheduled prefill tokens (and at least one token per PCP rank). Mixed
decode+prefill and smaller prefills use the baseline PCP path. Pure decode keeps
local replicated cache updates.

Current runahead validation requires:

- PCP > 1
- TP = 1
- PP = 1
- DP = 1
- DCP = 1
- no expert parallelism or MoE collectives
- no DBO
- no speculative decoding
- no async scheduling
- eager execution (`cudagraph_mode=NONE`)
- FlashAttention for standard MHA/GQA/MQA

## Profiling

NVTX ranges use the `pcp.*` prefix. Important ranges include:

```text
pcp.baseline_prefill_allgather
pcp.baseline_kv_allgather
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_concat
pcp.replica_forward_boundary
pcp.replica_broadcast
pcp.replica_cache_update
pcp.flush
```

Use Nsight Systems to verify that early PCP ranks enter later transformer layers
while later ranks are still completing the previous layer, and that replicated
KV repair occurs only at the forward/sampling boundary.
''',
)


# vLLMConfig: use the existing additional_config surface for the experimental gate.
replace_once(
    "vllm/config/vllm.py",
    '''        if self.parallel_config.prefill_context_parallel_size > 1 and not (\n            model_config is not None and model_config.use_mla\n        ):\n            unsupported.append("prefill context parallelism")\n''',
    '''        pcp_runahead = (\n            isinstance(self.additional_config, dict)\n            and bool(self.additional_config.get("pcp_runahead", False))\n        )\n        if (\n            self.parallel_config.prefill_context_parallel_size > 1\n            and not (model_config is not None and model_config.use_mla)\n            and not pcp_runahead\n        ):\n            unsupported.append("prefill context parallelism")\n''',
)


# PCP manager: keep the v0.27.1 implementation and only select the subclass.
replace_once(
    "vllm/v1/worker/gpu/pcp_manager.py",
    '''def maybe_build_pcp_manager(\n    vllm_config: VllmConfig,\n    device: torch.device,\n    supports_mm_inputs: bool,\n    req_states: RequestState,\n    block_tables: BlockTables,\n) -> PCPManager | None:\n    parallel_config = vllm_config.parallel_config\n    pcp_size = parallel_config.prefill_context_parallel_size\n    if pcp_size <= 1:\n        return None\n\n    PCPManager.validate_config(vllm_config, supports_mm_inputs)\n\n    pcp_rank = get_pcp_group().rank_in_group\n    dcp_size = parallel_config.decode_context_parallel_size\n    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0\n\n    return PCPManager(\n        pcp_world_size=pcp_size,\n        pcp_rank=pcp_rank,\n        device=device,\n        req_states=req_states,\n        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,\n        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,\n        block_tables=block_tables,\n        dcp_world_size=dcp_size,\n        dcp_rank=dcp_rank,\n        cp_interleave=parallel_config.cp_kv_cache_interleave_size,\n    )\n''',
    '''def maybe_build_pcp_manager(\n    vllm_config: VllmConfig,\n    device: torch.device,\n    supports_mm_inputs: bool,\n    req_states: RequestState,\n    block_tables: BlockTables,\n) -> PCPManager | None:\n    parallel_config = vllm_config.parallel_config\n    pcp_size = parallel_config.prefill_context_parallel_size\n    if pcp_size <= 1:\n        return None\n\n    runahead = (\n        isinstance(vllm_config.additional_config, dict)\n        and bool(vllm_config.additional_config.get("pcp_runahead", False))\n    )\n    manager_cls: type[PCPManager] = PCPManager\n    if runahead:\n        from vllm.v1.worker.gpu.pcp_runahead_manager import RunaheadPCPManager\n\n        manager_cls = RunaheadPCPManager\n\n    manager_cls.validate_config(vllm_config, supports_mm_inputs)\n\n    pcp_rank = get_pcp_group().rank_in_group\n    dcp_size = parallel_config.decode_context_parallel_size\n    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0\n\n    manager = manager_cls(\n        pcp_world_size=pcp_size,\n        pcp_rank=pcp_rank,\n        device=device,\n        req_states=req_states,\n        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,\n        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,\n        block_tables=block_tables,\n        dcp_world_size=dcp_size,\n        dcp_rank=dcp_rank,\n        cp_interleave=parallel_config.cp_kv_cache_interleave_size,\n    )\n    if runahead:\n        manager.set_standard_attention(  # type: ignore[attr-defined]\n            not vllm_config.model_config.use_mla\n        )\n    return manager\n''',
)


# FlashAttention: declare PCP support and reuse the existing cache writer.
replace_once(
    "vllm/v1/attention/backends/flash_attn.py",
    '''class FlashAttentionImpl(AttentionImpl):\n    can_return_lse_for_decode: bool = True\n''',
    '''class FlashAttentionImpl(AttentionImpl):\n    can_return_lse_for_decode: bool = True\n    supports_pcp: bool = True\n''',
)
replace_once(
    "vllm/v1/attention/backends/flash_attn.py",
    '''        vllm_config = get_current_vllm_config_or_none()\n        dcp_a2a = (\n''',
    '''        vllm_config = get_current_vllm_config_or_none()\n        self.use_pcp = (\n            vllm_config is not None\n            and vllm_config.parallel_config.prefill_context_parallel_size > 1\n        )\n        dcp_a2a = (\n''',
)
replace_once(
    "vllm/v1/attention/backends/flash_attn.py",
    '''        # Scatter write into the KV cache using slot_mapping indices.\n        # No TMA kernel is invoked here, so stride canonicalization is not needed.\n''',
    '''        if self.use_pcp:\n            from vllm.v1.attention.ops.pcp_standard import (\n                prepare_standard_pcp_kv_cache_inputs,\n            )\n\n            key, value, slot_mapping = prepare_standard_pcp_kv_cache_inputs(\n                key, value, slot_mapping, kv_cache\n            )\n\n        # Scatter write into the KV cache using slot_mapping indices.\n        # No TMA kernel is invoked here, so stride canonicalization is not needed.\n''',
)


# MLA: preserve the existing gather/cache-write structure and only register backing.
replace_once(
    "vllm/model_executor/layers/attention/mla_attention.py",
    '''from vllm.model_executor.layers.attention.pcp import (\n    finalize_mla_pcp_decode,\n    maybe_gather_mla_latent_cache_inputs,\n)\n''',
    '''from vllm.model_executor.layers.attention.pcp import (\n    finalize_mla_pcp_decode,\n    maybe_gather_mla_latent_cache_inputs,\n    maybe_register_pcp_runahead_cache,\n)\n''',
)
replace_once(
    "vllm/model_executor/layers/attention/mla_attention.py",
    '''            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n                kv_for_cache,\n''',
    '''            maybe_register_pcp_runahead_cache(self_kv_cache)\n            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n                kv_for_cache,\n''',
)
replace_once(
    "vllm/model_executor/layers/attention/mla_attention.py",
    '''        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n            kv_c_normed,\n''',
    '''        maybe_register_pcp_runahead_cache(kv_cache)\n        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n            kv_c_normed,\n''',
)


# Sparse indexer: keep the original cache insertion and register backing only.
replace_once(
    "vllm/model_executor/layers/sparse_attn_indexer.py",
    '''from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k\n''',
    '''from vllm.model_executor.layers.attention.pcp import (\n    maybe_gather_indexer_k,\n    maybe_register_pcp_runahead_cache,\n)\n''',
)
replace_once(
    "vllm/model_executor/layers/sparse_attn_indexer.py",
    '''    if not skip_k_cache_insert:\n        assert k is not None\n        k, slot_mapping_for_cache = maybe_gather_indexer_k(\n''',
    '''    if not skip_k_cache_insert:\n        assert k is not None\n        maybe_register_pcp_runahead_cache(kv_cache)\n        k, slot_mapping_for_cache = maybe_gather_indexer_k(\n''',
)

print("PCP v0.27.1 minimization applied")

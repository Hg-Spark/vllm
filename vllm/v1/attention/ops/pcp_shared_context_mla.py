# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inject the PCP shared-history planner into the canonical MLA builder."""

from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

import torch

from vllm.v1.attention.ops.pcp_shared_context import (
    combine_pcp_consumer_token_slices,
    plan_pcp_shared_context,
)

_pcp_source_row: ContextVar[int | None] = ContextVar(
    "pcp_shared_mla_source_row", default=None
)
_original_chunk_builder = None


def _build_pcp_or_default_chunked_context_metadata(
    *,
    context_lens_cpu: torch.Tensor,
    prefill_query_start_loc_cpu: torch.Tensor,
    chunked_prefill_workspace: torch.Tensor,
    chunked_prefill_workspace_size: int,
    block_size: int,
    align_chunk_to_block: bool,
    device: torch.device,
    dcp_world_size: int,
    dcp_local_block_size: int,
    dcp_virtual_block_size: int,
    dcp_manager=None,
):
    assert _original_chunk_builder is not None
    kwargs = dict(
        context_lens_cpu=context_lens_cpu,
        prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
        chunked_prefill_workspace=chunked_prefill_workspace,
        chunked_prefill_workspace_size=chunked_prefill_workspace_size,
        block_size=block_size,
        align_chunk_to_block=align_chunk_to_block,
        device=device,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=dcp_local_block_size,
        dcp_virtual_block_size=dcp_virtual_block_size,
        dcp_manager=dcp_manager,
    )

    source_row = _pcp_source_row.get()
    if source_row is None or dcp_world_size != 1 or not align_chunk_to_block:
        return _original_chunk_builder(**kwargs)

    jobs = plan_pcp_shared_context(
        source_row=source_row,
        context_lens=context_lens_cpu.tolist(),
        query_start_locs=prefill_query_start_loc_cpu.tolist(),
        row_budget=chunked_prefill_workspace_size,
        split_alignment=block_size,
    )
    if not jobs:
        return _original_chunk_builder(**kwargs)

    token_slices = tuple(
        combine_pcp_consumer_token_slices(job.consumers) for job in jobs
    )
    if any(
        token_slice is None or token_slice.stop <= token_slice.start
        for token_slice in token_slices
    ):
        return _original_chunk_builder(**kwargs)

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
        _flat_int32,
    )

    starts = _flat_int32([job.kv_start for job in jobs]).to(
        device, non_blocking=True
    )
    seq_lens = _flat_int32([job.kv_length for job in jobs])
    cu_seq_lens = _flat_int32(
        [value for job in jobs for value in (0, job.kv_length)]
    ).to(device, non_blocking=True)
    query_start_locs = _flat_int32(
        [
            value
            for token_slice in token_slices
            for value in (0, token_slice.stop - token_slice.start)  # type: ignore[union-attr]
        ]
    ).to(device, non_blocking=True)
    token_to_seq = torch.zeros(
        sum(job.kv_length for job in jobs), dtype=torch.int32, device=device
    )

    chunks: list[MLACommonPrefillMetadata.ContextChunk] = []
    context_token_offset = 0
    for index, (job, token_slice) in enumerate(zip(jobs, token_slices)):
        assert token_slice is not None
        kv_len = job.kv_length
        boundary_slice = slice(2 * index, 2 * index + 2)
        context_token_slice = slice(
            context_token_offset, context_token_offset + kv_len
        )
        chunks.append(
            MLACommonPrefillMetadata.ContextChunk(
                index=index,
                request_slice=slice(job.source_row, job.source_row + 1),
                token_slice=token_slice,
                continuation_token_end=token_slice.stop,
                is_continuation=job.kv_start > 0,
                num_context_tokens=kv_len,
                query_start_loc=query_start_locs[boundary_slice],
                max_query_len=token_slice.stop - token_slice.start,
                cu_seq_lens=cu_seq_lens[boundary_slice],
                starts=starts[index : index + 1],
                max_seq_len=kv_len,
                seq_lens=seq_lens[index : index + 1],
                token_to_seq=token_to_seq[context_token_slice],
                num_local_context_tokens=kv_len,
            )
        )
        context_token_offset += kv_len

    context_lens = context_lens_cpu.tolist()
    query_starts = prefill_query_start_loc_cpu.tolist()
    return MLACommonPrefillMetadata.ChunkedContextMetadata(
        context_lens=context_lens_cpu.to(device, non_blocking=True),
        workspace=chunked_prefill_workspace,
        chunks=chunks,
        context_lens_list=context_lens,
        empty_token_slices=[
            slice(query_starts[i], query_starts[i + 1])
            for i, context_len in enumerate(context_lens)
            if context_len == 0
        ],
        dcp_manager=None,
    )


def _install_chunk_planner_hook() -> None:
    global _original_chunk_builder
    if _original_chunk_builder is not None:
        return
    import vllm.model_executor.layers.attention.mla_attention as mla

    _original_chunk_builder = mla.build_mla_chunked_context_metadata
    mla.build_mla_chunked_context_metadata = (  # type: ignore[assignment]
        _build_pcp_or_default_chunked_context_metadata
    )


def _eligible(builder: Any, metadata: Any, source_ids: Sequence[int] | None) -> bool:
    if builder.__class__.__name__ != "MLACommonMetadataBuilder":
        return False
    metadata_cls = getattr(builder, "metadata_cls", None)
    if metadata_cls is None or metadata_cls.__name__ != "MLACommonMetadata":
        return False
    if not getattr(builder, "use_pcp", False) or getattr(builder, "dcp_world_size", 1) != 1:
        return False
    backend = getattr(builder, "_prefill_backend", None)
    if backend is None or backend.get_name() != "FLASH_ATTN":
        return False
    if hasattr(builder.model_config.hf_text_config, "index_topk"):
        return False
    if builder.model_config.get_sliding_window() is not None:
        return False
    if metadata.causal is not True or int(metadata.num_reqs) != 2:
        return False
    is_prefilling = getattr(metadata, "is_prefilling", None)
    if is_prefilling is None or is_prefilling.numel() < 2:
        return False
    if not bool(torch.all(is_prefilling[:2])):
        return False
    return (
        source_ids is not None
        and len(source_ids) >= 2
        and source_ids[0] == source_ids[1]
    )


def maybe_build_pcp_mla_shared_metadata(
    builder: Any,
    metadata: Any,
    source_ids: Sequence[int] | None,
) -> Any | None:
    """Run the normal MLA builder with the shared chunk planner active."""
    if not _eligible(builder, metadata, source_ids):
        return None

    _install_chunk_planner_hook()
    token = _pcp_source_row.set(0)
    try:
        return builder.build(common_prefix_len=0, common_attn_metadata=metadata)
    finally:
        _pcp_source_row.reset(token)

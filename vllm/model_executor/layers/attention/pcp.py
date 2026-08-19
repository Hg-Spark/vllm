# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm import _custom_ops as ops
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


def update_standard_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_layer: Any,
    cache_writer: Any,
    kv_cache: torch.Tensor,
) -> None:
    """Update standard MHA/GQA/MQA KV cache through PCP policy.

    A rank-local slot mapping keeps pure decode local. A gathered mapping selects
    the baseline synchronous K/V AllGather. Active runahead replaces that
    collective with causal-prefix P2P on the critical path plus asynchronous
    full-cache replication.
    """
    layer_name = getattr(attn_layer, "layer_name", "unknown")

    def apply(
        tensors: tuple[torch.Tensor, ...],
        cache_slot_mapping: torch.Tensor,
    ) -> None:
        cache_key, cache_value = tensors
        with pcp_nvtx_range(f"pcp.standard.cache_write:{layer_name}"):
            cache_writer(
                attn_layer,
                cache_key,
                cache_value,
                kv_cache,
                cache_slot_mapping,
            )

    runtime = get_pcp_runahead_runtime()
    if runtime is not None:
        with pcp_nvtx_range(f"pcp.standard.runahead:{layer_name}"):
            runtime.update_and_replicate((key, value), slot_mapping, apply)
        return

    if slot_mapping.shape[0] > key.shape[0]:
        pcp_group = get_pcp_group()
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
        with pcp_nvtx_range(f"pcp.baseline_kv_allgather:{layer_name}"):
            key = pcp_group.all_gather(key[:local_rows].contiguous(), dim=0)
            value = pcp_group.all_gather(value[:local_rows].contiguous(), dim=0)

    apply((key, value), slot_mapping)


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
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def update_mla_kv_cache(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
    impl: Any,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> None:
    """Update MLA KV cache through baseline PCP or experimental runahead PCP."""
    runtime = get_pcp_runahead_runtime() if use_pcp else None
    if runtime is not None:
        if num_decode_tokens not in (0, None):
            raise RuntimeError("runahead PCP only supports fresh prefill cache updates")
        assert slot_mapping is not None

        def apply(
            tensors: tuple[torch.Tensor, ...],
            cache_slot_mapping: torch.Tensor,
        ) -> None:
            cache_kv_c, cache_k_pe = tensors
            with pcp_nvtx_range("pcp.mla.cache_write"):
                impl.do_kv_cache_update(
                    cache_kv_c,
                    cache_k_pe,
                    kv_cache,
                    cache_slot_mapping,
                    kv_cache_dtype,
                    k_scale,
                )

        with pcp_nvtx_range("pcp.mla.runahead"):
            runtime.update_and_replicate(
                (kv_c_normed, k_pe),
                slot_mapping,
                apply,
            )
        return

    kv_for_cache, kpe_for_cache, cache_slot_mapping = (
        maybe_gather_mla_latent_cache_inputs(
            kv_c_normed,
            k_pe,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
    )
    with pcp_nvtx_range("pcp.mla.cache_write"):
        impl.do_kv_cache_update(
            kv_for_cache,
            kpe_for_cache,
            kv_cache,
            cache_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def update_indexer_k_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Insert sparse-indexer K with the same runahead/fallback policy as MLA."""
    runtime = get_pcp_runahead_runtime() if use_pcp else None
    if runtime is not None:
        if num_decode_tokens != 0:
            raise RuntimeError("runahead PCP only supports fresh prefill indexer updates")

        def apply(
            tensors: tuple[torch.Tensor, ...],
            cache_slot_mapping: torch.Tensor,
        ) -> None:
            (cache_k,) = tensors
            with pcp_nvtx_range("pcp.indexer.cache_write"):
                ops.indexer_k_quant_and_cache(
                    cache_k,
                    kv_cache,
                    cache_slot_mapping,
                    quant_block_size,
                    scale_fmt,
                )

        with pcp_nvtx_range("pcp.indexer.runahead"):
            runtime.update_and_replicate((k,), slot_mapping, apply)
        return

    cache_k, cache_slot_mapping = maybe_gather_indexer_k(
        k,
        slot_mapping,
        num_decode_tokens,
        use_pcp,
    )
    with pcp_nvtx_range("pcp.indexer.cache_write"):
        ops.indexer_k_quant_and_cache(
            cache_k,
            kv_cache,
            cache_slot_mapping,
            quant_block_size,
            scale_fmt,
        )


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

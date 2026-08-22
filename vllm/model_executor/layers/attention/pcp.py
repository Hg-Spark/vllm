# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.v1.attention.ops.pcp_profile import pcp_nvtx_range
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


_MLA_RUNAHEAD_RANGES = {
    "full_kv_collective": "pcp.mla_full_kv_exchange",
    "prefix_p2p": "pcp.mla_prefix_exchange",
    "direct_p2p": "pcp.mla_direct_exchange",
}


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
    with pcp_nvtx_range(
        "pcp.mla_baseline_kv_allgather",
        rank=pcp_group.rank_in_group,
        rows=local_num_tokens - num_decode_tokens,
    ):
        gathered_prefills = tuple(
            pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
            for tensor in tensors
        )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

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


def _ensure_mla_page_push_cache_write_hooks() -> None:
    """Wrap MLA native KV writes with the producer-push layer lifecycle."""
    forward_context = get_forward_context()
    for layer in forward_context.no_compile_layers.values():
        impl = getattr(layer, "impl", None)
        if (
            impl is None
            or not hasattr(layer, "kv_lora_rank")
            or not hasattr(layer, "kv_cache_dtype")
            or not hasattr(impl, "do_kv_cache_update")
            or getattr(impl, "_pcp_mla_page_push_wrapped", False)
        ):
            continue

        cache_dtype = str(layer.kv_cache_dtype)
        if cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "MLA PCP page-push currently supports unquantized FP16/BF16 "
                f"KV cache only, got cache_dtype={cache_dtype}"
            )

        original_cache_update = impl.do_kv_cache_update

        def page_push_cache_update(*args, _original=original_cache_update, **kwargs):
            runtime = get_pcp_runahead_runtime()
            if runtime is None or runtime.transport != "page_pull":
                return _original(*args, **kwargs)

            if len(args) >= 3:
                kv_cache = args[2]
            else:
                kv_cache = kwargs.get("kv_cache")
            if not isinstance(kv_cache, torch.Tensor):
                raise RuntimeError(
                    "MLA PCP page-push could not identify the native KV cache tensor"
                )

            runtime.page_push_prepare_layer(kv_cache)
            result = _original(*args, **kwargs)
            runtime.page_push_after_cache_write(kv_cache)
            return result

        impl.do_kv_cache_update = page_push_cache_update
        impl._pcp_mla_page_push_wrapped = True


def _exchange_runahead_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor] | None:
    """Exchange rank-local cache rows according to the active runahead policy."""
    runtime = get_pcp_runahead_runtime()
    if runtime is None:
        return None
    if runtime.transport == "page_pull":
        if any(tensor.shape[0] != runtime.local_rows for tensor in tensors):
            raise RuntimeError(
                "MLA PCP page-push expects configured rank-local latent rows: "
                f"rows={[tensor.shape[0] for tensor in tensors]}, "
                f"expected={runtime.local_rows}"
            )
        _ensure_mla_page_push_cache_write_hooks()
        return tensors, runtime.rank_local_slot_mapping(slot_mapping)
    range_name = _MLA_RUNAHEAD_RANGES.get(runtime.transport)
    if range_name is None:
        raise RuntimeError(f"unexpected PCP runahead transport: {runtime.transport!r}")
    with pcp_nvtx_range(
        range_name,
        e=runtime.epoch,
        rank=runtime.rank,
        rows=runtime.local_rows,
    ):
        return runtime.exchange_cache_inputs(tensors, slot_mapping)


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

    runahead_result = _exchange_runahead_cache_inputs(
        (kv_c_normed, k_pe_flat), slot_mapping
    )
    if runahead_result is not None:
        if num_decode_tokens != 0:
            raise RuntimeError(
                "PCP runahead MLA cache exchange requires a fresh prefill batch"
            )
        (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = runahead_result
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
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output

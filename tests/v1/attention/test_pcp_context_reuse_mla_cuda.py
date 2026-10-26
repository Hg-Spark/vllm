# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical reference for PCP MLA cached-context prefill on CUDA."""

import pytest
import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonPrefillMetadata,
    _build_pcp_packed_context_metadata,
    _plan_pcp_context_reuse,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA MLA")
@torch.inference_mode()
def test_pcp_cuda_context_matches_independent_reference(default_vllm_config) -> None:
    """Real gather, projection, attention, and restore for one PCP pair.

    Only cached context is compared. In particular, [64, 80) holds the first
    virtual row's new tokens and is not part of the second row's context
    partial; the tail begins at the absolute cache offset 80.
    """
    device = torch.device("cuda")
    heads, rank, nope, rope, value_dim = 4, 512, 64, 64, 128
    qk_dim = nope + rope
    scale = qk_dim**-0.5
    generator = torch.Generator(device=device).manual_seed(2026)

    with set_current_vllm_config(default_vllm_config):
        projection = ColumnParallelLinear(
            input_size=rank,
            output_size=heads * (nope + value_dim),
            bias=False,
            params_dtype=torch.bfloat16,
            disable_tp=True,
        ).to(device)
        projection.weight.copy_(
            torch.randn(
                projection.weight.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ) * 0.08
        )
        impl = TritonMLAImpl(
            num_heads=heads,
            head_size=rank + rope,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=rank,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            qk_head_dim=qk_dim,
            v_head_dim=value_dim,
            kv_b_proj=projection,
        )
        assert FlashAttnPrefillBackend.is_available()
        backend = FlashAttnPrefillBackend(
            num_heads=heads,
            scale=scale,
            kv_lora_rank=rank,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=value_dim,
            vllm_config=default_vllm_config,
        )
        assert backend.supports_out()

        cache = (
            torch.randn(
                (8, 16, rank + rope),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.7
        )
        query = torch.randn(
            (32, heads, qk_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        block_table = torch.arange(
            8, dtype=torch.int32, device=device
        ).repeat(2, 1)
        plan = _plan_pcp_context_reuse([7, 7], [64, 112], [0, 16, 32])
        assert plan is not None

        # Reference does not use the packed query selector, native planner,
        # context gather kernel, or optimized restore code.
        flat_cache = cache.view(-1, rank + rope)
        projected = torch.nn.functional.linear(
            flat_cache[:, :rank], projection.weight
        ).view(-1, heads, nope + value_dim)
        k_nope, values = projected.split([nope, value_dim], dim=-1)
        k_rope = flat_cache[:, rank:].unsqueeze(1).expand(-1, heads, -1)
        keys = torch.cat((k_nope, k_rope), dim=-1)
        expected_outputs = []
        expected_lses = []
        for query_slice, positions in (
            (slice(0, 16), list(range(64))),
            (slice(16, 32), list(range(64)) + list(range(80, 112))),
        ):
            scores = (
                torch.einsum(
                    "qhd,khd->hqk",
                    query[query_slice].float(),
                    keys[positions].float(),
                )
                * scale
            )
            weights = scores.softmax(-1)
            expected_outputs.append(
                torch.einsum(
                    "hqk,khd->qhd", weights, values[positions].float()
                )
            )
            expected_lses.append(torch.logsumexp(scores, dim=-1))
        expected_output = torch.cat(expected_outputs, dim=0)
        expected_lse = torch.cat(expected_lses, dim=1)
        assert not torch.allclose(expected_output[0], expected_output[1])

        results = []
        for workspace_size, expected_chunks in ((128, 1), (32, 3)):
            context = _build_pcp_packed_context_metadata(
                plan,
                chunked_prefill_workspace=torch.empty(
                    (workspace_size, rank + rope),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                chunked_prefill_workspace_size=workspace_size,
                page_size=16,
                device=device,
                dcp_local_block_size=1,
                dcp_virtual_block_size=1,
                align_indptr=False,
                block_table=block_table,
            )
            assert context is not None
            assert len(context.chunks) == expected_chunks
            prefill = MLACommonPrefillMetadata(
                block_table=block_table,
                query_start_loc=torch.tensor(
                    [0, 16, 32], device=device, dtype=torch.int32
                ),
                max_query_len=16,
                chunked_context=context,
                q_data_type=torch.bfloat16,
                output_dtype=torch.bfloat16,
                prefill_backend=backend,
            )
            metadata = MLACommonMetadata(
                num_reqs=2,
                max_query_len=16,
                max_seq_len=128,
                num_actual_tokens=32,
                query_start_loc=prefill.query_start_loc,
                slot_mapping=torch.arange(32, device=device),
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=2,
                prefill=prefill,
            )
            output, lse = impl._compute_prefill_context(
                query, cache, metadata, torch.ones((), device=device)
            )
            torch.testing.assert_close(
                output.float(), expected_output, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                lse.float(), expected_lse, rtol=2e-2, atol=2e-2
            )
            results.append((output, lse))

        for single, split in zip(*results):
            torch.testing.assert_close(
                single.float(), split.float(), rtol=2e-2, atol=2e-2
            )

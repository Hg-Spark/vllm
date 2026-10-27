# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP MLA shared/tail job planning with the native chunk planner."""

import pytest
import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonPrefillMetadata,
    _build_pcp_job_context_metadata,
    _plan_pcp_context_reuse,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType


def _native_chunks(plan, context_lens=(64, 112), workspace=128):
    return _build_pcp_job_context_metadata(
        plan,
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        chunked_prefill_workspace=torch.empty((workspace, 1)),
        chunked_prefill_workspace_size=workspace,
        page_size=16,
        device=torch.device("cpu"),
    )


def test_pcp_shared_tail_jobs_and_native_chunks() -> None:
    plan = _plan_pcp_context_reuse([7, 7], [64, 112], [0, 16, 32])
    assert plan is not None
    assert plan.merged_starts == [0, 32]
    assert [
        (job.source_row, job.context_start, job.context_len, job.query_slice)
        for job in plan.jobs
    ] == [
        (0, 0, 64, slice(0, 32)),
        (0, 80, 32, slice(16, 32)),
    ]
    assert [job.continuation for job in plan.jobs] == [False, True]
    chunks = _native_chunks(plan)
    assert chunks is not None
    assert [chunk.starts.tolist() for chunk in chunks.chunks] == [[0], [80]]
    assert [chunk.is_continuation for chunk in chunks.chunks] == [False, True]


def test_pcp_unaligned_tail_keeps_absolute_offset() -> None:
    plan = _plan_pcp_context_reuse([7, 7], [70, 100], [0, 8, 24])
    assert plan is not None
    chunks = _native_chunks(plan, context_lens=(70, 100))
    assert chunks is not None
    assert [chunk.starts.tolist() for chunk in chunks.chunks] == [[0], [78]]


def test_pcp_singleton_and_zero_context_preserve_original_queries() -> None:
    plan = _plan_pcp_context_reuse(
        [10, 11, 20, 20], [32, 0, 64, 112], [0, 16, 32, 48, 64]
    )
    assert plan is not None
    assert plan.merged_starts == [0, 16, 32, 64]
    assert plan.empty_token_slices == [slice(16, 32)]
    assert [(job.source_row, job.context_start) for job in plan.jobs] == [
        (0, 0),
        (2, 0),
        (2, 80),
    ]


def test_pcp_invalid_pairs_fall_back() -> None:
    assert _plan_pcp_context_reuse([7, 7, 7], [64, 80, 96], [0, 16, 32, 48]) is None
    assert _plan_pcp_context_reuse([7, 8, 7], [64, 64, 64], [0, 16, 32, 48]) is None
    assert _plan_pcp_context_reuse([7, 7], [64, 70], [0, 16, 32]) is None


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@torch.inference_mode()
def test_pcp_context_matches_reference(default_vllm_config, device_type: str) -> None:
    """Compare the real PCP context path with independent attention references.

    The second row attends to [0, 64) and [80, 112), excluding the first
    row's new tokens at [64, 80). Both native workspace splits are covered.
    """
    if device_type == "cpu":
        if not current_platform.is_cpu():
            pytest.skip("requires CPU MLA")
        from vllm.v1.attention.backends.mla.cpu_mla import CPUMLAImpl
        from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import (
            CPUSDPAMLAPrefillBackend,
        )

        impl_cls = CPUMLAImpl
        backend_cls = CPUSDPAMLAPrefillBackend
        heads, rank, nope, rope, value_dim = 3, 16, 8, 0, 8
        dtype = torch.float32
        scale = 1.0
        weight_scale, cache_scale = 0.35, 0.75
        max_seq_len = 144
        rtol, atol = 1e-5, 2e-5
    else:
        if not current_platform.is_cuda():
            pytest.skip("requires CUDA MLA")
        from vllm.v1.attention.backends.mla.prefill.flash_attn import (
            FlashAttnPrefillBackend,
        )
        from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

        impl_cls = TritonMLAImpl
        backend_cls = FlashAttnPrefillBackend
        assert backend_cls.is_available()
        heads, rank, nope, rope, value_dim = 4, 512, 64, 64, 128
        dtype = torch.bfloat16
        scale = (nope + rope) ** -0.5
        weight_scale, cache_scale = 0.08, 0.7
        max_seq_len = 128
        rtol, atol = 2e-2, 2e-2

    device = torch.device(device_type)
    generator = torch.Generator(device=device).manual_seed(2026)
    qk_dim = nope + rope

    with set_current_vllm_config(default_vllm_config):
        projection = ColumnParallelLinear(
            input_size=rank,
            output_size=heads * (nope + value_dim),
            bias=False,
            params_dtype=dtype,
            disable_tp=True,
        ).to(device)
        projection.weight.copy_(
            torch.randn(
                projection.weight.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * weight_scale
        )
        impl = impl_cls(
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
        backend = backend_cls(
            num_heads=heads,
            scale=scale,
            kv_lora_rank=rank,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=value_dim,
            vllm_config=default_vllm_config,
        )

        plan = _plan_pcp_context_reuse([7, 7], [64, 112], [0, 16, 32])
        assert plan is not None
        block_table = torch.arange(
            8, dtype=torch.int32, device=device
        ).repeat(2, 1)
        cache = (
            torch.randn(
                (8, 16, rank + rope),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * cache_scale
        )
        query = torch.randn(
            (32, heads, qk_dim), generator=generator, device=device, dtype=dtype
        )

        # Independent reference: projection and attention on absolute KV indices.
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
            weights = scores.softmax(dim=-1)
            expected_outputs.append(
                torch.einsum(
                    "hqk,khd->qhd", weights, values[positions].float()
                )
            )
            expected_lses.append(torch.logsumexp(scores, dim=-1))
        expected_output = torch.cat(expected_outputs)
        expected_lse = torch.cat(expected_lses, dim=1)
        assert not torch.allclose(expected_output[0], expected_output[1])
        assert not torch.allclose(expected_output[0, 0], expected_output[0, 1])

        results = []
        for workspace_size, expected_chunks in ((128, 2), (32, 3)):
            context = _build_pcp_job_context_metadata(
                plan,
                context_lens_cpu=torch.tensor([64, 112], dtype=torch.int32),
                chunked_prefill_workspace=torch.empty(
                    (workspace_size, rank + rope), device=device, dtype=dtype
                ),
                chunked_prefill_workspace_size=workspace_size,
                page_size=16,
                device=device,
                align_indptr=False,
            )
            assert context is not None
            assert len(context.chunks) == expected_chunks
            query_start_loc = torch.tensor(
                [0, 16, 32], dtype=torch.int32, device=device
            )
            prefill = MLACommonPrefillMetadata(
                block_table=block_table,
                query_start_loc=query_start_loc,
                max_query_len=16,
                chunked_context=context,
                q_data_type=dtype,
                output_dtype=dtype,
                prefill_backend=backend,
            )
            metadata = MLACommonMetadata(
                num_reqs=2,
                max_query_len=16,
                max_seq_len=max_seq_len,
                num_actual_tokens=32,
                query_start_loc=query_start_loc,
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
                output.float(), expected_output, rtol=rtol, atol=atol
            )
            torch.testing.assert_close(
                lse.float(), expected_lse, rtol=rtol, atol=atol
            )
            results.append((output, lse))

        split_rtol = 0 if device_type == "cpu" else rtol
        split_atol = 1e-4 if device_type == "cpu" else atol
        for single, split in zip(*results):
            torch.testing.assert_close(
                single.float(), split.float(), rtol=split_rtol, atol=split_atol
            )

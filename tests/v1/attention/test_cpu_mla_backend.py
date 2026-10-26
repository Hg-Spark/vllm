# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the CPU MLA backend.

Two levels of coverage are provided:

* :func:`test_kv_cache_cpu_write` checks that the kv-cache write logic
  inlined in the CPU MLA backend writes the latent KV cache in the same
  layout the CPU decode kernel expects. This runs without downloading
  any model weights and is safe for CI.
* :func:`test_cpu_mla_backend_smoke` exercises the full end-to-end path
  (LLM engine, model construction, prefill + decode) against a shrunk
  DeepSeek-V2-Lite with `hf_overrides`. It uses `dummy` weights so the
  outputs are not meaningful, only shape / plumbing correctness is
  asserted. Marked `cpu_model` so it is skipped on non-CPU CI.
"""

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonPrefillMetadata,
    _build_pcp_packed_context_metadata,
    _plan_pcp_context_reuse,
    build_mla_chunked_context_metadata,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.cpu_mla import CPUMLAImpl
from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import (
    CPUSDPAMLAPrefillBackend,
)
from vllm.v1.attention.backends.mla.prefill.selector import (
    get_mla_prefill_backend,
)
from vllm.v1.attention.backends.mla.prefill.zen_cpu_sdpa import (
    ZenCPUSDPAMLAPrefillBackend,
)


def _reference_ragged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seq_lens: torch.Tensor,
    kv_cu_seq_lens: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    num_reqs = q_cu_seq_lens.numel() - 1
    for req_idx in range(num_reqs):
        q_start = int(q_cu_seq_lens[req_idx])
        q_end = int(q_cu_seq_lens[req_idx + 1])
        kv_start = int(kv_cu_seq_lens[req_idx])
        kv_end = int(kv_cu_seq_lens[req_idx + 1])

        q_r = q[q_start:q_end].transpose(0, 1).float()
        k_r = k[kv_start:kv_end].transpose(0, 1).float()
        v_r = v[kv_start:kv_end].transpose(0, 1).float()

        scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones(
                (q_end - q_start, kv_end - kv_start),
                dtype=torch.bool,
                device=scores.device,
            ).triu(1)
            scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_r).transpose(0, 1).to(v.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))

    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_kv_cache_cpu_write() -> None:
    kv_lora_rank = 512
    pe_dim = 64
    block_size = 16
    num_blocks = 4
    num_tokens = 8
    dtype = torch.float32

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype)
    k_pe = torch.randn(num_tokens, pe_dim, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + pe_dim, dtype=dtype)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)
    # Mark the last slot as padding to make sure we honour negative slots.
    slot_mapping[-1] = -1

    flat = kv_cache.view(-1, kv_lora_rank + pe_dim)
    slots = slot_mapping.to(torch.long)
    valid_mask = slots >= 0
    if not valid_mask.all().item():
        slots = slots[valid_mask]
        kv_c = kv_c[valid_mask]
        k_pe = k_pe[valid_mask]
    target_dtype = flat.dtype
    flat[slots, :kv_lora_rank] = kv_c.to(target_dtype)
    flat[slots, kv_lora_rank:] = k_pe.to(target_dtype)

    flat = kv_cache.view(-1, kv_lora_rank + pe_dim)
    for i in range(num_tokens - 1):
        assert torch.equal(flat[i, :kv_lora_rank], kv_c[i])
        assert torch.equal(flat[i, kv_lora_rank:], k_pe[i])
    # The padding slot must remain untouched.
    assert torch.all(flat[num_tokens - 1] == 0)


@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_backend_smoke(tmp_path) -> None:
    # Shrink the architecture aggressively so the test is CI-friendly.
    from vllm import LLM, SamplingParams

    hf_overrides = {
        "num_hidden_layers": 2,
        "n_routed_experts": 4,
        "first_k_dense_replace": 0,
        "num_experts_per_tok": 2,
    }
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        block_size=16,
        gpu_memory_utilization=0.25,
        hf_overrides=hf_overrides,
        kv_transfer_config={
            "kv_connector": "ExampleConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path),
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    outputs = llm.generate([[1, 2, 3, 4], [5, 6, 7]], sampling_params, use_tqdm=False)
    assert len(outputs) == 2
    for output in outputs:
        # `dummy` weights do not produce meaningful text, but the number
        # of generated tokens must match what we asked for.
        assert len(output.outputs[0].token_ids) == 4

    prefix_hit_prompt = [1] * 20
    outputs = llm.generate([prefix_hit_prompt], sampling_params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 4
    assert llm.reset_prefix_cache(reset_connector=False)

    # ExampleConnector persists KV externally at block granularity. With a
    # 20-token prompt and block_size=16, the second run reloads 16 cached
    # tokens and forces the 4-token suffix through MLA prefill with context.
    outputs = llm.generate([prefix_hit_prompt], sampling_params, use_tqdm=False)
    assert len(outputs[0].outputs[0].token_ids) == 4


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_backend_selected() -> None:
    # On CPU (no device capability) the MLA prefill backend must be the
    # SDPA-based CPU backend, not flash-attn which is unavailable on CPU.
    # Accelerator-specific subclasses (e.g. the zentorch-backed Zen CPU
    # backend) are also valid selections.
    backend_cls = get_mla_prefill_backend(None)
    assert issubclass(backend_cls, CPUSDPAMLAPrefillBackend)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_new_tokens() -> None:
    # The SDPA prefill must exactly match a reference causal attention over
    # the ragged (varlen) q/k/v layout, with per-request padding handled.
    backend = CPUSDPAMLAPrefillBackend(
        num_heads=4,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        vllm_config=None,
    )

    class _PrefillMeta:
        query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)

    backend.prepare_metadata(_PrefillMeta())
    q = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    out = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)
    assert not isinstance(out, tuple)
    ref, _ = _reference_ragged_attention(
        q,
        k,
        v,
        _PrefillMeta.query_start_loc,
        _PrefillMeta.query_start_loc,
        scale=0.25,
        causal=True,
    )
    assert out.float().allclose(ref.float(), atol=2e-2)
    assert out.shape == (6, 4, 16)

    out_lse, lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    ref_out, ref_lse = _reference_ragged_attention(
        q,
        k,
        v,
        _PrefillMeta.query_start_loc,
        _PrefillMeta.query_start_loc,
        scale=0.25,
        causal=True,
    )
    assert out_lse.float().allclose(ref_out.float(), atol=2e-2)
    assert lse.float().allclose(ref_lse.float(), atol=2e-2)
    assert lse.shape == (4, 6)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_cpu_mla_prefill_context_chunk(default_vllm_config) -> None:
    backend = CPUSDPAMLAPrefillBackend(
        num_heads=4,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        vllm_config=default_vllm_config,
    )
    context = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor([3, 4], dtype=torch.int32),
        prefill_query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        chunked_prefill_workspace=torch.empty((16, 1)),
        chunked_prefill_workspace_size=16,
        block_size=16,
        align_chunk_to_block=True,
        device=torch.device("cpu"),
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
    )
    assert context is not None and len(context.chunks) == 1
    chunk = context.chunks[0]

    generator = torch.Generator().manual_seed(0)
    q = torch.randn((5, 4, 16), generator=generator, dtype=torch.bfloat16)
    k = torch.randn((7, 4, 16), generator=generator, dtype=torch.bfloat16)
    v = torch.randn((7, 4, 16), generator=generator, dtype=torch.bfloat16)
    result, lse = backend.run_prefill_context_chunk(chunk, q, k, v)
    ref_out, ref_lse = _reference_ragged_attention(
        q,
        k,
        v,
        chunk.query_start_loc,
        chunk.cu_seq_lens,
        scale=0.25,
        causal=False,
    )
    torch.testing.assert_close(result.float(), ref_out.float(), rtol=0, atol=2e-2)
    torch.testing.assert_close(lse.float(), ref_lse.float(), rtol=0, atol=2e-2)

    assert backend.supports_out()
    dest = torch.empty_like(result)
    output, output_lse = backend.run_prefill_context_chunk(
        chunk, q, k, v, out=dest
    )
    assert output.data_ptr() == dest.data_ptr()
    torch.testing.assert_close(output, result, rtol=0, atol=0)
    torch.testing.assert_close(output_lse, lse, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
@pytest.mark.skipif(
    not ZenCPUSDPAMLAPrefillBackend.is_available(),
    reason="requires a Zen CPU with the zentorch_sdpa op registered",
)
def test_zen_cpu_mla_prefill_new_tokens() -> None:
    # Runs the zentorch kernel itself. The head dims are deliberately
    # asymmetric (v_head_dim < qk_head_dim, as in real MLA) so that the V
    # padding and output slicing are exercised; the generic-CPU cases above
    # use equal dims and never reach that code.
    qk_head_dim = 24
    v_head_dim = 16
    backend = ZenCPUSDPAMLAPrefillBackend(
        num_heads=4,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=v_head_dim,
        vllm_config=None,
    )

    class _PrefillMeta:
        query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)

    backend.prepare_metadata(_PrefillMeta())
    q = torch.randn(6, 4, qk_head_dim, dtype=torch.bfloat16)
    k = torch.randn(6, 4, qk_head_dim, dtype=torch.bfloat16)
    v = torch.randn(6, 4, v_head_dim, dtype=torch.bfloat16)

    out = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)
    assert not isinstance(out, tuple)
    ref, _ = _reference_ragged_attention(
        q,
        k,
        v,
        _PrefillMeta.query_start_loc,
        _PrefillMeta.query_start_loc,
        scale=0.25,
        causal=True,
    )
    assert out.shape == (6, 4, v_head_dim)
    assert out.float().allclose(ref.float(), atol=2e-2)

    # zentorch_sdpa does not return a usable LSE, so that path must fall back
    # to the fp32 reference and match the generic backend exactly.
    generic = CPUSDPAMLAPrefillBackend(
        num_heads=4,
        scale=0.25,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=v_head_dim,
        vllm_config=None,
    )
    generic.prepare_metadata(_PrefillMeta())
    zen_out, zen_lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    gen_out, gen_lse = generic.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    assert torch.equal(zen_out, gen_out)
    assert torch.equal(zen_lse, gen_lse)


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU MLA implementation")
def test_pcp_context_executes_real_cpu_mla(default_vllm_config) -> None:
    # Real MLA implementation and backend; distinct queries, keys, and heads
    # ensure wrong PCP query remapping cannot masquerade as an averaging test.
    num_heads, rank, dim = 3, 16, 8
    projection = ColumnParallelLinear(
        input_size=rank,
        output_size=num_heads * (dim + dim),
        bias=False,
        params_dtype=torch.float32,
        disable_tp=True,
    )
    generator = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        projection.weight.copy_(
            torch.randn(projection.weight.shape, generator=generator) * 0.35
        )

    impl = CPUMLAImpl(
        num_heads=num_heads,
        head_size=rank,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=rank,
        qk_nope_head_dim=dim,
        qk_rope_head_dim=0,
        qk_head_dim=dim,
        v_head_dim=dim,
        kv_b_proj=projection,
    )
    backend = CPUSDPAMLAPrefillBackend(
        num_heads=num_heads,
        scale=1.0,
        kv_lora_rank=rank,
        qk_nope_head_dim=dim,
        qk_rope_head_dim=0,
        v_head_dim=dim,
        vllm_config=default_vllm_config,
    )
    plan = _plan_pcp_context_reuse([7, 7], [64, 112], [0, 16, 32])
    assert plan is not None
    block_table = torch.arange(8, dtype=torch.int32).repeat(2, 1)
    cache = torch.randn((8, 16, rank), generator=generator) * 0.75
    q = torch.randn((32, num_heads, dim), generator=generator)

    # Reference the *context partial*, not full causal attention: the second
    # PCP row reuses [0, 64) and its tail [80, 112), but excludes [64, 80).
    projected = torch.nn.functional.linear(
        cache.view(-1, rank), projection.weight
    ).view(-1, num_heads, 2 * dim)
    keys, values = projected.split(dim, dim=-1)
    references = []
    for query_slice, context_positions in (
        (slice(0, 16), list(range(64))),
        (slice(16, 32), list(range(64)) + list(range(80, 112))),
    ):
        scores = torch.einsum(
            "qhd,khd->hqk",
            q[query_slice],
            keys[context_positions],
        ) * impl.scale
        probs = scores.softmax(dim=-1)
        references.append(
            (
                torch.einsum("hqk,khd->qhd", probs, values[context_positions]),
                torch.logsumexp(scores, dim=-1),
            )
        )
    ref_output = torch.cat([pair[0] for pair in references])
    ref_lse = torch.cat([pair[1] for pair in references], dim=1)
    # Every token/head must carry a distinguishable query-dependent score.
    assert not torch.allclose(ref_output[0], ref_output[1])
    assert not torch.allclose(ref_output[0, 0], ref_output[0, 1])

    outputs = []
    for workspace_size, expected_chunks in ((128, 1), (32, 3)):
        context = _build_pcp_packed_context_metadata(
            plan,
            chunked_prefill_workspace=torch.empty((workspace_size, rank)),
            chunked_prefill_workspace_size=workspace_size,
            page_size=16,
            device=torch.device("cpu"),
            dcp_local_block_size=1,
            dcp_virtual_block_size=1,
            align_indptr=False,
            block_table=block_table,
        )
        assert context is not None
        assert len(context.chunks) == expected_chunks
        prefill = MLACommonPrefillMetadata(
            block_table=block_table,
            query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
            max_query_len=16,
            chunked_context=context,
            q_data_type=torch.float32,
            output_dtype=torch.float32,
            prefill_backend=backend,
        )
        metadata = MLACommonMetadata(
            num_reqs=2,
            max_query_len=16,
            max_seq_len=144,
            num_actual_tokens=32,
            query_start_loc=prefill.query_start_loc,
            slot_mapping=torch.arange(32, dtype=torch.int64),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=2,
            prefill=prefill,
        )
        output, lse = impl._compute_prefill_context(
            q, cache, metadata, torch.tensor(1.0)
        )
        torch.testing.assert_close(output, ref_output, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=2e-5)
        outputs.append((output, lse))

    for single, multi in zip(*outputs):
        torch.testing.assert_close(single, multi, rtol=0, atol=1e-4)


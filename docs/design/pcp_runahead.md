# PCP Prefill Runahead

This document describes the experimental prefill context parallel (PCP) runahead path, using the vLLM 0.27 PCP design as the baseline.

## Goal

Baseline PCP materializes the full current-layer KV image before attention:

```text
Layer L local KV
        |
        v
  PCP AllGather
        |
        v
replicated KV cache
        |
        v
    attention
        |
        v
    Layer L+1
```

For causal attention, PCP rank `r` only needs the current-step KV prefix from ranks `0..r`. Runahead uses that dependency directly:

```text
Layer L local KV
        |
        +---- async full-KV AllGather ----------------> replicated KV cache
        |
        +---- causal prefix P2P ----> attention -----> Layer L+1
```

Earlier PCP ranks can therefore enter later transformer layers while later ranks are still finishing the current layer.

## Current Architecture

The implementation is split by responsibility:

```text
RunaheadPCPManager
  |- workload eligibility
  |- contiguous PCP partition
  |- slot-mapping policy
  `- runahead runtime lifecycle
          |
          v
model_executor/layers/attention/pcp.py
  |- standard MHA/GQA/MQA PCP cache policy
  |- MLA PCP cache policy
  `- sparse-indexer PCP cache policy
          |
          v
PCPRunaheadRuntime
  |- causal-prefix exchange
  |- pending P2P sends
  |- asynchronous full-cache replication
  |- replica completion callbacks
  `- bounded backpressure
          |
          v
pcp_transport.py
  |- PCP-local -> global rank mapping
  |- batched tensor P2P
  `- async all_gather_into_tensor
```

The former `pcp_runahead_ext.py` import-time class patch has been removed. Eligibility, validation, partitioning, slot mapping, and manager construction now live directly in `pcp_manager.py`.

`pcp_standard.py` remains a narrow compatibility shim for the current MRV2 early config/backend gate and the standard Attention cache-write hook. The communication policy itself lives in the existing attention PCP helper module alongside MLA.

## Supported Standard-Attention Workloads

Standard FlashAttention MHA/GQA/MQA can use PCP with TP=1.

Runahead is selected for homogeneous prefill/extend batches when the aggregate number of scheduled prefill tokens is at least 1024. Existing context is intentionally absent from the eligibility predicate, so the same path covers:

- fresh long prefill,
- chunked or continued prefill,
- prefix-cache hit plus a long suffix,
- long existing context plus a long extend,
- multiple independent requests that are all still in prefill/extend.

Mixed decode+prefill batches keep the baseline PCP path. Pure decode keeps rank-local slot mappings and local cache writes.

MLA retains the original single fresh full-prefill runahead eligibility until that path is validated for the broader workload set.

## Contiguous Runahead Partition

For PCP size 4, a request is partitioned into four contiguous current-step chunks:

```text
full:  | chunk0 | chunk1 | chunk2 | chunk3 |
rank:      0        1        2        3
```

For multiple homogeneous prefill/extend requests:

```text
A -> A0 A1 A2 A3
B -> B0 B1 B2 B3
C -> C0 C1 C2 C3

rank0 = [A0 B0 C0]
rank1 = [A1 B1 C1]
rank2 = [A2 B2 C2]
rank3 = [A3 B3 C3]
```

The padded rank-major slot mapping is the contract between the manager and the cache-update policy.

## Layer Dependency

The causal visibility at PCP size 4 is:

```text
rank0: KV0             -> attention0 -> next layer
rank1: KV0 + KV1       -> attention1 -> next layer
rank2: KV0 + KV1 + KV2 -> attention2 -> next layer
rank3: KV0..KV3        -> attention3 -> next layer
```

The current-step prefix moves left-to-right:

```text
rank0 ---- KV0 ----> rank1 ---- KV0+KV1 ----> rank2 ---- KV0+KV1+KV2 ----> rank3
```

Previously computed context remains in the paged KV cache. Runahead communicates only the K/V generated in the current scheduler step.

## Replicated KV Cache

For each layer the runtime:

1. launches asynchronous full-KV AllGather from the local padded slab,
2. exchanges the causal prefix needed by the local PCP rank,
3. writes that visible prefix through the existing cache writer,
4. lets current-layer attention execute,
5. commits the full gathered KV image when the async collective completes,
6. applies backpressure when too many layer replicas remain pending,
7. flushes outstanding replication before the forward pass completes.

This preserves replicated-cache semantics required by later decode.

## Transport

P2P peers are specified in PCP-local rank space and mapped through `GroupCoordinator.ranks`; the runtime does not assume that PCP rank equals global rank.

Known-shape tensors use tensor-only communication. Multiple tensors belonging to one layer are submitted through `torch.distributed.batch_isend_irecv`. Full-cache reconstruction uses asynchronous `all_gather_into_tensor` so the collective can overlap with later computation.

## NVTX Profiling

NVTX ranges are disabled by default. Enable them for Nsight Systems runs with:

```bash
VLLM_PCP_NVTX=1
```

The runtime emits:

```text
pcp.runahead_kv_update
pcp.replica_launch
pcp.prefix_exchange
pcp.visible_cache_update
pcp.replica_commit
pcp.flush
```

Standard Attention additionally emits layer-qualified ranges:

```text
pcp.standard.runahead:<layer_name>
pcp.baseline_kv_allgather:<layer_name>
```

`pcp.replica_launch` marks CPU-side enqueue of the asynchronous collective. The actual NCCL kernel lifetime should be read from the CUDA/NCCL stream in Nsight. `pcp.replica_commit` shows when the program waits for or consumes the completed replicated image.

The desired trace shows layer skew, for example:

```text
rank0: L0 KV/attn -> L1 KV/attn -> L2 KV/attn
rank1:   recv L0 -> L0 attn -> recv L1 -> L1 attn
rank2:                recv L0 -> L0 attn -> ...
```

If every PCP rank continues entering each layer together, the synchronization target has not been achieved even when outputs are correct.

## Current Execution Envelope

The runahead MVP currently requires:

- TP=1,
- DP=1,
- DCP=1,
- PP=1,
- no EP/MoE,
- no DBO,
- no asynchronous scheduling,
- eager execution,
- FlashAttention for standard MHA/GQA/MQA.

Mixed decode+prefill runahead and broader topology support are deferred.

## Validation Gates

Correctness validation should compare baseline PCP and runahead at PCP=2 and PCP=4, covering fresh prefill, chunked prefill, existing-context extend, prefix-cache hits, homogeneous multi-request batches, pure decode fallback, and mixed fallback.

Validate model outputs within numerical tolerance and, where practical, compare final replicated KV-cache contents across PCP ranks.

Timeline validation should use Nsight Systems with NVTX enabled and confirm that earlier ranks execute later-layer GEMM/attention while later PCP ranks are still processing earlier layers.

Performance benchmarking should keep model, prompt/suffix lengths, PCP size, eager mode, cache dtype, and hardware identical between baseline and runahead. Measure TTFT/prefill latency, communication overlap, GPU utilization, peak temporary replication memory, and sensitivity to `max_pending_replica_layers`.

The fixed 1024-token runahead threshold is an initial guardrail and should be replaced or tuned from measured crossover data.

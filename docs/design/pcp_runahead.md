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
  |- optional weighted partition
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
  |- equal-width async all_gather_into_tensor
  `- variable-width async all_gather
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

MLA retains the original single fresh full-prefill runahead eligibility and fixed-width partition. Weighted variable-width partitioning is currently limited to standard FlashAttention MHA/GQA/MQA.

## Contiguous Runahead Partition

Without an explicit load-weight override, PCP size 4 preserves the existing contiguous equal-chunk behavior:

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

The rank-major slot mapping is the contract between the manager and the cache-update policy.

## Weighted Variable-Width Partition

Standard-attention runahead can optionally assign a different current-step token load to each PCP rank. Set one positive numeric weight per PCP rank before starting vLLM:

```bash
VLLM_PCP_RUNAHEAD_LOAD_WEIGHTS=4,2.5,1.9,1.6
```

The number of values must equal the PCP world size. Integer and floating-point values are accepted. Only the relative values matter:

```text
weight_i / sum(weights)
```

For 10,000 scheduled tokens and weights `4,2.5,1.9,1.6`:

```text
rank0 = 4000
rank1 = 2500
rank2 = 1900
rank3 = 1600
```

Integer token counts are produced with a largest-remainder allocation so every scheduled token is assigned exactly once. Each request is partitioned independently and remains contiguous within that request.

When `VLLM_PCP_RUNAHEAD_LOAD_WEIGHTS` is unset, the existing equal-chunk/fixed-width behavior is preserved. Setting explicit equal weights such as `1,1,1,1` enables the compact weighted layout while targeting equal proportions.

Weighted runahead requires every rank to receive at least one token in the aggregate batch. If rounding or an extreme weight vector produces an empty rank, that scheduler step falls back to baseline PCP.

## Variable-Width Layer Dependency

For rows per rank `[4000,2500,1900,1600]`, compact offsets are:

```text
[0, 4000, 6500, 8400, 10000]
```

Causal visibility becomes:

```text
rank0: rows [0:4000]                 -> attention0 -> next layer
rank1: rows [0:6500]                 -> attention1 -> next layer
rank2: rows [0:8400]                 -> attention2 -> next layer
rank3: rows [0:10000]                -> attention3 -> next layer
```

The current-step prefix moves left-to-right without padding:

```text
rank0 -- 4000 --> rank1 -- 6500 --> rank2 -- 8400 --> rank3
```

Previously computed context remains in the paged KV cache. Runahead communicates only the K/V generated in the current scheduler step.

## Replicated KV Cache

For each layer the runtime:

1. launches asynchronous full-KV replication from the local slab,
2. exchanges the causal prefix needed by the local PCP rank,
3. writes that visible prefix through the existing cache writer,
4. lets current-layer attention execute,
5. commits the full gathered KV image when the async collective completes,
6. applies backpressure when too many layer replicas remain pending,
7. flushes outstanding replication before the forward pass completes.

For the default fixed-width path, full-cache reconstruction keeps `all_gather_into_tensor`. Weighted variable-width standard attention uses `torch.distributed.all_gather` with correctly sized output views, producing one compact rank-major output buffer:

```text
[K0 4000][K1 2500][K2 1900][K3 1600]
```

No transport padding is inserted in the weighted path. Hidden-state restoration uses the same compact rank offsets before applying the restore index.

This preserves replicated-cache semantics required by later decode.

## Transport

P2P peers are specified in PCP-local rank space and mapped through `GroupCoordinator.ranks`; the runtime does not assume that PCP rank equals global rank.

Known-shape tensors use tensor-only communication. Multiple tensors belonging to one layer are submitted through `torch.distributed.batch_isend_irecv`. Equal-width full-cache reconstruction uses asynchronous `all_gather_into_tensor`; weighted variable-width reconstruction uses asynchronous uneven `all_gather`.

## NVTX Profiling

NVTX ranges are disabled by default. Enable them for Nsight Systems runs with:

```bash
VLLM_PCP_NVTX=1
```

Important runtime and transport ranges include:

```text
pcp.runahead_kv_update
pcp.replica_launch
pcp.replica_buffer_prepare
pcp.replica_allgather_enqueue
pcp.replica_backpressure
pcp.replica_wait
pcp.replica_cache_update
pcp.replica_commit
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_concat
pcp.prefix_local_prepare
pcp.visible_cache_update
pcp.send_wait
pcp.flush
pcp.transport.recv_enqueue
pcp.transport.send_enqueue
pcp.transport.allgather_enqueue
pcp.transport.variable_allgather_enqueue
pcp.restore_hidden_variable_allgather
```

Standard Attention additionally emits layer-qualified ranges:

```text
pcp.standard.runahead:<layer_name>
pcp.standard.cache_write:<layer_name>
pcp.baseline_kv_allgather:<layer_name>
```

The enqueue ranges mark CPU-side submission of asynchronous communication. The actual NCCL kernel lifetime should be read from the CUDA/NCCL streams in Nsight Systems. Wait ranges show when an asynchronous operation becomes a dependency on the forward path.

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

For weighted standard attention, validation should additionally sweep explicit load vectors such as `1,1,1,1`, `4,2.5,1.9,1.6`, and `4,3,2,1`. Confirm that the per-rank token counts sum to every request length, compact rank offsets agree across ranks, final hidden-state ordering is restored, and the replicated KV cache matches baseline PCP within numerical tolerance.

Timeline validation should use Nsight Systems with NVTX enabled and confirm that earlier ranks execute later-layer GEMM/attention while later PCP ranks are still processing earlier layers. Weighted runs should also compare final-layer completion skew across PCP ranks.

Performance benchmarking should keep model, prompt/suffix lengths, PCP size, eager mode, cache dtype, and hardware identical between baseline and runahead. Measure TTFT/prefill latency, communication overlap, GPU utilization, peak temporary replication memory, per-rank completion time, and sensitivity to both `max_pending_replica_layers` and the load-weight vector.

The fixed 1024-token runahead threshold is an initial guardrail and should be replaced or tuned from measured crossover data.

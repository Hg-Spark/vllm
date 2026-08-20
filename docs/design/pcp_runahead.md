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

For causal attention, PCP rank `r` only needs the current-step KV prefix from ranks `0..r`. Runahead follows that dependency directly:

```text
rank0: L0 KV/attn -> L1 KV/attn -> L2 KV/attn -> ...
          |
          v
rank1:   L0 KV/attn -> L1 KV/attn -> ...
                    |
                    v
rank2:              L0 KV/attn -> ...
```

The transformer-layer critical path contains causal prefix P2P only. Full replicated-cache repair is deferred until the model-forward boundary.

## Current Architecture

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
  |- visible-prefix cache update
  |- deferred replica descriptors
  `- forward-boundary replica repair
          |
          v
pcp_transport.py
  |- PCP-local -> global rank mapping
  |- batched tensor P2P
  |- equal-width async all_gather_into_tensor
  `- variable-width async all_gather
```

The former `pcp_runahead_ext.py` import-time class patch has been removed. Eligibility, validation, partitioning, slot mapping, and manager construction live directly in `pcp_manager.py`.

`pcp_standard.py` remains a compatibility shim for the MRV2 config/backend gate and the standard Attention cache-write hook. The communication policy lives in the existing attention PCP helper module alongside MLA.

## Layer Critical Path

For each participating attention layer, the runtime now executes:

```text
receive prefix from rank r-1
        |
        v
append local current-step KV
        |
        +---- nonblocking send to rank r+1
        |
        v
write causal-visible KV into paged cache
        |
        v
attention
        |
        v
next transformer layer
```

No full-KV AllGather is submitted from `update_and_replicate()` during layer execution.

The runtime stores a deferred replica descriptor containing the local current-step tensors, the compact rank-major slot mapping, and the cache-update callback. The descriptor is repaired after model forward.

## Forward-Boundary Repair

`RunaheadPCPManager.finish_forward()` calls `PCPRunaheadRuntime.flush()`.

The repair sequence is:

1. wait for outstanding prefix sends,
2. rendezvous all PCP ranks on the PCP CPU/Gloo process group,
3. after every rank has exited layer P2P, launch full-cache repair AllGathers on the PCP device/NCCL process group,
4. wait for each repair collective,
5. write the gathered rank-major image into persistent KV cache.

The CPU/Gloo rendezvous is required because earlier PCP ranks can finish all transformer layers while later ranks are still submitting trailing-layer P2P. Submitting an NCCL repair collective immediately from an early rank would re-enter the same device communicator before all ranks had finished the P2P phase and could violate operation ordering.

The boundary is therefore:

```text
                 transformer forward
rank0  L0 -> L1 -> L2 -> ... -> LN ----\
rank1    L0 -> L1 -> ... -> LN ---------+--> CPU/Gloo rendezvous
rank2      L0 -> ... -> LN -------------+          |
rank3        ... -> LN -----------------/           v
                                             NCCL cache repair
```

This boundary synchronization does not prevent cross-layer runahead inside the transformer stack.

## Replicated KV Cache Semantics

During forward, rank `r` contains the current-step prefix owned by ranks `0..r` for each completed layer. After boundary repair every PCP rank receives the complete current-step KV image, preserving the replicated-cache contract used by later chunked prefill and decode steps.

The current MVP reconstructs the full image with AllGather:

```text
[K0][K1][K2][K3]
```

For equal-width partitions this uses `all_gather_into_tensor`. Weighted variable-width standard attention uses uneven `torch.distributed.all_gather` output views and produces one compact rank-major output buffer.

This repair is intentionally placed outside transformer-layer execution. It still contributes to end-to-end prefill latency and TTFT because the current model runner flushes before sampling.

## Current Repair-Memory Tradeoff

Deferred repair retains each layer's local current-step KV tensors until `finish_forward()`.

This is an MVP mechanism for proving the communication schedule. Its temporary-memory cost grows with:

```text
num_layers * local_prefill_tokens * local_kv_width
```

A production implementation should source repair payloads from the already-written persistent paged KV cache through explicit pack/unpack operations. That removes the need to retain every layer's activation tensors across the whole model forward.

A later optimization can replace full AllGather repair with reverse suffix propagation:

```text
forward prefix:
rank0 -> rank1 -> rank2 -> rank3

boundary repair:
rank3 -> rank2 -> rank1 -> rank0
```

That avoids retransmitting prefix data each rank already received during forward.

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

Without an explicit load-weight override, PCP size 4 preserves contiguous equal chunks:

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

The rank-major slot mapping is the contract between the manager and cache-update policy.

## Weighted Variable-Width Partition

Standard-attention runahead can assign a different current-step token load to each PCP rank. Set one positive numeric weight per PCP rank before starting vLLM:

```bash
VLLM_PCP_RUNAHEAD_LOAD_WEIGHTS=4,2.5,1.9,1.6
```

For 10,000 scheduled tokens and these weights:

```text
rank0 = 4000
rank1 = 2500
rank2 = 1900
rank3 = 1600
```

Integer token counts use largest-remainder allocation so every scheduled token is assigned exactly once. Each request is partitioned independently and remains contiguous within that request.

Weighted runahead requires every rank to receive at least one token in the aggregate batch. If rounding or an extreme weight vector produces an empty rank, that scheduler step falls back to baseline PCP.

For rows per rank `[4000,2500,1900,1600]`, compact offsets are:

```text
[0, 4000, 6500, 8400, 10000]
```

Causal visibility becomes:

```text
rank0: rows [0:4000]
rank1: rows [0:6500]
rank2: rows [0:8400]
rank3: rows [0:10000]
```

The current-step prefix moves left-to-right without padding:

```text
rank0 -- 4000 --> rank1 -- 6500 --> rank2 -- 8400 --> rank3
```

Previously computed context remains in paged KV cache. Runahead communicates only K/V generated in the current scheduler step.

## Transport Ordering

P2P peers are specified in PCP-local rank space and mapped through `GroupCoordinator.ranks`; the runtime does not assume PCP rank equals global rank.

Known-shape tensors use tensor-only communication. Multiple tensors belonging to one layer are submitted through `torch.distributed.batch_isend_irecv`.

During transformer forward, the PCP device group carries prefix P2P only. Deferred device-group AllGathers start only after the CPU/Gloo forward-boundary rendezvous. This separation in time is deliberate: it avoids inserting a full collective between consecutive layer-prefix operations on the same NCCL communicator.

## NVTX Profiling

NVTX ranges are disabled by default. Enable them for Nsight Systems runs with:

```bash
VLLM_PCP_NVTX=1
```

Important ranges include:

```text
pcp.runahead_kv_update
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_concat
pcp.prefix_local_prepare
pcp.visible_cache_update
pcp.replica_defer
pcp.send_wait
pcp.flush
pcp.replica_forward_boundary
pcp.replica_commit
pcp.replica_buffer_prepare
pcp.replica_allgather_enqueue
pcp.replica_wait
pcp.replica_cache_update
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

The desired trace shows layer skew before `pcp.replica_forward_boundary`, for example:

```text
rank0: L0 KV/attn -> L1 KV/attn -> L2 KV/attn
rank1:   recv L0 -> L0 attn -> recv L1 -> L1 attn
rank2:                recv L0 -> L0 attn -> ...
```

If every PCP rank continues entering each layer together, the synchronization target has not been achieved even when outputs are correct.

No `pcp.replica_allgather_enqueue` range should appear between transformer layers. All repair ranges should occur after the forward-boundary rendezvous.

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

Mixed decode+prefill runahead and broader topology support remain deferred.

## Validation Gates

Correctness validation should compare baseline PCP and runahead at PCP=2 and PCP=4, covering fresh prefill, chunked prefill, existing-context extend, prefix-cache hits, homogeneous multi-request batches, pure decode fallback, and mixed fallback.

For weighted standard attention, additionally sweep explicit load vectors such as `1,1,1,1`, `4,2.5,1.9,1.6`, and `4,3,2,1`. Confirm that per-rank token counts sum to every request length, compact rank offsets agree across ranks, final hidden-state ordering is restored, and the repaired KV cache matches baseline PCP within numerical tolerance.

Timeline validation should use Nsight Systems with NVTX enabled and confirm:

1. earlier ranks execute later-layer GEMM/attention while later PCP ranks are still processing earlier layers,
2. no replica AllGather is submitted from the transformer-layer critical path,
3. every rank reaches `pcp.replica_forward_boundary` before device-group repair begins,
4. repaired cache contents match baseline before the next model forward.

Performance benchmarking should keep model, prompt/suffix lengths, PCP size, eager mode, cache dtype, and hardware identical between baseline and runahead. Measure TTFT/prefill latency, communication overlap, GPU utilization, peak deferred-KV memory, per-rank completion time, forward-boundary wait, repair time, and sensitivity to the load-weight vector.

The fixed 1024-token runahead threshold is an initial guardrail and should be replaced or tuned from measured crossover data.

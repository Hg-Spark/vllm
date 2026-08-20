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
  |- persistent paged-cache repair metadata
  `- forward-boundary raw-page repair
          |
          v
vLLM KV backing storage
  `- raw uint8 block/page view
```

The former `pcp_runahead_ext.py` import-time class patch has been removed. Eligibility, validation, partitioning, slot mapping, and manager construction live directly in `pcp_manager.py`.

`pcp_standard.py` remains a compatibility shim for the MRV2 config/backend gate and the standard Attention cache-write hook. The communication policy lives in the existing attention PCP helper module alongside MLA.

## Layer Critical Path

For each participating attention layer, the runtime executes:

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
record lightweight repair metadata
        |
        v
attention
        |
        v
next transformer layer
```

No full-KV repair communication is submitted from `update_visible_and_defer_repair()` during layer execution.

The deferred metadata contains references to the persistent KV backing storage and slot mappings. Block-ID extraction (`torch.unique`) and scalar range checks are postponed until the forward boundary. Per-layer K/V activation tensors are free to follow their normal lifetime after the cache update.

## Persistent Paged-KV Repair Source

The repair path mirrors vLLM's storage-level block-copy model. For supported block-major PCP cache layouts, `kv_cache.stride(0) * element_size` is the physical distance between consecutive cache pages. The runtime exposes the backing allocation as:

```text
uint8 [num_physical_pages, bytes_per_page]
```

through `untyped_storage()`.

This handles normal, padded, and compatible packed KV backing allocations without reconstructing logical K and V tensors. Multiple layer views that share the same backing-storage data pointer register one deferred repair buffer. When vLLM packs several layers into one physical page stride, one raw-page transfer carries those packed layer bytes together.

Standard FlashAttention uses its logical block size from the cache view. MLA and the sparse DeepSeek indexer use their block-major cache slot width; compressed DeepSeek layouts expose the storage block size through that view.

## Last-Rank Cache Authority

After causal prefix propagation for PCP size 4, the current-step cache state for a completed layer is:

```text
rank0: K0
rank1: K0 K1
rank2: K0 K1 K2
rank3: K0 K1 K2 K3
```

The final PCP rank therefore owns a complete current-step page image for every completed participating layer. Boundary repair uses the final rank as the source of truth and broadcasts every touched raw KV page to the other PCP ranks.

Using a complete-page authority also handles a PCP partition boundary that falls inside one KV page: the final rank has written every current-step token in that page before repair begins, so receivers replace the whole page with a complete image.

## Forward-Boundary Repair

`RunaheadPCPManager.finish_forward()` calls `PCPRunaheadRuntime.flush()`.

The current repair sequence is:

1. wait for outstanding causal-prefix P2P sends,
2. rendezvous all PCP ranks with `GroupCoordinator.barrier()` on the CPU/Gloo group,
3. derive the unique physical page IDs touched by the current step,
4. on the last PCP rank, gather those raw pages into a bounded temporary payload,
5. broadcast the payload over the PCP device communicator,
6. on earlier ranks, scatter the received raw pages back into the persistent KV backing storage.

The temporary broadcast payload is chunked to approximately 64 MiB. A single physical page larger than this limit is transferred as one page.

The CPU/Gloo rendezvous remains required because repair currently uses the same PCP device/NCCL group as layer-level prefix P2P. Earlier PCP ranks can finish all transformer layers while later ranks are still submitting trailing-layer P2P. The CPU rendezvous prevents a repair collective from entering that device-group operation stream before every rank has left the layer-P2P phase.

```text
                 transformer forward
rank0  L0 -> L1 -> L2 -> ... -> LN ----\
rank1    L0 -> L1 -> ... -> LN ---------+--> CPU/Gloo rendezvous
rank2      L0 -> ... -> LN -------------+          |
rank3        ... -> LN -----------------/           v
                                         raw KV-page broadcast
```

This boundary synchronization does not prevent cross-layer runahead inside the transformer stack.

## Repair Memory Behavior

The previous reference implementation retained every layer's local current-step K/V activation tensors until `finish_forward()`:

```text
O(num_layers * local_prefill_tokens * local_kv_width)
```

The current implementation retains persistent-cache references and lightweight slot metadata. The persistent KV cache already exists as part of normal vLLM execution, while temporary repair payload memory is bounded by the repair chunk size.

Consequently, the runahead-specific activation-retention term above is removed.

Raw-page transfer also operates on the cache's stored dtype. For an FP8 KV cache, repair transports the FP8 cache bytes rather than retaining and communicating the original BF16 K/V activation tensors.

## Current Repair Communication Cost

The current reference repair broadcasts touched full pages from the final PCP rank to every rank. It preserves the replicated-cache contract required by later chunked prefill and decode steps and avoids activation retention, while repair traffic still remains at the model-forward boundary.

A later optimization will use reverse suffix propagation:

```text
forward prefix:
rank0 -> rank1 -> rank2 -> rank3

boundary repair:
rank3 -- suffix3 --> rank2
rank2 -- suffix2+3 --> rank1
rank1 -- suffix1+2+3 --> rank0
```

That reduces redundant retransmission because each earlier rank already owns its causal prefix.

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

Previously computed context remains in paged KV cache. Runahead communicates only K/V generated in the current scheduler step on the layer critical path. Boundary repair operates on the physical pages touched by those slots.

## Transport Ordering

P2P peers are specified in PCP-local rank space and mapped through `GroupCoordinator.ranks`; the runtime does not assume PCP rank equals global rank.

Known-shape tensors use tensor-only communication. Multiple tensors belonging to one layer are submitted through `torch.distributed.batch_isend_irecv`.

During transformer forward, the PCP device group carries prefix P2P only. Raw-page repair starts only after the CPU/Gloo forward-boundary rendezvous. This separation in time avoids inserting a full repair collective between consecutive layer-prefix operations on the same NCCL communicator.

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
pcp.replica_block_index
pcp.replica_commit
pcp.replica_buffer_prepare
pcp.replica_broadcast
pcp.replica_cache_update
pcp.transport.recv_enqueue
pcp.transport.send_enqueue
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

No `pcp.replica_broadcast` range should appear between transformer layers. All repair communication should occur after the forward-boundary rendezvous.

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

For weighted standard attention, additionally sweep explicit load vectors such as `1,1,1,1`, `4,2.5,1.9,1.6`, and `4,3,2,1`. Confirm that per-rank token counts sum to every request length, compact rank offsets agree across ranks, final hidden-state ordering is restored, and the repaired paged KV cache matches baseline PCP byte-for-byte for cache formats where exact byte equality is expected.

Timeline validation should use Nsight Systems with NVTX enabled and confirm:

1. earlier ranks execute later-layer GEMM/attention while later PCP ranks are still processing earlier layers,
2. no replica repair communication is submitted from the transformer-layer critical path,
3. every rank reaches `pcp.replica_forward_boundary` before raw-page broadcast begins,
4. repaired cache contents match baseline before the next model forward.

Performance benchmarking should keep model, prompt/suffix lengths, PCP size, eager mode, cache dtype, and hardware identical between baseline and runahead. Measure TTFT/prefill latency, communication overlap, GPU utilization, peak temporary repair memory, per-rank completion time, forward-boundary wait, repair time, and sensitivity to the load-weight vector.

The fixed 1024-token runahead threshold is an initial guardrail and should be replaced or tuned from measured crossover data.

## Planned Evolution

The next optimization stages are intentionally separated so each can be validated independently:

1. replace final-rank full-page broadcast with reverse-suffix raw-page P2P,
2. create a sibling PCP device communicator and a dedicated repair CUDA stream, following the existing PP side-stream pattern,
3. use CUDA events to launch repair after the source pages are ready and wait only before the next operation that can mutate/reuse those pages,
4. overlap terminal-prefill repair with sampling,
5. evaluate block-aligned PCP partitioning so per-layer background repair writes only cache pages that current-layer attention cannot read.

# PCP causal-prefix runahead

This document describes the experimental `prefill_context_parallel_mode=runahead`
path in the V2 GPU model runner.

## Goal

Baseline PCP materializes every rank's newly generated prefill KV before each
attention layer. For a causal prefill, rank `r` only needs the new KV owned by
ranks `0..r`. Runahead replaces the layer-critical full AllGather with a
left-to-right causal-prefix P2P chain.

For PCP=4:

```text
rank0: K0 ───────────────►
rank1:    K0,K1 ─────────►
rank2:          K0,K1,K2 ─►
rank3:                K0,K1,K2,K3
```

Rank 0 can enter the next transformer layer without waiting for ranks 1-3 to
finish the current layer.

## Current execution

Runahead uses compact contiguous rank-major token slabs. Equal and weighted
partitions share the same layout; no padding is inserted into a runahead slab.

For one layer:

```text
local K/V
   │
   ├─ receive causal prefix from rank-1
   │
   ├─ concatenate prefix + local
   │
   ├─ enqueue send to rank+1
   │
   ├─ write causal-visible KV to paged cache
   │
   └─ attention
```

The send tensor is retained only until its NCCL P2P work completes.

## Step-level replica repair

Normal vLLM decode expects the PCP ranks to enter the next step with replicated
persistent KV state. Runahead restores that invariant at the forward boundary.

Repair planning is step-level:

1. `PCPManager` computes the global slot mappings once.
2. For every KV cache group, valid slots are converted to kernel-cache block IDs
   with `slot // kernel_block_size`.
3. The IDs are unioned once for the step.
4. Attention layers only register their persistent backing storage; they do not
   retain K/V activation tensors or per-layer slot metadata.
5. `finish_forward()` drains outstanding prefix sends.
6. PCP ranks rendezvous on the CPU/Gloo group.
7. The last PCP rank broadcasts the touched raw paged-KV blocks.
8. Other ranks write the received pages into their persistent backing storage.

The last rank is the repair authority because the causal-prefix chain gives it
the complete current-step KV image before each layer's cache write.

A union of block IDs across cache groups is allowed. A backing only consumes IDs
inside its block range. Extra valid pages are safe to refresh because the cache
is replicated before the step.

## Raw paged-cache repair

Repair communicates cache bytes rather than activation-shaped K/V tensors.

```text
KV backing storage
    ↓
uint8 [num_blocks, page_bytes]
    ↓
select touched blocks
    ↓
broadcast
    ↓
index_copy_ into receiver backing
```

This removes the previous memory overhead proportional to:

```text
num_layers × local_prefill_tokens × KV_width
```

Temporary repair payloads are chunked to approximately 64 MiB.

## Variable-width partitions

Runahead always uses compact rank-major layout. For rows:

```text
(4000, 2500, 1900, 1600)
```

the offsets are:

```text
(0, 4000, 6500, 8400, 10000)
```

Prefix sizes therefore naturally differ by rank. Empty participating ranks are
rejected and the step falls back to baseline PCP.

`VLLM_PCP_RUNAHEAD_LOAD_WEIGHTS` optionally supplies positive per-rank load
weights for standard attention. Without weights, the split is equal apart from
integer remainder.

## Hidden-state restore

Sampling still consumes the global batch order. Runahead restores hidden rows
with a variable-width all-gather and then applies `hidden_restore_idx`.

The preferred path uses the existing PyNccl `all_gatherv` implementation. A
process-group all-gather fallback remains for environments where PyNccl is
disabled.

## Standard attention

Standard MHA/GQA/MQA currently uses FlashAttention and TP=1. The cache writer
preserves the KV-head dimension; the runahead transport treats the dimensions
after the token row as opaque.

The current vLLM 0.27-derived V2 capability gate still requires a small
compatibility bridge for standard-attention PCP. The bridge installs one
class-level FlashAttention cache-update hook instead of patching every
attention instance.

## MLA and sparse indexer

MLA and sparse-indexer cache updates use the same causal-prefix runtime.
Their original baseline PCP AllGather remains the fallback when a step is not
eligible for runahead.

MLA currently keeps the narrower eligibility used by the original prototype:
a single fresh full-prefill request.

## Supported runahead configuration

Current runtime validation requires:

- PCP > 1
- TP = 1
- PP = 1
- DP = 1
- DCP = 1
- no expert parallelism / MoE collectives
- no DBO
- no async scheduling
- eager execution (`cudagraph_mode=NONE`)
- standard attention: FlashAttention
- homogeneous prefill/extend batches for standard attention

Mixed decode + prefill falls back to baseline PCP.

## Ordering invariant

Prefix P2P and boundary repair currently use the PCP device communicator. A rank
can finish its transformer stack earlier than another rank, so it must not
enqueue repair collectives while a later rank is still submitting layer-level
P2P on that communicator.

`flush()` therefore performs:

```text
wait local outstanding sends
        ↓
CPU/Gloo PCP barrier
        ↓
device-group repair broadcast
```

This barrier remains until repair moves to a dedicated sibling communicator.

## Profiling

NVTX ranges use the `pcp.*` prefix. Important ranges include:

```text
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_concat
pcp.visible_cache_update
pcp.replica_forward_boundary
pcp.replica_buffer_prepare
pcp.replica_broadcast
pcp.replica_cache_update
pcp.restore_hidden_variable_allgather
pcp.flush
```

Use Nsight Systems to verify rank skew, prefix P2P latency, boundary repair,
hidden-state restore, and host synchronization on the layer critical path.

## Next optimization boundary

The next transport change should be isolated inside `PCPRunaheadRuntime`:

```text
current:
last-rank touched-page broadcast

next:
reverse-suffix page P2P

later:
dedicated sibling communicator + repair CUDA stream
```

Manager partitioning, attention cache policy, and step-level repair planning
should not need to change when the repair transport changes.

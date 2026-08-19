# PCP Prefill Runahead

This document describes the experimental prefill context parallel (PCP) runahead path, using the vLLM 0.27 PCP design as the baseline.

## Goal

The goal is to remove the per-layer full-KV AllGather from the attention synchronization critical path during prefill.

Baseline PCP requires every rank to finish a full KV AllGather before attention can run:

```text
Layer L
  local KV on every PCP rank
        |
        v
  full PCP AllGather
        |
        v
  replicated KV cache update
        |
        v
     attention
        |
        v
     Layer L+1
```

For causal attention, rank `r` only needs the prefix from PCP ranks `0..r`. Runahead uses this dependency directly:

```text
Layer L local KV
        |
        +---- async full-KV AllGather ----------------> replicated KV cache
        |
        +---- causal prefix P2P ----> attention -----> Layer L+1
```

This allows earlier PCP ranks to advance into later layers while later ranks are still processing the current layer.

## Current Status

The mechanism target is implemented.

- Current-layer attention waits only for the causal KV prefix required by the local PCP rank.
- Full replicated KV reconstruction runs through asynchronous AllGather outside the current-layer attention critical path.
- Replicated cache completion is bounded by a pending-replica queue and is flushed before the forward pass completes.
- Existing MLA and sparse-indexer cache update primitives remain the source of truth for physical KV-cache writes.
- PCP topology comes from `GroupCoordinator`; the transport does not assume PCP ranks are identical to world ranks.
- Multiple tensors belonging to one layer use `torch.distributed.batch_isend_irecv` to avoid unnecessary per-tensor P2P launch overhead.

The implementation should still be treated as experimental until real multi-GPU correctness and performance validation pass.

## Architecture

### `RunaheadPCPManager`

`RunaheadPCPManager` extends the existing `PCPManager` and reuses the existing PCP batch-layout machinery for:

- `InputBatch` partitioning,
- padding,
- block tables,
- slot mappings,
- request-state handling,
- input buffers,
- restore indices.

Runahead only changes the fresh full-prefill partition to a contiguous causal layout.

The current MVP enables runahead only for a single fresh full-prefill request. Unsupported combinations continue to fall back or are rejected by configuration validation.

### `PCPRunaheadRuntime`

`PCPRunaheadRuntime` owns runahead-specific scheduling state:

```text
PCPRunaheadRuntime
  |- causal-prefix exchange
  |- pending P2P sends
  |- pending replicated-KV AllGathers
  |- cache-completion callbacks
  `- replication backpressure
```

It intentionally does not duplicate MLA or sparse-indexer cache insertion logic.

### `pcp_transport.py`

Low-level tensor transport remains PCP-private for the MVP:

```text
PCPRunaheadRuntime
        |
        v
pcp_transport.py
  |- PCP-local -> global rank mapping
  |- batched tensor P2P
  `- asynchronous all_gather_into_tensor
        |
        v
GroupCoordinator
  |- ranks
  |- rank_in_group
  `- device_group
```

The transport is kept private because these helpers currently assume GPU tensor communication and PyTorch distributed handles. They are not yet a complete generic vLLM distributed abstraction.

## Layer Dependency

For PCP size 4, the causal dependency is:

```text
rank0: KV0             -> attention0 -> next layer
rank1: KV0 + KV1       -> attention1 -> next layer
rank2: KV0 + KV1 + KV2 -> attention2 -> next layer
rank3: KV0..KV3        -> attention3 -> next layer
```

The prefix is propagated left-to-right:

```text
rank0 ---- KV0 ----> rank1 ---- KV0+KV1 ----> rank2 ---- KV0+KV1+KV2 ----> rank3
```

Rank 0 has no inter-rank causal dependency and can therefore become the leading runahead rank.

## Replicated KV Cache

Runahead changes the timing of replication, while preserving the replicated-cache requirement for subsequent decode.

For each layer:

1. Launch asynchronous full-KV AllGather from the local padded KV chunk.
2. Exchange only the causal prefix needed by the current rank.
3. Apply the causal prefix to the existing cache-update primitive for current-layer attention.
4. When the asynchronous AllGather completes, apply the gathered KV image through the same existing cache-update primitive.
5. Apply backpressure if too many layer replications are outstanding.
6. Flush all outstanding replication before forward completion.

This preserves the expected full replicated cache before decode begins.

## Communication Decisions

### PCP group ownership

P2P peers are expressed in PCP-local rank space and mapped through `GroupCoordinator.ranks`.

The transport no longer requires:

```text
PCP rank == world rank
```

This removes a transport-level topology restriction and keeps PCP membership owned by vLLM distributed state.

### Tensor-only P2P

The existing tensor-dict P2P APIs exchange CPU metadata before sending tensors. Runahead already knows tensor shapes, dtypes, and peers on every layer, so that protocol would add unnecessary synchronization to the critical path.

The runahead path therefore uses tensor-only P2P.

### Batched P2P

All tensors belonging to the same prefix exchange are submitted through one `batch_isend_irecv` call. This keeps the original grouped P2P launch behavior for MLA KV components.

### Asynchronous AllGather

Full-KV replication must remain asynchronous. Replacing it with the synchronous `GroupCoordinator.all_gather()` path would restore the layer-level collective barrier that runahead is intended to remove.

## Current Scope

The MVP intentionally keeps a narrow execution envelope. Current validation rejects or does not enable runahead for combinations including:

- tensor parallel execution,
- data parallel execution,
- pipeline parallel execution,
- decode context parallel execution,
- expert parallel / MoE,
- dual-batch overlap,
- asynchronous scheduling,
- CUDA graph execution,
- unsupported request mixtures.

These restrictions should be relaxed only after the base PCP runahead path is validated.

## Validation Gates

### Gate 1: PCP=2 correctness

Run the same fresh full-prefill input with baseline PCP and runahead PCP.

Validate:

- final model output within the expected numerical tolerance,
- per-layer attention output where practical,
- final replicated MLA KV cache on both ranks,
- sparse-indexer cache when the model uses it,
- padding and slot-mapping behavior,
- correctness under allocator pressure after asynchronous sends.

Expected final cache state:

```text
rank0 replicated cache == rank1 replicated cache == baseline PCP cache
```

### Gate 2: PCP=4 correctness

Repeat the comparison at PCP size 4 to exercise the complete causal chain:

```text
0 -> 1 -> 2 -> 3
```

This validates increasing prefix sizes and multiple outstanding replication layers.

### Gate 3: Timeline validation

Use Nsight Systems or equivalent tracing to confirm real layer skew.

The key expected property is that an earlier rank can execute a later-layer GEMM while later PCP ranks are still communicating or computing an earlier layer.

Example expected overlap:

```text
rank0: L0 compute -> L1 compute -> L2 compute
rank1:     recv L0 -> L0 compute -> recv L1 -> L1 compute
rank2:                    recv ...
```

If all ranks continue to enter each layer together, the synchronization target has not been achieved even if correctness passes.

### Gate 4: Performance benchmark

Compare baseline PCP and runahead using identical model, prompt lengths, PCP size, eager mode, and hardware.

Measure at least:

- prefill latency,
- prefill throughput,
- communication time on the attention critical path,
- GPU utilization,
- peak temporary memory from pending replication,
- sensitivity to `max_pending_replica_layers`.

## Completion Criteria

The current code satisfies the mechanism and code-boundary goals.

The feature should be considered complete only after:

1. PCP=2 numerical and replicated-cache correctness pass,
2. PCP=4 causal-chain correctness passes,
3. traces show genuine cross-layer runahead,
4. benchmark results show a meaningful prefill improvement without unacceptable memory growth or regression.

Until these gates pass, further expansion into TP/DP, asynchronous scheduling, DBO, CUDA graphs, or forward-context lifecycle refactoring is lower priority than validating the existing path.

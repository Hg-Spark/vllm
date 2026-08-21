# PCP causal-prefix runahead

This experimental path extends vLLM V2 PCP for standard FlashAttention prefill.
The design keeps stock PCP batch/block-table machinery and adds three narrowly
scoped pieces:

1. a contiguous logical-segment compiler;
2. a logical PCP communicator for one-segment-per-rank bindings;
3. causal-prefix transports, including one-sided paged-KV pull.

## Configuration

A non-empty `pcp_runahead` object enables the path. Runahead has a fixed compact
layout and requires a fresh complete prefill, so those are derived invariants
rather than independent experiment axes.

### One segment per rank with arbitrary binding

```json
{
  "pcp_runahead": {
    "transport": "direct_p2p",
    "partition": {
      "segments": [
        {"weight": 4.0, "pcp_rank": 2},
        {"weight": 2.5, "pcp_rank": 0},
        {"weight": 1.9, "pcp_rank": 3},
        {"weight": 1.6, "pcp_rank": 1}
      ]
    },
    "eligibility": {"min_tokens": 1024},
    "runtime": {"max_inflight_sends": 4}
  }
}
```

The configured physical ownership is:

```text
logical segment:  0     1     2     3
physical rank:    2     0     3     1
```

For a permutation, runahead creates a PCP communicator whose member order is
`[rank2, rank0, rank3, rank1]`. Inside the runahead hot path:

```text
logical segment index == communicator rank
```

Tensor collectives and P2P therefore need no per-layer `segment_to_rank`
translation or rank-major-to-segment-major reorder. The original PCP group is
left unchanged and remains available for fallback execution.

`partition.weights` is still accepted as a compact compatibility form and
implies identity binding.

### Repeated ownership

Repeated ownership is supported only by `page_pull`:

```json
{
  "pcp_runahead": {
    "transport": "page_pull",
    "partition": {
      "segments": [
        {"weight": 1, "pcp_rank": 1},
        {"weight": 1, "pcp_rank": 0},
        {"weight": 1, "pcp_rank": 1}
      ]
    },
    "runtime": {
      "nixl_backends": ["UCX"],
      "max_inflight_reads": 4
    }
  }
}
```

A process group cannot contain one process at two logical ranks, so repeated
ownership remains local to `SegmentLayout` and `PCPPagePlan`. Every physical PCP
rank must own at least one segment.

## Segment compilation

The manager partitions every scheduled request once and compiles a
`SegmentLayout` containing:

```text
segments_by_rank
rows_per_rank
rows_per_segment
logical_segment_slices
page-boundary validity
```

This replaces repeated per-rank/per-feature partition passes. The cost is
`O(batch * logical_segments)` for the layout compiler.

PCPManager now has a generic compact rank-major layout mode. Stock PCP remains
padded. Runahead uses cumulative actual rank widths directly:

```text
rank0 rows | rank1 rows | ... | rankN rows
```

No max-width slab is materialized and sliced afterward. Restore and full-KV
collectives keep the fixed-size fast path:

```text
equal rank widths     -> all_gather
variable rank widths  -> all_gatherv
```

## Transport modes

| Transport | Data unit | Route | Repeated ownership |
| --- | --- | --- | --- |
| `full_kv_collective` | contiguous K/V rows | logical PCP collective | no |
| `prefix_p2p` | accumulated prefix rows | logical rank chain | no |
| `direct_p2p` | one logical segment | owner to later logical ranks | no |
| `page_pull` | physical KV pages | consumer NIXL READ | yes |

All runahead transports require:

```text
num_computed_tokens == 0
num_scheduled_tokens == prefill_len
```

Continued prefill and decode remain unsupported after a causal-prefix transport
because persistent KV is intentionally sharded/asymmetric.

## Prefix P2P

For four logical ranks:

```text
rank0 -> rank1 -> rank2 -> rank3
  S0      S0+S1   S0+S1+S2
```

The communicator already encodes configured physical binding, so neighbor
selection is simply `rank-1` / `rank+1`.

## Direct P2P

Each logical rank sends only its local segment to every later logical rank:

```text
S0 -> S1, S2, S3
S1 ->     S2, S3
S2 ->         S3
```

This removes relay dependencies. It does not reduce the fresh-prefill
information volume by itself.

## Page pull

`page_pull` moves completed physical KV pages instead of temporary K/V rows.
Internal segment boundaries are aligned to the common kernel page granularity
from `BlockTables.kernel_block_sizes`. A step whose required internal cuts are
not page aligned falls back.

### Stable registration phase

The first layer-cache discovery registers all write-owning attention KV caches.
NIXL agent metadata and layer memory geometry are exchanged once:

```text
layer_id -> base address, block bytes, block count, device
```

Remote transfer descriptor lists are prepared from this stable metadata and
reused. Cache addresses must remain stable while the transport is alive.

### Per-layer protocol

```text
projection K/V
    |
    v
prepare local slot mapping
    |
    v
FlashAttention native reshape_and_cache_flash   (single local KV write)
    |
    v
record CUDA event
    |
    +-----------------------------> model/progress overlap
                                      |
                              progress thread event.query()
                                      |
                                      v
                         NIXL READY(epoch, layer, source)
                                      |
                                      v
                         consumer NIXL one-sided READ
                                      |
                                      v
                         destination paged KV cache
                                      |
                              required reads DONE
                                      |
                                      v
                              current attention
```

The old early manual cache write has been removed. READY is published only after
the backend's native cache-write event completes.

The model thread does not call `event.synchronize()` for READY. A background
progress thread queries the event and sends the notification when the write is
complete.

### NIXL control and data plane

READY uses NIXL notifications rather than a parallel Gloo `isend/irecv` control
path. The per-layer message contains only changing state:

```text
epoch, layer_id, source_rank
```

Address and page geometry stay in the registration phase.

The data path uses prepared NIXL READ descriptors and polls transfer handles for
`DONE`.

### Block-plan invariant

All PCP workers execute the same scheduler block allocation for the supported
configuration. Each rank derives physical block IDs from the same global slot
mapping. `PCPPagePlan` therefore stores one deterministic page list per logical
segment:

```text
segment -> owner rank -> block IDs
```

Source and destination descriptor IDs are identical. Before installing a plan,
workers exchange only a fixed-size fingerprint of the derived mapping. A
mismatch fails fast instead of falling back to the previous Python-object block
metadata all-gather.

### Repeated-owner short circuit

For:

```text
segment_to_rank = [1, 0, 1]
```

rank 1 owns `S0` and `S2`. While executing its latest segment it already has
`S0`; only `S1` is pulled from rank 0.

## Performance and scaling notes

The refactor removes hot-path work that grows with PCP size:

- repeated segment partition passes;
- rank/segment reorder for permutation bindings;
- padded compact staging;
- duplicate local KV-cache write;
- per-layer Gloo READY tensors and receives;
- per-step Python-object block-map exchange.

The long-term page-pull scaling limit is still producer fanout. With `P` logical
segments, early segments serve more downstream consumers:

```text
S0: P-1 consumers
S1: P-2 consumers
...
```

At large PCP sizes this can saturate one source GPU/NIC even when aggregate
fabric bandwidth remains available. Hierarchical relay, replication, or
resident-page-aware source selection can be added later without changing the
segment compiler.

Physical binding should also be topology aware for multi-node runs so adjacent
causal ranks stay on high-bandwidth links where possible.

## Current validation constraints

- standard MHA/GQA/MQA with FlashAttention;
- PCP > 1;
- TP = 1, PP = 1, DP = 1, DCP = 1;
- no EP/MoE, DBO, speculative decoding, or async scheduling;
- eager execution (`cudagraph_mode=NONE`);
- fresh complete prefill only for runahead;
- `page_pull`: NIXL available, one standard-attention KV-cache group,
  contiguous block-major FP16/BF16 cache, page-aligned internal cuts, stable KV
  cache addresses.

## Persistent KV semantics

`full_kv_collective` reconstructs a replicated current-step cache. Causal
transports leave rank-specific prefix state:

- `prefix_p2p`: prefix through this logical rank;
- `direct_p2p`: same semantic state without relay;
- `page_pull`: pages required through the rank's latest owned logical segment.

Sampling the current forward is valid. A later model step is rejected until a
persistent sharded-KV continuation/decode path is implemented.

## Profiling

Set:

```bash
VLLM_PCP_NVTX=1
```

Profiling helpers are direct call-site ranges/marks only. They do not monkey
patch FlashAttention or the page-pull progress engine and do not introduce CUDA
synchronization.

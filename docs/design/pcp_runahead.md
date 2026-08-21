# PCP causal-prefix runahead

This experimental path extends vLLM V2 PCP for standard FlashAttention prefill.
It reuses stock PCP batch/block-table machinery and keeps four transport choices:
`full_kv_collective`, `prefix_p2p`, `direct_p2p`, and `page_pull`.

The refactor has three core rules:

1. compile logical segments once per batch;
2. encode one-segment-per-rank permutations in the primary PCP group order;
3. keep repeated ownership local to the page-pull plan.

## Configuration

A non-empty `pcp_runahead` object enables the path. Runahead always uses compact
rank-major execution and requires a fresh complete prefill.

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

For a permutation, startup constructs the **primary PCP group** with member order
`[rank2, rank0, rank3, rank1]`. No second runahead communicator exists. After
startup:

```text
logical segment index == get_pcp_group().rank_in_group
```

Tensor collectives and both P2P transports therefore use ordinary PCP ranks with
no per-layer rank translation or rank-major/segment-major reorder.

`partition.weights` remains accepted and implies identity binding.

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

A process cannot occupy two ranks in one process group. Repeated ownership thus
leaves the primary PCP group in its normal order and remains in `SegmentLayout`
and `PCPPagePlan`. Every physical PCP rank must own at least one segment.

## Segment compilation and compact layout

The manager partitions every scheduled request once into a `SegmentLayout`:

```text
segments_by_rank
rows_per_rank
logical_segments(req, start_pos, end_pos, owner_rank)
```

Internal page-alignment validity is checked during this same pass. The compiler
cost is `O(batch * logical_segments)`.

Runahead uses exact rank widths:

```text
rank0 rows | rank1 rows | ... | rankN rows
```

No max-width padded slab is created. Hidden-state restore and full-KV transport
retain the equal-width fast path:

```text
equal rank widths     -> all_gather
variable rank widths  -> all_gatherv
```

## Transport modes

| Transport | Data unit | Route | Repeated ownership |
| --- | --- | --- | --- |
| `full_kv_collective` | contiguous K/V rows | primary PCP collective | no |
| `prefix_p2p` | accumulated prefix rows | logical rank chain | no |
| `direct_p2p` | one logical segment | rank to all later ranks | no |
| `page_pull` | physical KV pages | consumer NIXL READ | yes |

All runahead transports require:

```text
num_computed_tokens == 0
num_scheduled_tokens == prefill_len
```

Continued prefill and decode remain unsupported after a causal-prefix transport
because persistent KV state is rank-specific.

## Prefix P2P

For four logical PCP ranks:

```text
rank0 -> rank1 -> rank2 -> rank3
  S0      S0+S1   S0+S1+S2
```

Configured physical binding is already represented by the primary PCP group, so
neighbors are simply `rank-1` and `rank+1`.

## Direct P2P

Each logical rank sends only its local segment directly to later ranks:

```text
S0 -> S1, S2, S3
S1 ->     S2, S3
S2 ->         S3
```

This removes relay dependencies while preserving the fresh-prefill information
volume.

## Page pull

`page_pull` moves completed physical KV pages instead of temporary K/V rows.
Internal segment boundaries are aligned to the common kernel-page granularity
from `BlockTables.kernel_block_sizes`. An unaligned step falls back.

### CPU block plan

`BlockTables` maintains a lightweight CPU mirror of the kernel block IDs written
to its staged GPU table. Page-plan compilation therefore stays on CPU:

```text
logical segment
  -> request state index
  -> token-position block interval
  -> BlockTables.get_block_ids_cpu(...)
  -> physical block IDs
```

The supported PCP execution receives the same scheduler block allocation on all
model workers, so source and destination descriptor IDs are identical. There is
no per-step GPU slot-map readback, Python-object block-map all-gather, or block
fingerprint collective.

### Stable registration phase

The first layer-cache discovery registers every write-owning attention KV cache.
NIXL agent metadata and stable layer geometry are exchanged once:

```text
layer_id -> base address, block bytes, block count, device
```

Remote descriptor lists are prepared once and reused. Cache addresses must stay
stable while the transport is alive.

NHD and HND logical views are accepted when `stride(0)` proves that each physical
block still occupies one dense whole-page byte span.

### Per-layer ordering

The normal FlashAttention KV update remains the only local cache write:

```text
projection K/V
    |
    v
prepare_standard_pcp_kv_cache_inputs
    |  - select local slots
    |  - register pending page-pull layer
    v
native reshape_and_cache_flash
    |
    v
unified KV-update dummy dependency
    |
    v
attention-entry PCP hook
    |  - page_pull_after_cache_write
    |  - record CUDA event
    v
progress thread event.query()
    |
    v
NIXL READY(epoch, layer, source)
    |
    v
consumer NIXL READ -> destination KV pages
    |
    v
all required reads DONE
    |
    v
FlashAttention
```

The attention custom op already depends on the unified KV-update custom op via
`kv_cache_dummy_dep`; the page-pull hook uses that existing ordering instead of
patching FlashAttention. READY becomes externally visible only after the CUDA
event recorded after the native write reports completion. The model thread does
not call `event.synchronize()`.

READY uses NIXL notifications. Per-layer messages contain only changing state:

```text
epoch, layer_id, source_rank
```

Memory geometry remains in the registration phase.

### Repeated-owner short circuit

For:

```text
segment_to_rank = [1, 0, 1]
```

rank 1 owns `S0` and `S2`. While executing `S2`, `S0` is already local and only
`S1` is pulled from rank 0.

## Performance and scaling notes

The refactor removes several PCP-size-sensitive costs:

- repeated partition passes;
- runtime rank/segment reorder for permutations;
- padded compact staging;
- duplicate local KV-cache writes;
- per-layer Gloo READY tensors/receives;
- per-step GPU block-map readback/object exchange.

Page-pull still has a producer-fanout scaling limit. With `P` logical segments:

```text
S0: P-1 consumers
S1: P-2 consumers
...
```

At larger PCP sizes this can saturate an early source GPU or NIC. Hierarchical
relay, replication, or resident-page-aware source selection are future options.
Physical rank binding should also consider multi-node topology.

## Current validation constraints

- standard MHA/GQA/MQA with FlashAttention;
- PCP > 1;
- TP = 1, PP = 1, DP = 1, DCP = 1;
- no EP/MoE, DBO, speculative decoding, async scheduling, or request-level KV
  transfer connector;
- eager execution (`cudagraph_mode=NONE`);
- fresh complete prefill only;
- `page_pull`: NIXL available, one standard-attention KV-cache group,
  block-major dense FP16/BF16 physical pages, page-aligned internal cuts, stable
  KV-cache addresses.

## Persistent KV semantics

`full_kv_collective` reconstructs a replicated current-step cache. Causal
transports leave rank-specific prefix state:

- `prefix_p2p`: prefix through this logical rank;
- `direct_p2p`: the same state without relay;
- `page_pull`: pages required through the rank's latest owned logical segment.

Sampling the current forward is valid. A later model step is rejected until a
persistent sharded-KV continuation/decode path is implemented.

## Profiling

Set:

```bash
VLLM_PCP_NVTX=1
```

Profiling consists only of direct call-site ranges/marks. It does not patch
FlashAttention or the page-pull progress engine and does not add CUDA
synchronization.

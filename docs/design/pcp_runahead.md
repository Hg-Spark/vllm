# PCP causal-prefix runahead

This experimental path is based on vLLM 0.27.1 V2 PCP. It reuses the existing
PCP batch construction, `RankSegment` mapping compiler, block tables, slot
mapping, sampling restore lifecycle, and CUDA `GroupCoordinator.all_gatherv()`
implementation.

The prefix-P2P path intentionally leaves persistent KV cache non-replicated
after prefill. This branch therefore targets single-step full-prefill/runahead
measurement. A later decode/continue step requires a sharded-KV consumer and is
rejected explicitly.

## Configuration

A non-empty `pcp_runahead` object enables the experimental standard-attention
PCP path. Omit the key or set it to `false` to disable it. Boolean `true` is not
a valid configuration; the nested object form is required.

```bash
--prefill-context-parallel-size 4 \
--additional-config '{
  "pcp_runahead": {
    "transport": "prefix_p2p",
    "partition": {
      "policy": "weighted_contiguous",
      "weights": [4, 2.5, 1.9, 1.6],
      "page_align": true
    },
    "layout": "compact",
    "eligibility": {
      "require_full_prefill": true,
      "min_tokens": 1024
    },
    "runtime": {
      "max_inflight_sends": 4
    }
  }
}' \
--enforce-eager
```

### Experiment axes

`transport`:

- `full_kv_collective`: every layer reconstructs full K/V with a collective.
- `prefix_p2p`: each rank receives only the causal prefix from rank-1 and
  forwards its enlarged visible prefix to rank+1.

`partition.policy`:

- `stock`: reuse PCP DualChunkSwap.
- `equal_contiguous`: one contiguous interval per rank with equal token weights.
- `weighted_contiguous`: one contiguous interval per rank using manual weights.

`layout`:

- `padded`: retain the original PCP equal-width padded slabs.
- `compact`: expose only actual local rows to the Transformer.

`prefix_p2p` currently requires a contiguous partition, compact layout, and
`require_full_prefill=true`. `stock` is intentionally restricted to
`full_kv_collective + padded`.

## Recommended controls

Use the following matrix to separate the effects of partition/layout,
communication, and weighting:

| Case | Partition | Layout | Transport | Purpose |
| --- | --- | --- | --- | --- |
| A | stock | padded | full_kv_collective | stock-style baseline |
| B | equal_contiguous | compact | full_kv_collective | layout/partition control |
| C | equal_contiguous | compact | prefix_p2p | isolate runahead transport |
| D | weighted_contiguous | compact | prefix_p2p | add load balancing |

The key causal comparison is B versus C: token ownership and execution layout
are identical; only the per-layer full-KV collective is replaced by causal
prefix P2P.

## Full-prefill eligibility

The default experiment gate requires every request in the selected batch to be
a fresh, complete prefill:

```text
num_computed_tokens == 0
num_scheduled_tokens == prefill_len
```

This prevents a chunked prefill from entering prefix-P2P on its first scheduler
chunk and then reaching a second model step with sharded persistent KV. Long
prompt benchmarks must therefore configure enough scheduler token capacity for
the target prompt if they intend to measure runahead.

## Runahead transport

For PCP=4:

```text
rank0: K0 ───────────────►
rank1:    K0,K1 ─────────►
rank2:          K0,K1,K2 ─►
rank3:                K0,K1,K2,K3
```

Each layer:

```text
local K/V
   │
   ├─ receive causal prefix from rank-1
   ├─ append local K/V into one visible buffer
   ├─ enqueue nonblocking send to rank+1
   ├─ write causal-visible KV to paged cache
   └─ attention
```

Receive completion is a causal dependency and remains on the critical path.
Send completion is deferred. Outstanding sends are bounded by
`runtime.max_inflight_sends`; exceeding that bound waits the oldest send and
creates controlled backpressure. `flush()` waits any remaining prefix sends at
the forward boundary.

No KV repair, broadcast, or PCP barrier is performed.

## Manual weighted partition

Weights determine contiguous per-rank token ownership. For:

```text
weights = [4, 2.5, 1.9, 1.6]
tokens  = 10000
```

the ideal split is approximately:

```text
4000 / 2500 / 1900 / 1600
```

Internal boundaries are aligned to the common KV kernel-page granularity
derived from `BlockTables.kernel_block_sizes`. Alignment uses absolute sequence
positions. When a request is too small for page-aligned cuts, the implementation
falls back to exact integer weighted allocation.

## Mapping reuse and compact execution

The custom contiguous policies override only PCP token ownership through
`_get_rank_segments()`. The original `PCPManager._build_batch_layout()` still
compiles those segments into `padded_gather_idx`, `hidden_restore_idx`, and
`gathered_kv_write_mask`.

For compact execution, the normal padded PCP batch is built first, then only the
actual local rows are exposed to the model. Slot mappings are compacted with the
existing PCP write mask. The padded hidden restore index is remapped into compact
rank-major coordinates rather than rebuilt independently.

Variable-width hidden restore continues to use:

```python
get_pcp_group().all_gatherv(
    hidden_states,
    dim=0,
    sizes=rows_per_rank,
)
```

## Full-KV collective control

For compact unequal widths, `full_kv_collective` uses vLLM's existing batched
`GroupCoordinator.all_gatherv()` to reconstruct full K/V. For padded equal-width
PCP it keeps the baseline `all_gather()` path.

This provides a same-partition, same-layout control for measuring the effect of
removing the per-layer full-KV synchronization barrier.

## Persistent KV semantics

After one `prefix_p2p` prefill:

```text
rank0: prefix owned/visible by rank0
rank1: prefix visible through rank1
rank2: prefix visible through rank2
rank3: full current-step prefix
```

The cache is intentionally left in that state. Sampling the current forward
output is valid; a subsequent model step is rejected until a sharded-KV
decode/continue path is implemented.

`full_kv_collective` writes a replicated full current-step cache and does not set
the sharded-history guard.

## Current validation constraints

- standard MHA/GQA/MQA with FlashAttention
- PCP > 1
- TP = 1
- PP = 1
- DP = 1
- DCP = 1
- no EP/MoE collectives
- no DBO
- no speculative decoding
- no async scheduling
- eager execution (`cudagraph_mode=NONE`)
- prefix-P2P: single-step full prefill only

MLA continues to use stock PCP when this experimental config is disabled; the
runahead config itself is currently standard-attention-only.

## Profiling

Important NVTX ranges:

```text
pcp.baseline_kv_allgather
pcp.full_kv_allgatherv
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_visible_alloc
pcp.prefix_local_append
pcp.compact_slot_mapping
pcp.restore_hidden_variable_allgather
pcp.send_wait
pcp.flush
```

Nsight should show early PCP ranks entering later Transformer layers while later
ranks are still completing earlier layers. There should be no forward-boundary
KV repair or broadcast in the prefix-P2P case.

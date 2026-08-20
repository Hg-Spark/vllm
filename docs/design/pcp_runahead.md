# PCP causal-prefix runahead

This experimental path is based on vLLM 0.27.1 V2 PCP. It reuses the existing
PCP batch construction, block tables, slot mapping, sampling restore lifecycle,
and CUDA `GroupCoordinator.all_gatherv()` implementation.

The runahead path intentionally leaves persistent KV cache non-replicated after
prefill. This branch therefore targets prefill/runahead measurement. A later
decode/continue step requires a sharded-KV consumer and is rejected explicitly.

## Enable

```bash
--prefill-context-parallel-size 4 \
--additional-config '{"pcp_runahead":true,"pcp_runahead_weights":[4,2.5,1.9,1.6]}' \
--enforce-eager
```

`pcp_runahead_weights` is optional. Omitting it selects equal weights. Values
must be positive and the list length must equal PCP world size.

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
   ├─ append local K/V directly into one visible buffer
   ├─ enqueue nonblocking send to rank+1
   ├─ write causal-visible KV to paged cache
   └─ attention
```

Receive completion is a causal dependency and remains on the critical path.
Send completion is deferred; `flush()` only waits outstanding prefix sends.

No forward-boundary KV repair, broadcast, or PCP barrier is performed.

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
positions, so continued-prefill cuts honor existing-context page boundaries.
When a request is too small for page-aligned cuts, the implementation falls
back to exact integer weighted allocation.

## Compact variable-width execution

vLLM's original PCP batch builder is still used first. It produces its normal
padded rank-major buffers and restore metadata. Runahead then exposes only the
actual rank-local rows to the model and compacts the gathered slot mapping with
the original PCP write mask.

This keeps original PCP request ordering, block-table handling, and metadata
generation while avoiding Transformer compute on padding rows.

## Hidden-state restore

Variable rank widths use the existing vLLM collective abstraction:

```python
get_pcp_group().all_gatherv(
    hidden_states,
    dim=0,
    sizes=rows_per_rank,
)
```

The original padded `hidden_restore_idx` is remapped into compact rank-major
space, then reused to restore global token order.

## Persistent KV semantics

After one runahead prefill:

```text
rank0: prefix owned/visible by rank0
rank1: prefix visible through rank1
rank2: prefix visible through rank2
rank3: full current-step prefix
```

The cache is intentionally left in that state. The branch does not attempt to
replicate pages before sampling. Sampling the current forward output is valid;
a subsequent model step is rejected until a sharded-KV decode/continue path is
implemented.

## Standard attention

Standard MHA/GQA/MQA uses FlashAttention with TP=1. FlashAttention declares PCP
capability through the existing `AttentionBackend.supports_pcp()` mechanism.
Its cache-update path calls a small PCP input adapter. When runahead is inactive,
that adapter uses the baseline PCP K/V AllGather.

## Current validation constraints

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
- FlashAttention for standard MHA/GQA/MQA

## Profiling

Important NVTX ranges:

```text
pcp.baseline_prefill_allgather
pcp.baseline_kv_allgather
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
ranks are still completing earlier layers. There should be no
`pcp.replica_*` ranges or forward-boundary KV broadcast.

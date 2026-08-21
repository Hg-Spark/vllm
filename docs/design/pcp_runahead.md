# PCP causal-prefix runahead

This experimental path is based on vLLM 0.27.1 V2 PCP. It reuses the existing
PCP batch construction, `RankSegment` mapping compiler, block tables, slot
mapping, sampling restore lifecycle, and `GroupCoordinator` collectives, while
separating two coordinate systems that stock PCP normally treats as identical:

1. **logical segment order** owns causal order and load weights;
2. **physical PCP rank** owns the worker, process-group membership, and GPU.

The bridge is `segment_to_rank`. Causal dependency never implies a fixed network
route.

## Configuration

A non-empty `pcp_runahead` object enables the experimental standard-attention
PCP path. Omit the key or set it to `false` to disable it. Boolean `true` is not
a valid configuration.

### One logical segment per physical rank

```bash
--prefill-context-parallel-size 4 \
--additional-config '{
  "pcp_runahead": {
    "transport": "direct_p2p",
    "partition": {
      "policy": "weighted_contiguous",
      "segments": [
        {"weight": 4.0, "pcp_rank": 2},
        {"weight": 2.5, "pcp_rank": 0},
        {"weight": 1.9, "pcp_rank": 3},
        {"weight": 1.6, "pcp_rank": 1}
      ],
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

The example above produces logical causal order:

```text
segment 0 -> segment 1 -> segment 2 -> segment 3
rank 2       rank 0       rank 3       rank 1
```

The old `partition.weights` form remains supported and implies identity mapping
`segment_to_rank = [0, 1, ...]`.

### Repeated physical ownership with page pull

`page_pull` additionally allows more logical segments than PCP ranks and lets a
physical rank own multiple logical segments:

```json
{
  "pcp_runahead": {
    "transport": "page_pull",
    "partition": {
      "policy": "weighted_contiguous",
      "segments": [
        {"weight": 1, "pcp_rank": 1},
        {"weight": 1, "pcp_rank": 0},
        {"weight": 1, "pcp_rank": 1}
      ],
      "page_align": true
    },
    "layout": "compact",
    "runtime": {
      "page_pull_backend": "nixl",
      "nixl_backends": ["UCX"],
      "max_inflight_reads": 4
    }
  }
}
```

Every physical PCP rank must own at least one logical segment. Repeated bindings
are intentionally rejected by tensor transports because they do not have a
single rank-to-segment inverse.

## Transport modes

| Transport | Data unit | Route | Repeated rank binding |
| --- | --- | --- | --- |
| `full_kv_collective` | contiguous K/V rows | all-gatherv | no |
| `prefix_p2p` | accumulated causal-prefix rows | logical chain | no |
| `direct_p2p` | one logical segment's rows | owner -> all downstream consumers | no |
| `page_pull` | physical KV-cache blocks | consumer one-sided READ from owner | yes |

All non-stock runahead modes use compact execution. `prefix_p2p`, `direct_p2p`,
and `page_pull` currently require a fresh complete prefill:

```text
num_computed_tokens == 0
num_scheduled_tokens == prefill_len
```

This intentionally excludes chunked-prefill continuation and decode until a
persistent sharded-KV consumer path is implemented.

## Page-aligned partition

Internal logical boundaries are aligned to the common kernel-page granularity
from `BlockTables.kernel_block_sizes`. Alignment uses absolute sequence
positions. For a fresh prefill this means every segment except the final tail is
composed of full physical KV pages.

That property is important for `page_pull`: all logical segments that can be
needed by a downstream consumer end at page boundaries. The final segment may
contain a partial page, but it has no later consumer, so no remote partial-page
merge is required.

When a request is too small for page-aligned internal cuts,
`weighted_partition_lengths()` falls back to exact token allocation. Such a step
is not eligible for `page_pull`; the manager falls back rather than transferring
a partial shared block.

## Prefix chain transport

`prefix_p2p` preserves the original experimental relay algorithm. For logical
segments `0..3`:

```text
S0 owner -> S1 owner -> S2 owner -> S3 owner
    S0        S0+S1      S0+S1+S2
```

Each receiver allocates one visible-prefix buffer, receives the accumulated
prefix, appends local K/V, and forwards the enlarged buffer. Logical adjacency
is translated through `segment_to_rank`, so a physical permutation such as
`[2, 0, 3, 1]` uses chain `rank2 -> rank0 -> rank3 -> rank1`.

This mode is useful as a reference but still pays relay latency and repeated
receive/copy/send work on intermediate ranks.

## Direct logical fanout

`direct_p2p` removes relay dependency without changing the K/V tensor data unit.
Each logical owner sends only its newly produced segment directly to every later
logical segment:

```text
S0 -> S1, S2, S3
S1 ->     S2, S3
S2 ->         S3
```

A consumer receives preceding segments independently and places them into one
logical-order cache-write buffer. Network completion order therefore no longer
has to match causal order.

For a fresh prefill, direct fanout does **not** reduce the theoretical payload.
With four equal segments both chain and direct fanout move `6 * segment_bytes`.
Its benefit is removing relay dependencies, accumulated-prefix allocations, and
serial hop latency.

## Consumer-driven page pull

`page_pull` changes the data plane from temporary K/V rows to physical paged-KV
cache blocks.

### Per-layer protocol

```text
producer projection K/V
        |
        v
write producer's LOCAL KV pages
        |
        v
CUDA write-completion event
        |
        v
READY(epoch, layer, cache address/geometry)
        |
        +-----------------------+
                                |
                      consumer local scheduler
                                |
                 missing remote segments only
                                |
                                v
                     asynchronous NIXL READ
                                |
                                v
                    destination paged KV cache
                                |
                    all required pages READY
                                |
                                v
                         FlashAttention
```

The implementation deliberately keeps the request-level `KVConnector`
scheduler out of PCP. PCP is a layer-level protocol. It reuses the same NIXL
ideas and primitives instead:

- registered device KV-cache memory;
- one descriptor per physical cache block;
- remote/local block-ID lists;
- `make_prepped_xfer("READ", ...)`;
- asynchronous transfer handles and completion polling.

### Control plane versus data plane

The control plane addresses **logical segments**. The data plane addresses
**physical KV block IDs**.

At `prepare_slot_mappings()` time, every rank already knows the same logical
segment slices. Its normal vLLM slot mapping converts those token positions into
local physical block IDs. Each owner publishes the source block IDs of the
segments it owns. The resulting `PCPPagePlan` contains, for every logical
segment:

```text
logical segment
    -> owner physical rank
    -> source physical block IDs on the owner
    -> destination physical block IDs on this consumer
```

No second page-address system is introduced; the plan is compiled from existing
vLLM `BlockTables` / slot mappings.

### Completion-driven ordering

Consumers pre-post READY receives for every required source. READY messages are
queued in arrival order. `progress()` starts reads while the configured
`max_inflight_reads` budget has room, so a consumer can fetch whichever producer
has completed first:

```text
S2 READY first -> pull S2 pages
S0 READY next  -> pull S0 pages
S1 READY last  -> pull S1 pages
```

Causal correctness only requires all pages needed by the current query to be
local before attention starts. Network transfer order does not have to be
logical order.

Layer cache registrations persist across scheduler steps. Once the layer set is
known, the next full-prefill step can pre-post READY receives for future layers;
progress at earlier layer boundaries can therefore launch already-ready future
layer reads before the consumer reaches that layer. The first pass lazily learns
those layer registrations, so same-layer completion ordering works immediately
while full future-layer prefetch becomes effective after registration is known.

### Repeated-rank local short circuit

For:

```text
segment_to_rank = [1, 0, 1]
```

rank 1 owns both `S0` and `S2`. When computing the later `S2` query it requires
`S0 + S1`, but `S0` is already local. Its page plan therefore issues only:

```text
pull S1 from rank0
```

There is no network transfer for `S0`. This is the first case where repeated
ownership directly reduces payload rather than only shortening the critical
path.

### READY ordering and memory safety

A producer must not publish READY merely because the cache-write kernel was
submitted. `page_pull` records a CUDA event after the early local cache write and
waits for that event before sending READY. Thus remote NIXL READ starts only
after source data is visible.

The consumer considers a source ready only after the NIXL handle reports DONE.
Only then can FlashAttention read those destination blocks. Remote prefix blocks
and locally owned segment blocks are page-disjoint, allowing local cache writes
and remote movement to target independent physical pages.

## Cache-write integration

The existing FlashAttention hook calls
`prepare_standard_pcp_kv_cache_inputs(key, value, slot_mapping, kv_cache)` before
its normal `reshape_and_cache_flash()` call. `page_pull` uses this hook without a
large backend rewrite:

1. slice the compact slot mapping to this physical rank;
2. perform an early local K/V cache write;
3. publish READY and complete required NIXL reads;
4. return local K/V and local slot mapping;
5. let the normal `reshape_and_cache_flash()` write the local rows again.

The duplicated local write is deliberate in this experimental version. It keeps
the standard backend path intact while making cache pages available to remote
readers before attention. A later optimization can move the READY hook directly
after the backend's native cache-write kernel and remove the duplicate write.

## Payload and reuse semantics

For a completely fresh prefill, every downstream rank still needs its causal
prefix. Page pull therefore cannot beat the information-theoretic payload solely
by changing the route:

```text
bytes = sum_i (number_of_downstream_consumers_i * segment_bytes_i)
```

Actual byte reduction comes from eliminating transfers that are unnecessary:

- an earlier segment is owned by the same physical rank;
- a future implementation retains a resident page from a previous step;
- a page has another usable replica;
- prefix-cache state proves the destination already contains the required page.

The current implementation includes the first optimization (local repeated-rank
short circuit). Resident/replica-aware source selection is the natural next
extension of `PCPPagePlan`; the transport API does not require a causal-chain
rewrite to add it.

## Current validation constraints

Common runahead constraints:

- standard MHA/GQA/MQA with FlashAttention;
- PCP > 1;
- TP = 1, PP = 1, DP = 1, DCP = 1;
- no EP/MoE collectives;
- no DBO;
- no speculative decoding;
- no async scheduling;
- eager execution (`cudagraph_mode=NONE`);
- causal-prefix transports are full-fresh-prefill only.

Additional `page_pull` constraints:

- NIXL must be installed and able to register the accelerator's KV memory;
- `partition.page_align=true`;
- one standard-attention KV cache group;
- homogeneous PCP model/cache geometry;
- unquantized FP16/BF16 cache (`cache_dtype=auto|float16|bfloat16`);
- no partial internal page transfer;
- registered layer cache addresses must remain stable while the transport is
  alive (cache reallocation/wake-up integration is not implemented yet).

If page-aligned full-prefill eligibility is not met, the step falls back instead
of issuing a partial-page pull.

## Persistent KV semantics

`full_kv_collective` reconstructs and writes a replicated current-step cache.
The three causal-prefix transports intentionally leave asymmetric persistent KV:

- `prefix_p2p`: each rank stores the prefix through its logical segment;
- `direct_p2p`: same semantic result without relay;
- `page_pull`: each rank has all pages needed through its latest owned logical
  segment, while locally repeated earlier segments are reused in place.

Sampling the current forward is valid. A later model step requiring decode or
continued prefill is rejected until the persistent sharded-KV consumer path is
implemented.

## Profiling

Enable detailed PCP NVTX ranges with:

```bash
VLLM_PCP_NVTX=1 ...
```

Relevant ranges include:

```text
pcp.baseline_kv_allgather
pcp.full_kv_allgatherv
pcp.prefix_exchange
pcp.prefix_recv_wait
pcp.prefix_send_enqueue
pcp.direct_exchange
pcp.direct_recv_enqueue
pcp.direct_recv_wait
pcp.direct_send_enqueue
pcp.page_pull_plan
pcp.page_pull_local_cache_write
pcp.page_pull_local_ready_wait
pcp.page_pull_read_submit
pcp.page_pull_exchange
pcp.page_pull_wait
pcp.compact_slot_mapping
pcp.restore_hidden_variable_allgather
pcp.send_wait
pcp.flush
```

Nsight should make the distinction clear: `prefix_p2p` exposes serial relay
hops, `direct_p2p` exposes independent owner-to-consumer transfers, and
`page_pull` exposes local cache writes followed by one-sided block reads whose
submission order follows producer readiness rather than logical rank order.

# PCP causal-prefix runahead

This experimental path extends vLLM V2 PCP for long-prefill execution. It reuses
stock PCP batch/block-table machinery and keeps four transport choices:
`full_kv_collective`, `prefix_p2p`, `direct_p2p`, and `page_pull`.

The implementation follows four rules:

1. compile logical compute segments once per scheduler step;
2. encode one-segment-per-rank permutations in the primary PCP group order;
3. keep repeated ownership local to the page-pull plan;
4. once `pcp_runahead` selects the runahead manager, unsupported steps fail
   explicitly instead of silently switching back to baseline PCP.

## Configuration

A non-empty `pcp_runahead` object enables the path.

```json
{
  "pcp_runahead": {
    "transport": "page_pull",
    "partition": {
      "segments": [
        {"weight": 4.0, "pcp_rank": 2},
        {"weight": 2.5, "pcp_rank": 0},
        {"weight": 1.9, "pcp_rank": 3},
        {"weight": 1.6, "pcp_rank": 1}
      ]
    },
    "runtime": {
      "nixl_backends": ["UCX"],
      "max_inflight_reads": 4,
      "max_inflight_sends": 4
    }
  }
}
```

`partition.weights` remains accepted and implies identity binding. Repeated
physical ownership is supported only by `page_pull`.

## Step admission semantics

Runahead is fail-closed. `RunaheadPCPManager.partition_batch()` either constructs
an active runahead step or raises. There is no `eligible=False -> baseline PCP`
path after the runahead manager has been selected.

The tensor transports remain fresh-prefill-only:

```text
full_kv_collective
prefix_p2p
direct_p2p

require:
  all requests are prefilling
  num_computed_tokens == 0
  num_scheduled_tokens == prefill_len
```

`page_pull` supports both fresh prefill and tracked chunked-prefill continuation:

```text
chunk N starts at scheduler.num_computed_tokens
                == PCPPageStateTracker.committed_tokens
```

A new request with a nonzero untracked prefix is rejected. APC/external-prefix
migration into PCP runahead is not implemented. A scheduler rewind or forward
jump relative to the committed PCP page state is also rejected.

Decode and mixed decode/prefill after rank-sharded page-pull prefill remain
unsupported.

The current model path still requires every physical PCP rank to receive at least
one token in a runahead step. Inactive/zero-row rank execution is a separate
follow-up; a too-small or otherwise unrepresentable chunk raises rather than
switching transport or baseline mode.

`eligibility.min_tokens` is retained as a compatibility configuration field, but
it is not used to redirect an enabled runahead manager to baseline PCP.

## Segment compilation and compact layout

The manager compiles every scheduled request into a `SegmentLayout`:

```text
segments_by_rank
rows_per_rank
logical_segments(req, start_pos, end_pos, owner_rank)
causal_segments_by_request
page_owner_updates
```

For `page_pull`, existing partial tail pages stay on their previously committed
owner. Internal ownership cuts are aligned to the kernel block size so two ranks
never write different token ranges into the same physical KV page.

Runahead uses exact rank widths:

```text
rank0 rows | rank1 rows | ... | rankN rows
```

Hidden-state restore and full-KV transport retain the equal-width fast path:

```text
equal rank widths     -> all_gather
variable rank widths  -> all_gatherv
```

## Transport modes

| Transport | Data unit | Route | Chunked continuation |
| --- | --- | --- | --- |
| `full_kv_collective` | contiguous K/V rows | primary PCP collective | no |
| `prefix_p2p` | accumulated prefix rows | logical rank chain | no |
| `direct_p2p` | one logical segment | rank to later ranks | no |
| `page_pull` | physical KV pages | consumer NIXL READ | yes |

### Prefix P2P

For four logical PCP ranks:

```text
rank0 -> rank1 -> rank2 -> rank3
  S0      S0+S1   S0+S1+S2
```

### Direct P2P

Each logical rank sends its local segment directly to later ranks:

```text
S0 -> S1, S2, S3
S1 ->     S2, S3
S2 ->         S3
```

This removes the relay dependency while keeping causal-prefix semantics.

## Chunked page pull

`page_pull` moves completed physical KV pages rather than temporary K/V rows.
The scheduler remains authoritative for chunk size and physical block allocation;
PCP adds only ownership and rank-local validity metadata.

### Persistent page state

For each live request slot, `PCPPageStateTracker` stores:

```text
logical page -> authoritative owner rank
logical page -> local valid physical block id
committed_tokens
```

The physical block id comes from vLLM `BlockTables`. If vLLM reallocates/COWs a
logical page to another physical block, the saved local-valid id no longer
matches and the page is demanded again from its authoritative owner.

Ownership is transactional. New `page_owner_updates`, mutable-tail invalidation,
rank-local validity, and `committed_tokens` are committed only after the forward
and required page reads complete. A failed forward therefore does not publish
future page ownership.

### Historical and current routes

Every page-pull step compiles two dependency classes:

```text
history route
  page was produced by an earlier chunk
  source page is already readable
  NIXL READ can be submitted immediately

current route
  page is produced in the current chunk/layer
  consumer waits for READY(epoch, layer, source)
  then submits NIXL READ
```

There is no background fill or chunk-end replication. A page is pulled only when
the current rank's attention requires it and the rank-local physical block does
not already contain a valid copy.

### CPU block plan

`BlockTables` maintains a CPU mirror of kernel block IDs. Page-plan compilation
therefore stays on CPU:

```text
logical request page
  -> request state index
  -> BlockTables.get_block_ids_cpu(...)
  -> physical block id
  -> local-valid check
  -> authoritative source rank
  -> PCPPageRoute
```

The currently supported replicated scheduler allocation gives identical source
and destination physical block IDs. `PCPPageRoute` nevertheless keeps source and
destination IDs separate for future asymmetric allocation/residency work.

### Stable NIXL registration

The first layer-cache discovery registers every write-owning attention KV cache.
NIXL agent metadata and stable layer geometry are exchanged once:

```text
layer_id -> base address, block bytes, block count, device
```

Cache addresses must remain stable while the transport is alive. NHD and HND
logical views are accepted when each physical block is one dense whole-page byte
span.

### Per-layer ordering and runahead

The normal native KV cache update remains the only local write:

```text
projection K/V
    |
    v
native KV cache write
    |
    v
CUDA event / READY for current-chunk producer
    |
    +-------------------------+
    |                         |
historical READs        current READY-gated READs
    |                         |
    +------------+------------+
                 v
          wait_layer(layer)
                 |
                 v
             attention
```

`wait_layer(L)` waits only for this rank's required routes for layer `L`. It does
not wait for all PCP ranks to finish `L`, so ranks naturally advance to different
layers and retain the diagonal cross-layer runahead pipeline.

Historical reads for later layers may already be queued while the model thread
computes an earlier layer. The progress engine prioritizes the currently demanded
layer without making prefetched later-layer reads part of that layer's completion
condition.

READY uses NIXL notifications containing only changing state:

```text
epoch, layer_id, source_rank
```

The model thread does not call `event.synchronize()`.

## Repeated ownership

Repeated ownership remains page-pull-only. For:

```text
segment_to_rank = [1, 0, 1]
```

rank 1 owns both `S0` and `S2`; while executing `S2`, its own `S0` pages are local
and only missing predecessor pages are pulled.

## MLA and MoE

Dense MLA `page_pull` uses the native latent KV cache write and supports
unquantized FP16/BF16 cache pages. Sparse MLA/indexer remains unsupported.

Runahead does not use PCP ranks as a MoE tensor/expert-parallel group. With the
supported `TP=1`, `DP=1`, and `EP=off` configuration, each PCP rank loads the
complete routed-expert set and executes routing/expert kernels locally. This
avoids layer-synchronous PCP collectives while ranks are on different layers.

## Current validation constraints

- PCP > 1;
- TP = 1, PP = 1, DP = 1, DCP = 1;
- EP off; MoE experts replicated locally;
- no DBO, speculative decoding, async scheduling, LoRA, MM, or encoder-decoder;
- eager execution (`cudagraph_mode=NONE`);
- no request-level KV transfer connector;
- tensor transports: fresh complete prefill only;
- `page_pull`: fresh or exactly tracked chunked prefill, one KV-cache group,
  NIXL available, dense unquantized FP16/BF16 pages, stable KV-cache addresses;
- `page_pull`: no APC/external-prefix adoption, preemption rewind migration,
  decode, mixed decode/prefill, or inactive/zero-row rank execution yet.

## Profiling

Set the PCP profiling flag before worker startup:

```bash
VLLM_PCP_NVTX=1
```

Important scopes/marks include runahead step begin/end, whole decoder layers,
page-plan compilation, READY publication, READ submission/completion, and
`pcp.page_pull_wait`.

For generic vLLM execution scopes it can be combined with:

```bash
VLLM_NVTX_SCOPES_FOR_PROFILING=1
```
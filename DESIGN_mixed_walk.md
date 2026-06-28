# Mixed prefill+decode step (MSTAR_MIXED_WALK) — continuous batching for TTFT

Branch: `exp/mixed-walk-piggyback` (from `integration-mnew` / M\*-new).
Status: **scheduler slice landed + forward-build core landed + replay stubbed.**
Env flag `MSTAR_MIXED_WALK` (default OFF → behavior byte-identical to base M\*).

## 1. Problem

M\*'s worker scheduler enforces **one `(node, graph_walk)` per scheduler step**:

- `MicroScheduler._select_node_priority` (`mstar/worker/micro_scheduler.py`)
  picks the single most-common `graph_walk` among ready requests; every request
  on a *different* walk stays queued for a later cycle.
- `_select_node_rr` likewise returns a single `(node, walk)`.
- CUDA graphs are keyed `(graph_walk, requires_cfg, bs, num_tokens)`
  (`CudaGraphKey`), so a step is one captured shape for one walk.

Consequence: a freshly-arrived request must wait for the running **decode**
batch to yield the node before its **prefill** can run. At batch this inflates
TTFT — worst for **S2T** (the Thinker prefill of an audio request queues behind
in-flight `thinker_decode`) and compounding for **I2T** (vision prefill behind
decode behind other prefills).

vLLM v1 avoids this with **continuous batching**: one `token_budget` per step;
running decodes consume 1 token each, then waiting prefills consume the
remainder, all emitted in a single `SchedulerOutput` → one varlen forward.

M\*'s topology makes this natural for Qwen3-Omni: `thinker_decode` and every
Thinker prefill walk (`prefill_text` / `prefill_audio` / `prefill_vision`) run
on the **same node** ("Thinker") with the **same submodule** and **same KV
cache**. So a decode batch and a waiting prefill can be served by one mixed
varlen forward — they only differ in `graph_walk` and per-request query length.

## 2. What is implemented vs stubbed

| Piece | State | Location |
|---|---|---|
| Token-budget admission (pure) | **done, tested** | `micro_scheduler.plan_mixed_budget` |
| Decode/prefill walk split | **done, tested** | `micro_scheduler.is_decode_walk` |
| Scheduler emits mixed batch | **done, tested** | `MicroScheduler._maybe_plan_mixed`, `MixedBatchPlan` |
| Mixed `CudaGraphKey` variant | **done, tested** | `cuda_graph_runner.CudaGraphKey` |
| Flat varlen layout builder | **done, tested** | `mstar/engine/mixed_walk.py` |
| Prefill capture bucketing | **done, tested** | `mixed_walk.pad_prefill_tokens_to_bucket` |
| Mixed graph capture | **stub / TODO** | (new MIXED `CudaGraphConfig`) |
| Mixed graph replay | **stub / TODO** | `CudaGraphRunner.run_mixed` |
| Worker consumption | **guarded NotImplemented** | `Worker._build_node_batch` |

Flag OFF: `_maybe_plan_mixed` returns `None` immediately; every `ScheduledBatch`
carries `mixed_plan=None`; no new code path is taken. Flag ON: the scheduler
produces a mixed batch and the worker raises `NotImplementedError` at the build
boundary (honest stub — never silently mis-executes prefill tokens as decode).

## 3. Token-budget loop (scheduler)

`plan_mixed_budget(decode_count, prefill_candidates, token_budget,
prefill_chunk_cap, max_prefill_requests, token_count_fn)`:

```
used = decode_count                 # each running decode = 1 query token
for cand in prefill_candidates:     # scan order = priority order
    if len(admitted) >= max_prefill_requests: break
    cost = min(token_count_fn(cand), prefill_chunk_cap)
    if used + cost > token_budget: continue
    admitted.append(cand); used += cost
```

`_maybe_plan_mixed` runs only when the just-selected primary is a decode
(`is_decode_walk`) on a KV-cache node. It collects prefill candidates = ready
entries on the **same node** with a non-decode walk, admits them via
`plan_mixed_budget`, pops their ready nodes into `node_objects` (same mutation
the decode loop does), and returns a `MixedBatchPlan`.

Env knobs: `MSTAR_MIXED_TOKEN_BUDGET` (8192), `MSTAR_MIXED_PREFILL_CHUNK` (512),
`MSTAR_MIXED_MAX_PREFILL_REQS` (1, matching the minimal "one prefill chunk"
slice).

**Limitation (documented):** exact per-request prefill length is not reliably
available at schedule time (it lives in the request's pending input tensors, not
the ready-node metadata). `token_count_fn` therefore defaults to the
conservative `prefill_chunk_cap`. Exact-token accounting belongs at build time
(`build_mixed_varlen_layout` already takes real lengths); the scheduler only
needs to bound how many prefills ride along.

## 4. Mixed `CudaGraphKey`

```python
@dataclass(frozen=True)
class CudaGraphKey:
    graph_walk: str; requires_cfg: bool; bs: int; num_tokens: int
    mixed: bool = False           # additive, default-inert
    num_decode: int = 0
    num_prefill_tokens: int = 0
```

Invariants for a mixed key: `bs == num_decode + 1` (one piggybacked prefill in
the minimal slice) and `num_tokens == num_decode + num_prefill_tokens`. The
defaults mean every existing construction (which omits the new fields) hashes /
compares exactly as before — verified in
`test_cuda_graph_key_mixed_default_inert`.

## 5. Varlen tensor layout (`build_mixed_varlen_layout`)

Order matches vLLM v1: **decodes first (1 token each), then each prefill chunk.**

```
flat tokens: [d0 .. d(D-1) | p0_0 .. p0_(P0-1) | p1_0 .. ]
qo_indptr:   [0, 1, .., D, D+P0, D+P0+P1, ..]        # FlashInfer query offsets
qo_seq_lens: [1]*D + [P0, P1, ..]
kv_seq_lens: [Ld_i + 1]*D + [kv_start_r + Pr]        # post-step KV per request
```

Returns a `MixedWalkLayout` (`qo_indptr`, `kv_seq_lens`, `qo_seq_lens`,
`mrope_positions [3, T]`, `positions [T]`, `request_token_spans [bs, 2]`). Pure,
device-agnostic, allocates only small index tensors — CPU-unit-tested.

`request_token_spans` gives each request's `[start, end)` rows so replay can (a)
copy input embeddings into the right rows and (b) scatter sampled logits back —
the decode rows and the **final** row of each prefill chunk (the token that
becomes that request's first decode input).

## 6. M-RoPE position handling

Qwen3-Omni uses interleaved 3D M-RoPE (`components/rope.py`,
`mrope_section=(24,20,20)`), so positions are `[3, T]` (temporal, height,
width). The builder writes:

- **decode token i**: all 3 rows = `decode_positions[i]` (the scalar next-token
  position; passed explicitly because M-RoPE position ids diverge from raw token
  counts after multimodal spans).
- **text prefill**: all 3 rows = `arange(pos_start, pos_start + length)`.
- **audio/vision prefill**: caller supplies `prefill_mrope_fn(req_idx,
  pos_start, length) -> [3, length]` so spatial/temporal deltas are honored
  (the builder validates the shape). Default is text.

This keeps the mixed batch's positions correct per request even though their
walks differ — the decode rows and prefill rows are independent slices of the
same `[3, T]` tensor.

## 7. KV append

Each request appends into its own paged KV via the existing
`BatchedCacheManager` / `PagedAllocationManager` machinery — **no shared
sequence**. Decodes append 1 page-slot's worth (1 token) at offset
`decode_kv_lens[i]`; prefills append `Pr` tokens at offset `kv_start_r`. The
matching paged `kv_indptr` is built by the cache manager from
`layout.kv_seq_lens` at plan time (out of scope for the pure builder).
FlashInfer's varlen prefill wrapper handles the ragged `(qo_indptr, kv_indptr)`
in a single attention call; the 1-token decode rows are just length-1 query
segments — no special-casing needed.

## 8. Capture-shape explosion risk + mitigation

A mixed graph is keyed on **both** `num_decode` and `num_prefill_tokens`. Naively
that is `|decode bs| × |distinct prompt lengths|` captures — and prompt length is
unbounded, so capture count and warmup time/memory would blow up.

**Mitigation (implemented in the builder, to be consumed by the capture loop):**

1. **Cap** each prefill chunk at `MSTAR_MIXED_PREFILL_CHUNK` (512). Longer
   prompts are not piggybacked in the minimal slice (they run as a normal
   separate prefill step) — full chunked-prefill is future work (overlaps the
   independent `exp/chunked-prefill` branch).
2. **Bucket** `num_prefill_tokens` to a fixed set
   `DEFAULT_MIXED_PREFILL_BUCKETS = (64, 128, 256, 512)` via
   `pad_prefill_tokens_to_bucket` (pad up to next bucket; FlashInfer ignores the
   padding tokens through `qo_indptr`). Capture count is now
   `|decode bs buckets| × 4` — finite and small.
3. Reuse the existing **prefill** FlashInfer persistent wrapper
   (`_capture_one_flashinfer_packed`): varlen already covers the 1-token decode
   rows, so no separate decode wrapper is needed in the mixed graph.

## 9. Remaining work (TODO in `CudaGraphRunner.run_mixed`)

1. **Capture**: a MIXED `CudaGraphConfig` whose `get_total_tokens` enumerates the
   `(num_decode_bucket, num_prefill_bucket)` grid; capture via the existing
   FlashInfer-packed path with mixed `qo_indptr`/`kv_indptr`.
2. **Replay** (`run_mixed`): pad `(decode_bs, prefill_tokens)` to the captured
   bucket, pack the flat token tensor + M-RoPE positions from
   `build_mixed_varlen_layout` into the slot's static buffers, plan the prefill
   wrapper, copy, replay.
3. **Sample**: only the decode rows + each prefill's final row.
4. **Worker**: build the mixed `NodeBatch` in `_build_node_batch` (replace the
   guard), route decode vs prefill outputs by `request_token_spans`.

## 10. GPU validation

Parity (correctness), once `run_mixed` lands:

```
pytest test/modular/test_qwen3_omni_mixed_walk.py::test_mixed_step_logits_match_separate_steps
# asserts mixed-step per-request logits == separate-step logits within tol
```

TTFT A/B (the win), S2T under staggered arrival, flag ON vs OFF:

```
# Fixed devices, idle-verified, per workspace GPU conventions.
for FLAG in 0 1; do
  MSTAR_MIXED_WALK=$FLAG \
  timeout 1800 python benchmark/ttft_bench.py \
    --model qwen3-omni --task s2t \
    --batch-sizes 8,16,32 \
    --arrival staggered \
    --metric ttft_p50 \
    --out benchmark-personal/mixed_walk/raw_flag${FLAG}.json
done
# Compare ttft_p50 at B=8,16,32 (expect ON < OFF, largest gap at high B);
# also report decode-throughput delta to confirm no regression.
```

(Adapt `benchmark/ttft_bench.py` to the actual harness in this repo;
the contract is: staggered arrivals, S2T, p50 TTFT at B∈{8,16,32}, ON vs OFF.)

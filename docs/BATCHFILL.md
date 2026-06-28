# Batch-fill instrumentation (`MSTAR_BATCHFILL_STATS`)

## What it answers

Does the Talker AR-decode CUDA graph actually **fill** at batch (B=32), or do
requests desync so the engine runs small batched replays back-to-back, wasting
the batched-decode advantage?

The Talker (and Thinker) decode steps each capture CUDA graphs at batch sizes
`[1, 2, 4, 8, 16, 32]` and pad the live batch up to the next captured size. If
the scheduler keeps all in-flight requests in lock-step, the Talker decode
replays should mostly run at `bs=32`. If requests desync (different lengths,
staggered arrivals, prefill/decode interleaving), the engine ends up replaying
lots of small graphs (`bs=1`, `bs=2`, ...) back-to-back, and the big batched
graph never pays off. That desync is exactly what the mixed-walk / Lever-2
scheduler change is meant to fix; this instrumentation tells you how much it is
even needed.

## How it works

Every node forward (Talker decode, Thinker decode, encoder forwards, prefills)
goes through `KVCacheEngine.execute_forward`. When the flag is set, that hook
records the **actual** batch size used per replay, keyed by
`(node_name, graph_walk, dispatch_path)` where `dispatch_path` is one of
`cuda_graph` / `batched` / `sequential`. At engine shutdown a per-key histogram
over the capture buckets plus the mean fill ratio is logged.

The flag defaults to **OFF**. When unset, the only hot-path cost is one module
attribute load and a boolean check, so there is effectively zero overhead.

- Hook: `mstar/engine/kv_cache_engine.py` — `execute_forward` (records
  `len(batch.request_ids)` + the dispatch path), `shutdown` (dumps summary).
- Logic: `mstar/engine/batchfill_stats.py` — pure, CPU-only aggregation
  (`fill_bucket`, `aggregate_fill_stats`) plus the process-global
  `BatchFillRecorder`.

## Turning it on

```bash
export MSTAR_BATCHFILL_STATS=1            # 1/true/yes/on enable it
# optional: also dump the summary as JSON at shutdown
export MSTAR_BATCHFILL_STATS_JSON=/path/to/batchfill.json
```

The summary is emitted at `INFO` level from `mstar.engine.batchfill_stats` when
the engine shuts down (and to the JSON file if `MSTAR_BATCHFILL_STATS_JSON` is
set). Make sure the engine is shut down cleanly so the dump runs.

## Reading the output

Each line looks like:

```
talker/talker_decode/cuda_graph: n=12000 mean_bs=30.50 fill=0.97 frac@32=0.91 hist[1:120 2:60 4:40 8:80 16:300 32:11400]
```

- `n` — number of replays recorded for this stage.
- `mean_bs` — mean actual batch size across replays.
- `fill` — mean fill ratio `actual_bs / capture_bucket` (1.0 = perfect).
- `frac@32` — fraction of replays that landed in the top (`bs=32`) bucket. This
  is the headline number for "is the big batch filling?".
- `hist[...]` — count of replays per capture bucket `[1,2,4,8,16,32]`.

### Good vs bad fill

At offered concurrency `B=32` with `max_concurrent_requests >= 32`:

- **Good fill (mixed-walk not urgently needed):** Talker `talker_decode`
  replays are mostly `bs=32` — `frac@32` near 1.0, `mean_bs` near 32, histogram
  dominated by the `32` bucket. The batched-decode graph is paying off.
- **Bad fill / desync (mixed-walk / Lever-2 needed):** lots of `bs=1` / `bs=2`
  replays, `frac@32` low (e.g. < 0.5), histogram spread across small buckets.
  The engine is running small graphs back-to-back; the batched advantage is
  wasted and the scheduler change is justified.

Compare the Talker line against the Thinker line (`thinker/thinker_decode/...`)
and the encoder lines: if the Thinker fills but the Talker does not, the desync
is happening specifically in the Talker AR loop.

> Note on `bs=1`: a single-request replay pads to the `bs=1` bucket, so its
> *fill ratio* is 1.0 even though it is the worst case for throughput. Always
> read `frac@32` and the histogram alongside `fill`, not `fill` alone.

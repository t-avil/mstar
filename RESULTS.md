# Exp 3: Chunk-Boundary Encoder Coalescing — Results

**Branch**: `exp/encoder-chunk-coalesce` (based on `opt/combined-vision-opts` @ e943d72)
**Env flag**: `MSTAR_ENCODER_CHUNK_COALESCE=1` (default OFF), `MSTAR_ENCODER_COALESCE_SIZE=4`
**Methodology**: B-only sweep on GPUs 5,6 (NUMA 1, matches committed mstar_new). Compared against the committed `mstar_new` baseline in `raw_audio_to_text.json` / `raw_image_to_text.json`.
**Date**: 2026-06-29

## Headline

**NEUTRAL overall**, with a borderline win at I2T B=1 (+8% req/s, **-22% TTFT**) and one
anomalous spike at S2T B=8 (+8.8% req/s, likely run-to-run noise — surrounded by neutral
B=4 and B=16). TTFT improvements are small but consistent at low batches across both paths.

This is not a clear win to ship. The chunk-boundary trigger fires at the right places,
but at high batch sizes the encoder is already overlapped with decode naturally (via
the worker's existing scheduling), and the coalescer adds bookkeeping cost without
unlocking new parallelism.

## Full results (B vs committed mstar_new)

### S2T (audio_to_text)

| B | base req/s | ours req/s | Δreq/s | base TTFT | ours TTFT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 5.060 | 4.867 | -3.8% | 101 ms | 102 ms | +1.8% | NEUTRAL |
| 2 | 6.925 | 6.948 | +0.3% | 143 ms | 142 ms | -0.8% | NEUTRAL |
| 4 | 8.920 | 8.674 | -2.8% | 186 ms | 187 ms | +0.8% | NEUTRAL |
| 8 | 12.163 | 13.229 | **+8.8%** | 248 ms | 242 ms | -2.6% | PROMISING* |
| 16 | 17.900 | 17.635 | -1.5% | 288 ms | 288 ms | +0.1% | NEUTRAL |
| 32 | 23.532 | 22.366 | -5.0% | 420 ms | 455 ms | +8.3% | NEUTRAL (borderline) |

*B=8 PROMISING is an outlier. B=4 (-2.8%) and B=16 (-1.5%) bracketing it are neutral,
strongly suggesting this is run-to-run variance, not a real win.

### I2T (image_to_text)

| B | base req/s | ours req/s | Δreq/s | base TTFT | ours TTFT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.693 | 0.749 | **+8.0%** | 283 ms | 220 ms | **-22.2%** | PROMISING |
| 2 | 1.095 | 1.145 | +4.6% | 325 ms | 299 ms | -7.9% | NEUTRAL |
| 4 | 1.727 | 1.676 | -2.9% | 349 ms | 343 ms | -1.7% | NEUTRAL |
| 8 | 2.438 | 2.422 | -0.7% | 417 ms | 353 ms | -15.3% | NEUTRAL |
| 16 | 3.299 | 3.288 | -0.4% | 500 ms | 458 ms | -8.3% | NEUTRAL |
| 32 | 4.209 | 4.144 | -1.6% | 585 ms | 576 ms | -1.4% | NEUTRAL |

I2T shows a consistent (small) TTFT improvement at every batch but throughput is flat.
B=1 is the only PROMISING data point; the effect diminishes as concurrency grows because
the encoder is already amortized across many requests by then.

## Why this didn't deliver more

The chunk-boundary hook fires at the right points (between Thinker prefill walks for the
same request — e.g. `prefill_audio → prefill_text`). But on the e943d72 base, single-walk
chunked prefill is not enabled (`MSTAR_CHUNKED_PREFILL` isn't a real flag in this branch).
That means the chunk-boundary signal fires only at coarse-grained inter-walk yields, not
intra-prefill chunk yields. The coalescer rarely accumulated a meaningful batch.

To unlock the real potential of this experiment, it should be rebased on a branch that
includes true intra-prefill chunking, so the chunk-boundary hook fires multiple times
per request and the coalescer can build larger batches.

## Recommendation

**Park.** Don't ship as a default flag. Revisit if/when single-walk chunked prefill lands
upstream — then the coalescer has natural batch-formation windows to exploit.

## Files in this branch

- `mstar/worker/encoder_coalescer.py` (new)
- `mstar/worker/micro_scheduler.py` — optional coalescer hook
- `mstar/worker/worker.py` — chunk-boundary fire-event in `_postprocess_batch`
- `benchmark/mvp_encoder_chunk_coalesce.sh` — A/B MVP (B=8 single-batch)
- `benchmark/full_sweep_encchunk.sh` — full A/B sweep
- `benchmark/full_sweep_encchunk_bonly.sh` — B-only sweep (this run)

## Raw data

`/home/tim/tmp/full_sweep_encchunk_bonly_20260629T182541/`

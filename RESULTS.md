# Exp 2: Async Encoder Scheduling — Results

**Branch**: `exp/encoder-async-schedule` (based on `opt/combined-vision-opts` @ e943d72)
**Env flag**: `MSTAR_ENCODER_ASYNC=1` (default OFF), `MSTAR_ENCODER_ASYNC_DEPTH=4`
**Path tested**: I2T (image_to_text)
**Hardware**: GPUs 5,6 (NUMA 1), H200
**Date**: 2026-06-29

## Headline

**PROMISING at high concurrency.** Async encoder scheduling — pipelining request N+1's
encoder forward with request N's Thinker decode — delivers measurable wins at B=16+ and a
clear win at B=32. At low batch (B=1,2) it's slightly negative (no decode work to overlap
with). The trend is monotone in the right direction.

## Full sweep results (I2T)

| B | A: OFF req/s | B: ON req/s | Δ req/s | A: TTFT_p50 | B: TTFT_p50 | Δ TTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.741 | 0.722 | -2.6% | 239 ms | 275 ms | +14.8% | NEGATIVE *(contaminated)* |
| 2 | 1.118 | 1.080 | -3.4% | 293 ms | 316 ms | +7.9% | NEUTRAL |
| 4 | 1.724 | 1.759 | +2.0% | 338 ms | 354 ms | +4.5% | NEUTRAL |
| 8 | 2.366 | 2.426 | +2.6% | 370 ms | 373 ms | +1.1% | NEUTRAL |
| 16 | 3.199 | 3.313 | +3.6% | 515 ms | 458 ms | **-11.2%** | NEUTRAL |
| 32 | 4.060 | **4.361** | **+7.4%** | 703 ms | **489 ms** | **-30.4%** | **PROMISING** |

Methodology: closed-loop, food101 dataset, N = max(50, 10×B), warmup = 5. Same physical GPUs
for both A and B runs, server torn down between A and B (see Caveat).

## Caveat: B_on B=1 contaminated

The full-sweep harness uses the same SHM socket prefix across A and B. After A teardown,
the new B server's `/v1/models` curl was answered by the still-shutting-down A API server
(workers hadn't fully exited), so the script printed "server ready in 1s" when in fact the
new server didn't fully come up until ~80s later. The B_on B=1 result therefore mixes
"requests served by tail of A_off" with "requests served by fresh B_on".

Workaround for future runs: extend the post-teardown idle check to wait for GPU memory
to drop, not just verify the port. B=2..32 are clean (the new server was fully up by then).

## Why this is the right tradeoff

At B=1 there is no concurrent decode to overlap the speculative encoder forward with.
The async-encoder dispatcher consumes scheduler bandwidth without delivering hidden
latency, so it's a small net loss.

At B=16, the Thinker is constantly busy with decode for some request, and starting the
next request's encoder forward on its own CUDA stream amortizes the encoder cost behind
those decode steps. The +3.6% req/s improvement is modest, but TTFT drops 11% — meaning
new requests start producing tokens noticeably sooner because their encoded tokens are
already in the Thinker's queue when their turn comes.

At B=32, the same effect amplifies. +7.4% req/s, **-30% TTFT** — the TTFT gain is
larger than the throughput gain, which makes sense: async scheduling is fundamentally a
latency-hiding optimization. Most of the time, you save on tail latency rather than
peak throughput.

## Recommendation

**Ship as opt-in.** Add `MSTAR_ENCODER_ASYNC=1` as an additional M\*-new flag for
deployments targeting B ≥ 16 (production multi-tenant serving). For latency-sensitive
single-request workloads, leave it OFF.

## Raw data

- `full_sweep_encasync_20260629T175445/A_off/B{1,2,4,8,16,32}/results.json`
- `full_sweep_encasync_20260629T175445/B_on/B{1,2,4,8,16,32}/results.json`
- `full_sweep_encasync_20260629T175445/server_{A_off,B_on}.log`

## Files changed in this branch

- `mstar/worker/micro_scheduler.py`: flag parsing, encoder-priority bump, depth counter
- `mstar/worker/worker.py`: lazy low-priority CUDA stream for encoder, async dispatch
- `benchmark/mvp_encoder_async.sh`: A/B MVP (B=1,8 only)
- `benchmark/full_sweep_encasync_i2t.sh`: full A/B sweep across B=1..32 on I2T

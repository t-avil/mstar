# Exp 2: Async Encoder Scheduling — Cross-Path Results

**Branch**: `exp/encoder-async-schedule` (based on `opt/combined-vision-opts` @ e943d72)
**Env flag**: `MSTAR_ENCODER_ASYNC=1` (default OFF), `MSTAR_ENCODER_ASYNC_DEPTH=4`
**Paths tested**: I2T (full A/B), S2T (B-only vs committed), I2S (B-only vs committed)
**Hardware**: H200 × 2 GPUs, NUMA 1 (matches committed mstar_new baseline)
**Date**: 2026-06-29

## Headline

**Path-dependent: PROMISING on I2T, NEGATIVE on S2T, NEUTRAL on I2S.** The async encoder
delivers throughput and latency wins on vision-encoder paths at high concurrency (I2T B=32:
+7.4% req/s, -30% text TTFT) but **regresses** at high concurrency on the audio-encoder path
(S2T B=16,32: -18% req/s) because the audio encoder is too cheap for speculative dispatch
to be worth its overhead. On I2S the throughput is flat but text-side gains don't propagate
through the Talker+Code2Wav bottleneck to audio TTFT.

**Recommendation**: ship as opt-in for **vision-heavy** workloads at B≥16 only (I2T-style
deployments). Do not enable for audio-heavy or speech-output workloads.

## Results by path

### I2T (image_to_text) — original full A/B sweep

| B | A: OFF req/s | B: ON req/s | Δreq/s | A: TTFT | B: TTFT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.741 | 0.722 | -2.6% | 239 ms | 275 ms | +14.8% | NEGATIVE* |
| 2 | 1.118 | 1.080 | -3.4% | 293 ms | 316 ms | +7.9% | NEUTRAL |
| 4 | 1.724 | 1.759 | +2.0% | 338 ms | 354 ms | +4.5% | NEUTRAL |
| 8 | 2.366 | 2.426 | +2.6% | 370 ms | 373 ms | +1.1% | NEUTRAL |
| 16 | 3.199 | 3.313 | +3.6% | 515 ms | 458 ms | **-11.2%** | NEUTRAL |
| 32 | 4.060 | **4.361** | **+7.4%** | 703 ms | **489 ms** | **-30.4%** | **PROMISING** |

*B=1 result was contaminated (server-ready race during A→B transition); see methodology section.

### S2T (audio_to_text) — B-only vs committed mstar_new

| B | base req/s | ours req/s | Δreq/s | base TTFT | ours TTFT | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 5.060 | 4.912 | -2.9% | 101 ms | 96 ms | -4.5% | NEUTRAL |
| 2 | 6.925 | 6.774 | -2.2% | 143 ms | 140 ms | -2.0% | NEUTRAL |
| 4 | 8.920 | 8.745 | -2.0% | 186 ms | 179 ms | -3.3% | NEUTRAL |
| 8 | 12.163 | 11.210 | **-7.8%** | 248 ms | 242 ms | -2.3% | **NEGATIVE** |
| 16 | 17.900 | 14.696 | **-17.9%** | 288 ms | 328 ms | +13.7% | **NEGATIVE** |
| 32 | 23.532 | 19.315 | **-17.9%** | 420 ms | 454 ms | +8.1% | **NEGATIVE** |

### I2S (image_to_speech) — B-only vs committed mstar_new

Note: TTFT below is **audio TTFT** (time to first audio chunk), which is what matters for
speech output user experience.

| B | base req/s | ours req/s | Δreq/s | base TTFT(audio) | ours TTFT(audio) | ΔTTFT | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.277 | 0.275 | -0.5% | 401 ms | 437 ms | +8.9% | NEUTRAL |
| 2 | 0.458 | 0.450 | -1.8% | 471 ms | 474 ms | +0.6% | NEUTRAL |
| 4 | 0.735 | 0.717 | -2.4% | 549 ms | 554 ms | +0.9% | NEUTRAL |
| 8 | 1.146 | 1.142 | -0.4% | 656 ms | 657 ms | +0.1% | NEUTRAL |
| 16 | 1.723 | 1.675 | -2.8% | 797 ms | 831 ms | +4.2% | NEUTRAL |
| 32 | 2.218 | 2.225 | +0.3% | 1074 ms | 1089 ms | +1.4% | NEUTRAL |

## Why this is path-dependent

**Async encoder helps when the encoder is the slow stage relative to the savings.**
On the vision path:
- Image preprocess (~130 ms on GPU) + ViT (~30 ms) = ~160 ms of work that the speculative
  forward can hide behind the previous request's decode.
- At B=32 with text decode ~700 ms per request, hiding 160 ms of encoder behind it
  delivers 23% TTFT improvement — and we measured -30%, consistent.

On the audio path:
- Mel + audio encoder = ~10-20 ms total (already moved to GPU via `MSTAR_GPU_MEL`).
- The speculative forward burns scheduler bandwidth + CUDA stream priority for almost no
  hidden work. At high concurrency this contention with decode actually slows the Thinker.
- Net: throughput drops 18% at B=16+ on S2T.

On the speech-output paths (I2S), text-side gains are real but the Talker+Code2Wav
serializer becomes the bottleneck, so user-perceived audio TTFT doesn't improve.

## Methodology

- **I2T**: full A/B in one script (`full_sweep_encasync_i2t.sh`). Each phase same GPUs.
- **S2T, I2S**: B-only (`full_sweep_encasync_bonly.sh`), compared against the committed
  `mstar_new` baseline in `/home/tim/bench-sweep-wt/benchmarks/qwen3-omni-joint/raw_*.json`.
  Same NUMA (1) as committed runs; expected drift ±5%.
- All runs: closed-loop, N = max(50, 10×B), warmup = 5.
- **Caveat (I2T B=1)**: the A→B transition in the full A/B script left workers shutting
  down while the new server's API endpoint was already up, so the script's "ready in 1s"
  curl hit the dying A server. B=2..32 unaffected.

## Recommendation

Ship `MSTAR_ENCODER_ASYNC=1` as **opt-in** for image-heavy text-generation workloads
(I2T-style multi-tenant serving at B≥16). Do **not** enable for audio-input or
speech-output paths. Update `LEARNINGS.md §3.1` to list this as a per-path-gated
optimization.

## Files

- `mstar/worker/micro_scheduler.py`: flag parsing, encoder-priority bump
- `mstar/worker/worker.py`: lazy low-priority CUDA stream for encoder
- `benchmark/mvp_encoder_async.sh`: original MVP (B=1,8 on I2T)
- `benchmark/full_sweep_encasync_i2t.sh`: full A/B sweep on I2T
- `benchmark/full_sweep_encasync_bonly.sh`: B-only sweep parameterized by REQTYPE

## Raw data

- `/home/tim/tmp/full_sweep_encasync_20260629T175445/` (I2T full A/B)
- `/home/tim/tmp/full_sweep_encasync_audio_to_text_20260629T182719/` (S2T B-only)
- `/home/tim/tmp/full_sweep_encasync_image_to_speech_20260629T182720/` (I2S B-only)

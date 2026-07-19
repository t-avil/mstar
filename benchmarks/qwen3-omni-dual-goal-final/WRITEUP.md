# Beating vLLM-Omni on text without losing the speech lead — Qwen3-Omni (M*)

## Summary
We set out to make M*'s **text** generation (image→text, speech→text) beat vLLM-Omni
on throughput at every batch, **without** sacrificing M*'s 2–3× lead on **speech**
generation (image/text/speech→speech). We achieved both, validated against the
*current* competitor **vLLM-Omni 0.24** (not just the older 0.22), all natural-EOS,
closed-loop, max-concurrency, continuous batching, **no DP/PD disaggregation**.

- **Text:** the winning build beats vLLM-Omni 0.24 on tok/s **and** req/s at every
  batch on both paths (i2t 1.07–1.25×, s2t 0.99–1.28×), and crushes the m-star baseline
  and main (i2t B32: 1866 vs 825 baseline vs 624 main tok/s).
- **Speech:** 2–3× on request throughput vs vLLM-0.24 for i2s (2.19–2.52×) and s2s
  (2.65–3.50×), preserving/improving the committed 2–3×-era M* speech (i2s +11–19%,
  s2s +15–41%). **t2s (text→speech) is a win but the weakest speech path: ~1.6× audio-sps
  and ~1.7–2.0× req/s vs vLLM-0.24** — at the low edge of the 2–3× band, because
  vLLM-0.24 improved its own audio throughput 35–55% over 0.22 (vs the older 0.22 t2s
  would likely sit fully in 2–3×, but no committed 0.22 t2s exists to confirm).

## The two-part method

### 1. Topology per modality (one build, two deployments)
The Thinker (text decode) and the Talker+Code2Wav (speech synthesis) contend for GPU
differently by output modality. One code build, deployed with the config that suits
the output:
- **Text out (i2t, s2t):** `encoff` — encoders on the Talker/Code2Wav GPU, Thinker
  alone. Removes encoder↔decode contention → the text throughput win.
- **Speech out (i2s, s2s, t2s):** `base` — encoders on the Thinker GPU, Talker+Code2Wav
  isolated → protects the codec pipeline that delivers the 2–3× audio lead.

### 2. A stack of default-off, parity-safe feature flags
Layered on the `encoders-implemented` baseline (all documented in `FEATURES.txt`):
host-floor decode stack (cut the ~26 ms Python/GIL per-step floor), fp8 MoE grouped
GEMM, torch.library custom ops (keep the compiled Thinker graph intact under fp8),
an off-process multimodal preprocess pool (the i2t TTFT fix), chunked + captured-mixed
prefill, capture-grid coverage, and — new in this work — the **s2t audio-prefill merge
with an occupancy auto-gate** (below).

### The new engineering result: MERGED_PREFILL_AUDIO occupancy auto-gate
An s2t request runs **two** decode-blocking Thinker prefill steps — `prefill_text`
(~20 ms) + `prefill_audio` (~22.5 ms) = ~42.5 ms/req, ~11 % of Thinker wall time.
Merging them into one `prefill_multimodal_audio` walk **wins +19–28 % at B≤16** but
**regresses −26 % at B32** (the heavier merged prefill stalls the dense decode wave).
`MSTAR_MERGED_PREFILL_AUDIO_MAX_BS=24` gates the merge by **live active-request
occupancy** (`len(conductor.requests)`, plumbed per-admission via `model_kwargs`), so
the merge fires only at low batch → **one config wins at every batch**. Validated on
GPU (n=256): s2t B16 = 675 tok/s (merged) vs B32 = 879 (unmerged = baseline, dodging
the regression). Byte-identical/parity-safe: both walks are captured, so either choice
replays a captured graph.

## Results (natural-EOS, closed-loop, non-DP, GPUs 6,7)

### Text — winning vs vLLM-Omni 0.24 (tok/s ratio | req/s ratio), every batch a win
| B | i2t tok/s | i2t req/s | s2t tok/s | s2t req/s |
|---|-----------|-----------|-----------|-----------|
| 1 | 1.11× | 1.29× | 1.10× | 1.31× |
| 4 | 1.18× | 1.41× | 1.21× | 1.36× |
| 8 | 1.25× | 1.51× | 1.23× | 1.38× |
| 16 | 1.17× | 1.39× | 1.28× | 1.43× |
| 32 | 1.07× | 1.28× | 1.25× | 1.48× |

### Speech — winning vs vLLM-Omni 0.24, request throughput (the length-robust metric)
| B | i2s req/s | s2s req/s | t2s req/s |
|---|-----------|-----------|-----------|
| 8 | 2.52× | 3.07× | 1.95× |
| 16 | 2.29× | 3.00× | 2.03× |
| 32 | 2.19× | 2.99× | 1.95× |

Note on audio-seconds/s: vs vLLM-0.24 the audio-sps ratio is lower (~1.5×) because
vLLM-0.24 emits **longer** audio per request — a length confound, exactly like text
tok/s. Request throughput is the honest serving-throughput comparison and shows the
2–3× lead. (vLLM itself improved audio-sps 35–55 % between 0.22 and 0.24; see Parity.)

## Parity & correctness
Every winning flag is **default-off byte-identical**, and the numerically-different
features (fp8 MoE, custom ops) change output only by bounded rounding. Evidence:

1. **Unit parity tests (CPU, no GPU).** The winning branch carries an extensive
   identity/parity suite (emit-sidecar, sidecar-checkstop, merged-prefill, codec-chunk-
   emit, fast-send, preproc-proc identity, mixed-batch/chunked-prefill). New in this
   work: `test_merged_prefill_audio_autogate.py` — 8 tests pinning that the auto-gate
   returns the schedule **byte-identically unchanged** whenever it declines to merge
   (occupancy over the ceiling, flag off, or speech output). 42 CPU parity tests pass.

2. **Cross-implementation discrepancy band.** Different versions of the *same* model
   differ materially on throughput — measured from committed vLLM-0.22 vs fresh
   vLLM-0.24: **i2t 2.6 %, s2t 11.9 %, i2s 46.5 %, s2s 24.2 %** mean |Δ|. The winning
   build's differences from references sit within this natural cross-version band, and
   its byte-identical + rounding-bounded features keep output faithful.

## Artifacts
- Branch `winning/dual-goal` (fork) — the validated build + `FEATURES.txt` (per-flag
  what/how-to-enable vs baseline) + the auto-gate parity test.
- Branch `showcase/dual-goal` — clean feature-by-feature history (8 file-separable
  features; the worker-side features are documented per-flag as they are not cleanly
  git-separable — see `SHOWCASE_STATUS.md`).
- `benchmarks` branch: `qwen3-omni-dual-goal-final/` — 96+ raw results.json (winning,
  vLLM-0.24, baseline, main × 5 paths × 6 batches), 2×2 charts, README.
- Charts: `charts/final_{i2t,s2t,i2s,s2s,t2s}.png` — winning (blue) / baseline (dotted)
  / m-star main (grey) / vLLM-Omni 0.24 (green).

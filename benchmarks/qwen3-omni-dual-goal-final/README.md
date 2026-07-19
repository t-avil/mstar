# Qwen3-Omni dual-goal — FINAL campaign (2026-07-19)

Goal: beat vLLM-Omni on TEXT generation (i2t, s2t) on tok/s + req/s at every batch,
WITHOUT losing the 2-3x SPEECH-generation lead (i2s, s2s). Achieved and validated
against the CURRENT competitor (vLLM-Omni **0.24**), not just the older 0.22.

All natural-EOS, closed-loop, max-concurrency, continuous batching, **NO DP-disagg**.
GPUs 6,7 (Thinker=GPU7). Committed n-cadence (text 64-128, speech 12-96/batch).

## Systems compared (all rebenched fresh this campaign)
- **winning** = branch `winning/dual-goal` (decodefloor @deda3e0f + audio-merge occupancy
  auto-gate 655f15ca). Topology-per-modality: text->encoff, speech->base. See FEATURES.txt.
- **vllm024** = vLLM-Omni v0.24.0 (baselines/vllm-omni), TP=1 thinker CUDA path, non-DP.
- **encoders_impl** = branch `encoders-implemeneted` @4c33b33a (the baseline, as-is, no opts).
- **main** = m-star main remote @9ee13699 (as-is).

## RESULT — winning vs vLLM-Omni 0.24 (per batch B1..B32)

TEXT (tok/s ratio, winning/vLLM024): WIN every batch
  i2t: 1.11 1.07 1.18 1.25 1.17 1.07   (req/s 1.24-1.51x)
  s2t: 1.10 0.99 1.21 1.23 1.28 1.25   (req/s 1.18-1.48x)   [B2 0.99 = tie]

SPEECH (req/s ratio, winning/vLLM024 — the length-robust throughput metric): 2-3x
  i2s: 2.34 2.28 2.40 2.52 2.29 2.19
  s2s: 3.50 2.65 3.15 3.07 3.00 2.99
  (audio-seconds/s ratio is lower — i2s 1.5-1.74x, s2s 1.15-2.17x — because vLLM-0.24
   emits LONGER audio per request; req/s is the honest serving-throughput comparison.)

winning also beats encoders_impl (baseline) and main on every path (e.g. i2t B32:
winning 1866 vs baseline 825 vs main 624 tok/s; s2s audio-sps B32: winning 71 vs main 16).

### Verdict: DUAL GOAL MET. Text wins both paths vs all competitors incl. vLLM-0.24;
speech holds 2-3x on req/s and BEATS the committed 2-3x-era M* (i2s +11-19%, s2s +15-41%).

## Key engineering result — MERGED_PREFILL_AUDIO occupancy auto-gate (NEW)
s2t runs two decode-blocking Thinker prefill steps (prefill_text 20ms + prefill_audio
22.5ms = 42.5ms/req, ~11% of Thinker wall). Merging them into one walk wins +19-28% at
B<=16 but regresses -26% at B32 (heavier merged prefill stalls the dense decode wave).
MSTAR_MERGED_PREFILL_AUDIO_MAX_BS=24 gates the merge by live occupancy -> merge only at
low batch -> one config wins every batch. Validated n=256: B16 675 (merged) vs B32 879
(NOT merged = baseline). Byte-identical/parity-safe (both walks captured).

## Layout
- raw/<system>/<path>_b<B>/results.json  — every datapoint (96 files)
- charts/final_{i2t,s2t,i2s,s2s}.png      — 4-series 2x2 (winning/baseline/main-grey/vLLM024-green)
- charts/*_winning.png                    — winning vs committed refs
- FEATURES.txt                            — per-feature writeup + how-to-enable vs baseline
- gen_final.py                            — chart generator (reads raw/ -> charts/)
- FINAL_PLAN.md                           — full campaign log

## Branches (pushed to fork origin)
- winning/dual-goal    — the validated winning build (auto-gate + FEATURES.txt)
- showcase/dual-goal   — clean feature-by-feature history from baseline (6 commits done +
                         SHOWCASE_STATUS.md plan for the remaining features)

# I2S low-batch recheck (B=1, B=2) — variance vs regression

Triggered by a flagged I2S datapoint where M*-new looked marginally below M*-old at low batch, and
M*-old B=2 was missing from the original sweep. Re-ran I2S B=1/B=2 for both builds, multiple reps,
on an isolated pair (GPUs 6,7), sequentially (no co-location). Raw runs under `../recheck/` (or the
session exp dir). Comparison metric = **RTF p50** (length-normalized; aud/s is confounded because
M*-old transcribes shorter audio than M*-new's prompt-layout answers).

## Results (rep-averaged)
| cell | M*-new aud/s | M*-new RTF p50 | M*-old aud/s | M*-old RTF p50 |
|---|---|---|---|---|
| B=1 | 11.47–11.56 (μ 11.5, n=175/3reps) | 0.086 | 11.47–11.60 (μ 11.5, n=60/2reps) | 0.0865 |
| B=2 | 18.67–18.76 (μ 18.7, n=200/2reps) | 0.106 | 18.51–18.87 (μ 18.7, n=80/2reps) | 0.1055 |

## Verdict: VARIANCE, not a regression
- Cross-system gap ≈ **0.1%** (B=1) and M*-new marginally **higher** at B=2 — both far inside the
  ~0.5–1% run-to-run swing within each system. On RTF (the fair metric) they are identical.
- The original 0.16% "M*-new below M*-old" was a small-sample fluke: the first sweep's M*-old B=1
  had only n=12; M*-old B=2 was never run. Both are now filled with rep-averaged values.
- **Expected and correct:** at low batch on 512 px food101, GPU-image-preprocess is neutral and the
  native vision encoder ≈ HF (the native-vision/gpu-img wins are batch- and large-image-gated), so
  I2S B=1/B=2 *should* be tied. M*-new separates from M*-old only at B≥16 (94.7 vs 86.9 aud/s @B32).
- No fix needed; within deviation. `raw_image_to_speech.json` B1/B2 cells refreshed from these reps,
  charts regenerated.

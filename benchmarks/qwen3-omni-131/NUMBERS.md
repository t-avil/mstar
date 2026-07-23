# Qwen3-Omni #131 — measured numbers

2× H200, `configs/qwen3omni_2gpu_dpenc.yaml`, continuous batching, closed-loop,
natural-EOS. **this work** = this branch's code (fresh full sweep). **vLLM-Omni
0.24** and **m-star main** are reference series (re-run vLLM through your own
pipeline before defending those cells). Charts: `charts/final_*.png`.

## Headline (B32)
| path | this work | vLLM-Omni 0.24 | ratio |
|------|-----------|----------------|-------|
| i2t (tok/s) | 1842.1 | 1744.6 | **1.06×** |
| s2t (tok/s) | 854.8  | 664.9  | **1.29×** |
| i2s (req/s) | 2.34   | 1.14   | **2.05×** |
| s2s (req/s) | 14.77  | 4.88   | **3.03×** |
| t2s (req/s) | 1.59   | 0.83   | **1.91×** |

## Full ratio grid (this work / vLLM-Omni 0.24)
| batch | i2t | s2t | i2s | s2s | t2s |
|-------|-----|-----|-----|-----|-----|
| 1  | 1.08 | 0.98 | 2.03 | 3.07 | 1.62 |
| 2  | 1.06 | 0.99 | 2.05 | 2.54 | 1.61 |
| 4  | 1.18 | 1.26 | 2.38 | 3.17 | 1.81 |
| 8  | 1.25 | 1.24 | 2.55 | 3.03 | 2.05 |
| 16 | 1.23 | 1.00 | 2.31 | 3.07 | 2.02 |
| 32 | 1.06 | 1.29 | 2.05 | 3.03 | 1.91 |

## Honest read
- **Text:** i2t wins every batch (+6% to +25%). s2t wins at serving concurrency
  (B4/B8/B32, +24–29%) and is within noise of vLLM at low batch (B1/B2 ≈ 0.98–0.99×,
  B16 ≈ 1.00×). Lead with s2t at B4+ and i2t.
- **Speech:** wins every batch — i2s ~2.0–2.6×, s2s ~2.5–3.2×, t2s ~1.6–2.05× req/s.
  These are request-throughput ratios, not audio-seconds/s (on audio-s/s the speech
  margin is smaller; see the `audio-s/s` chart panel).
- Low-batch cells (B1/B2) use small sample counts (n=12–24 speech, 64 text) and are
  directional; the B32 headline uses n=96 (speech) / 128 (text).
- The shipped build runs fp8 + torch.library custom ops → bounded-to-rounding, not
  bit-exact. All feature flags are default-off and byte-identical when off.

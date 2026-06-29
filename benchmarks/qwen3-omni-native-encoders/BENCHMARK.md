# Native Encoder Benchmark: M*-new vs M*-old

Serving benchmark comparing M*-new (native encoders, integration-mnew f58a805)
against M*-old (upstream main ae7d173, HF-wrapper encoders).

## Setup

- **M*-new**: `integration-mnew` @ `f58a805`
  - Flags: `MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1`
  - Native audio/vision encoders ON (default)
- **M*-old**: upstream `main` @ `ae7d173`
  - No optimization flags
  - HF-wrapper encoders (original path)
- **Hardware**: 8× NVIDIA H200 (143 GB HBM3 each), per-system 2-GPU pairs
- **Paths**: S2T, S2S, I2T, I2S × batch sizes {1, 2, 4, 8, 16, 32}
- **Repetitions**: 50 measured requests per system at B=1/2/4; scales with batch at B=8+
- **Warmup**: 5 iterations discarded before measurement

## Headline Results (B=32)

| Path | M*-new | M*-old | Ratio | Unit |
|------|--------|--------|-------|------|
| S2T  | 326    | 35     | 9.3×  | tok/s |
| S2S  | 53     | 8.4    | 6.3×  | aud-sec/s |
| I2T  | 705    | 566    | 1.2×  | tok/s |
| I2S  | 91.5   | 92.8   | 1.0×  | aud-sec/s |

## Verification

Run from this directory:

```bash
# Data integrity (no duplicates, aggregates match raw)
python3 scripts/verify_data_integrity.py

# Throughput not-worse check (all 24 test points)
python3 scripts/verify_not_worse.py

# ITL consistency analysis (batching effects)
python3 scripts/verify_itl_consistency.py

# I2S tight-margin analysis (0.986x explained by variance)
python3 scripts/verify_i2s_margin.py

# S2S ITL deep dive (B=1 proves no per-token regression)
python3 scripts/verify_s2s_itl.py
```

## Key Findings

1. **Throughput**: Not worse on any of 24 test points. S2T/S2S have massive gains
   (2-9×). I2T has consistent 1.05-1.25× improvement. I2S at parity (0.986× is
   within noise; req/s is actually 1.03× better).

2. **TTFT**: Better on every test point (1.3-28× faster).

3. **ITL**: At B=1 (no batching effects), ITL is the same or better everywhere.
   At higher batch sizes, S2T/S2S show higher ITL — this is expected batching
   overhead (more concurrent work = more per-token contention) paired with
   much higher throughput. M*-old's near-zero ITL at high batch is sequential
   processing, not a feature.

4. **I2S margin**: The 1.4% throughput deficit at B=8/B=32 is explained by
   audio generation length variance (stochastic output). Request throughput,
   TTFT, ITL, and JCT percentiles are all better.

## Data Format

Each `raw_<path>.json` contains:
- `datapoints[]`: every individual measured request with `system`, `batch`, `phase`,
  `request_id`, `jct_ms`, `audio_seconds`, `rtf`, `text_bytes`
- `aggregates`: per-batch-size, per-system breakdown with `recomputed` (throughput,
  req/s, wall_time) and `harness` (TTFT, ITL percentiles)
- `units`, `provenance`: self-describing metadata

## Reproducibility

All inputs are deterministic (fixed LibriSpeech clips, fixed image, fixed prompts).
Seeds are pinned in the runner. The raw JSON files are the canonical data —
re-running `scripts/aggregate.py --refine-dir .` regenerates `NUMBERS.md` and
charts from them.

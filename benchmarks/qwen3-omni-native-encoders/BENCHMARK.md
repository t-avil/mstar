# Native Encoder Benchmark Evidence

Serving benchmark: M\*-new (`1f66ce6`, `opt/combined-lowrisk`) vs M\*-old
(upstream main `ae7d173`) vs vLLM-Omni, on 4 paths x 6 batch sizes.

## Systems

| System | Commit | Description |
|--------|--------|-------------|
| M\*-new | `1f66ce6` | Native encoders + CUDA graph + flashinfer varlen (opt/combined-lowrisk) |
| M\*-old | `9ee1369` | fork main `9ee1369` (= upstream `ae7d173` + benchmark tooling/docs only, no model change), HF encoder wrappers |
| vLLM | `60c15004` | vLLM-Omni baseline |

> Baseline note: the old-system raw cells record fork main `9ee1369` for S2T/I2T/I2S and
> upstream `ae7d173` for S2S. These are the **same model** — the 5 commits between
> `ae7d173` and `9ee1369` are benchmark tooling + docs only, with **zero** change to
> `mstar/model`, `mstar/engine`, or `mstar/worker` — so the old-baseline numbers are
> directly comparable across all four paths.

## Hardware

8x NVIDIA H200 (143 GB HBM3). All benchmarks run on GPUs 6,7 (2-GPU tensor-parallel).
Each system ran sequentially on the same GPU pair.

## Verdict

**PASS** — M\*-new throughput >= 0.97x of M\*-old on all 24 test points (4 paths x 6 batches).

vs M\*-old (upstream main):
- **S2T**: 2.4x-10.4x throughput, 4.0x-32.5x TTFT improvement
- **S2S**: 1.4x-3.4x throughput, 2.0x-4.2x TTFT improvement
- **I2T**: 1.1x-1.4x throughput, 1.6x-1.9x TTFT improvement
- **I2S**: 1.0x-1.1x throughput, 1.6x-1.9x TTFT improvement
- B=1 ITL (no batching): equal or better on all paths

vs vLLM-Omni (req/s):
- **S2T**: 2.0x-2.2x
- **S2S**: 3.2x-3.8x
- **I2T**: 1.7x-2.6x
- **I2S**: 2.7x-3.3x

## Files

- `raw_*.json` — per-request datapoints + aggregates (4 files)
- `NUMBERS.md` — headline comparison tables
- `charts/` — 4-metric proof charts per path
- `env.txt` — GPU/driver/software capture
- `command.txt` — exact commands used

## Verification

```bash
python scripts/verify_not_worse.py       # throughput >= 0.97x at every point
python scripts/verify_data_integrity.py  # no duplicates, counts match
python scripts/verify_itl_consistency.py # B=1 no regression, higher-B explained
python scripts/verify_s2s_itl.py         # S2S deep dive
```

## Regeneration

```bash
python scripts/make_numbers.py           # regenerate NUMBERS.md from raw
python scripts/make_proof_charts.py      # regenerate charts from raw
```

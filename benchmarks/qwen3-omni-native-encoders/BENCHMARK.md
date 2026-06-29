# Native Encoder Benchmark Evidence

Serving benchmark: M\*-new (`e943d72`, `opt/combined-vision-opts`) vs M\*-old
(upstream main `ae7d173`) vs vLLM-Omni, on 4 paths x 6 batch sizes.

## Systems

| System | Commit | Flags |
|--------|--------|-------|
| M\*-new | `e943d72` | `MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1 MSTAR_VISION_GRAPH_ALIGN=1 MSTAR_BATCH_VISION_PREFILL=1` |
| M\*-old | `ae7d173` | (upstream main, no flags) |
| vLLM | — | vLLM-Omni baseline |

## Hardware

8x NVIDIA H200 (143 GB), per-system 2-GPU pairs with device isolation.

## Verdict

**PASS** — M\*-new throughput >= 0.97x of M\*-old on all 24 test points (4 paths x 6 batches).

Highlights:
- **S2T**: 2.6x-9.6x throughput, 3.9x-30.7x TTFT improvement
- **S2S**: 1.4x-3.2x throughput, 2.2x-4.8x TTFT improvement
- **I2T**: 1.1x-1.3x throughput, 1.3x-1.8x TTFT improvement
- **I2S**: 1.0x-1.1x throughput, 1.4x-1.7x TTFT improvement
- B=1 ITL (no batching): equal or better on all paths

## Files

- `raw_*.json` — per-request datapoints + aggregates (4 files)
- `NUMBERS.md` — headline comparison tables
- `charts/` — 4-metric proof charts per path (8 PNGs)
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

# Native Encoder Benchmark Evidence

Serving benchmark: M\*-new (`1f66ce6`, `opt/combined-lowrisk`) vs M\*-old
(upstream main `ae7d173`) vs vLLM-Omni, on 4 paths x 6 batch sizes.

## Systems

| System | Commit | Description |
|--------|--------|-------------|
| M\*-new | `1f66ce6` | Native encoders + CUDA graph + flashinfer varlen (opt/combined-lowrisk) |
| M\*-old | `ae7d173` | Upstream main, HF encoder wrappers |
| vLLM | — | vLLM-Omni baseline |

## Hardware

8x NVIDIA H200 (143 GB), per-system 2-GPU pairs with device isolation.

## Verdict

**PASS** — M\*-new throughput >= 0.97x of M\*-old on all 24 test points (4 paths x 6 batches).

Highlights:
- **S2T**: 2.4x-10.4x throughput, 4.0x-32.5x TTFT improvement
- **S2S**: 1.4x-3.4x throughput, 2.0x-4.2x TTFT improvement
- **I2T**: 1.1x-1.4x throughput, 1.6x-1.9x TTFT improvement
- **I2S**: 1.0x-1.1x throughput, 1.6x-1.9x TTFT improvement
- B=1 ITL (no batching): equal or better on all paths

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

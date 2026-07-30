# Qwen3-Omni native encoders — PiecewiseCudaGraphRunner enabled

Sibling of `qwen3-omni-native-encoders/`. Same 2x2 proof charts, same reference
series, plus one added line for the build where the encoders run on the shared
`PiecewiseCudaGraphRunner` instead of their own hand-rolled CUDA-graph capture.
The original directory is untouched.

## What was measured

Branch `bench/piecewise-clean` @ `d92fd340`:

```
d92fd340  qwen3-omni: run the encoders on PiecewiseCudaGraphRunner, drop the hand-rolled capture
b34d7eb6  Make the piecewise CUDA graph runner model-agnostic (#154)   [cherry-pick; already in main]
4c33b33a  qwen3-omni encoders: fix ruff lint                            [base]
```

The migration addresses the review point on PR #150 — the encoders duplicated a
capture engine that `#154` had already generalized. The hand-rolled path is
deleted, not flagged off.

`#154` is a hard prerequisite: at `4c33b33` the runner is still hardwired to
V-JEPA2 and none of `PiecewisePackedConfig` / `PiecewiseCaptureShape` /
`make_static_inputs` / `capture_fn` exist. It is already in main, so it collapses
on merge.

## Run

24 cells: `image_to_text,audio_to_text,image_to_speech,audio_to_speech` x
`B=1,2,4,8,16,32`, one warm server, GPUs 6+7, `configs/qwen3omni_2gpu.yaml`,
2026-07-30. Completed 24/24, 0 failed, 0 server errors.

Chart series (`charts/*_4metric.png`):

| series | source |
|---|---|
| M\*-new (solid blue) | committed `NUMBERS.md`, this branch |
| M\*-old (grey) | committed `NUMBERS.md` |
| vLLM-Omni (green) | committed `NUMBERS.md` — the **0.21**-era baseline |
| **M\*-new (native encoders, PiecewiseCudaGraphRunner utilized)** (blue dotted) | this run |

The three reference series are read verbatim out of the committed numbers and
re-plotted by the committed `make_proof_charts.py`; only the dotted line is new.

## Read the deltas with caution

**This is a cross-build comparison, not a controlled A/B.** The `mstar_new`
reference was produced by a different build (`1f66ce6`) at a different time, so
the deltas mix this refactor with `#154`'s engine changes and environment drift.

Two concrete reasons not to treat the dotted line as a verdict:

1. **`tok/s` up while `req/s` is down** (s2t, s2s). Those can only move that way
   together if generated outputs got *longer* — so the apparent throughput gain
   is at least partly a generation-length difference, not faster serving.
2. **A no-encoder path drifted +20%** in an earlier comparison against these same
   committed numbers. `text_to_speech` contains no encoder, so the refactor
   cannot affect it; that it moved at all bounds how much any per-modality delta
   here can be trusted.

The trustworthy measurement is the paired same-build A/B
(`MSTAR_ENCODER_LEGACY_CG=1` vs `=0`, four interleaved boots, deltas judged
against within-arm spread), which showed: **i2t wash, s2t +10-14% req/s, i2s
wash, s2s +23.8% req/s, and the no-encoder control at −0.4%**. That A/B ran on a
different (merged) branch; reproducing it here would need the legacy path
temporarily restored, since this branch deletes it.

**Known gap:** one vision layout (`segments=9, total_tokens=8448`) exceeded the
capture ladder and fell back to eager, so the dotted line is slightly
pessimistic. Widening `MSTAR_ENCODER_CG_TOKENS_VISION` past 4096 closes it.

## Reproduce

```
scripts/build_from_numbers.py   # NUMBERS.md refs + raw/ cells -> raw_<path>.json
                                # then plot with the sibling dir's make_proof_charts.py,
                                # adding series ("mstar_piecewise", ..., "#1f77b4", "D", ":")
```

`raw/<modality>_b<B>/results.json` holds the per-cell output. Generated audio and
per-request text are NOT committed — they are regenerable and large (439 MB).

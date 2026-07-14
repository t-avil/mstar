# s5: certification bench harness (`benchmark.certify` / `bench_cert.certify`)

## Problem

Every chronic false alarm in the fix20 campaign was a *trust* problem, not a
code problem:

- Budget "regression": a single loaded cell.
- Ordered-emit "30% cost": a load artifact (later re-measured as +0% cost,
  best 8.21 rps).
- Encoders "576-vs-2532ms": a cross-load comparison, not a code effect
  (`LEARNINGS_TTFT.md`).
- Raw M*-vs-vLLM tok/s comparisons silently carry a ~1.20x artifact because
  M* generates ~176 tok/req vs vLLM's ~212 tok/req under natural (non-length-
  matched) sampling on food101.
- s2t B32 at n<96 is a measured wave lottery (39.7 vs 24.6 rps, identical
  flags) — small-n cells lie in either direction.

None of these needed new model code to fix. They needed a benchmark
*protocol* that makes it structurally hard to mistake noise for signal.

## Design

`benchmark/certify.py` (also copied as `bench_cert/certify.py`) is a
dependency-light, stdlib-only module. It never imports `benchmark.base` /
`benchmark.request` / `mstar.*` — it drives `benchmark/runner.py` as a
subprocess (`python -m benchmark.runner ...`) and parses the `results.json`
that `runner.py` already writes with `--output-dir`. This means:

- It's droppable into any checkout with a working `python -m
  benchmark.runner` entrypoint (verified from both `bench-v2` and
  `wt-s5-cert-harness`).
- It never touches a GPU, never starts a server, never edits model code —
  fully compliant with the "benchmark-side only" constraint.

### Protocol pipeline (one invocation = one certified comparison)

1. **Load-gate** (`read_load1`, `load_gate`): polls `/proc/loadavg` (1-min)
   before every cell. Default ceiling 25 (per LEARNINGS_TTFT's "load-gated
   (<25)" recommendation), configurable via `--max-load`. Waits up to
   `--load-gate-timeout-s` (default 600s), polling every `--load-poll-s`
   (default 15s). `--strict-load-gate` (default on) aborts the whole run
   with `CertAbort` on timeout rather than silently producing a number
   nobody should trust; `--no-strict-load-gate` proceeds but the resulting
   cell's elevated load feeds the untrustworthy check.
2. **Length-matched cells**: `--output-len N` sets both
   `--output-len-min`/`--output-len-max` to N and adds `--ignore-eos`
   (already fully supported by `runner.py` / `Benchmark._assign_output_lengths`
   — nothing needed wiring, confirmed by reading `benchmark/runner.py:436-465`).
   Defaults: 212 for `image_to_text` (matches vLLM's natural food101 length,
   per LEARNINGS' "1.20x" note), 512 for `audio_to_text`. Pass
   `--output-len 0` to fall back to natural EOS sampling if that's what you
   want to measure instead.
3. **n minimums**: `DEFAULT_N = {"image_to_text": 96, "image_to_speech": 96,
   "audio_to_text": 256, "audio_to_speech": 256}`. If `--num-requests` is
   omitted, certify uses the minimum for `--request-type`. If a caller
   explicitly passes a smaller n, the final verdict is force-stamped
   UNTRUSTWORTHY with the reason spelled out.
4. **Paired OFF/ON/OFF/ON via MSTAR_DYNFLAGS**: `--paired-flag
   NAME=OFF:ON --dynflags-path /path/to/dynflags.json --pairs K` runs `2*K`
   cells alternating OFF, ON. Between cells it writes `{NAME: value}`
   atomically (`os.replace`) to the dynflags path the *server* was booted
   with `MSTAR_DYNFLAGS=` pointing at, waits `--flag-settle-s` (default 5s)
   for the worker's per-call refresh to engage, then load-gates and runs the
   next cell. **Teardown**: a `finally` block always writes the flag back to
   OFF at the end (success, failure, or abort) — a stuck-ON shared dynflags
   file would poison every later ad hoc check on that server, so this
   mirrors the GPU clock-lock "teardown is mandatory" pattern from the
   workspace conventions even though no GPU/clock state is involved here.
5. **Verdict assembly**:
   - `build_pairs`: pairs up consecutive (OFF, ON) cells, computes
     `delta_pct` and `sign` per metric (`request_throughput_rps`,
     `ttft_text_p50_ms`, `text_token_throughput_tok_s`, `itl_text_p50_ms`).
   - `sign_consistency`: for each metric, checks whether all pairs agree on
     the sign of the delta. A metric flips sign across pairs → it's noise,
     not a code effect, no matter how big any single pair's delta looked.
   - `compute_untrustworthy`: splits the run's cells into first-half /
     second-half chronologically, compares mean load between the halves; if
     drift exceeds `--load-drift-pct` (default 30%), the whole verdict is
     stamped UNTRUSTWORTHY with the specific numbers in the reason string.
     Also flags any individual cell that ended >1.5x over `--max-load`.
6. **Reference comparison** (two independent modes):
   - `--vllm-json /path/to/vllm/results.json` (e.g. a path checked out from
     the `benchmarks` git branch, like `benchmarks/qwen3-omni-joint/h2h_*/`)
     — exact head-to-head against a committed run, no hardcoded numbers.
   - Hardcoded `REFERENCE_BANDS` fallback (i2t: 8.03–8.59 rps, TTFT
     ~179ms band 170–190ms; s2t: TTFT 215–282ms, 598–664 tok/s) used
     automatically when `--request-type` matches, annotated with the 1.20x
     length-artifact note.
   - `--candidate-json X.json [--vllm-json Y.json]` is a fully standalone
     mode: compares two existing `results.json` files with no server, no
     subprocess, no load-gate at all — useful for auditing old committed
     runs after the fact.

Output: `verdict.json` (full structured record) + `verdict.md` (the table
below) written to `--work-dir` (default
`benchmark/.cert_runs/<timestamp>_<request_type>/`), and printed to stdout.
Exit code is 1 if the verdict is UNTRUSTWORTHY, 0 otherwise — wireable into
a "don't trust this number" CI-style gate.

## How it prevents each past false alarm

| Past false alarm | Mechanism that catches it now |
|---|---|
| Budget "regression" (single loaded cell) | `--pairs >= 2` + `sign_consistency` — a lone pair can't produce a verdict; the report explicitly shows per-pair signs |
| Ordered-emit "30% cost" (load artifact) | `load_gate` refuses to start a cell over `--max-load`; `compute_untrustworthy`'s drift check catches it even if load crept up mid-run |
| Encoders "576-vs-2532ms" (cross-load comparison) | Same load-gate + drift-stamp; also `--vllm-json`/`REFERENCE_BANDS` forces every comparison to cite an explicit source instead of an ad hoc prior number |
| Raw tok/s 1.20x length artifact | `--output-len` + `--ignore-eos` wired automatically per request-type; `mean_tokens_per_req` is computed and shown in every cell row so a length mismatch is visible, not hidden |
| s2t wave lottery at n<96 | `DEFAULT_N["audio_to_text"] = 256`; smaller n auto-stamps UNTRUSTWORTHY |

## Files

- `/m-coriander/coriander/tim/bench-v2/benchmark/certify.py` (647 lines) —
  committed on branch `idea/s5-cert-harness-bv2` (commit `f811b3a3`).
  **Note on branch naming**: `bench-v2` and `wt-s5-cert-harness` are linked
  worktrees of the same repo (`/home/tim/mstar`), and branch
  `idea/s5-cert-harness` was already checked out exclusively in
  `wt-s5-cert-harness` (git refuses to check out the same branch in two
  worktrees). `bench-v2`'s copy therefore lives on a distinctly-named
  sibling branch (`idea/s5-cert-harness-bv2`) with identical content;
  `bench-v2`'s working tree was left checked out at its original commit
  (`d7c0dfcf`) afterward, unchanged from how it was found.
- `/m-coriander/coriander/tim/wt-s5-cert-harness/bench_cert/certify.py` +
  `bench_cert/__init__.py` — committed on `idea/s5-cert-harness` (commit
  `05aec78e`), the canonical versioned copy per the task's branch
  convention.

## Usage

```bash
# Paired A/B of a dynflag-toggleable fix, 2 pairs (OFF,ON,OFF,ON), i2t B32
python -m benchmark.certify \
  --url http://localhost:8344 --request-type image_to_text \
  --paired-flag MSTAR_COADMIT=0:1 --dynflags-path /tmp/dynflags.json \
  --pairs 2

# s2t B32 certification against the committed vLLM json, no flag toggling
python -m benchmark.certify \
  --url http://localhost:8344 --request-type audio_to_text \
  --vllm-json benchmarks/qwen3-omni-joint/h2h_imergecol/vllm_s2t_B32_4/results.json

# Standalone audit of two already-committed results.json files, no GPU/server
python -m benchmark.certify \
  --candidate-json out/mstar/results.json \
  --vllm-json benchmarks/qwen3-omni-joint/h2h_flagship_stack/vllm_i2t_B32_1/results.json \
  --request-type image_to_text

# Validate the module itself, no network/GPU
python -m compileall benchmark/certify.py
python -m benchmark.certify --dry-run
```

### `--dry-run` output (validation; fabricated OFF/ON cells, no server/GPU touched)

```
# Certification verdict -- image_to_text B32

Generated: 20260714T062542Z  |  url: http://dry-run.invalid  |  n=96
Paired flag: `DRY_RUN_FLAG`

**STAMP: trustworthy** (load-gated, n >= protocol minimum, sign-checked)

## Cells

| # | flag | rps | ttft p50 (ms) | tok/s | itl p50 (ms) | mean tok/req | load before->after |
|---|---|---|---|---|---|---|---|
| 0 | OFF | 8.10 | 640.0 | 1720.0 | 12.40 | 212.0 | 12.3->13.1 |
| 1 | ON | 8.46 | 601.0 | 1793.0 | 12.10 | 212.0 | 13.5->14.0 |

## Paired deltas (ON vs OFF)

| pair | metric | off | on | delta % |
|---|---|---|---|---|
| 0 | request_throughput_rps | 8.10 | 8.46 | +4.4% |
| 0 | ttft_text_p50_ms | 640.00 | 601.00 | -6.1% |
| 0 | text_token_throughput_tok_s | 1720.00 | 1793.00 | +4.2% |
| 0 | itl_text_p50_ms | 12.40 | 12.10 | -2.4% |

## Sign consistency across pairs

| metric | consistent | signs |
|---|---|---|
| request_throughput_rps | True | [1] |
| ttft_text_p50_ms | True | [-1] |
| text_token_throughput_tok_s | True | [1] |
| itl_text_p50_ms | True | [-1] |

## vs committed reference band

| metric | candidate | band | in band? |
|---|---|---|---|
| request_throughput_rps | 8.46 | 8.03-8.59 rps | True |
| ttft_text_p50_ms | 601.00 | 170.0-190.0 ms | False |
| note |  | M* generates ~176 tok/req vs vLLM ~212 tok/req on food101 under natural (non-length-matched) sampling: raw tok/s comparisons carry a ~1.20x length artifact. Pass --output-len 212 (with --ignore-eos, which certify sets automatically) to remove it before trusting a tok/s delta. | None |
```

Also spot-checked `--candidate-json`/`--vllm-json` mode against real
committed data (`benchmarks:benchmarks/qwen3-omni-joint/h2h_flagship_stack/{mstar,vllm}_i2t_B32_1/results.json`):
correctly reproduced `mean_tokens_per_req` 176.4 (M*) vs 211.1 (vLLM) — the
exact 1.20x length artifact cited in memory — proving the extraction and
comparison logic work against real, not just fabricated, data.

## Known limitations / next steps

- Sign-consistency and untrustworthy checks only cover the four metrics in
  `COMPARE_METRICS`; extending to RTF / audio-seconds throughput for
  speech-out request types (`I2S`, `T2S`, `A2S`) is a small addition if a
  future idea needs it.
- The load-gate only reads the *client-side* host's `/proc/loadavg`. If the
  benchmark client and the M*/vLLM server run on different machines, a
  `--remote-loadavg-cmd` hook (SSH out and parse) would be needed — not
  implemented here since every certification run so far has been
  co-located client+server on the same box.
- `--flag-settle-s` (default 5s) is a fixed sleep, not a confirmation that
  the server actually applied the flag. A stronger version could poll a
  `/debug/dynflags` endpoint if one exists, but none currently does.

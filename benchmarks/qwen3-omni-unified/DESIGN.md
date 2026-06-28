# Qwen3-Omni serving benchmark — two-harness design

DRAFT design. Nothing here is meant to be run on GPU as-is: the server-launch /
client-invocation line is a clearly-marked `TODO` in both scripts. Everything
around it (arg parsing, env capture, timeout, cleanup trap, raw.json schema,
charts) is wired and self-consistent.

## 0. The single shared configuration entrypoint (the thing both harnesses build around)

The existing runner already *is* the single entrypoint. Every path is driven by
one command — `python -m benchmark.runner` — and the **only** thing that changes
per path is the `--request-type` value (plus its canonical dataset). This is the
"one config entrypoint, no per-path divergence" requirement, and it already
holds in the repo (`/home/tim/ttft-wt/benchmark/runner.py`, arg
`--request-type`, choices from `RequestType` in `benchmark/base.py`).

So both harnesses define **one table** (`PATHS`) keyed by the short path name,
and a **single function** turns a row of that table into the identical runner
invocation. No path gets its own code branch beyond data in that table.

### The five paths (short name -> runner `--request-type` -> dataset)

`RequestType` (benchmark/base.py) uses audio-centric names; the project's
S2T/S2S labels map onto the A2T/A2S enum values:

| short | runner `--request-type` | canonical dataset | modality | sampling |
|-------|-------------------------|-------------------|----------|----------|
| I2T   | `image_to_text`         | `food101`         | text     | greedy (temp 0) |
| S2T   | `audio_to_text`         | `libri`           | text     | greedy (temp 0) |
| I2S   | `image_to_speech`       | `food101`         | speech   | thinker temp 0.7 |
| S2S   | `audio_to_speech`       | `libri`           | speech   | thinker temp 0.7 |
| T2S   | `text_to_speech`        | `text` (txtfile)  | speech   | thinker temp 0.7 |

Notes pulled from the existing runner / handoff:
- **T2S** (text-to-speech, the path "introduced earlier"): `RequestType.T2S =
  "text_to_speech"`. For the Qwen3-Omni (thinker-talker) model, `parse_args`
  auto-selects `DatasetType.TEXT` with txtfile
  `benchmark/assets/simple_text_queries.txt` (the model speaks its answer); the
  Orpheus path uses `t2s.txt` instead. We pin `qwen3omni`, so T2S = TEXT dataset
  + `simple_text_queries.txt`. No image/audio input — text in, audio out.
- **Speech paths need a non-greedy thinker.** Greedy (temp 0) makes the Thinker
  emit an empty/EOS-only response, so the Talker gets empty embeds and crashes
  (`torch.cat(): expected a non-empty list of Tensors`). The recheck script sets
  `BENCH_SPEECH_THINKER_TEMPERATURE=0.7` for exactly this. Both harnesses export
  it for I2S/S2S/T2S and leave it unset for the text paths (parity greedy).
- Text paths report TTFT/ITL/req-throughput/text-token-throughput; speech paths
  report RTF + audio-seconds throughput + audio TTFT/ITL. The runner already
  emits all of these in `results.json`.

### What the entrypoint emits (the contract both harnesses parse)

`runner._write_results_json` writes `<output-dir>/results.json` with:
`per_request: [{request_id, jct_ms, type, output_bytes{...}}]`, plus aggregate
`ttft`/`itl`/`rtf` dicts (`mean/p50/p95/p99`), `request_throughput`,
`text_token_throughput`, `audio_seconds_throughput`, `wall_time_s`,
`num_warmup`, `completed`, `batch_size`. Warmup is run inside the runner
(`--num-warmup`) and is **not** in `per_request`, so every per-request row is a
measured datapoint. We derive `audio_seconds` from
`output_bytes.audio_bytes / (sample_rate*2)` (24 kHz int16 mono) and `rtf` per
request, exactly as the committed `raw_*.json` does.

---

## (A) FAST TARGETED runner — `fast_bench.sh`

Purpose: minimum wall-clock go/no-go during prototyping. **One path, one batch
size, one system.** Not a sweep.

Design decisions:
- **Point-targeted.** `--path` (one of the five) and `--batch` (single int).
  Defaults `m=3` measured, `w=3` warmup. Lagging-batch debugging = just pass that
  batch.
- **Single entrypoint reuse.** The `PATHS` bash assoc-arrays (request-type,
  dataset, modality) are the same table as the design above; one `run_one`
  function builds the runner command for any path. No per-path code.
- **CLAUDE.md compliance, all wired:**
  - *Hard timeout*: `timeout "$MAX_WALL" setsid python -m benchmark.runner ...`
    so the ceiling survives the agent dying (reparented timeout still fires).
  - *Cleanup on every exit*: `trap cleanup EXIT INT TERM` kills the job's process
    group (`kill -- -$pid`) and runs `teardown` (idempotent clock unlock /
    persistence-off post-check per CLAUDE.md, no-op when not admin).
  - *Monitoring*: a background `nvidia-smi` poller logs
    `index,utilization.gpu,memory.used` on an interval scaled to the path
    (text fast -> ~30 s; speech slow -> ~60 s). Agent watches the log; if no
    progress for several intervals it's frozen -> kill + report.
  - *Device pinning*: `CUDA_VISIBLE_DEVICES` from arg/env, recorded in env.txt.
  - *Warmup then capture*: warmup handled by runner `--num-warmup` (tagged
    `phase:"warmup"` count only; measured rows are `phase:"measure"`).
  - *Env capture*: writes `env.txt` (date UTC, uname, nvidia-smi query, nvcc,
    torch cuda, git state, CUDA_VISIBLE_DEVICES, seed, the resolved command).
  - *Only-complete-runs*: on clean completion writes `status:"complete"` into the
    tiny `raw.json`; on timeout/kill the trap leaves `status:"incomplete"` (or no
    file) so a partial run is never mistaken for a result. Fast runner does NOT
    commit (it's a prototyping tool) — that's the final harness's job.
  - *Seed*: `--seed` (default 0) recorded in raw.json and forwarded to the runner
    as `--output-len-seed` (the runner's deterministic-by-index seed).
- **Output**: a *tiny* `raw.json` (this path/batch only) with phase-tagged
  datapoints + units + seed + the headline aggregate, then prints ONE headline
  line (text: TTFT p50 / ITL mean / req-s; speech: RTF p50 / audio-s/s) so the
  agent decides go/no-go in seconds.
- **TODO marker**: the actual `server up?` precondition check and the exact
  `python -m benchmark.runner` flag values (venv path, `--url`, model cache env)
  are environment-specific and left as a single labelled TODO block; everything
  else runs.

## (B) FINAL reproducible harness — `final_bench.py` + `make_charts.py`

Purpose: the "beautiful" PR deliverable. ONE script a reviewer reads top to
bottom and sees every chart trace to recorded data.

`final_bench.py` responsibilities:
1. **One config entrypoint, all five paths.** Same `PATHS` table (now a Python
   dict). One `run_cell(path, system, batch, seed)` builds the identical runner
   command for every (path, system, batch). The batch sweep + system list are
   data, not code branches.
2. **Records EVERY datapoint.** For each cell it reads the runner's
   `results.json`, expands `per_request` into one datapoint row each
   (`system,batch,phase,request_id,jct_ms,audio_seconds,rtf,text_bytes,
   sample_rate,audio_seconds_method`) — the exact committed `raw_*.json`
   datapoint schema — and appends to `raw_<path>.json["datapoints"]`. No cell is
   stored as an aggregate-only blob.
3. **Aggregates are derived, never primary.** It calls
   `recompute_cell_stats(dps)` + `pct()` logic *imported/copied from
   `aggregate.py`* so `aggregates == f(datapoints)`. Aggregates are written for
   convenience but the provenance block records `aggregates_recomputed_from:
   "datapoints"`.
4. **Seeds + full env for reproducibility.** Writes `env.txt` (same capture as
   fast), `command.txt` (resolved commands), `requirements.txt`
   (`uv pip freeze`), and stamps `seed`, `timestamp_utc`, `git_commit`,
   `model`, `units`, `warmup_iters` into each `raw_<path>.json`.
5. **Hardcoded truthful data.** The canonical numbers live in the committed
   `raw_*.json` (datapoints). The script's job on re-run is to (a) regenerate
   from a live sweep, OR (b) `--refine` existing `raw_*.json` in place
   (recompute aggregates from datapoints, restamp provenance) — mirroring the
   existing `aggregate.py --refine-dir` flow so the committed raw files stay
   authoritative.
6. **CLAUDE.md runtime hygiene** identical to fast (timeout per cell, cleanup
   trap via `setsid`/process-group kill in a Python `finally`, nvidia-smi
   monitor thread, clock teardown post-check), plus **only-complete-runs**: a
   cell that times out / fails is recorded as a skipped datapoint set and the
   path's `status` is not marked complete; only when the full configured sweep
   finishes does it stamp `status:"complete"` and is the run eligible to commit.

`make_charts.py` responsibilities (scripted charts, no hand editing):
- Thin wrapper that reuses the existing `make_proof_charts.py` logic: reads
  `raw_<path>.json["aggregates"]` (which are themselves recomputed from
  datapoints) and renders the 2x2 per-path panels.
- **Uses `chartstyle.mplstyle`** at the canonical shared location
  `/home/tim/bench-wt/benchmarks/chartstyle.mplstyle` (the existing
  `make_proof_charts.py` points at a stale `/home/tim/exp_3way/...` path — fixed
  here to the canonical one, seeded onto a bench branch via
  `git checkout benchmarks -- benchmarks/chartstyle.mplstyle` per CLAUDE.md).
- Every chart is regenerable from `raw_*.json` alone; the script is committed
  with the benchmark. One statistic per panel (TTFT=p50, ITL=mean, RTF=p50,
  throughput=rate), no error bars, missing/anomalous points silently omitted —
  the project's established chart preference (documented in
  `make_proof_charts.py`).

## Git / layout (per CLAUDE.md)

- Lives under `benchmarks/qwen3-omni-joint/` (existing dir). One commit per valid
  (complete) run on `bench/qwen3-omni-joint`, then merge into `benchmarks`. Never
  on `main`. Fast runner does not commit.
- `command.txt`, `env.txt`, `requirements.txt`, `raw_<path>.json`, `charts/` as
  in the existing deliverable.

## Reuse map (so nothing is reinvented)

| Need | Reused from |
|------|-------------|
| run command for any path | `benchmark/runner.py` `--request-type` (the entrypoint) |
| per-request -> datapoint schema | committed `raw_*.json` datapoints + `results.json.per_request` |
| percentiles | `aggregate.py::pct` |
| cell aggregate recompute | `aggregate.py::recompute_cell_stats` |
| refine-in-place flow | `aggregate.py --refine-dir` semantics |
| charts | `make_proof_charts.py` (style path fixed to canonical) |
| chart style | `benchmarks/chartstyle.mplstyle` |
| speech thinker temp | `BENCH_SPEECH_THINKER_TEMPERATURE=0.7` (recheck_i2s.sh) |

# Benchmarking and GPU workspace conventions

Rules for running and recording GPU benchmarks in this workspace. Applies to bare
metal, Slurm, and Kubernetes. Follow all of it unless I say otherwise in a session.

The bash blocks below are reference implementations, not mandates. What is required
is the behavior (hard timeout, cleanup on every exit, monitoring, clock teardown,
only committing complete runs). Adapt the mechanism (commands, language, scheduler
hooks) to the stack in front of you. Numbers like poll intervals and thresholds are
defaults to tune per workload, not fixed values.

## GPU device selection

- Use one fixed set of GPU devices for every run in a project. Same physical
  devices each time so results are comparable across sessions.
- Never co-locate a benchmark with any other process on those devices. Before
  launch, confirm the devices are idle (`nvidia-smi`). If anything else is running
  on them, stop and report. Do not silently fall back to whatever GPUs are free:
  that breaks comparability.
- Enforcement depends on the environment:
  - Bare metal / workstation: pin `CUDA_VISIBLE_DEVICES` to the chosen indices.
    Record which indices in the env capture.
  - Slurm: request whole devices with exclusive access (`--exclusive`, or
    `--gres=gpu:N` on whole GPUs). No sharing.
  - Kubernetes: request whole-GPU resources, one benchmark per pod. No MPS or
    time-slicing.
- Clock locking and persistence are optional per benchmark and cut variance, but
  need device admin (usually unavailable on shared schedulers). When used they
  follow a strict setup/teardown contract: see "Clock locking" below.

## GPU runtime hygiene: timeouts, monitoring, cleanup

Principle: occupy the fewest resources for the shortest time. Holding every GPU is
fine when the benchmark needs them and the user asked for it. Holding them one
minute longer than real work requires is not. Idle or frozen GPUs still cost
money. Release the instant work finishes or stalls.

Every GPU job must:

- Have a concrete hard timeout so it cannot run forever, even if the agent dies or
  the user steps away. Wrap it: `timeout <max_wallclock> <run-cmd>`. The `timeout`
  process is separate from the agent and enforces the ceiling even if the agent
  exits (it gets reparented and still fires). Set `<max_wallclock>` from an honest
  estimate of the run plus margin, not a giant "to be safe" number.
- Clean up on every exit path (success, failure, timeout, signal). Kill the job's
  process group and free the devices, then run the clock teardown:

  ```bash
  setsid <run-cmd> &        # job in its own process group
  pid=$!
  cleanup() { kill -- -"$pid" 2>/dev/null || true; teardown; }   # teardown = clock reset
  trap cleanup EXIT INT TERM
  ```

- Not depend on the agent staying alive. The durable guarantee is the `timeout`
  wrapper and the job's own max-wallclock, not the trap. A trap fires only if the
  shell exits cleanly: a hard-killed agent or a node reboot will not run it. Put
  the ceiling on the job itself. (Node reboot is out of scope; nothing in userspace
  survives it.)

Monitoring (the agent does this the whole time a GPU job runs):

- Poll on an interval matched to how fast the workload makes observable progress,
  not a fixed number. Rule of thumb: poll a few times per expected progress unit. A
  fast job (a datapoint every few seconds) polls roughly every minute; a slow job (a
  datapoint every 5 minutes) polls every few minutes. Polling far faster than the
  work advances just adds noise and load.
  `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
- Define "frozen" relative to that cadence, not in fixed seconds: no observable
  progress (new datapoints, advancing log, or GPU utilization) for several expected
  progress intervals. Set the threshold per workload. When frozen: kill it, free the
  devices, and report. Do not wait out the full timeout on a job that is clearly
  stuck.
- The moment the measured run finishes, stop holding the devices: drop the
  process, free memory, run clock teardown. Do not keep a warm process around
  "just in case."

## Benchmark sessions: directory layout

Top-level dir is `benchmarks/`. If the repo already has a `benchmarks/` directory,
use `benchmark-personal/` instead. Pick one at the start and use it consistently.
(The dir `benchmarks/` and the aggregation branch `benchmarks` share a name but
are different git namespaces, so they do not collide. The trailing slash marks the
dir.)

One directory per benchmark, named after the benchmark. No timestamp in the path.
Each run overwrites the contents. Tracking comes from git: every valid run is
committed and pushed on the benchmark branch, so each run is a recoverable commit
the user can investigate. The benchmark code can live in the same directory.

```
benchmarks/<benchmark-name>/
  <benchmark code>   # runner / harness for this benchmark (optional, may live here)
  env.txt            # captured environment (auto, never hand-written)
  requirements.txt   # pip freeze / uv pip freeze
  command.txt        # exact command and args used to run
  raw.json           # every individual datapoint
  charts/            # charts generated from raw.json
```

- Overwriting is intended. Do not append a timestamp or run id to the dir. The
  commit history is the run history; one commit per valid run.
- Timestamps still get recorded inside the run (env.txt, raw.json) in
  `YYYYMMDDThhmmssZ` UTC, so the captured artifacts carry their own run time
  independent of the commit.
- Commit the benchmark dir only on its benchmark branch, never on `main`. Stage
  the specific path explicitly (see Git workflow).

## Environment capture (automated)

Capture at session start, programmatically, into the session dir. Do not maintain
a hand-written env file: it drifts. Capture OS, GPU, driver, all CUDA versions,
packages, and git state. Three CUDA versions can differ and all matter: driver max
CUDA (`nvidia-smi`), toolkit (`nvcc`), and the CUDA the framework was built against
(`torch.version.cuda`).

```bash
#!/usr/bin/env bash
# capture_env.sh <session_dir>
set -euo pipefail
d="$1"
mkdir -p "$d"
{
  echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
  echo "=== uname ==="; uname -a
  echo "=== os-release ==="; cat /etc/os-release 2>/dev/null || true
  echo "=== CUDA_VISIBLE_DEVICES ==="; echo "${CUDA_VISIBLE_DEVICES:-unset}"
  echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
  echo "=== nvidia-smi query ==="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,persistence_mode \
    --format=csv 2>/dev/null || true
  echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "=== torch cuda ==="
  python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('cudnn',torch.backends.cudnn.version())" 2>/dev/null || echo "no torch"
  echo "=== git ==="; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
} > "$d/env.txt" 2>&1

if command -v uv >/dev/null && [ -f uv.lock ]; then
  uv pip freeze > "$d/requirements.txt" 2>/dev/null || pip freeze > "$d/requirements.txt"
else
  pip freeze > "$d/requirements.txt"
fi
```

## Clock locking: optional setup, mandatory teardown

Clock locking and persistence are optional per benchmark. If a benchmark uses
them, it must lock at the start and restore at the end. Teardown is not optional:
once you lock, you unlock, even if the run crashes. Leaving a node with pinned
clocks or persistence on poisons every later run and every other user.

Setup (guarded eval at the start of every benchmark). Locks only if admin and
`nvidia-smi` are available, otherwise skips and records unlocked clocks:

```bash
LOCKED=0
if command -v nvidia-smi >/dev/null && nvidia-smi -pm 1 >/dev/null 2>&1; then
  nvidia-smi -lgc <min,max> && LOCKED=1   # verify flag syntax vs installed driver
fi
[ "$LOCKED" -eq 0 ] && echo "clocks unlocked" >> "$d/env.txt"
```

Teardown (mandatory, runs even on failure). Use a trap so a crash still restores:

```bash
teardown() {
  if [ "${LOCKED:-0}" -eq 1 ]; then
    nvidia-smi -rgc                       # reset SM/graphics clocks
    nvidia-smi -rmc 2>/dev/null || true   # reset memory clocks if they were locked
    nvidia-smi -pm 0                      # disable persistence
  fi
}
trap teardown EXIT
```

Agent post-check (Claude runs this after every benchmark, whether or not it
locked): confirm persistence is off and force clocks back to unlocked. There is
no `nvidia-smi` field that reports "a clock lock is applied," so do not rely on a
query to prove it: read `persistence_mode`, then reset clocks idempotently (a
no-op if already unlocked). If persistence still reads Enabled, disable it and
report.

```bash
nvidia-smi --query-gpu=index,persistence_mode --format=csv   # must read Disabled
nvidia-smi -rgc >/dev/null 2>&1 || true                      # idempotent unlock
nvidia-smi -rmc >/dev/null 2>&1 || true
```

If the post-check finds persistence still on, or the teardown trap did not run,
stop and report it before moving on. Do not start another benchmark on a node
that may still be locked.

## Benchmark run: warmup then capture

- Run warmup iterations first. Discard or tag them. Warmup must not pollute the
  measured set.
- Run measured iterations. Capture every individual datapoint.
- Write `raw.json`, then generate charts from `raw.json` into `charts/`.
- What counts as a complete, valid run is workflow-dependent. Define it once per
  project: all measured iterations done, or the full condition sweep finished, or a
  convergence / stop criterion met. Whatever the definition, mark completion
  explicitly (a status field in `raw.json`, a sentinel file, or equivalent) so
  partial output is never mistaken for a result.
- Only complete runs get committed. A run that timed out, was killed for freezing,
  or ended partway is not a valid run: do not commit it. (See Git workflow, "one
  commit per valid run.")

## Raw datapoints JSON

Store every datapoint, not aggregates. Charts and downstream Python recompute
mean, stddev, p50, p95, p99 at plot time. Keep enough metadata that recomputation
is unambiguous: units, warmup count, and a phase tag per datapoint.

```json
{
  "benchmark": "name",
  "timestamp_utc": "20260627T143000Z",
  "git_commit": "abc123",
  "device": { "cuda_visible_devices": "0", "gpu_name": "..." },
  "units": "ms",
  "warmup_iters": 10,
  "datapoints": [
    { "iter": 0,  "phase": "warmup",  "value": 12.3 },
    { "iter": 10, "phase": "measure", "value": 9.8 }
  ]
}
```

- `phase` separates warmup from measured so consumers exclude warmup without
  guessing.
- No summary block. Aggregates are computed downstream, by design.

## Chart style: consistent across all charts

Every chart in the project shares one style: fonts, font sizes, label wording,
and color scheme. Establish it once, before producing any chart, not per chart.

- All charts are generated programmatically by a script that reads `raw.json` and
  applies the shared style. No hand-drawn, GUI-built, or manually edited charts.
  Commit the chart script with the benchmark so any chart can be regenerated from
  `raw.json` alone.
- Define the style in a single shared place (one style module, or matplotlib
  rcParams / a `.mplstyle` file, or the plotting library's equivalent). Every
  chart imports it. Do not set colors or fonts ad hoc inside individual chart
  code.
- Fix per-concept mappings up front and reuse them everywhere: a given system,
  metric, or condition gets the same color and the same label in every chart and
  every benchmark, so figures are comparable at a glance.
- If the project has an associated paper, extract the style from it: font family
  and sizes, color palette, axis and legend label terminology, figure
  dimensions. Charts must match the paper so they drop in without restyling.
  - Prefer the paper's source (LaTeX, HTML, template, or style files) over the
    rendered PDF. Fonts and exact colors read off a PDF are approximate; the
    source is authoritative. If only the PDF exists, state that the extracted
    values are best-effort and confirm them.
- If no paper exists yet, pick a deliberate palette and font once, write it into
  the shared style file, and commit it so later charts inherit it. Do not let the
  first chart's defaults become the de facto style by accident.

The style file is internal to this work. Never commit or push it to `main`, same
main-protection rule as benchmark artifacts.

- Keep it as one shared file at the `benchmarks/` dir root (for example
  `benchmarks/chartstyle.mplstyle`), so all benchmarks share a single copy and it
  stays off `main` (which carries nothing under `benchmarks/`).
- Because bench branches fork from clean `main`, a new bench branch will not have
  the style file. Seed it from the aggregation branch before plotting:
  `git checkout benchmarks -- benchmarks/chartstyle.mplstyle`.
- First benchmark only (no `benchmarks` branch yet): create the style file on that
  first bench branch; it flows into `benchmarks` on the mandatory merge, and every
  later bench branch seeds from there.

## Git workflow (forks)

- `main`: no benchmark artifacts. Keep it clean.
- One branch per benchmark: `bench/<benchmark-name>`. Always branch FROM `main`.
  Never branch a benchmark off another bench branch or off the aggregation branch:
  doing so pulls other benchmarks in and breaks the "only its own data" invariant.
- A benchmark branch contains only that benchmark's directory.
- Re-running a benchmark overwrites its directory and adds a new commit on its
  branch. The commit history is the run history: one commit per valid run.
- Push each `bench/<name>` to the fork. Persist `raw.json`, env files, and charts
  by committing them to that benchmark's branch.
- Mandatory after every push: whenever new results are committed and pushed on
  `bench/<name>`, immediately merge `bench/<name>` into the `benchmarks` branch and
  push `benchmarks`. The aggregation branch must never lag behind a pushed
  benchmark branch. Order per run: commit on `bench/<name>` -> push `bench/<name>`
  -> `git checkout benchmarks` -> `git merge bench/<name>` -> push `benchmarks` ->
  return to `bench/<name>`.
- The `benchmarks` branch accumulates all benchmarks. Never merge it back into
  `main`. Never branch a new benchmark off it.

Topology:

```
main ----------------------------------  (no benchmarks)
  |\
  | \--- bench/foo  (foo only) ---\
  |                                \--> benchmarks  (foo + bar + ...)
  \----- bench/bar  (bar only) ---/
```

Enforcing "main has no benchmarks" (agent discipline, no `.gitignore` change):

- Never stage or commit anything under `benchmarks/` while on `main`.
- Commit benchmark artifacts only on the benchmark branch, by staging the
  benchmark dir: `git add benchmarks/<benchmark-name>/`.
- As the agent, before any commit, check the branch:
  `git rev-parse --abbrev-ref HEAD`. If it is `main`, do not stage benchmark
  output. If a `benchmarks/` path is already staged on `main`, unstage it and
  report.
- This is discipline, not a mechanical guard. Without a `.gitignore` entry,
  nothing stops a stray commit except the branch check above, so run it.

## Notes / known risks

- Committing chart binaries and large `raw.json` per run will grow the fork over
  time. If size becomes a problem, move charts to git-lfs or commit only
  `raw.json` and regenerate charts on demand.
- Clock locking and persistence need device admin and are usually unavailable on
  shared Slurm/k8s. When used they are bound by the setup/teardown contract in
  "Clock locking": always restore, and the agent verifies the off-state after
  every benchmark. When unavailable, the env file records unlocked clocks; treat
  cross-run variance accordingly.
- `nvidia-smi -lgc` / `-pm` flag syntax: verify against the installed driver
  before relying on it.

## Disk / home free space

Keep `/home` with generous free space at all times. `/home` is a small, shared
volume (~819G) and benchmark byproducts fill it fast; a full `/home` wedges
servers, JIT compiles, and HF downloads for every user on the box. Treat it as a
standing constraint, not a one-off cleanup.

- Target: keep **at least ~50G free** on `/home` (more is better). Check before
  and after any run that writes artifacts:

  ```bash
  df -h /home | tail -1                      # quick look: last col = free
  df -BG --output=avail,pcent /home | tail -1 \
    | awk '{a=$1+0; print a"G free ("$2" used)"; if(a<50) print "LOW"}'
  ```

- Big data lives on the coriander pool (`/m-coriander/coriander`, ~37T), NOT on
  `/home`. Established offloads (symlinked back so paths keep working):
  - HF cache/datasets -> `/m-coriander/coriander/hf` (`HF_HOME` points here).
  - `~/baselines` (vLLM-Omni checkout etc.) -> `/m-coriander/coriander/tim/baselines`.
  - Use `/m-coriander/coriander/tim/` as the general offload target for anything
    large. Move with `rsync -a` to the pool, verify (fast dry-run
    `rsync -an --delete --itemize-changes` shows nothing left + matching file
    counts), then `rm` source and symlink back. Never `du -sb`-compare across the
    two filesystems: metadata accounting differs, so sizes never match exactly
    even when content is identical.

- Regenerable-first cleanup order when space is low: `uv cache clean` and JIT
  caches (`~/.cache/flashinfer`, `~/.cache/vllm`) first (safe, rebuild on demand;
  note `uv` hardlinks into venvs so it may free less than reported), then
  generated benchmark **audio/media byproducts** (`*.wav`/`*.flac`/... under
  `exp_*`/`sweep_*`/`tmp/` output dirs), then move (don't delete) real data to the
  pool. Never delete committed benchmark evidence (`raw.json`, `results.json`,
  `charts/`) or reusable inputs (`libri_wavs`).

## Experimental Discipline (MANDATORY — full text imported below)

Never stand still. Maximum parallelism: multiple experiments running,
multiple code paths explored, multiple hypotheses tested at all times. On
any unknown, immediately spawn four parallel agents (Idiot = inventory of
what we already measured; Research = theory from papers/docs/architecture;
Code = implementation points + instrumentation targets; Propose = ranked
testable branches) and continue working while they run. Never idle: while
agents execute, check GPU capacity and schedule the next minimal validation
experiment. Validation = WARMUP_RUNS + 1 run on minimal config; expand to
REAL_RUNS across batch sizes only if it passes and NUMA overhead ≤ ~5%;
on failure, don't expand — respawn agents on the new unknown. Idiot Agent
must cross-reference git history/remote branches/existing benchmarks before
any experiment: never re-run prior sweeps, only missing or invalidated
datapoints. Co-schedule validated configs per NUMA cluster, independent
sweeps on separate clusters, batch similar configs to amortize warmups.

@/m-coriander/coriander/tim/EXPERIMENTAL_DISCIPLINE.md

## vLLM benchmarking — OWNER RULE (2026-07-05)

Do NOT benchmark, boot, or race vLLM. The user runs all vLLM measurements
themselves. For any comparison, use the COMMITTED vLLM reference values on the
benchmarks branch (benchmarks/qwen3-omni-joint/h2h_* raw results.json; latest
fresh-boot live band for i2t B32: 8.03-8.50 req/s, see h2h_steadyflag).
M*-side A/Bs are unaffected — iterate M* code freely with M*-only cells.

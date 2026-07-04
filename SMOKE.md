# SMOKE — V2 budgeted chunked-prefill admission (opt/v2-policy)

GPU A/B recipe for `MSTAR_MIXED_BUDGET_TOKENS`. **Code-only branch; no GPU was
used to produce it.** All commands assume the standard mstar bench env (SHM
protocol, `HF_HOME=/m-coriander/coriander/hf`, `ninja` on PATH,
`PYTHONPATH=<this worktree>` so spawned GPU workers load THIS worktree's code —
without it the A/B is invalid, see the worktree-PYTHONPATH trap).

## What the flag does

W5 P2 folds a ready prefill chunk into the running decode spec chain ONLY at a
fairness *yield boundary* (`must_yield_away`, ~8% of steps). Under continuous
arrivals a mixable chunk then sits idle for several steps before it rides a
decode step. `MSTAR_MIXED_BUDGET_TOKENS=N` (default 0=off) makes the worker
probe on EVERY spec chain step and fold a ready chunk NOW, capping the mixed
step at `N` total tokens (`n_decode` 1-token rows + the `C`-token chunk).

It does NOT change how standalone/unchunked prefills admit — it only accelerates
the drain of chunks that already exist in the pipeline (long prefills the V2
chunker split, and vision chunks under the vision stack). This is the sole
difference from the CLOSED `MSTAR_MIXED_SINGLE_CHUNK`, which routed short
standalone prefills through the chunk planner and starved decode occupancy ~10%.

Expected value: TTFT / admission latency at **B2-B8** and arrival-heavy
patterns. A fold is ~compute-neutral (mixed ~30-36ms vs prefill+decode ~29ms
replaced), so **B32 closed-loop is expected ~neutral — the sentinel is
no-regression, not a win.**

## Fixed setup (every cell)

```bash
WT=/m-coriander/coriander/tim/mstar-v2pol
PY=/m-coriander/coriander/tim/mstar-new/.venv/bin/python
GPUS=${GPUS:-6,7}                 # one fixed canonical pair for the whole session
PORT=${PORT:-8240}
SOCK=/home/tim/tmp/sk_v2pol_${PORT}
LIBRI=/home/tim/tmp/libri_wavs    # reuse; do not re-download
export PYTHONPATH=$WT HF_HOME=/m-coriander/coriander/hf
# Confirm the devices are idle BEFORE launch (CLAUDE.md GPU-selection rule):
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i $GPUS
```

**Base stack held constant on both arms** (the locked FINAL STACK, encoff
config — this is what produces the chunks the budget accelerates; vision
chunking is part of it, so i2t has vision chunks to fold):

```bash
BASE_FLAGS="MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 \
  MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_CHUNKED_PREFILL_V2_VISION=1 \
  MSTAR_MIXED_BATCH=1 MSTAR_MIXED_BATCH_VISION=1 MSTAR_MIXED_SPEC=1 \
  MSTAR_SLIM_EMIT=1 MSTAR_FAST_ROUTE=1 MSTAR_SAMPLER_CFG_CACHE=1 \
  MSTAR_FAST_CHECKSTOP=1 MSTAR_WALK_STATS=1"
CONFIG=configs/qwen3omni_2gpu_encoff.yaml
```

Serve:

```bash
setsid env CUDA_VISIBLE_DEVICES=$GPUS $BASE_FLAGS $EXTRA_FLAGS MSTAR_DYNFLAGS=$DYN \
  timeout 5400 $PY -m mstar.cli.main serve qwen3_omni \
    --gpus $GPUS --port $PORT --tensor-comm-protocol SHM \
    --socket-path-prefix $SOCK --config $CONFIG > $SERVER_LOG 2>&1 &
SERVER_PID=$!
cleanup(){ kill -- -$SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT INT TERM
```

Bench one cell (`$B` in 2,4,8,32; i2t = `image_to_text`):

```bash
timeout 1800 $PY -m benchmark.runner \
  --url http://127.0.0.1:$PORT --model qwen3omni \
  --request-type image_to_text \
  --dataset libri --profiling-type closed_loop \
  --max-concurrency $B --num-requests $((B*10)) --num-warmup 5 \
  --inference-system ours --local-cache $LIBRI \
  --output-dir $ODIR
```

Record the FULL metric set from `results.json` (TTFT/first-token, ITL, RTF, JCT,
request throughput) for every cell — not just the targeted one. Free `*.wav`
after each cell to protect `/home`.

## Arms

Three arms. The budget is dynflags-refreshable (it bakes nothing into capture),
so **off vs budget is a one-server interleaved dyn_ab**. Split-attn / preplan
bake the capture layout and are process-static, so they need their **own
server**.

- **Arm 1 — off**: base stack, `MSTAR_MIXED_BUDGET_TOKENS` unset (=0).
- **Arm 2 — budget-on**: base stack, `MSTAR_MIXED_BUDGET_TOKENS=512`.
- **Arm 3 — budget + split + preplan**: base stack, `MSTAR_MIXED_BUDGET_TOKENS=512`,
  `EXTRA_FLAGS="MSTAR_MIXED_SPLIT_ATTN=1 MSTAR_MIXED_PREPLAN=1"` (raised fold
  volume is exactly where these two pay off; separate server).

### Cells

- **Primary (where P2 leaves value): i2t B2, i2t B4, i2t B8.**
- **Sentinel (must not regress): i2t B32.**

### Server 1 (no split/preplan) — interleaved off vs budget

`EXTRA_FLAGS=""`. Point `MSTAR_DYNFLAGS=$DYN` at a JSON file and flip the budget
between cells (no restart, adjacent cells cancel box noise):

```bash
echo '{"MSTAR_MIXED_BUDGET_TOKENS":"0"}'   > $DYN   # off cell
# ... run B2/B4/B8/B32 ...
echo '{"MSTAR_MIXED_BUDGET_TOKENS":"512"}' > $DYN   # budget cell (worker
#   picks it up within ~50 iters; _refresh_dynamic_flags re-reads the budget
#   and resets the min-decode cache)
# ... run B2/B4/B8/B32 ... then flip back and repeat for >=3 pairs/cell.
```

### Server 2 (split+preplan baked ON) — interleaved off vs budget

Restart with `EXTRA_FLAGS="MSTAR_MIXED_SPLIT_ATTN=1 MSTAR_MIXED_PREPLAN=1"`, same
dyn_ab off-vs-budget flip. This isolates whether split+preplan turns the raised
fold volume net-positive (the retained-lever hypothesis).

## Proof the mechanism fired (WALK_STATS, every 200 steps at WARNING)

New counters in the server log — read them on the budget arm:

```bash
grep -oE 'budget_folds[^,}]*|budget_fold_tokens[^,}]*|budget_skips_floor[^,}]*' $SERVER_LOG | tail -6
grep -oE '_fold_ok[^,}]*|_mix_opp[^,}]*' $SERVER_LOG | tail -4
```

- **`budget_folds` must be > 0 and climb on the budget arm, and stay 0 on the
  off arm** (it is only bumped for a fold on a non-yield step). If it stays 0 on
  the budget arm, the workload produced no chunks to accelerate — check that
  prefills actually chunk (long text spans > `MSTAR_PREFILL_CHUNK_TOKENS`, or the
  vision stack on); the flag is inert without chunks, which is a workload
  finding, not a bug.
- `_fold_ok` / `_mix_opp` (total folds / opportunities) should be **higher on the
  budget arm than off** — that is the accelerated drain.
- `budget_fold_tokens` sizes the chunk tokens the policy admitted.
- `budget_skips_floor` counts steps where a chunk was ready but the decode side
  was under the occupancy floor (the anti-lesson guard working). A large value
  relative to `budget_folds` means ramp-up is dominated by sub-floor batches —
  consider lowering `MSTAR_MIXED_BUDGET_MIN_DECODE` (default 24).

## Proof of correctness (outputs unchanged)

Folding only changes WHEN a chunk runs, not the math. Under a fixed seed the
per-request token sequences must be identical off vs budget; at minimum tok/req
and request/throughput must match within sampling noise:

```bash
$PY - <<PY
import json
off=json.load(open("$ODIR_OFF/results.json")); on=json.load(open("$ODIR_BUDGET/results.json"))
for k in ("request_throughput","tokens_per_request"):
    print(k, "off", off.get(k), "budget", on.get(k))
PY
```

Run one budget cell with `MSTAR_MIXED_BATCH_ASSERT=1` (server env) and confirm
**zero** assert failures / "fold missed a peeked chunk" warnings in the log
(a lost fold race is tolerable but must be rare):

```bash
grep -cE 'AssertionError|fold missed a peeked chunk' $SERVER_LOG   # expect 0
```

## Proof of win / no-regression

- **B2-B8:** TTFT / first-token latency should drop on the budget arm (a waiting
  chunk admits sooner); request throughput flat-to-up. This is the target.
- **B32 sentinel:** request throughput must be within noise of off (±~3% on this
  box). A fold is ~compute-neutral, so a real B32 drop beyond noise is a
  regression — do not ship budget-on for B32-heavy configs if it appears.
- **Arm 3:** if split+preplan lifts the budget arm above off where Arm 2 was
  neutral/negative, that is the retained-lever payoff at raised fold volume.

Report per-cell deltas as geomean over >=3 interleaved pairs/cell; single cells
on this box swing ±40% under foreign load (only interleaved dyn_ab counts).

## Hard rules (CLAUDE.md) for every cell

One fixed GPU pair; confirm idle before launch; `timeout` wrapper on server +
bench; kill the process group + free devices on every exit path; poll
`nvidia-smi` while running and kill on freeze; commit only complete runs. No
clock locking needed (shared box, no device admin).

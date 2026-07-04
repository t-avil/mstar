# SMOKE — merged multimodal prefill (`MSTAR_MERGED_PREFILL`, opt/prefill-merge)

GPU A/B recipe for the merged text+vision prefill walk (old plan rows B1/B5).
**Code-only branch; no GPU was used to produce it.** All commands assume the
standard mstar bench env (SHM protocol, `HF_HOME=/m-coriander/coriander/hf`,
`ninja` on PATH, `PYTHONPATH=<this worktree>` so spawned GPU workers load THIS
worktree's code — without it the A/B is invalid, see the worktree-PYTHONPATH
trap). Design + invariants: `DESIGN_merged_prefill.md`.

## What the flag does

An i2t admission runs `prefill_text` AND `prefill_vision` as SEPARATE Thinker
graph walks. Each walk boundary is a conductor round-trip (worker→conductor
`WORKER_GRAPHS_DONE` → `get_partition_forward_pass_args` → conductor→worker
`InputSignals`, `conductor.py:918-990`) on the TTFT critical path.
`MSTAR_MERGED_PREFILL=1` collapses that exact one-text+one-vision schedule into a
SINGLE `prefill_multimodal` walk that runs both spans in ONE Thinker forward,
dropping the round-trip. The merged walk reuses the `prefill_vision` CUDA-graph
capture (identical post-preprocess signature), so there is NO new capture and NO
extra warmup.

Numerically it is the same KV cache: the per-span embeds / positions / deepstack
are computed by the same helpers with the same threaded MRoPE start position, and
causal attention over the concatenated `[text][vision]` span equals each span
attending to what precedes it — so the merged forward is equivalent to the two
sequential walks modulo kernel-tiling ULP drift (the property accepted for
chunked prefill).

Expected value: **i2t TTFT / JCT at B1-B4** (the round-trip is a large fraction of
a low-batch admission). B32 is the no-regression sentinel.

## Eligibility (why it is a vision-strategy SWAP, not an add-on)

The merge fires only when the schedule is EXACTLY one `prefill_text` + one
`prefill_vision` (either order), output is text (no Talker), AND
`MSTAR_CHUNKED_PREFILL_V2_VISION` is OFF. The vision-chunking stack
(`CHUNKED_PREFILL_V2_VISION` + `MIXED_BATCH_VISION`) rewrites the vision walk into
`encode_vision` + a chunkable Thinker walk — a 3-entry schedule the merge does
not match. So merged prefill is an ALTERNATIVE to the vision-fold strategy, not
stacked on it. The A/B base therefore runs VISION as plain separate walks (vision
chunking off) on BOTH arms, isolating the round-trip.

At B1-B4 the vision-fold path has almost no decode to fold vision chunks into, so
it degenerates toward separate walks anyway — exactly where removing the
round-trip outright should help most.

## Fixed setup (every cell)

```bash
WT=/m-coriander/coriander/tim/mstar-pmerge
PY=/m-coriander/coriander/tim/mstar-new/.venv/bin/python
GPUS=${GPUS:-6,7}                 # one fixed canonical pair for the whole session
PORT=${PORT:-8250}
SOCK=/home/tim/tmp/sk_pmerge_${PORT}
LIBRI=/home/tim/tmp/libri_wavs    # reuse; do not re-download
export PYTHONPATH=$WT HF_HOME=/m-coriander/coriander/hf
# Confirm the devices are idle BEFORE launch (CLAUDE.md GPU-selection rule):
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i $GPUS
```

**Base stack held constant on both arms** — the locked FINAL STACK MINUS the
vision-chunking levers (so vision runs as separate walks and the merge is
eligible). Text chunking / mixed(text) / spec stay on; they do not touch the
vision walk:

```bash
BASE_FLAGS="MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 \
  MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 \
  MSTAR_SLIM_EMIT=1 MSTAR_FAST_ROUTE=1 MSTAR_SAMPLER_CFG_CACHE=1 \
  MSTAR_FAST_CHECKSTOP=1 MSTAR_WALK_STATS=1"
CONFIG=configs/qwen3omni_2gpu_encoff.yaml
```

`MSTAR_MERGED_PREFILL` registers a graph walk + capture at startup, so it is
**process-static (NOT dynflags-refreshable)** — each arm needs its OWN server.

Serve (per arm; `$ARM_FLAGS` is the only difference):

```bash
setsid env CUDA_VISIBLE_DEVICES=$GPUS $BASE_FLAGS $ARM_FLAGS \
  timeout 5400 $PY -m mstar.cli.main serve qwen3_omni \
    --gpus $GPUS --port $PORT --tensor-comm-protocol SHM \
    --socket-path-prefix $SOCK --config $CONFIG > $SERVER_LOG 2>&1 &
SERVER_PID=$!
cleanup(){ kill -- -$SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT INT TERM
```

Bench one cell (`$B` in 1,2,4,32; i2t = `image_to_text`):

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

- **Arm A — baseline (separate walks)**: `ARM_FLAGS=""`. Each i2t admission runs
  `prefill_text` + `prefill_vision` (the round-trip under test). Own server.
- **Arm B — merged**: `ARM_FLAGS="MSTAR_MERGED_PREFILL=1"`. Each i2t admission
  runs ONE `prefill_multimodal` walk. Own server.

Because both arms are static servers, interleave at the ROUND level: A-cell,
B-cell, A-cell, ... for >=3 rounds/cell (single cells on this box swing ±40%
under foreign load; only interleaved deltas count). Keep both servers on the
SAME fixed GPU pair, one benchmark at a time (never co-locate — CLAUDE.md).

### Cells

- **Primary (where the round-trip dominates): i2t B1, i2t B2, i2t B4** — TTFT +
  JCT focus.
- **Sentinel (must not regress): i2t B32.**

## Proof the mechanism fired (WALK_STATS, every 200 steps at WARNING)

```bash
# Arm B: the merged walk ran; the separate walks did not.
grep -oE "merged_prefill_walks[^,}]*" $SERVER_LOG_B | tail -3      # > 0 and climbs
grep -oE "'prefill_multimodal'[^,}]*"  $SERVER_LOG_B | tail -3      # present
grep -oE "'prefill_vision'[^,}]*"      $SERVER_LOG_B | tail -3      # ~0
# Arm A: the two separate walks ran; no merge.
grep -oE "'prefill_text'[^,}]*|'prefill_vision'[^,}]*" $SERVER_LOG_A | tail -4
grep -c  "merged_prefill_walks"        $SERVER_LOG_A                # expect 0
```

- On Arm B, `merged_prefill_walks` must be > 0 and climb, and the
  `("Thinker","prefill_multimodal")` counter must appear while
  `("Thinker","prefill_vision")` stays ~0. If `merged_prefill_walks` is 0 on
  Arm B, the merge never fired — check that vision chunking is OFF and the i2t
  request really is one text + one image (multi-image / audio / video-second
  prompts fall back by design), a workload finding, not a bug.
- On Arm B the Thinker runs ~ONE prefill step per admission where Arm A runs
  TWO (`prefill_text` + `prefill_vision`); the per-admission Thinker-prefill
  step-count halving IS the removed round-trip.

## Proof of correctness (outputs unchanged)

The merge changes HOW the prefill runs, not the math: same embeds at the same
MRoPE positions, same causal KV. Under a fixed seed the first sampled token
should match and per-request output should be semantically identical; bitwise
divergence after a few tokens is EXPECTED and acceptable (the merged single
forward routes through different FlashInfer split-KV / fp8 tile schedules than
two shorter forwards, so greedy amplifies ULP drift — the same property as
chunked prefill / vLLM chunked prefill). At minimum tok/req and request
throughput match within sampling noise:

```bash
$PY - <<PY
import json
a=json.load(open("$ODIR_A/results.json")); b=json.load(open("$ODIR_B/results.json"))
for k in ("request_throughput","tokens_per_request"):
    print(k, "A", a.get(k), "B", b.get(k))
PY
```

Optionally dump the first-token logits on a fixed i2t prompt on each arm
(`MSTAR_DUMP_DIR=...`) and confirm the argmax matches and the top-token gap is
within ULP-scale drift.

## Proof of win / no-regression

- **B1-B4:** TTFT / first-token latency and JCT should DROP on Arm B (one fewer
  conductor round-trip per admission); request throughput flat-to-up. This is
  the target.
- **B32 sentinel:** request throughput within noise of Arm A (±~3% on this box).
  The round-trip is a smaller fraction of a full B32 admission, so B32 is
  expected neutral-to-slightly-positive; a real drop beyond noise is a
  regression — investigate before shipping default-on.

Report per-cell deltas as geomean over >=3 interleaved rounds/cell.

## Hard rules (CLAUDE.md) for every cell

One fixed GPU pair; confirm idle before launch; `timeout` wrapper on server +
bench; kill the process group + free devices on every exit path; poll
`nvidia-smi` while running and kill on freeze; commit only complete runs. No
clock locking needed (shared box, no device admin).

#!/usr/bin/env bash
# MVP launcher for the encoder-placement profiling experiment.
#
#   Launches M*-new (chunked-prefill + all combined-vision-opts +
#   MSTAR_PROFILE_ENCODER_PLACEMENT=1) on the worktree at $WORKTREE,
#   waits for /v1/models, runs benchmark.runner for I2T B=1 and B=8
#   (5 warmup + 30 measured each), tears the server down cleanly,
#   then parses the per-request JSONL to print median ms per stage at
#   B=1 vs B=8 and the encoder share of total time.
#
# Hard wall-clock cap: 15 minutes (timeout(1) on the whole script).
# Per CLAUDE.md (GPU hygiene): always free devices on every exit path
# (success, failure, timeout, signal); reset clocks idempotently.
#
# Usage:
#     bash /home/tim/exp_encplace/benchmark/mvp_encoder_placement.sh
#
# Env overrides: GPUS (default "1,2"), PORT (default 8170), BENCH_TIMEOUT,
# SERVER_TIMEOUT, JSONL_PATH.  Keep defaults stable so the run is
# comparable across sessions (per the GPU device-selection rule in
# CLAUDE.md).

set -uo pipefail

# ─── Self-imposed wall clock ───────────────────────────────────────────
# 15 min hard cap covering the entire MVP (server startup + 2 batches +
# teardown + analysis).  We re-exec under `timeout` so the ceiling
# survives even if this shell dies (timeout(1) reparents and still
# fires).
HARD_TIMEOUT="${HARD_TIMEOUT:-900}"
if [[ "${_MVP_UNDER_TIMEOUT:-0}" != "1" ]]; then
    export _MVP_UNDER_TIMEOUT=1
    exec timeout --foreground --signal=TERM --kill-after=30 "$HARD_TIMEOUT" \
        bash "$0" "$@"
fi

# ─── Constants ─────────────────────────────────────────────────────────
WORKTREE="${WORKTREE:-/home/tim/exp_encplace}"
GPUS="${GPUS:-1,2}"
PORT="${PORT:-8170}"
SERVER_CONFIG="${SERVER_CONFIG:-configs/qwen3omni_2gpu.yaml}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"   # generous: warmup + CUDA-graph capture
BENCH_TIMEOUT="${BENCH_TIMEOUT:-300}"     # per-batch
HF_DATASETS="${HF_DATASETS:-/home/tim/hf_datasets}"
HF_HOME_DIR="${HF_HOME:-/m-coriander/coriander/hf}"
PYTHON="${PYTHON:-/home/tim/mstar-encoders/.venv/bin/python}"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"  # ninja for FlashInfer JIT

# Output dir for this MVP run.
OUTPUT="${OUTPUT:-/home/tim/tmp/mvp_encoder_placement_$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTPUT"

# Profiling JSONL: append.  Wipe any prior run so the analysis below sees
# only this MVP's records.
JSONL_PATH="${JSONL_PATH:-/home/tim/tmp/encoder_placement_profile.jsonl}"
mkdir -p "$(dirname "$JSONL_PATH")"
: > "$JSONL_PATH"

# ─── NUMA from first GPU (GPUs 0..3 -> NUMA 0; 4..7 -> NUMA 1) ─────────
IFS=',' read -ra GPU_IDS <<< "$GPUS"
FIRST_GPU="${GPU_IDS[0]}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi

GIT_COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "========================================"
echo "MVP encoder-placement profiling"
echo "  Worktree: $WORKTREE ($GIT_COMMIT)"
echo "  GPUs: $GPUS (NUMA $NUMA)"
echo "  Port: $PORT"
echo "  Output: $OUTPUT"
echo "  JSONL:  $JSONL_PATH"
echo "  Hard timeout: ${HARD_TIMEOUT}s"
echo "========================================"

# ─── Env capture (per CLAUDE.md) ───────────────────────────────────────
{
    echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
    echo "=== git_commit ==="; echo "$GIT_COMMIT"
    echo "=== branch ==="; git -C "$WORKTREE" rev-parse --abbrev-ref HEAD 2>/dev/null || true
    echo "=== uname ==="; uname -a
    echo "=== CUDA_VISIBLE_DEVICES ==="; echo "$GPUS"
    echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
    echo "=== nvidia-smi query ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,persistence_mode \
        --format=csv 2>/dev/null || true
} > "$OUTPUT/env.txt" 2>&1

# ─── Pre-launch GPU idleness check ─────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Checking GPUs $GPUS are idle..."
for gid in "${GPU_IDS[@]}"; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ -z "$mem" ]]; then
        echo "ERROR: could not read GPU $gid memory; aborting."
        exit 1
    fi
    if [[ "$mem" -gt 100 ]]; then
        echo "ERROR: GPU $gid has ${mem} MiB in use; aborting (per CLAUDE.md, no co-location)."
        exit 1
    fi
done
echo "[$(date -u +%H:%M:%S)] GPUs idle."

# ─── Cleanup trap: kill server, free GPUs, reset clocks idempotently ───
SERVER_PID=""
cleanup() {
    local ec=$?
    echo ""
    echo "[$(date -u +%H:%M:%S)] Cleanup: tearing down server..."
    if [[ -n "$SERVER_PID" ]]; then
        kill -- -"$SERVER_PID" 2>/dev/null || true
    fi
    sleep 2
    # Kill any straggler processes still holding our GPUs.
    for gid in "${GPU_IDS[@]}"; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null \
            | while read -r pid; do
                [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
            done
    done
    # Clock teardown is idempotent per CLAUDE.md: reset SM/mem clocks and
    # disable persistence even if we never set them (we don't lock clocks
    # in this MVP).  A no-op if already unlocked.
    nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi -rmc >/dev/null 2>&1 || true
    # Don't toggle persistence_mode if it was already off; just print the
    # state so the agent's post-check can verify.
    nvidia-smi --query-gpu=index,persistence_mode --format=csv 2>/dev/null \
        | tee -a "$OUTPUT/post_check.txt" || true
    echo "[$(date -u +%H:%M:%S)] Cleanup done (exit=$ec)."
    return $ec
}
trap cleanup EXIT INT TERM

# ─── Server env ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS"
# PYTHONPATH MUST point at the worktree under test, NOT some sibling
# checkout (see LEARNINGS.md §5.2 "PYTHONPATH trap").
export PYTHONPATH="$WORKTREE"

# Combined-vision-opts + chunked prefill + our new profiling flag.
export MSTAR_GPU_MEL=1
export MSTAR_GPU_IMAGE_PREPROCESS=1
export MSTAR_CHUNKED_PREFILL=1
export MSTAR_VISION_GRAPH_ALIGN=1
export MSTAR_BATCH_VISION_PREFILL=1
export MSTAR_PROFILE_ENCODER_PLACEMENT=1
export MSTAR_PROFILE_ENCODER_PLACEMENT_PATH="$JSONL_PATH"

SOCK="/home/tim/tmp/sk_encplace_${PORT}"
rm -rf "$SOCK"; mkdir -p "$SOCK"

SERVER_LOG="$OUTPUT/server.log"
echo "[$(date -u +%H:%M:%S)] Launching M*-new server (port $PORT)..."
setsid bash -c "cd $WORKTREE && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    $PYTHON -m mstar.cli.main serve qwen3_omni \
        --config $SERVER_CONFIG \
        --host 0.0.0.0 --port $PORT \
        --tensor-comm-protocol SHM \
        --socket-path-prefix $SOCK \
        --log-level INFO \
    > $SERVER_LOG 2>&1" </dev/null &
SERVER_PID=$!

# ─── Wait for /v1/models (up to 180s, plus crash detection) ────────────
echo "[$(date -u +%H:%M:%S)] Waiting for /v1/models on port $PORT..."
READY=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        READY=1
        echo "[$(date -u +%H:%M:%S)] Server ready after ~$((i*3))s."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died before becoming ready. Last 40 log lines:"
        tail -40 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 3
done
if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server did not become ready within 180s.  Last 60 log lines:"
    tail -60 "$SERVER_LOG" || true
    exit 1
fi

# ─── Run I2T B=1 and B=8 ───────────────────────────────────────────────
# IMPORTANT: cd into the worktree before invoking benchmark.runner.
# Python prepends CWD to sys.path, so `python -m benchmark.runner` from
# the worktree resolves to the worktree's runner.  Avoiding the
# PYTHONPATH trap (LEARNINGS §5.2): server PYTHONPATH is set above, so
# spawned GPU workers also load the worktree's code.
last_progress_ts=0
note_progress() { last_progress_ts=$(date +%s); }

run_i2t() {
    local b="$1"
    local n=30 warm=5
    local odir="$OUTPUT/i2t/B${b}"
    mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] i2t B=$b N=$n (warm=$warm) ... "

    # Per CLAUDE.md monitoring: bench is short (~minutes), poll cadence
    # built into the benchmark itself (per-request prints).  Hard cap
    # per batch via timeout; if it exceeds, kill and continue.
    if ( cd "$WORKTREE" && \
        timeout "$BENCH_TIMEOUT" "$PYTHON" -m benchmark.runner \
            --url "http://127.0.0.1:$PORT" \
            --model qwen3omni \
            --request-type image_to_text \
            --dataset food101 \
            --profiling-type closed_loop \
            --max-concurrency "$b" \
            --num-requests "$n" \
            --num-warmup "$warm" \
            --inference-system ours \
            --local-cache "$HF_DATASETS" \
            --output-dir "$odir" \
            ) > "$odir/run.log" 2>&1
    then
        echo "done"
        note_progress
    else
        echo "FAILED (see $odir/run.log)"
        if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "  Server appears dead; aborting remaining batches."
            return 1
        fi
    fi
}

run_i2t 1 || true
run_i2t 8 || true

# ─── Server teardown (cleanup trap handles freeing GPUs / clocks) ──────
echo "[$(date -u +%H:%M:%S)] Stopping server cleanly..."
kill -- -"$SERVER_PID" 2>/dev/null || true
# Give it a moment to drain; the trap will catch the rest.
for _ in $(seq 1 10); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

# ─── Analyse the JSONL ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Stage analysis  (JSONL: $JSONL_PATH)"
echo "========================================"
"$PYTHON" - <<PY
import json, statistics, sys
from collections import defaultdict
path = "$JSONL_PATH"
recs_by_rid = defaultdict(dict)
for line in open(path):
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    rid = d.get("request_id")
    if rid is None: continue
    # Per-process records are written separately; merge by request_id by
    # taking the union of stage timestamps.  Each stage is written by
    # exactly one process so no conflicts.
    for k, v in d.items():
        recs_by_rid[rid].setdefault(k, v)

# Stage order + the spans we summarise.  All values in ms.
STAGE_TS = [
    ("ts_arrived_ns",              "arrived"),
    ("ts_preprocess_start_ns",     "preprocess_start"),
    ("ts_preprocess_end_ns",       "preprocess_end"),
    ("ts_vision_fwd_start_ns",     "vision_fwd_start"),
    ("ts_vision_fwd_end_ns",       "vision_fwd_end"),
    ("ts_vision_delivered_ns",     "vision_delivered"),
    ("ts_thinker_prefill_text_start_ns", "prefill_text_start"),
    ("ts_thinker_prefill_text_end_ns",   "prefill_text_end"),
    ("ts_first_decode_ns",         "first_decode"),
    ("ts_complete_ns",             "complete"),
]
SPANS = [
    ("preprocess",     "ts_preprocess_start_ns",     "ts_preprocess_end_ns"),
    ("vision_fwd",     "ts_vision_fwd_start_ns",    "ts_vision_fwd_end_ns"),
    ("vision_handoff", "ts_vision_fwd_end_ns",      "ts_vision_delivered_ns"),
    ("prefill_text",   "ts_thinker_prefill_text_start_ns", "ts_thinker_prefill_text_end_ns"),
    ("ttft",           "ts_arrived_ns",             "ts_first_decode_ns"),
    ("e2e",            "ts_arrived_ns",             "ts_complete_ns"),
]
ENCODER_STAGES = ("preprocess", "vision_fwd")  # what "encoder share" sums

def split_by_batch(recs):
    by_bs = defaultdict(list)
    for r in recs.values():
        bs = r.get("batch_size")
        if bs is None: continue
        by_bs[bs].append(r)
    return by_bs

by_bs = split_by_batch(recs_by_rid)

def spans_ms(records):
    out = defaultdict(list)
    for r in records:
        for label, s, e in SPANS:
            if s in r and e in r:
                out[label].append((r[e] - r[s]) / 1e6)
    return out

print(f"\nTotal records (rids) with batch_size: {sum(len(v) for v in by_bs.values())}\n")
for bs in sorted(by_bs):
    recs = by_bs[bs]
    print(f"--- batch_size = {bs}  (n = {len(recs)}) ---")
    spans = spans_ms(recs)
    enc_total = 0.0
    e2e_total = 0.0
    rows = []
    for label, s, e in SPANS:
        vals = spans.get(label, [])
        if not vals:
            rows.append((label, None, None, None))
            continue
        m = statistics.median(vals)
        rows.append((label, m, min(vals), max(vals)))
        if label in ENCODER_STAGES:
            enc_total += m
        if label == "e2e":
            e2e_total = m
    for label, m, lo, hi in rows:
        if m is None:
            print(f"  {label:<16}  (no data)")
        else:
            print(f"  {label:<16}  median {m:8.2f} ms   min {lo:7.2f}   max {hi:7.2f}")
    if e2e_total:
        share = 100.0 * enc_total / e2e_total
        print(f"  encoder share of e2e: {enc_total:6.2f} ms  /  {e2e_total:6.2f} ms  =  {share:5.1f}%")
    print()
PY

echo "[$(date -u +%H:%M:%S)] MVP complete."
echo "Server log: $SERVER_LOG"
echo "JSONL:      $JSONL_PATH"

#!/usr/bin/env bash
# mvp_encoder_chunk_coalesce.sh — A/B MVP for MSTAR_ENCODER_CHUNK_COALESCE.
#
# What this measures
#   Pre-vs-post comparison of two server launches on the SAME worktree at the
#   SAME git SHA, identical CLI args, the only delta being whether
#   MSTAR_ENCODER_CHUNK_COALESCE=1 is exported. Both audio and vision encoder
#   paths are exercised against each launch (S2T → audio encoder; I2T → vision
#   encoder; same server, same warmup, same B/N).
#
# Why MVP (not full sweep)
#   This is the gate test before a full sweep. We only ask: at B=8 (chunked
#   prefill active, real concurrency for the chunk-boundary trigger to fire),
#   does req/s move at least +5% with TTFT not regressed >5%? If yes,
#   PROMISING → invest in a full B sweep. Otherwise NEUTRAL/NEGATIVE → stop.
#
# GPU hygiene (from /home/tim/CLAUDE.md)
#   * Hard timeout: 20 min global, plus per-bench timeouts.
#   * setsid + trap cleanup on every exit path; kills server group.
#   * GPU-idle precheck before each launch.
#   * Persistence/clock teardown attempted idempotently post-run.
#   * cd to the worktree (PYTHONPATH trap) before benchmark.runner.

set -uo pipefail

WORKTREE="/home/tim/exp_encchunk"
GPUS="${GPUS:-1,2}"
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
PORT="${PORT:-8172}"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"
B=8
N_BASE=30
N_WARMUP=5
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/tim/tmp/mvp_encoder_chunk_coalesce_$(date -u +%Y%m%dT%H%M%S)}"

# Global cap on the whole MVP. The two A/B runs together must finish here.
GLOBAL_TIMEOUT="${GLOBAL_TIMEOUT:-1200}"  # 20 min
SERVER_TIMEOUT=900
BENCH_TIMEOUT=600
SERVER_READY_TIMEOUT=420

# Same baseline opts as the M*-new label (LEARNINGS §2). The only delta the
# A/B can attribute its result to is MSTAR_ENCODER_CHUNK_COALESCE.
BASE_FLAGS=(
    "MSTAR_GPU_MEL=1"
    "MSTAR_GPU_IMAGE_PREPROCESS=1"
    "MSTAR_VISION_GRAPH_ALIGN=1"
    "MSTAR_BATCH_VISION_PREFILL=1"
    # NOTE: MSTAR_CHUNKED_PREFILL is NOT in this base set because the worktree
    # base (e943d72, opt/combined-vision-opts) does not include the
    # chunked-prefill commits. The chunk-boundary hook in this experiment
    # still fires at prefill-walk boundaries; on a build that has chunked
    # prefill enabled, those become per-chunk boundaries automatically.
)

BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-/home/tim/mstar-encoders/.venv/bin/python}"
if [[ ! -x "$BENCHMARK_PYTHON" ]]; then
    BENCHMARK_PYTHON="$(command -v python3)"
fi

LIBRI_CACHE="${LIBRI_CACHE:-/home/tim/tmp/libri_wavs}"
HF_DATASETS="${HF_DATASETS:-/home/tim/hf_datasets}"
HF_HOME_DIR="${HF_HOME_DIR:-/m-coriander/coriander/hf}"

mkdir -p "$OUTPUT_ROOT"

echo "================================================================"
echo "MVP: MSTAR_ENCODER_CHUNK_COALESCE  A/B"
echo "  worktree: $WORKTREE"
echo "  git SHA:  $(cd "$WORKTREE" && git rev-parse --short HEAD)"
echo "  GPUs:     $GPUS (NUMA $NUMA)"
echo "  port:     $PORT"
echo "  B=$B  N=$((N_BASE + N_WARMUP)) (warmup=$N_WARMUP, measured=$N_BASE)"
echo "  output:   $OUTPUT_ROOT"
echo "================================================================"

# ─── Global hard-timeout wall ──────────────────────────────────────────
# Run the body as a child shell wrapped by `timeout`. The wrapper survives
# even if this controller dies (it reparents to init). Cleanup of GPU procs
# is done by the cleanup function below, which fires on every exit path.
script_pid=$$
(
    sleep "$GLOBAL_TIMEOUT"
    echo "!! GLOBAL TIMEOUT ($GLOBAL_TIMEOUT s) — killing MVP"
    kill -TERM -- -"$script_pid" 2>/dev/null || true
    sleep 5
    kill -KILL -- -"$script_pid" 2>/dev/null || true
) &
TIMEOUT_PID=$!

IFS=',' read -ra GPU_IDS <<< "$GPUS"

SERVER_PID=""
cleanup() {
    rc=$?
    echo ""
    echo "[$(date -u +%H:%M:%S)] CLEANUP (rc=$rc)"
    if [[ -n "$SERVER_PID" ]]; then
        kill -- -"$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 -- -"$SERVER_PID" 2>/dev/null || true
    fi
    # belt+suspenders: kill anything still on these GPUs
    for gid in "${GPU_IDS[@]}"; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null | while read -r pid; do
            [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        done
    done
    # idempotent clock/persistence teardown (CLAUDE.md hygiene)
    nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi -rmc >/dev/null 2>&1 || true
    sleep 1
    # Confirm GPUs free (best-effort)
    echo "Post-cleanup GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,persistence_mode --format=csv,noheader 2>/dev/null | grep -E "^($(echo "$GPUS" | tr ',' '|'))," || true
    kill "$TIMEOUT_PID" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

# ─── GPU idle precheck ─────────────────────────────────────────────────
check_gpus_idle() {
    for gid in "${GPU_IDS[@]}"; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
        if [[ -z "$mem" ]] || [[ "$mem" -gt 200 ]]; then
            echo "!! GPU $gid not idle (mem=${mem} MiB). Aborting."
            return 1
        fi
    done
    return 0
}

check_gpus_idle || exit 1

# ─── server / bench helpers ────────────────────────────────────────────
launch_server() {
    local label="$1"; shift
    local log="$OUTPUT_ROOT/server_${label}.log"
    local env_kv=("$@")

    echo "[$(date -u +%H:%M:%S)] Launching server ($label) on port $PORT..."
    echo "  Env: ${env_kv[*]}"

    # setsid → own process group, so cleanup() can kill the whole tree.
    # The PYTHONPATH trap: spawned GPU workers MUST load this worktree's
    # mstar/ code, not whatever the ambient PYTHONPATH points at.
    setsid env "${env_kv[@]}" \
        PYTHONPATH="$WORKTREE" \
        HF_HOME="$HF_HOME_DIR" \
        HF_HUB_OFFLINE=1 \
        bash -c "cd $WORKTREE && timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
            $BENCHMARK_PYTHON -m mstar.cli.main serve qwen3_omni \
                --gpus $GPUS \
                --port $PORT \
                --tensor-comm-protocol SHM \
                --socket-path-prefix /home/tim/tmp/sk_encchunk_${PORT} \
            > $log 2>&1" </dev/null &
    SERVER_PID=$!

    # Poll for /v1/models — fast cadence at first, slower later.
    for i in $(seq 1 "$SERVER_READY_TIMEOUT"); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "[$(date -u +%H:%M:%S)] Server ($label) ready (~${i}s)"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "!! Server ($label) died during startup. Tail of log:"
            tail -40 "$log"
            return 1
        fi
        sleep 1
    done
    echo "!! Server ($label) did not become ready within ${SERVER_READY_TIMEOUT}s"
    tail -40 "$log"
    return 1
}

teardown_server() {
    local label="$1"
    echo "[$(date -u +%H:%M:%S)] Tearing down server ($label)..."
    if [[ -n "$SERVER_PID" ]]; then
        kill -- -"$SERVER_PID" 2>/dev/null || true
        # Wait briefly for it to die
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 -- -"$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    # Wait for GPU memory to fully release.
    for _ in $(seq 1 30); do
        if check_gpus_idle 2>/dev/null; then
            echo "[$(date -u +%H:%M:%S)] GPUs free after teardown"
            return 0
        fi
        sleep 1
    done
    echo "!! GPUs still busy after teardown ($label)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep -E "^($(echo "$GPUS" | tr ',' '|')),"
    return 1
}

run_bench() {
    local reqtype="$1"   # audio_to_text or image_to_text
    local dataset="$2"   # libri or food101
    local cache="$3"
    local out="$4"
    mkdir -p "$out"
    local nreq=$((N_BASE + N_WARMUP))

    echo "[$(date -u +%H:%M:%S)] bench: $reqtype B=$B N=$nreq"
    # cd to the worktree — PYTHONPATH trap.
    timeout "$BENCH_TIMEOUT" bash -c "cd $WORKTREE && \
        PYTHONPATH=$WORKTREE \
        HF_HOME=$HF_HOME_DIR \
        HF_DATASETS_CACHE=$HF_DATASETS \
        $BENCHMARK_PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model qwen3omni \
            --request-type $reqtype \
            --dataset $dataset \
            --profiling-type closed_loop \
            --max-concurrency $B \
            --num-requests $nreq \
            --num-warmup $N_WARMUP \
            --inference-system ours \
            --local-cache $cache \
            --output-dir $out" \
        > "$out/run.log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  !! bench FAILED (rc=$rc), tail:"
        tail -10 "$out/run.log"
    fi
    return $rc
}

# ─── A: server flag OFF ────────────────────────────────────────────────
echo ""
echo "================  A: MSTAR_ENCODER_CHUNK_COALESCE = OFF  ================"
launch_server "A_off" "${BASE_FLAGS[@]}" || exit 2

run_bench audio_to_text libri      "$LIBRI_CACHE"  "$OUTPUT_ROOT/A_s2t" || true
run_bench image_to_text food101    "$HF_DATASETS"  "$OUTPUT_ROOT/A_i2t" || true

teardown_server "A_off" || true

# ─── B: server flag ON ─────────────────────────────────────────────────
echo ""
echo "================  B: MSTAR_ENCODER_CHUNK_COALESCE = ON   ================"
launch_server "B_on" "${BASE_FLAGS[@]}" "MSTAR_ENCODER_CHUNK_COALESCE=1" "MSTAR_ENCODER_COALESCE_SIZE=4" || exit 3

run_bench audio_to_text libri      "$LIBRI_CACHE"  "$OUTPUT_ROOT/B_s2t" || true
run_bench image_to_text food101    "$HF_DATASETS"  "$OUTPUT_ROOT/B_i2t" || true

teardown_server "B_on" || true

# ─── Verdict ───────────────────────────────────────────────────────────
echo ""
echo "================  VERDICT  ================"

verdict_one() {
    local label="$1"
    local a_json="$OUTPUT_ROOT/A_${label}/results.json"
    local b_json="$OUTPUT_ROOT/B_${label}/results.json"
    if [[ ! -f "$a_json" || ! -f "$b_json" ]]; then
        echo "VERDICT_${label^^}: ABORTED (missing results)"
        return
    fi
    "$BENCHMARK_PYTHON" - <<PY
import json
A = json.load(open("$a_json"))
B = json.load(open("$b_json"))
def pct(b, a):
    if not a:
        return float('inf') if b else 0.0
    return 100.0 * (b - a) / a
a_rps = A.get("request_throughput", 0.0) or 0.0
b_rps = B.get("request_throughput", 0.0) or 0.0
a_ttft = ((A.get("ttft") or {}).get("text") or {}).get("p50")
b_ttft = ((B.get("ttft") or {}).get("text") or {}).get("p50")
rps_d  = pct(b_rps, a_rps)
ttft_d = pct(b_ttft or 0.0, a_ttft or 0.0) if (a_ttft is not None and b_ttft is not None) else 0.0
if rps_d >= 5.0 and ttft_d <= 5.0:
    v = "PROMISING"
elif rps_d <= -1.0 or ttft_d >= 5.0:
    v = "NEGATIVE"
else:
    v = "NEUTRAL"
print(f"VERDICT_${label^^}: {v} (req/s_delta={rps_d:+.1f}%, TTFT_delta={ttft_d:+.1f}%)")
print(f"  A: req/s={a_rps:.3f}  TTFT_p50={a_ttft}")
print(f"  B: req/s={b_rps:.3f}  TTFT_p50={b_ttft}")
PY
}

verdict_one "s2t"
verdict_one "i2t"

echo ""
echo "Output dir: $OUTPUT_ROOT"
echo "If NEUTRAL: check server logs for flush-reason counters from"
echo "  EncoderCoalescer (grep 'MSTAR_ENCODER_CHUNK_COALESCE' in"
echo "  server_B_on.log). Specifically: did 'flushed_by_chunk' fire, or did"
echo "  all flushes fall through to 'flushed_by_size'? Without chunked"
echo "  prefill landing, chunk-boundary events fire only at full-walk"
echo "  boundaries, which may coincide with size-cap anyway."
echo ""

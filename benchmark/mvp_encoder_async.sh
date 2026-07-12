#!/usr/bin/env bash
# mvp_encoder_async.sh — A/B MVP test for MSTAR_ENCODER_ASYNC on Qwen3-Omni.
#
# Single script. Runs:
#   A) flag OFF baseline:  I2T B=1 N=30+5w, I2T B=8 N=30+5w
#   B) flag ON pipelined:  I2T B=1 N=30+5w, I2T B=8 N=30+5w
# Then prints PROMISING / NEUTRAL / NEGATIVE verdict per batch on TTFT and
# req/s deltas (>=5% improvement -> PROMISING; <=-5% -> NEGATIVE; else NEUTRAL).
#
# Hygiene (per CLAUDE.md):
#   - 20 min hard timeout on the whole script via the calling timeout wrapper
#     plus a per-server SERVER_TIMEOUT and per-bench BENCH_TIMEOUT.
#   - setsid + per-process-group cleanup trap so the server cannot leak GPUs
#     if this script is killed mid-run.
#   - Confirms both GPUs idle before EACH server launch (A and B).
#   - Idempotent clock reset at exit; this node doesn't lock clocks, so the
#     reset is a no-op when persistence is already off but keeps the contract.
#
# IMPORTANT (LEARNINGS §5.2 — PYTHONPATH trap): benchmark.runner runs from
# /home/tim/exp_encasync so spawned GPU workers load THIS worktree's code.
# Setting PYTHONPATH=/home/tim/exp_encasync makes the trap explicit even if
# the working directory ever drifts.
#
# Usage (recommended, with the 20-min ceiling):
#   timeout 1200 bash /home/tim/exp_encasync/benchmark/mvp_encoder_async.sh
#
# Or to override defaults:
#   PORT=8171 GPUS=1,2 OUTPUT=/tmp/mvp_encasync \
#       timeout 1200 bash benchmark/mvp_encoder_async.sh

set -uo pipefail

# ─── Constants ─────────────────────────────────────────────────────────
WORKTREE="${WORKTREE:-/home/tim/exp_encasync}"
PYTHON="${PYTHON:-/home/tim/mstar-encoders/.venv/bin/python}"
MODEL=qwen3omni
INFERENCE_SYS=ours
PROFILING=closed_loop
WARMUP=5
NUM_REQUESTS=30
REQTYPE=image_to_text
SHORT=i2t
DATASET=food101
LIBRI_CACHE=/home/tim/tmp/libri_wavs
HF_DATASETS=/home/tim/hf_datasets
HF_HOME_DIR=/m-coriander/coriander/hf
SERVER_CONFIG=configs/qwen3omni_2gpu.yaml
SERVER_TIMEOUT=540       # per server launch; both A and B share this
BENCH_TIMEOUT=420        # per (B=1|B=8) benchmark
SERVER_READY_TIMEOUT=180 # /v1/models poll budget

PORT="${PORT:-8171}"
GPUS="${GPUS:-1,2}"
OUTPUT="${OUTPUT:-/home/tim/tmp/mvp_encasync_$(date -u +%Y%m%dT%H%M%S)}"
SOCK="/home/tim/tmp/sk_mvp_encasync_${PORT}"

# Baseline flag-set: match the integration-mnew shipping stack so the only
# variable between A and B is MSTAR_ENCODER_ASYNC.
BASELINE_FLAGS=(
    "MSTAR_GPU_MEL=1"
    "MSTAR_GPU_IMAGE_PREPROCESS=1"
    "MSTAR_CHUNKED_PREFILL=1"
    "MSTAR_VISION_GRAPH_ALIGN=1"
    "MSTAR_BATCH_VISION_PREFILL=1"
)

mkdir -p "$OUTPUT"
echo "========================================"
echo "MVP encoder-async A/B"
echo "  Worktree: $WORKTREE"
echo "  GIT SHA:  $(git -C "$WORKTREE" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  GPUs:     $GPUS (NUMA 0; GPUs 1,2 are on NUMA 0 in this node)"
echo "  Port:     $PORT"
echo "  Output:   $OUTPUT"
echo "  Baseline flags: ${BASELINE_FLAGS[*]}"
echo "========================================"

# ─── Persistent env (set once; per-server overrides go in launch_server) ──
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS"
export PYTHONPATH="$WORKTREE"             # PYTHONPATH trap — explicit
export PATH="/home/tim/mstar-encoders/.venv/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

IFS=',' read -ra GPU_IDS <<< "$GPUS"

# Auto-detect NUMA from first GPU; GPUs 0..3 -> NUMA 0, 4..7 -> NUMA 1.
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi

# ─── Cleanup (always runs, every exit path) ────────────────────────────
SERVER_PID=""
gpu_idle_check() {
    local label="$1"
    echo "[$(date -u +%H:%M:%S)] [$label] checking GPUs $GPUS are idle..."
    for gid in "${GPU_IDS[@]}"; do
        local mem
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
        if [[ -z "$mem" ]]; then
            echo "WARN: could not read memory for GPU $gid"
            continue
        fi
        if [[ "$mem" -gt 100 ]]; then
            echo "ERROR: GPU $gid has ${mem} MiB in use. Aborting."
            exit 1
        fi
    done
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[$(date -u +%H:%M:%S)] killing server group $SERVER_PID..."
        kill -- -"$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        # give the kernel a moment to release the GPU compute contexts
        sleep 4
        for gid in "${GPU_IDS[@]}"; do
            nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null \
                | while read -r pid; do
                    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
                done
        done
        sleep 2
    fi
}

teardown_clocks() {
    # Idempotent — this node doesn't lock clocks, but the contract still says
    # we reset on the way out.
    nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi -rmc >/dev/null 2>&1 || true
    # If persistence got flipped on by an earlier run, switch it off.
    nvidia-smi --query-gpu=index,persistence_mode --format=csv 2>/dev/null \
        | grep -i Enabled >/dev/null && nvidia-smi -pm 0 >/dev/null 2>&1 || true
}

cleanup() {
    local rc=$?
    kill_server
    teardown_clocks
    echo "[$(date -u +%H:%M:%S)] cleanup done (exit $rc)"
}
trap cleanup EXIT INT TERM

# ─── Server launch helper ──────────────────────────────────────────────
launch_server() {
    local label="$1"
    shift
    local extra_flags=("$@")
    rm -rf "$SOCK"; mkdir -p "$SOCK"

    # Compose the per-launch env. We deliberately build the command line
    # rather than using `export FLAG=...` so a server crash leaves no
    # stray exports in the script's environment for the NEXT launch.
    local flag_exports=""
    for f in "${BASELINE_FLAGS[@]}" "${extra_flags[@]}"; do
        flag_exports+=" $f"
    done
    echo "[$(date -u +%H:%M:%S)] [$label] launching server with flags:$flag_exports"

    local SERVER_LOG="$OUTPUT/server_${label}.log"
    # setsid + own process group so the cleanup trap can wipe every worker.
    setsid bash -c " \
        cd $WORKTREE && \
        $flag_exports \
        timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
        python -m mstar.cli.main serve qwen3_omni \
            --config $SERVER_CONFIG \
            --host 0.0.0.0 --port $PORT \
            --tensor-comm-protocol SHM \
            --socket-path-prefix $SOCK \
            --log-level INFO \
        > $SERVER_LOG 2>&1" </dev/null &
    SERVER_PID=$!

    echo "[$(date -u +%H:%M:%S)] [$label] server pid=$SERVER_PID; waiting up to ${SERVER_READY_TIMEOUT}s for /v1/models..."
    local i
    for i in $(seq 1 "$SERVER_READY_TIMEOUT"); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "[$(date -u +%H:%M:%S)] [$label] server ready after ~${i}s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: [$label] server process died early. Tail of log:"
            tail -40 "$SERVER_LOG"
            return 1
        fi
        sleep 1
    done
    echo "ERROR: [$label] server failed to become ready within ${SERVER_READY_TIMEOUT}s"
    tail -40 "$SERVER_LOG"
    return 1
}

# ─── Bench helper ──────────────────────────────────────────────────────
run_bench() {
    local label="$1"  # e.g. "A_b1"
    local batch="$2"
    local odir="$OUTPUT/$label"
    mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] $label (B=$batch N=$NUM_REQUESTS warmup=$WARMUP) ... "

    # cd to $WORKTREE so spawned GPU workers see this worktree's code on
    # PYTHONPATH (see sweep_vllm.sh ~line 183 and LEARNINGS §5.2).
    timeout "$BENCH_TIMEOUT" bash -c " \
        cd $WORKTREE && \
        $PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model $MODEL \
            --request-type $REQTYPE \
            --dataset $DATASET \
            --profiling-type $PROFILING \
            --max-concurrency $batch \
            --num-requests $NUM_REQUESTS \
            --num-warmup $WARMUP \
            --inference-system $INFERENCE_SYS \
            --local-cache $HF_DATASETS \
            --output-dir $odir" \
        > "$odir/run.log" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "FAILED (exit $rc; see $odir/run.log)"
        return 1
    fi
    if [[ ! -f "$odir/results.json" ]]; then
        echo "FAILED (no results.json)"
        return 1
    fi
    # Place a stable JSON copy at $OUTPUT/<label>.json so the verdict step
    # can find the file by label name as the spec calls out.
    cp "$odir/results.json" "$OUTPUT/${label}.json"
    local r
    r=$("$PYTHON" -c "
import json
d=json.load(open('$odir/results.json'))
ttft=d.get('ttft',{}).get('text') or d.get('ttft',{}).get('first')
ttft_mean = (ttft or {}).get('mean', 0.0)*1000 if ttft else 0
print(f\"req/s={d.get('request_throughput',0):.3f} ttft_mean_ms={ttft_mean:.2f}\")
" 2>/dev/null) || r="(parse failed)"
    echo "done — $r"
    return 0
}

# ─── A: baseline (flag OFF) ────────────────────────────────────────────
gpu_idle_check "A"
launch_server "A" "MSTAR_ENCODER_ASYNC=0"
run_bench "A_b1" 1 || true
run_bench "A_b8" 8 || true
kill_server
gpu_idle_check "post-A"

# ─── B: pipelined (flag ON) ────────────────────────────────────────────
launch_server "B" "MSTAR_ENCODER_ASYNC=1" "MSTAR_ENCODER_ASYNC_DEPTH=4"
run_bench "B_b1" 1 || true
run_bench "B_b8" 8 || true
kill_server
gpu_idle_check "post-B"
teardown_clocks

# ─── Verdict ───────────────────────────────────────────────────────────
emit_verdict() {
    local b="$1"
    local a_json="$OUTPUT/A_b${b}.json"
    local b_json="$OUTPUT/B_b${b}.json"
    if [[ ! -f "$a_json" || ! -f "$b_json" ]]; then
        echo "VERDICT_B${b}: SKIPPED (missing results)"
        return
    fi
    "$PYTHON" - <<EOF
import json
a=json.load(open("$a_json"))
b=json.load(open("$b_json"))
def ttft_ms(d):
    t=d.get("ttft",{})
    block=t.get("text") or t.get("first") or (next(iter(t.values())) if t else None)
    if not block: return None
    return block.get("mean",0.0)*1000
def rps(d): return d.get("request_throughput",0.0) or 0.0
ta, tb = ttft_ms(a), ttft_ms(b)
ra, rb = rps(a), rps(b)
def pct(a_,b_):
    if not a_: return 0.0
    return (b_-a_)/a_*100.0
# req/s: higher is better. ttft: lower is better, so we report the negative delta
# of B vs A and flip sign for the "better is positive" convention printed below.
rps_delta = pct(ra, rb)              # >0 means B is faster than A
ttft_delta = -pct(ta, tb) if ta and tb else 0.0  # >0 means B has LOWER ttft
if rps_delta >= 5.0 or ttft_delta >= 5.0:
    v="PROMISING"
elif rps_delta <= -5.0 or ttft_delta <= -5.0:
    v="NEGATIVE"
else:
    v="NEUTRAL"
print(f"VERDICT_B${b}: {v} (TTFT_delta={ttft_delta:+.1f}%, req/s_delta={rps_delta:+.1f}%)")
print(f"  A (flag OFF): ttft_mean_ms={ta:.2f}  req/s={ra:.3f}")
print(f"  B (flag ON ): ttft_mean_ms={tb:.2f}  req/s={rb:.3f}")
EOF
}

echo ""
echo "========================================"
echo "MVP encoder-async results"
echo "========================================"
emit_verdict 1
emit_verdict 8
echo ""
echo "Artifacts in: $OUTPUT"

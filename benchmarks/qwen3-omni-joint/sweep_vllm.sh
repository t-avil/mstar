#!/usr/bin/env bash
# sweep_vllm.sh — Automated sweep for vLLM-Omni (same interface as sweep.sh outputs)
#
# Usage:
#   ./sweep_vllm.sh --gpus 5,6 --port 8093 --paths s2t,i2t,s2s,i2s \
#       --output /home/tim/tmp/sweep_vllm_rerun
#
# Launches vllm-omni server, runs benchmark.runner for each path×batch,
# saves results.json in the same layout as sweep.sh so ingest_sweep.py works.
set -euo pipefail

# ─── Constants ─────────────────────────────────────────────────────────
VLLM_DIR="/home/tim/baselines/vllm-omni"
VLLM_PYTHON="$VLLM_DIR/.venv/bin/python"
VLLM_BIN="$VLLM_DIR/.venv/bin/vllm"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEPLOY_CONFIG="$VLLM_DIR/vllm_omni/deploy/qwen3_omni_moe.yaml"
BENCHMARK_PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
BENCHMARK_CODE="/home/tim/mstar"
LIBRI_CACHE="/home/tim/tmp/libri_wavs"
HF_DATASETS="/home/tim/hf_datasets"
HF_HOME_DIR="/m-coriander/coriander/hf"
SERVER_TIMEOUT=5400
BENCH_TIMEOUT=1800
WARMUP=5
SYSTEM="vllm"

declare -A SHORT_TO_REQTYPE=(
    [s2t]=audio_to_text  [s2s]=audio_to_speech
    [i2t]=image_to_text  [i2s]=image_to_speech
)
declare -A PATH_DATASET=(
    [audio_to_text]=libri    [audio_to_speech]=libri
    [image_to_text]=food101  [image_to_speech]=food101
)
declare -A PATH_CACHE=(
    [audio_to_text]="$LIBRI_CACHE"    [audio_to_speech]="$LIBRI_CACHE"
    [image_to_text]="$HF_DATASETS"    [image_to_speech]="$HF_DATASETS"
)

n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

# ─── Parse args ────────────────────────────────────────────────────────
GPUS="" PORT="" PATHS_CSV="" BATCHES_CSV="1,2,4,8,16,32" OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)    GPUS="$2"; shift 2;;
        --port)    PORT="$2"; shift 2;;
        --paths)   PATHS_CSV="$2"; shift 2;;
        --batches) BATCHES_CSV="$2"; shift 2;;
        --output)  OUTPUT="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

[[ -z "$GPUS" ]]      && { echo "ERROR: --gpus required"; exit 1; }
[[ -z "$PORT" ]]      && { echo "ERROR: --port required"; exit 1; }
[[ -z "$PATHS_CSV" ]] && { echo "ERROR: --paths required"; exit 1; }
OUTPUT="${OUTPUT:-/home/tim/tmp/sweep_vllm_$(date -u +%Y%m%dT%H%M%S)}"

# Resolve paths
IFS=',' read -ra PATHS_ARR <<< "$PATHS_CSV"
RESOLVED_PATHS=()
for p in "${PATHS_ARR[@]}"; do
    [[ -n "${SHORT_TO_REQTYPE[$p]+x}" ]] && RESOLVED_PATHS+=("${SHORT_TO_REQTYPE[$p]}") || RESOLVED_PATHS+=("$p")
done
IFS=',' read -ra BATCHES_ARR <<< "$BATCHES_CSV"
IFS=',' read -ra GPU_IDS <<< "$GPUS"

# Auto-detect NUMA
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi

GIT_COMMIT=$(cd "$VLLM_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "========================================"
echo "vLLM-Omni sweep"
echo "  GPUs: $GPUS (NUMA $NUMA)"
echo "  Port: $PORT"
echo "  Paths: ${RESOLVED_PATHS[*]}"
echo "  Batches: ${BATCHES_ARR[*]}"
echo "  vLLM: $VLLM_DIR ($GIT_COMMIT)"
echo "  Output: $OUTPUT"
echo "========================================"

mkdir -p "$OUTPUT"

# ─── Env capture ───────────────────────────────────────────────────────
{
    echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
    echo "=== system ==="; echo "$SYSTEM"
    echo "=== git_commit ==="; echo "$GIT_COMMIT"
    echo "=== vllm_version ==="; "$VLLM_PYTHON" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "?"
    echo "=== uname ==="; uname -a
    echo "=== CUDA_VISIBLE_DEVICES ==="; echo "$GPUS"
    echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || true
} > "$OUTPUT/env.txt" 2>&1
"$VLLM_PYTHON" -m pip freeze > "$OUTPUT/requirements.txt" 2>/dev/null || true

# ─── GPU check ─────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Checking GPUs $GPUS are idle..."
for gid in "${GPU_IDS[@]}"; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ "$mem" -gt 100 ]]; then
        echo "ERROR: GPU $gid has ${mem} MiB in use. Aborting."
        exit 1
    fi
done
echo "[$(date -u +%H:%M:%S)] GPUs idle. Launching vLLM-Omni server..."

# ─── Server launch ─────────────────────────────────────────────────────
# We need to remap GPU indices in the deploy config. vLLM uses CUDA_VISIBLE_DEVICES
# so devices "0","1" in the yaml map to the first two visible GPUs.
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS"
export PATH="$VLLM_DIR/.venv/bin:$PATH"

SERVER_LOG="$OUTPUT/server.log"
setsid bash -c "cd $VLLM_DIR && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    $VLLM_BIN serve $MODEL --omni \
        --port $PORT \
        --stage-configs-path $DEPLOY_CONFIG \
    > $SERVER_LOG 2>&1" </dev/null &
SERVER_PID=$!

cleanup() {
    echo "[$(date -u +%H:%M:%S)] Cleanup: killing server group..."
    kill -- -"$SERVER_PID" 2>/dev/null || true
    sleep 2
    for gid in "${GPU_IDS[@]}"; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null | while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done
    done
    echo "[$(date -u +%H:%M:%S)] Server stopped."
}
trap cleanup EXIT INT TERM

# Wait for server
echo "[$(date -u +%H:%M:%S)] Waiting for server on port $PORT..."
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] Server ready after ~${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. Check $SERVER_LOG"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 3
done

if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ERROR: Server failed to start within 540s"
    tail -30 "$SERVER_LOG"
    exit 1
fi

# ─── Run sweeps ────────────────────────────────────────────────────────
for reqtype in "${RESOLVED_PATHS[@]}"; do
    dataset="${PATH_DATASET[$reqtype]}"
    cache="${PATH_CACHE[$reqtype]}"
    short=""
    for k in "${!SHORT_TO_REQTYPE[@]}"; do
        [[ "${SHORT_TO_REQTYPE[$k]}" == "$reqtype" ]] && short="$k" && break
    done
    [[ -z "$short" ]] && short="$reqtype"

    echo ""
    echo "======== $short ($reqtype) ========"

    for b in "${BATCHES_ARR[@]}"; do
        n=$(n_requests "$b")
        odir="$OUTPUT/$short/B${b}"
        mkdir -p "$odir"

        echo -n "[$(date -u +%H:%M:%S)] $short B=$b N=$n ... "

        timeout "$BENCH_TIMEOUT" bash -c "cd $BENCHMARK_CODE && $BENCHMARK_PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model qwen3omni \
            --request-type $reqtype \
            --dataset $dataset \
            --profiling-type closed_loop \
            --max-concurrency $b \
            --num-requests $n \
            --num-warmup $WARMUP \
            --inference-system vllm_omni \
            --local-cache $cache \
            --output-dir $odir" \
            > "$odir/run.log" 2>&1
        rc=$?

        if [[ $rc -ne 0 ]]; then
            echo "FAILED (exit $rc)"
            if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
                echo "  Server crashed. Skipping remaining batches for $short."
                break
            fi
            continue
        fi

        if [[ -f "$odir/results.json" ]]; then
            "$BENCHMARK_PYTHON" -c "
import json
r = json.load(open('$odir/results.json'))
r['git_commit'] = '$GIT_COMMIT'
r['build'] = '$SYSTEM'
r['system'] = '$SYSTEM'
r['flags'] = 'vllm-omni $GIT_COMMIT'
json.dump(r, open('$odir/results.json','w'), indent=2)
"
            completed=$("$BENCHMARK_PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(d.get('completed',0))" 2>/dev/null)
            failed=$("$BENCHMARK_PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(d.get('failed',0))" 2>/dev/null)
            req_s=$("$BENCHMARK_PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(f\"{d.get('request_throughput',0):.2f}\")" 2>/dev/null)
            echo "done (${completed}/${n} ok, ${failed} fail, ${req_s} req/s)"
        else
            echo "done (no results.json)"
        fi
    done
done

echo ""
echo "========================================"
echo "SWEEP COMPLETE: $SYSTEM"
echo "========================================"
echo "[$(date -u +%H:%M:%S)] Tearing down server..."

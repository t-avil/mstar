#!/usr/bin/env bash
# benchmark/sweep.sh — Standardized benchmark entry point for Qwen3-Omni.
#
# Single script to launch a server, run a full sweep (all paths × all batches),
# capture env, inject metadata, and optionally auto-commit results.
#
# Usage:
#   benchmark/sweep.sh --system mstar_new --gpus 0,1 --port 8160 \
#       --paths s2t,i2t,s2s,i2s --batches 1,2,4,8,16,32 \
#       --flags "MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1" \
#       --output /home/tim/tmp/sweep_mnew
#
#   benchmark/sweep.sh --system mstar_old --gpus 2,3 --port 8161 \
#       --paths s2t,i2t --output /home/tim/tmp/sweep_mold
#
# All datasets, sample counts, seeds, and cache paths are hardcoded below
# so every run is reproducible without remembering flags.
set -euo pipefail

# ─── Hardcoded constants (do not change per-run) ───────────────────────
MODEL=qwen3omni
PROFILING=closed_loop
WARMUP=5
INFERENCE_SYS=ours
LIBRI_CACHE=/home/tim/tmp/libri_wavs
HF_DATASETS=/home/tim/hf_datasets
HF_HOME_DIR=/m-coriander/coriander/hf
SERVER_CONFIG=configs/qwen3omni_2gpu.yaml
SERVER_TIMEOUT=5400
BENCH_TIMEOUT=1800
HANG_TIMEOUT=300

# Path → dataset + local-cache mapping
declare -A PATH_DATASET=(
    [audio_to_text]=libri
    [audio_to_speech]=libri
    [image_to_text]=food101
    [image_to_speech]=food101
)
declare -A PATH_CACHE=(
    [audio_to_text]="$LIBRI_CACHE"
    [audio_to_speech]="$LIBRI_CACHE"
    [image_to_text]="$HF_DATASETS"
    [image_to_speech]="$HF_DATASETS"
)

# Short name → request-type mapping
declare -A SHORT_TO_REQTYPE=(
    [s2t]=audio_to_text
    [s2s]=audio_to_speech
    [i2t]=image_to_text
    [i2s]=image_to_speech
)

# Sample count: N = max(50, 10*B)
n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

# ─── Parse CLI args ────────────────────────────────────────────────────
SYSTEM=""
GPUS=""
PORT=""
PATHS_CSV=""
BATCHES_CSV="1,2,4,8,16,32"
FLAGS=""
OUTPUT=""
WORKTREE=""
NUMA=""

usage() {
    echo "Usage: $0 --system <name> --gpus <ids> --port <n> --paths <p1,p2,...> [options]"
    echo ""
    echo "Required:"
    echo "  --system    System label: mstar_new, mstar_old, vllm"
    echo "  --gpus      GPU indices (e.g. 0,1)"
    echo "  --port      Server port"
    echo "  --paths     Comma-separated: s2t,i2t,s2s,i2s (or audio_to_text,...)"
    echo ""
    echo "Optional:"
    echo "  --batches   Batch sizes (default: 1,2,4,8,16,32)"
    echo "  --flags     Space-separated env flags for server (e.g. 'MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1')"
    echo "  --output    Output directory (default: /home/tim/tmp/sweep_<system>)"
    echo "  --worktree  Code directory (default: cwd)"
    echo "  --numa      NUMA node for numactl (auto-detected if omitted)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --system)  SYSTEM="$2"; shift 2;;
        --gpus)    GPUS="$2"; shift 2;;
        --port)    PORT="$2"; shift 2;;
        --paths)   PATHS_CSV="$2"; shift 2;;
        --batches) BATCHES_CSV="$2"; shift 2;;
        --flags)   FLAGS="$2"; shift 2;;
        --output)  OUTPUT="$2"; shift 2;;
        --worktree) WORKTREE="$2"; shift 2;;
        --numa)    NUMA="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown arg: $1"; usage;;
    esac
done

[[ -z "$SYSTEM" ]] && { echo "ERROR: --system required"; usage; }
[[ -z "$GPUS" ]]   && { echo "ERROR: --gpus required"; usage; }
[[ -z "$PORT" ]]   && { echo "ERROR: --port required"; usage; }
[[ -z "$PATHS_CSV" ]] && { echo "ERROR: --paths required"; usage; }

WORKTREE="${WORKTREE:-$(pwd)}"
OUTPUT="${OUTPUT:-/home/tim/tmp/sweep_${SYSTEM}}"
GIT_COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Auto-detect NUMA from first GPU index
if [[ -z "$NUMA" ]]; then
    FIRST_GPU="${GPUS%%,*}"
    if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
fi

# Resolve short path names
IFS=',' read -ra PATHS_ARR <<< "$PATHS_CSV"
RESOLVED_PATHS=()
for p in "${PATHS_ARR[@]}"; do
    if [[ -n "${SHORT_TO_REQTYPE[$p]+x}" ]]; then
        RESOLVED_PATHS+=("${SHORT_TO_REQTYPE[$p]}")
    else
        RESOLVED_PATHS+=("$p")
    fi
done

IFS=',' read -ra BATCHES_ARR <<< "$BATCHES_CSV"

echo "========================================"
echo "Benchmark sweep: $SYSTEM"
echo "  GPUs: $GPUS (NUMA $NUMA)"
echo "  Port: $PORT"
echo "  Paths: ${RESOLVED_PATHS[*]}"
echo "  Batches: ${BATCHES_ARR[*]}"
echo "  Flags: ${FLAGS:-none}"
echo "  Worktree: $WORKTREE ($GIT_COMMIT)"
echo "  Output: $OUTPUT"
echo "========================================"

mkdir -p "$OUTPUT"

# ─── Environment capture ───────────────────────────────────────────────
{
    echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
    echo "=== system ===" ; echo "$SYSTEM"
    echo "=== git_commit ==="; echo "$GIT_COMMIT"
    echo "=== flags ==="; echo "${FLAGS:-none}"
    echo "=== uname ==="; uname -a
    echo "=== os-release ==="; cat /etc/os-release 2>/dev/null || true
    echo "=== CUDA_VISIBLE_DEVICES ==="; echo "$GPUS"
    echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
    echo "=== nvidia-smi query ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,persistence_mode \
        --format=csv 2>/dev/null || true
    echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "no nvcc"
    echo "=== torch cuda ==="
    python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('cudnn',torch.backends.cudnn.version())" 2>/dev/null || echo "no torch"
    echo "=== git ==="; git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || true
    git -C "$WORKTREE" status --short 2>/dev/null || true
} > "$OUTPUT/env.txt" 2>&1

pip freeze > "$OUTPUT/requirements.txt" 2>/dev/null || true

# ─── Confirm GPUs are idle ─────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Checking GPUs $GPUS are idle..."
IFS=',' read -ra GPU_IDS <<< "$GPUS"
for gid in "${GPU_IDS[@]}"; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ "$mem" -gt 100 ]]; then
        echo "ERROR: GPU $gid has ${mem} MiB in use. Aborting."
        exit 1
    fi
done
echo "[$(date -u +%H:%M:%S)] GPUs idle. Launching server..."

# ─── Server launch ─────────────────────────────────────────────────────
SOCK="/home/tim/tmp/sk_${SYSTEM}_${PORT}"
rm -rf "$SOCK"; mkdir -p "$SOCK"

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS"
export PYTHONPATH="$WORKTREE"
export PATH="/home/tim/mstar-encoders/.venv/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Export optimization flags
for flag in $FLAGS; do
    export "$flag"
done

SERVER_LOG="$OUTPUT/server.log"
setsid bash -c "cd $WORKTREE && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    python -m mstar.cli.main serve qwen3_omni \
        --config $SERVER_CONFIG \
        --host 0.0.0.0 --port $PORT \
        --tensor-comm-protocol SHM \
        --socket-path-prefix $SOCK \
        --log-level INFO \
    > $SERVER_LOG 2>&1" </dev/null &
SERVER_PID=$!

cleanup() {
    echo "[$(date -u +%H:%M:%S)] Cleanup: killing server group..."
    kill -- -"$SERVER_PID" 2>/dev/null || true
    sleep 2
    # Kill any orphan workers holding our GPUs
    for gid in "${GPU_IDS[@]}"; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null | while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done
    done
    echo "[$(date -u +%H:%M:%S)] Server stopped."
}
trap cleanup EXIT INT TERM

# Wait for server ready (poll /v1/models)
echo "[$(date -u +%H:%M:%S)] Waiting for server on port $PORT..."
for i in $(seq 1 120); do
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
    echo "ERROR: Server failed to start within 360s"
    tail -30 "$SERVER_LOG"
    exit 1
fi

# ─── Run sweeps ────────────────────────────────────────────────────────
PYTHON="/home/tim/mstar-encoders/.venv/bin/python"

for reqtype in "${RESOLVED_PATHS[@]}"; do
    dataset="${PATH_DATASET[$reqtype]}"
    cache="${PATH_CACHE[$reqtype]}"
    # Short name for directory
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

        timeout "$BENCH_TIMEOUT" "$PYTHON" -m benchmark.runner \
            --url "http://127.0.0.1:$PORT" \
            --model "$MODEL" \
            --request-type "$reqtype" \
            --dataset "$dataset" \
            --profiling-type "$PROFILING" \
            --max-concurrency "$b" \
            --num-requests "$n" \
            --num-warmup "$WARMUP" \
            --inference-system "$INFERENCE_SYS" \
            --local-cache "$cache" \
            --output-dir "$odir" \
            > "$odir/run.log" 2>&1
        rc=$?

        if [[ $rc -ne 0 ]]; then
            echo "FAILED (exit $rc)"
            # Check if server is still alive
            if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
                echo "  Server crashed. Skipping remaining batches for $short."
                break
            fi
            continue
        fi

        # Inject metadata
        if [[ -f "$odir/results.json" ]]; then
            "$PYTHON" -c "
import json, sys
r = json.load(open('$odir/results.json'))
r['git_commit'] = '$GIT_COMMIT'
r['flags'] = '${FLAGS:-none}'
r['build'] = '$SYSTEM'
r['system'] = '$SYSTEM'
json.dump(r, open('$odir/results.json','w'), indent=2)
"
            # Extract headline numbers
            completed=$("$PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(d.get('completed',0))" 2>/dev/null)
            failed=$("$PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(d.get('failed',0))" 2>/dev/null)
            req_s=$("$PYTHON" -c "import json; d=json.load(open('$odir/results.json')); print(f\"{d.get('request_throughput',0):.2f}\")" 2>/dev/null)
            echo "done (${completed}/${n} ok, ${failed} fail, ${req_s} req/s)"
        else
            echo "done (no results.json)"
        fi
    done
done

# ─── Summary ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "SWEEP COMPLETE: $SYSTEM"
echo "========================================"

for reqtype in "${RESOLVED_PATHS[@]}"; do
    short=""
    for k in "${!SHORT_TO_REQTYPE[@]}"; do
        [[ "${SHORT_TO_REQTYPE[$k]}" == "$reqtype" ]] && short="$k" && break
    done
    [[ -z "$short" ]] && short="$reqtype"

    echo ""
    echo "--- $short ($reqtype) ---"
    printf "  %-4s | %-10s | %-8s | %-8s | %-8s\n" "B" "TTFT mean" "req/s" "tok/s" "comp/fail"
    printf "  %-4s-+-%-10s-+-%-8s-+-%-8s-+-%-8s\n" "----" "----------" "--------" "--------" "--------"

    for b in "${BATCHES_ARR[@]}"; do
        rf="$OUTPUT/$short/B${b}/results.json"
        if [[ -f "$rf" ]]; then
            "$PYTHON" -c "
import json
d = json.load(open('$rf'))
ttft = d.get('ttft',{})
# Try text TTFT first, then audio
t = ttft.get('text',{}).get('mean') or ttft.get('audio',{}).get('mean') or 0
print(f'  {$b:<4d} | {t*1000:>8.1f}ms | {d.get(\"request_throughput\",0):>6.2f} | {d.get(\"text_token_throughput\",0):>6.1f} | {d.get(\"completed\",0)}/{d.get(\"failed\",0)}')
" 2>/dev/null || echo "  $b    | (parse err)"
        else
            echo "  $b    | (missing)"
        fi
    done
done

echo ""
echo "[$(date -u +%H:%M:%S)] Tearing down server..."
# cleanup runs via trap

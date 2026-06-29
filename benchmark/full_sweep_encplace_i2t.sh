#!/usr/bin/env bash
# full_sweep_encplace_i2t.sh — Full-batch I2T sweep with placement profiling ON.
# Single server, loops B=1,2,4,8,16,32. Dumps per-stage JSONL for every request.
set -uo pipefail

WORKTREE="/home/tim/exp_encplace"
GPUS="${GPUS:-1,2}"
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
PORT="${PORT:-8180}"
PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"

BATCHES=(1 2 4 8 16 32)
WARMUP=5
HF_DATASETS="/home/tim/hf_datasets"
HF_HOME_DIR="/m-coriander/coriander/hf"
SERVER_TIMEOUT=5400
SERVER_READY_TIMEOUT=420
BENCH_TIMEOUT=1800
OUTPUT="${OUTPUT:-/home/tim/tmp/full_sweep_encplace_$(date -u +%Y%m%dT%H%M%S)}"
JSONL="${JSONL:-$OUTPUT/encoder_placement_profile.jsonl}"
mkdir -p "$OUTPUT"; : > "$JSONL"
SOCK="/home/tim/tmp/sk_encplace_full_${PORT}"

n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

echo "=================================="
echo "Exp 1 FULL SWEEP I2T (placement profile)"
echo "  GPUs: $GPUS (NUMA $NUMA), Port: $PORT"
echo "  Batches: ${BATCHES[*]}"
echo "  Output: $OUTPUT"
echo "  JSONL: $JSONL"
echo "=================================="

# GPU idle precheck
for gid in ${GPUS//,/ }; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ "${mem:-0}" -gt 100 ]]; then echo "ERROR: GPU $gid busy ($mem MiB)"; exit 1; fi
done

# Server env
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS"
export MSTAR_GPU_MEL=1
export MSTAR_GPU_IMAGE_PREPROCESS=1
export MSTAR_VISION_GRAPH_ALIGN=1
export MSTAR_BATCH_VISION_PREFILL=1
export MSTAR_PROFILE_ENCODER_PLACEMENT=1
export MSTAR_PROFILE_ENCODER_PLACEMENT_PATH="$JSONL"

SERVER_LOG="$OUTPUT/server.log"
setsid bash -c "cd $WORKTREE && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    $PYTHON -m mstar.cli.main serve qwen3_omni \
        --config configs/qwen3omni_2gpu.yaml \
        --host 0.0.0.0 --port $PORT \
        --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
    > $SERVER_LOG 2>&1" </dev/null &
SERVER_PID=$!

cleanup() {
    echo "[$(date -u +%H:%M:%S)] CLEANUP"
    kill -- -"$SERVER_PID" 2>/dev/null || true
    sleep 3
    for gid in ${GPUS//,/ }; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null | while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done
    done
    nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi -rmc >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[$(date -u +%H:%M:%S)] Waiting for /v1/models on port $PORT..."
for i in $(seq 1 $SERVER_READY_TIMEOUT); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] Server ready (~${i}s)"; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died early"; tail -30 "$SERVER_LOG"; exit 1
    fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ERROR: server not ready after ${SERVER_READY_TIMEOUT}s"; tail -30 "$SERVER_LOG"; exit 1
fi

for b in "${BATCHES[@]}"; do
    n=$(n_requests "$b")
    odir="$OUTPUT/B${b}"; mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] B=$b N=$n ... "
    timeout $BENCH_TIMEOUT bash -c "cd $WORKTREE && \
        PYTHONPATH=$WORKTREE \
        HF_HOME=$HF_HOME_DIR \
        HF_DATASETS_CACHE=$HF_DATASETS \
        MSTAR_PROFILE_ENCODER_PLACEMENT=1 \
        MSTAR_PROFILE_ENCODER_PLACEMENT_PATH=$JSONL \
        $PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model qwen3omni \
            --request-type image_to_text \
            --dataset food101 \
            --profiling-type closed_loop \
            --max-concurrency $b --num-requests $n --num-warmup $WARMUP \
            --inference-system ours \
            --local-cache $HF_DATASETS \
            --output-dir $odir" \
        > "$odir/run.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc"; continue; fi
    r=$($PYTHON -c "import json; d=json.load(open('$odir/results.json')); print(f\"req/s={d.get('request_throughput',0):.3f} ttft={(d.get('ttft',{}).get('text') or {}).get('p50',0)*1000:.0f}ms\")")
    echo "done — $r"
done

echo "[$(date -u +%H:%M:%S)] SWEEP COMPLETE"
echo "JSONL records: $(wc -l < $JSONL)"

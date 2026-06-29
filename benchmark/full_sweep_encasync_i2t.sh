#!/usr/bin/env bash
# full_sweep_encasync_i2t.sh — A/B full-batch I2T sweep with MSTAR_ENCODER_ASYNC.
# Launches server twice (flag OFF, then ON), loops B=1,2,4,8,16,32 each time.
set -uo pipefail

WORKTREE="/home/tim/exp_encasync"
GPUS="${GPUS:-5,6}"
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
PORT="${PORT:-8181}"
PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"

BATCHES=(1 2 4 8 16 32)
WARMUP=5
HF_DATASETS="/home/tim/hf_datasets"
HF_HOME_DIR="/m-coriander/coriander/hf"
SERVER_TIMEOUT=5400
SERVER_READY_TIMEOUT=420
BENCH_TIMEOUT=1800
OUTPUT="${OUTPUT:-/home/tim/tmp/full_sweep_encasync_$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTPUT"
SOCK="/home/tim/tmp/sk_encasync_full_${PORT}"

n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

BASE_ENV=(
    "CUDA_VISIBLE_DEVICES=$GPUS"
    "HF_HOME=$HF_HOME_DIR"
    "HF_DATASETS_CACHE=$HF_DATASETS"
    "MSTAR_GPU_MEL=1"
    "MSTAR_GPU_IMAGE_PREPROCESS=1"
    "MSTAR_VISION_GRAPH_ALIGN=1"
    "MSTAR_BATCH_VISION_PREFILL=1"
)

echo "=================================="
echo "Exp 2 FULL SWEEP I2T (async encoder A/B)"
echo "  GPUs: $GPUS (NUMA $NUMA), Port: $PORT"
echo "  Batches: ${BATCHES[*]}"
echo "  Output: $OUTPUT"
echo "=================================="

# GPU idle precheck
for gid in ${GPUS//,/ }; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ "${mem:-0}" -gt 100 ]]; then echo "ERROR: GPU $gid busy ($mem MiB)"; exit 1; fi
done

SERVER_PID=""
cleanup() {
    echo "[$(date -u +%H:%M:%S)] CLEANUP"
    [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null || true
    sleep 3
    for gid in ${GPUS//,/ }; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gid" 2>/dev/null | while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done
    done
}
trap cleanup EXIT INT TERM

launch_server() {
    local label=$1
    shift
    local extra_env=("$@")
    SERVER_LOG="$OUTPUT/server_${label}.log"
    setsid env "${BASE_ENV[@]}" "${extra_env[@]}" \
        bash -c "cd $WORKTREE && \
        timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
        $PYTHON -m mstar.cli.main serve qwen3_omni \
            --config configs/qwen3omni_2gpu.yaml \
            --host 0.0.0.0 --port $PORT \
            --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
        > $SERVER_LOG 2>&1" </dev/null &
    SERVER_PID=$!
    echo "[$(date -u +%H:%M:%S)] [$label] server pid=$SERVER_PID; waiting up to ${SERVER_READY_TIMEOUT}s..."
    for i in $(seq 1 $SERVER_READY_TIMEOUT); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "[$(date -u +%H:%M:%S)] [$label] server ready (~${i}s)"; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: [$label] server died"; tail -30 "$SERVER_LOG"; return 1
        fi
        sleep 1
    done
    echo "ERROR: [$label] server not ready"; tail -30 "$SERVER_LOG"; return 1
}

teardown_server() {
    local label=$1
    echo "[$(date -u +%H:%M:%S)] [$label] tearing down..."
    kill -- -"$SERVER_PID" 2>/dev/null || true
    sleep 5
    SERVER_PID=""
}

run_bench() {
    local label=$1 b=$2
    local n=$(n_requests $b)
    local odir="$OUTPUT/${label}/B${b}"; mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] $label B=$b N=$n ... "
    timeout $BENCH_TIMEOUT bash -c "cd $WORKTREE && \
        PYTHONPATH=$WORKTREE \
        HF_HOME=$HF_HOME_DIR \
        HF_DATASETS_CACHE=$HF_DATASETS \
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
    if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc"; return; fi
    r=$($PYTHON -c "import json; d=json.load(open('$odir/results.json')); print(f\"req/s={d.get('request_throughput',0):.3f} ttft={(d.get('ttft',{}).get('text') or {}).get('p50',0)*1000:.0f}ms\")")
    echo "done — $r"
}

echo ""
echo "===== A: MSTAR_ENCODER_ASYNC=0 ====="
launch_server "A_off" "MSTAR_ENCODER_ASYNC=0" || exit 1
for b in "${BATCHES[@]}"; do run_bench "A_off" $b; done
teardown_server "A_off"

# idle check before next
sleep 5
for gid in ${GPUS//,/ }; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
    if [[ "${mem:-0}" -gt 1000 ]]; then echo "WARN: GPU $gid still has $mem MiB after teardown"; fi
done

echo ""
echo "===== B: MSTAR_ENCODER_ASYNC=1 ====="
launch_server "B_on" "MSTAR_ENCODER_ASYNC=1" "MSTAR_ENCODER_ASYNC_DEPTH=4" || exit 2
for b in "${BATCHES[@]}"; do run_bench "B_on" $b; done
teardown_server "B_on"

echo ""
echo "===== VERDICT ====="
for b in "${BATCHES[@]}"; do
    a_f="$OUTPUT/A_off/B${b}/results.json"
    b_f="$OUTPUT/B_on/B${b}/results.json"
    if [[ -f "$a_f" && -f "$b_f" ]]; then
        $PYTHON -c "
import json
a=json.load(open('$a_f')); b=json.load(open('$b_f'))
ar=a.get('request_throughput',0); br=b.get('request_throughput',0)
at=(a.get('ttft',{}).get('text') or {}).get('p50',0)*1000
bt=(b.get('ttft',{}).get('text') or {}).get('p50',0)*1000
dr = (br-ar)/ar*100 if ar else 0
dt = (bt-at)/at*100 if at else 0
v = 'PROMISING' if dr>=5 else ('NEGATIVE' if dr<=-5 or dt>=10 else 'NEUTRAL')
print(f'B={$b}: A={ar:.3f}req/s {at:.0f}ms  B={br:.3f}req/s {bt:.0f}ms  dr={dr:+.1f}%  dt={dt:+.1f}%  {v}')
"
    fi
done

echo "[$(date -u +%H:%M:%S)] FULL SWEEP COMPLETE"

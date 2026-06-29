#!/usr/bin/env bash
# sweep_d5_bonly.sh — B-only sweep with MSTAR_SPATIAL_MERGE_NODE=1.
# Param: REQTYPE (image_to_text / image_to_speech). Compares against the
# committed mstar_new baseline at /home/tim/bench-sweep-wt.
#
# D5 is a placement experiment: encoder + spatial_merge + Thinker all live on
# the same rank (Rank 1) so this should be a no-op vs the unsplit baseline.
# The goal is to verify zero regression and document the placement boundary.
set -uo pipefail

WORKTREE="/home/tim/exp_d5"
GPUS="${GPUS:-5,6}"
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
PORT="${PORT:-8194}"
REQTYPE="${REQTYPE:-image_to_text}"
PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"

# ninja on PATH (Python extension builds expect it; matches other sweep scripts)
if ! command -v ninja >/dev/null 2>&1; then
    NINJA_DIR=$(dirname "$(find /home/tim -name ninja -type f -executable 2>/dev/null | head -1)")
    [[ -n "$NINJA_DIR" ]] && export PATH="$NINJA_DIR:$PATH"
fi

BATCHES=(1 2 4 8 16 32)
WARMUP=5
HF_DATASETS="/home/tim/hf_datasets"
HF_HOME_DIR="/m-coriander/coriander/hf"
LIBRI_CACHE="/home/tim/tmp/libri_wavs"
SERVER_TIMEOUT=5400
SERVER_READY_TIMEOUT=420
BENCH_TIMEOUT=1800
OUTPUT="${OUTPUT:-/home/tim/tmp/sweep_d5_${REQTYPE}_$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTPUT"
SOCK="/home/tim/tmp/sk_d5_bonly_${PORT}"
CONFIG="$WORKTREE/configs/qwen3omni_2gpu_spatial_merge.yaml"

case "$REQTYPE" in
    audio_to_text|audio_to_speech) DATASET=libri; CACHE="$LIBRI_CACHE" ;;
    image_to_text|image_to_speech) DATASET=food101; CACHE="$HF_DATASETS" ;;
    *) echo "Unknown REQTYPE: $REQTYPE"; exit 1 ;;
esac

n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

echo "=================================="
echo "D5 B-only sweep: MSTAR_SPATIAL_MERGE_NODE=1 on $REQTYPE"
echo "  Worktree: $WORKTREE"
echo "  Config:   $CONFIG"
echo "  GPUs: $GPUS (NUMA $NUMA), Port: $PORT"
echo "  Batches: ${BATCHES[*]}"
echo "  Dataset: $DATASET, Cache: $CACHE"
echo "=================================="

# GPU idle check: refuse to launch on a shared device.
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
    rm -rf "${SOCK}"* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SERVER_LOG="$OUTPUT/server.log"
setsid env \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    HF_HOME="$HF_HOME_DIR" \
    HF_DATASETS_CACHE="$HF_DATASETS" \
    PYTHONPATH="$WORKTREE" \
    MSTAR_GPU_MEL=1 \
    MSTAR_GPU_IMAGE_PREPROCESS=1 \
    MSTAR_VISION_GRAPH_ALIGN=1 \
    MSTAR_BATCH_VISION_PREFILL=1 \
    MSTAR_SPATIAL_MERGE_NODE=1 \
    bash -c "cd $WORKTREE && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    $PYTHON -m mstar.cli.main serve qwen3_omni \
        --gpus $GPUS \
        --config $CONFIG \
        --port $PORT \
        --tensor-comm-protocol SHM \
        --socket-path-prefix $SOCK \
    > $SERVER_LOG 2>&1" </dev/null &
SERVER_PID=$!
echo "[$(date -u +%H:%M:%S)] server pid=$SERVER_PID; waiting up to ${SERVER_READY_TIMEOUT}s..."
sleep 5
for i in $(seq 1 $SERVER_READY_TIMEOUT); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        if grep -q "Conductor process started" "$SERVER_LOG" 2>/dev/null; then
            echo "[$(date -u +%H:%M:%S)] server ready (~${i}s)"; break
        fi
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "ERROR: server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ERROR: server not ready"; tail -30 "$SERVER_LOG"; exit 1
fi

for b in "${BATCHES[@]}"; do
    n=$(n_requests $b)
    odir="$OUTPUT/B${b}"; mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] $REQTYPE B=$b N=$n ... "
    timeout $BENCH_TIMEOUT bash -c "cd $WORKTREE && \
        PYTHONPATH=$WORKTREE \
        HF_HOME=$HF_HOME_DIR \
        HF_DATASETS_CACHE=$HF_DATASETS \
        $PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model qwen3omni \
            --request-type $REQTYPE \
            --dataset $DATASET \
            --profiling-type closed_loop \
            --max-concurrency $b --num-requests $n --num-warmup $WARMUP \
            --inference-system ours \
            --local-cache $CACHE \
            --output-dir $odir" \
        > "$odir/run.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc"; continue; fi
    r=$($PYTHON -c "import json; d=json.load(open('$odir/results.json')); print(f\"req/s={d.get('request_throughput',0):.3f} ttft={(d.get('ttft',{}).get('text') or d.get('ttft',{}).get('audio') or {}).get('p50',0)*1000:.0f}ms\")")
    echo "done — $r"
done

echo ""
echo "===== VERDICT vs committed mstar_new ====="
$PYTHON << PYEOF
import json, os
out = "$OUTPUT"
baseline = '/home/tim/bench-sweep-wt/benchmarks/qwen3-omni-joint/raw_${REQTYPE}.json'
base = json.load(open(baseline))['aggregates']
mod = 'audio' if '$REQTYPE'.endswith('speech') else 'text'
for b in [1,2,4,8,16,32]:
    op = f'{out}/B{b}/results.json'
    if not os.path.exists(op): print(f'B={b}: missing'); continue
    ours = json.load(open(op))
    bk = f'B{b}'
    if bk not in base or 'mstar_new' not in base[bk]: print(f'B={b}: baseline missing'); continue
    bl = base[bk]['mstar_new']['recomputed']
    ar = bl.get('request_throughput', 0)
    br = ours.get('request_throughput', 0)
    at = (base[bk]['mstar_new']['harness'].get(f'ttft_{mod}', {}) or {}).get('p50', 0) * 1000
    bt = (ours.get('ttft', {}).get(mod) or {}).get('p50', 0) * 1000
    dr = (br-ar)/ar*100 if ar else 0
    dt = (bt-at)/at*100 if at else 0
    # D5 is a placement-only change at same-rank, so any non-trivial regression
    # is a bug. Throughput within +/- 5% and TTFT within +10% counts as no-op.
    v = 'NEUTRAL' if -5 <= dr <= 5 and dt <= 10 else ('NEGATIVE' if dr < -5 or dt > 10 else 'PROMISING')
    print(f'B={b}: base={ar:.3f}req/s {at:.0f}ms  ours={br:.3f}req/s {bt:.0f}ms  dr={dr:+.1f}%  dt={dt:+.1f}%  {v}')
PYEOF

echo "[$(date -u +%H:%M:%S)] DONE"

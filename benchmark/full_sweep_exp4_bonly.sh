#!/usr/bin/env bash
# full_sweep_exp4_bonly.sh — Exp 4 B-only sweep.
# MSTAR_ENCODER_CHUNK_COALESCE=1 + MSTAR_CHUNKED_PREFILL=1 on top of the
# committed mstar_new baseline (combined vision opts). Pairs the encoder
# coalescer with REAL intra-prefill chunked Thinker: the coalescer's chunk
# trigger now fires on every intra-walk Thinker chunk boundary (not just
# walk-final), which is the whole point of Exp 4 vs Exp 3.
#
# Halves the time vs A/B variant (single arm, compares against committed
# baseline in /home/tim/bench-sweep-wt/...).
set -uo pipefail

WORKTREE="/home/tim/exp_chunkprefill"
GPUS="${GPUS:-5,6}"
FIRST_GPU="${GPUS%%,*}"
if [[ "$FIRST_GPU" -lt 4 ]]; then NUMA=0; else NUMA=1; fi
PORT="${PORT:-8184}"
PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
export PATH="/home/tim/mstar-encoders/.venv/bin:$PATH"

BATCHES=(1 2 4 8 16 32)
WARMUP=5
HF_DATASETS="/home/tim/hf_datasets"
HF_HOME_DIR="/m-coriander/coriander/hf"
LIBRI_CACHE="/home/tim/tmp/libri_wavs"
SERVER_TIMEOUT=5400
SERVER_READY_TIMEOUT=420
BENCH_TIMEOUT=1800
OUTPUT="${OUTPUT:-/home/tim/tmp/full_sweep_exp4_bonly_$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "$OUTPUT"
SOCK="/home/tim/tmp/sk_exp4_bonly_${PORT}"

n_requests() { local b=$1; local n=$((10 * b)); [ "$n" -lt 50 ] && n=50; echo "$n"; }

echo "=================================="
echo "Exp 4 B-only sweep: MSTAR_ENCODER_CHUNK_COALESCE=1 + MSTAR_CHUNKED_PREFILL=1"
echo "  GPUs: $GPUS (NUMA $NUMA, matches committed mstar_new), Port: $PORT"
echo "  Batches: ${BATCHES[*]}"
echo "  Compares vs: committed mstar_new in raw_audio_to_text.json / raw_image_to_text.json"
echo "=================================="

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
    MSTAR_GPU_MEL=1 \
    MSTAR_GPU_IMAGE_PREPROCESS=1 \
    MSTAR_VISION_GRAPH_ALIGN=1 \
    MSTAR_BATCH_VISION_PREFILL=1 \
    MSTAR_ENCODER_CHUNK_COALESCE=1 \
    MSTAR_ENCODER_COALESCE_SIZE=4 \
    MSTAR_CHUNKED_PREFILL=1 \
    bash -c "cd $WORKTREE && \
    timeout $SERVER_TIMEOUT numactl --cpunodebind=$NUMA --membind=$NUMA \
    $PYTHON -m mstar.cli.main serve qwen3_omni \
        --gpus $GPUS \
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

run_bench() {
    local reqtype=$1 dataset=$2 cache=$3 b=$4
    local n=$(n_requests $b)
    local odir="$OUTPUT/${reqtype}/B${b}"; mkdir -p "$odir"
    echo -n "[$(date -u +%H:%M:%S)] $reqtype B=$b N=$n ... "
    timeout $BENCH_TIMEOUT bash -c "cd $WORKTREE && \
        PYTHONPATH=$WORKTREE \
        HF_HOME=$HF_HOME_DIR \
        HF_DATASETS_CACHE=$HF_DATASETS \
        $PYTHON -m benchmark.runner \
            --url http://127.0.0.1:$PORT \
            --model qwen3omni \
            --request-type $reqtype \
            --dataset $dataset \
            --profiling-type closed_loop \
            --max-concurrency $b --num-requests $n --num-warmup $WARMUP \
            --inference-system ours \
            --local-cache $cache \
            --output-dir $odir" \
        > "$odir/run.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then echo "FAILED rc=$rc"; return; fi
    r=$($PYTHON -c "import json; d=json.load(open('$odir/results.json')); print(f\"req/s={d.get('request_throughput',0):.3f} ttft={(d.get('ttft',{}).get('text') or {}).get('p50',0)*1000:.0f}ms\")")
    echo "done — $r"
}

for b in "${BATCHES[@]}"; do run_bench audio_to_text libri "$LIBRI_CACHE" $b; done
for b in "${BATCHES[@]}"; do run_bench image_to_text food101 "$HF_DATASETS" $b; done

# Compare against committed baseline
echo ""
echo "===== VERDICT vs committed mstar_new ====="
SWEEP_OUT="$OUTPUT" $PYTHON << 'PYEOF'
import json, os
out = os.environ.get('SWEEP_OUT')
baseline_path = '/home/tim/bench-sweep-wt/benchmarks/qwen3-omni-joint'
for path, label in [('audio_to_text', 'S2T'), ('image_to_text', 'I2T')]:
    print(f'--- {label} ({path}) ---')
    base = json.load(open(f'{baseline_path}/raw_{path}.json'))['aggregates']
    for b in [1,2,4,8,16,32]:
        ours_p = f'{out}/{path}/B{b}/results.json'
        if not os.path.exists(ours_p):
            print(f'B={b}: ours missing'); continue
        ours = json.load(open(ours_p))
        bkey = f'B{b}'
        if bkey not in base or 'mstar_new' not in base[bkey]:
            print(f'B={b}: baseline missing'); continue
        bl = base[bkey]['mstar_new']['recomputed']
        ar = bl.get('request_throughput', 0)
        br = ours.get('request_throughput', 0)
        at = (base[bkey]['mstar_new']['harness'].get('ttft_text', {}) or {}).get('p50', 0) * 1000
        bt = (ours.get('ttft', {}).get('text') or {}).get('p50', 0) * 1000
        dr = (br-ar)/ar*100 if ar else 0
        dt = (bt-at)/at*100 if at else 0
        v = 'PROMISING' if dr>=5 else ('NEGATIVE' if dr<=-5 or dt>=10 else 'NEUTRAL')
        print(f'B={b}: base={ar:.3f}req/s {at:.0f}ms  ours={br:.3f}req/s {bt:.0f}ms  dr={dr:+.1f}%  dt={dt:+.1f}%  {v}')
PYEOF

echo "[$(date -u +%H:%M:%S)] EXP4 B-ONLY SWEEP COMPLETE"

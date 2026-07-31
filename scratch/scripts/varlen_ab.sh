#!/usr/bin/env bash
# Quick varlen-backend A/B: m*-new I2T, B=1, warmup=5, n=10.
# Usage: varlen_ab.sh <LABEL> <GPU_A,GPU_B> <PORT> [extra server flags KEY=VAL ...]
set -uo pipefail
LABEL="$1"; GPUS="$2"; PORT="$3"; shift 3; EXTRA=("$@")
WT=/home/tim/integration_mnew_v2
PYTHON=/home/tim/mstar-encoders/.venv/bin/python
SOCK=/home/tim/tmp/sk_vab_${LABEL}_${PORT}
TMPDIR_JIT=/m-coriander/coriander/tmp/vab-jit/${LABEL}      # fast storage — avoids the JIT hang
OUT=/home/tim/benchmarks-personal/varlen-ab/${LABEL}
FIRST=${GPUS%%,*}; [ "$FIRST" -lt 4 ] && NUMA=0 || NUMA=1
mkdir -p "$OUT" "$TMPDIR_JIT" /home/tim/tmp
export PATH=/home/tim/mstar-encoders/.venv/bin:$PATH
log(){ echo "[$(date -u +%H:%M:%S)][$LABEL] $*"; }
for g in ${GPUS//,/ }; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' '); { [ -z "$u" ]||[ "$u" -gt 1000 ]; } && { log "ABORT GPU $g busy"; exit 3; }; done
SPID=""; cleanup(){ [ -n "$SPID" ] && kill -- -"$SPID" 2>/dev/null; for g in ${GPUS//,/ }; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; done; rm -rf "${SOCK}"*; }
trap cleanup EXIT INT TERM
ENVV=( CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
       PYTHONPATH="$WT" TMPDIR="$TMPDIR_JIT" TORCHINDUCTOR_CACHE_DIR="$TMPDIR_JIT/ind" TRITON_CACHE_DIR="$TMPDIR_JIT/tri"
       MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_VLLM_PROMPT_LAYOUT=1 )
for f in "${EXTRA[@]}"; do ENVV+=( "$f" ); done
log "flags: ${EXTRA[*]:-(defaults)}"
t0=$(date +%s)
setsid env "${ENVV[@]}" bash -c "cd '$WT' && timeout 3600 numactl --cpunodebind=$NUMA --membind=$NUMA \
  $PYTHON -m mstar.cli.main serve qwen3_omni --config configs/qwen3omni_2gpu.yaml --host 0.0.0.0 --port $PORT \
  --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO" > "$OUT/server.log" 2>&1 </dev/null &
SPID=$!; log "server pid(group)=$SPID"
ready=0
for i in $(seq 1 300); do
  kill -0 "$SPID" 2>/dev/null || { log "SERVER DIED"; exit 4; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
[ "$ready" -eq 1 ] || { log "server NOT READY (possible flashinfer-JIT hang) after $(( ($(date +%s)-t0)/60 ))min"; exit 5; }
log "server READY in $(( ($(date +%s)-t0) ))s"
# encoder graphs actually captured?
ae=$(grep -c "Captured CUDA graph for audio_encoder" "$OUT/server.log" 2>/dev/null)
ve=$(grep -c "Captured CUDA graph for vision_encoder" "$OUT/server.log" 2>/dev/null)
log "encoder graphs captured: audio=$ae vision=$ve"
timeout 900 env PYTHONPATH="$WT" HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets \
  bash -c "cd '$WT' && $PYTHON -m benchmark.runner --url http://127.0.0.1:$PORT --model qwen3omni \
    --request-type image_to_text --dataset food101 --profiling-type closed_loop \
    --max-concurrency 1 --num-requests 10 --num-warmup 5 --inference-system ours \
    --local-cache /home/tim/hf_datasets --output-dir $OUT" > "$OUT/run.log" 2>&1
log "RUN done (rc=$?)"
echo "{\"label\":\"$LABEL\",\"audio_enc_graphs\":$ae,\"vision_enc_graphs\":$ve}" > "$OUT/meta.json"
log "COMPLETE -> $OUT"

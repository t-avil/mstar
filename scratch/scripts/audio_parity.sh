#!/usr/bin/env bash
# Audio parity driver: generate I2S + S2S audio from one M* build for A/B listening.
# Greedy Thinker (deterministic words) + default Talker sampling (realistic audio).
# Same client runner for every build so prompts/params are identical; identical
# server flags across builds so only the build differs. No warmup, n=3.
#
# Usage: audio_parity.sh <LABEL> <SERVER_WORKTREE> <GPU_A,GPU_B> <PORT>
set -uo pipefail
LABEL="$1"; SERVER_WT="$2"; GPUS="$3"; PORT="$4"

CLIENT_WT=/home/tim/integration_mnew_v2          # ONE runner used as client for ALL builds
PYTHON=/home/tim/mstar-encoders/.venv/bin/python
HF_HOME_DIR=/m-coriander/coriander/hf
DS_CACHE=/home/tim/hf_datasets
LIBRI_CACHE=/home/tim/tmp/libri_wavs
SOCK=/home/tim/tmp/sk_apar_${LABEL}_${PORT}
TMPDIR_JIT=/m-coriander/coriander/tmp/apar-jit/${LABEL}
OUT=/home/tim/benchmarks-personal/audio-parity/${LABEL}
FIRST=${GPUS%%,*}; [ "$FIRST" -lt 4 ] && NUMA=0 || NUMA=1
N=3; WARMUP=0; BATCHES="1 2"
# identical server flags across builds (both old & new support these)
SERVER_FLAGS="MSTAR_VLLM_PROMPT_LAYOUT=1 MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1"
# greedy Thinker so the spoken WORDS are deterministic and cross-build comparable
export BENCH_SPEECH_THINKER_TEMPERATURE=0.0
mkdir -p "$OUT" "$TMPDIR_JIT" /home/tim/tmp
export PATH=/home/tim/mstar-encoders/.venv/bin:$PATH
log(){ echo "[$(date -u +%H:%M:%S)][$LABEL] $*"; }

for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  { [ -z "$used" ] || [ "$used" -gt 1000 ]; } && { log "ABORT GPU $g busy (${used}MiB)"; exit 3; }
done
log "GPUs $GPUS idle (NUMA $NUMA); server from $SERVER_WT"

SPID=""
cleanup(){ log cleanup; [ -n "$SPID" ] && kill -- -"$SPID" 2>/dev/null
  for g in ${GPUS//,/ }; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; done
  rm -rf "${SOCK}"* 2>/dev/null; }
trap cleanup EXIT INT TERM

ENVV=( CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$HF_HOME_DIR" HF_DATASETS_CACHE="$DS_CACHE"
       PYTHONPATH="$SERVER_WT" TMPDIR="$TMPDIR_JIT"
       TORCHINDUCTOR_CACHE_DIR="$TMPDIR_JIT/ind" TRITON_CACHE_DIR="$TMPDIR_JIT/tri" )
for f in $SERVER_FLAGS; do ENVV+=( "$f" ); done
log "server flags: $SERVER_FLAGS"
setsid env "${ENVV[@]}" bash -c "cd '$SERVER_WT' && \
  timeout 5400 numactl --cpunodebind=$NUMA --membind=$NUMA \
  $PYTHON -m mstar.cli.main serve qwen3_omni --config configs/qwen3omni_2gpu.yaml \
    --host 0.0.0.0 --port $PORT --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO" \
  > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!; log "server pid(group)=$SPID"

ready=0
for i in $(seq 1 240); do
  kill -0 "$SPID" 2>/dev/null || { log "SERVER DIED, see server.log"; exit 4; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 5
done
[ "$ready" -eq 1 ] || { log "server not ready"; exit 5; }
log "server READY"

declare -A DS=( [image_to_speech]=food101 [audio_to_speech]=libri )
declare -A CACHE=( [image_to_speech]="$DS_CACHE" [audio_to_speech]="$LIBRI_CACHE" )
declare -A SHORT=( [image_to_speech]=i2s [audio_to_speech]=s2s )
status=complete
for reqtype in image_to_speech audio_to_speech; do
  for b in $BATCHES; do
    odir="$OUT/${SHORT[$reqtype]}/B${b}"; mkdir -p "$odir"
    log "run ${SHORT[$reqtype]} B=$b N=$N"
    if ! timeout 1800 env PYTHONPATH="$CLIENT_WT" HF_HOME="$HF_HOME_DIR" HF_DATASETS_CACHE="$DS_CACHE" \
         BENCH_SPEECH_THINKER_TEMPERATURE=0.0 \
         bash -c "cd '$CLIENT_WT' && $PYTHON -m benchmark.runner \
           --url http://127.0.0.1:$PORT --model qwen3omni \
           --request-type $reqtype --dataset ${DS[$reqtype]} --profiling-type closed_loop \
           --max-concurrency $b --num-requests $N --num-warmup $WARMUP \
           --inference-system ours --local-cache ${CACHE[$reqtype]} --output-dir $odir" \
         > "$odir/run.log" 2>&1; then
      log "RUN FAILED ${SHORT[$reqtype]} B=$b (see run.log)"; status=partial
    fi
    naudio=$(ls "$odir"/*.wav 2>/dev/null | wc -l); log "  saved $naudio wav"
  done
done
echo "{\"label\":\"$LABEL\",\"server_worktree\":\"$SERVER_WT\",\"status\":\"$status\"}" > "$OUT/arm_status.json"
log "AUDIO PARITY $status -> $OUT"

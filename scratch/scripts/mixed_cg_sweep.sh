#!/usr/bin/env bash
# Mixed-walk CUDA-graph I2T sweep driver. One arm per invocation.
# Honors CLAUDE.md GPU hygiene: fixed devices, idle precheck, hard timeout,
# process-group cleanup on every exit, no co-location.
#
# Usage:
#   mixed_cg_sweep.sh <ARM_NAME> <WORKTREE> <GPU_A,GPU_B> <PORT> [extra server env KEY=VAL ...]
# Example:
#   mixed_cg_sweep.sh bucketed /home/tim/mixed-cg-bucketed-wt 2,3 8261 MSTAR_MIXED_WALK=1
set -uo pipefail

ARM="$1"; WORKTREE="$2"; GPUS="$3"; PORT="$4"; shift 4
EXTRA_ENV=("$@")

PYTHON=/home/tim/mstar-encoders/.venv/bin/python
HF_HOME_DIR=/m-coriander/coriander/hf
DS_CACHE=/home/tim/hf_datasets
SOCK=/home/tim/tmp/sk_${ARM}_${PORT}
TMPDIR_JIT=/m-coriander/coriander/tmp/mixedcg-jit/${ARM}   # /home is small; JIT/inductor cache is GBs
OUT=/home/tim/benchmarks-personal/mixed-cg/${ARM}
FIRST_GPU=${GPUS%%,*}
if [ "$FIRST_GPU" -lt 4 ]; then NUMA=0; else NUMA=1; fi
mkdir -p "$OUT" "$TMPDIR_JIT" /home/tim/tmp
export PATH=/home/tim/mstar-encoders/.venv/bin:$PATH

log(){ echo "[$(date -u +%H:%M:%S)][$ARM] $*"; }

# ---- idle GPU precheck (abort if either device busy) ----
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  if [ -z "$used" ] || [ "$used" -gt 1000 ]; then
    log "ABORT: GPU $g not idle (used=${used} MiB). Will not co-locate."; exit 3
  fi
done
log "GPUs $GPUS idle (NUMA $NUMA). Launching server on port $PORT."

SERVER_PID=""
cleanup(){
  log "cleanup: tearing down"
  if [ -n "$SERVER_PID" ]; then kill -- -"$SERVER_PID" 2>/dev/null; fi
  # kill any stray compute apps we own on these GPUs
  for g in ${GPUS//,/ }; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null
    done
  done
  rm -rf "${SOCK}"* 2>/dev/null
  log "cleanup done"
}
trap cleanup EXIT INT TERM

# ---- launch server (hard 90-min ceiling incl. MoE load + graph capture warmup) ----
ENV_ARGS=( CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$HF_HOME_DIR" HF_DATASETS_CACHE="$DS_CACHE"
           PYTHONPATH="$WORKTREE" TMPDIR="$TMPDIR_JIT"
           TORCHINDUCTOR_CACHE_DIR="$TMPDIR_JIT/inductor" TRITON_CACHE_DIR="$TMPDIR_JIT/triton"
           MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_VLLM_PROMPT_LAYOUT=1 )
ENV_ARGS+=( "${EXTRA_ENV[@]}" )
log "server env extras: ${EXTRA_ENV[*]}"
setsid env "${ENV_ARGS[@]}" bash -c "cd '$WORKTREE' && \
  timeout 5400 numactl --cpunodebind=$NUMA --membind=$NUMA \
  $PYTHON -m mstar.cli.main serve qwen3_omni \
    --gpus $GPUS --port $PORT \
    --tensor-comm-protocol SHM --socket-path-prefix $SOCK" \
  > "$OUT/server.log" 2>&1 < /dev/null &
SERVER_PID=$!
log "server pid(group)=$SERVER_PID"

# ---- readiness gate (up to 12 min for MoE load + bucket capture warmup) ----
ready=0
for i in $(seq 1 ${READY_TRIES:-144}); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "SERVER DIED during startup, see server.log"; exit 4; fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
     && grep -q "Conductor process started" "$OUT/server.log" 2>/dev/null; then
    ready=1; break
  fi
  sleep 5
done
[ "$ready" -eq 1 ] || { log "server not ready after timeout"; exit 5; }
log "server READY"

# ---- client sweep: configurable batches (BATCHES), n auto-scaled per batch ----
status="complete"
BATCHES="${BATCHES:-4 8}"
for b in $BATCHES; do
  case $b in
    1) ns="20";;
    16) ns="48";;
    32) ns="64";;
    *) ns="10 20";;
  esac
  for n in $ns; do
    odir="$OUT/B${b}_N${n}"; mkdir -p "$odir"
    log "run B=$b N=$n"
    if ! timeout 1800 env PYTHONPATH="$WORKTREE" HF_HOME="$HF_HOME_DIR" HF_DATASETS_CACHE="$DS_CACHE" \
      bash -c "cd '$WORKTREE' && $PYTHON -m benchmark.runner \
        --url http://127.0.0.1:$PORT --model qwen3omni \
        --request-type image_to_text --dataset food101 \
        --profiling-type closed_loop \
        --max-concurrency $b --num-requests $n --num-warmup 5 \
        --inference-system ours --local-cache $DS_CACHE \
        --output-dir $odir" > "$odir/run.log" 2>&1; then
      log "RUN FAILED B=$b N=$n (see $odir/run.log)"; status="partial"
    fi
  done
done
echo "{\"arm\":\"$ARM\",\"worktree\":\"$WORKTREE\",\"gpus\":\"$GPUS\",\"status\":\"$status\"}" > "$OUT/arm_status.json"
log "SWEEP $status -> $OUT"

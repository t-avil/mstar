#!/usr/bin/env bash
# boot_pd.sh <worktree> <config> <port> <megacache_prefix> [EXTRA_FLAGS...]
# Generic fast-boot: full ship COMMON + preproc pool + gpu-img-preproc, configurable
# worktree/config/port/mega-cache. Mirrors fix20_boot7 gates+readiness but parameterized.
set -uo pipefail
WT=${1:?worktree}; CFG=${2:?config}; PORT=${3:?port}; MC=${4:?megacache_prefix}; shift 4; EXTRA="${*:-}"
NAME="pd$(basename "$PORT")"
COMMON="MSTAR_ORDERED_EMIT=1 MSTAR_CUSTOM_OPS=1 MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_SAMPLER_CFG_CACHE_V2=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1 MSTAR_EMIT_SIDECAR=1 MSTAR_SIDECAR_CHECKSTOP=1 MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_PREP_DEVICE_POS=1 MSTAR_PREP_DEVICE_POS_BATCHED=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 MSTAR_PREFILL_CHUNK_TOKENS=2048 MSTAR_MIXED_BUDGET_TOKENS=4096 TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache_shared TORCHDYNAMO_CACHE_SIZE_LIMIT=128"
PREPROC="MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_PREPROC_PROC=1 MSTAR_PREPROC_PROCS=8 MSTAR_BURST_CAP=1 MSTAR_BURST_THREADS=8"
GRIDS="MSTAR_BATCH_VISION_PREFILL=1 MSTAR_VIS_BATCH_SIZES=1,2,4,8,16,32 MSTAR_PREFILL_BATCH_SIZES=1,2,4,8,16,32"
INFRA="MSTAR_BOOT_PHASES=1 MSTAR_MEGA_CACHE=$MC"
for g in 6 7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$m" -lt 1000 ] || { echo "GPU $g BUSY ($m MiB) — ABORT"; exit 1; }
done
echo "PHASE boot_start $(date -u +%H:%M:%SZ) cfg=$CFG port=$PORT mc=$MC"
setsid bash /m-coriander/coriander/tim/lab_server.sh "$NAME" "$WT" \
  "$COMMON $PREPROC $GRIDS $INFRA $EXTRA" "$CFG" 6,7 "$PORT" 4 < /dev/null &
echo "launched lab_server name=$NAME log=/m-coriander/coriander/tim/lab_${NAME}/server.log"

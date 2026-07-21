#!/usr/bin/env bash
# Boot THE single winning config (replicated-encoder + output-modality routing).
# ONE server serves ALL 5 paths (i2t, s2t, i2s, s2s, t2s). No topology swap.
#
#   Usage: boot_single_config.sh <gpus e.g. 6,7> <port e.g. 8296>
#
# Requires: server venv with M* installed (SVENV below) and this repo checked out
# at branch submission/single-config (== exp/replicated-encoder code). Adjust the
# three paths at the top for your box.
set -uo pipefail
GPUS=${1:?gpus e.g. 6,7}; PORT=${2:?port e.g. 8296}
REPO=$(cd "$(dirname "$0")/../.." && pwd)          # this repo root
SVENV=${SVENV:-/m-coriander/coriander/tim/mstar-new/.venv}   # venv with M* installed
export HF_HOME=${HF_HOME:-/m-coriander/coriander/hf}
export TMPDIR=${TMPDIR:-/m-coriander/coriander/tim/tmp}; mkdir -p "$TMPDIR"
SOCK=$TMPDIR/sk_single_$PORT; rm -rf "$SOCK"; mkdir -p "$SOCK"

# ---- THE WINNING FLAG STACK (all default-off in baseline; enabling = the delta) ----
DECODE="MSTAR_FAST_POSTPROC=1 MSTAR_BATCH_EMIT=1 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1 MSTAR_FAST_SEND=1 MSTAR_EMIT_SIDECAR=1 MSTAR_SIDECAR_CHECKSTOP=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_CODEC_CHUNK_EMIT=1"
EMIT="MSTAR_ORDERED_EMIT=1"
SAMPLER="MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_SAMPLER_CFG_CACHE_V2=1 MSTAR_PREP_DEVICE_POS=1 MSTAR_PREP_DEVICE_POS_BATCHED=1"
KERNELS="MSTAR_MOE_FP8=1 MSTAR_CUSTOM_OPS=1"
PREPROC="MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_PREPROC_PROC=1 MSTAR_PREPROC_PROCS=8 MSTAR_BURST_CAP=1 MSTAR_BURST_THREADS=8"
PREFILL="MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 MSTAR_PREFILL_CHUNK_TOKENS=2048 MSTAR_MIXED_BUDGET_TOKENS=4096"
GRIDS="MSTAR_BATCH_VISION_PREFILL=1 MSTAR_VIS_BATCH_SIZES=1,2,4,8,16,32 MSTAR_PREFILL_BATCH_SIZES=1,2,4,8,16,32"
S2TMERGE="MSTAR_MERGED_PREFILL_AUDIO=1 MSTAR_MERGED_PREFILL_AUDIO_MAX_BS=24"   # s2t-only, occupancy auto-gated
COMPILE="TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_cache_shared TORCHDYNAMO_CACHE_SIZE_LIMIT=128"
FLAGS="$DECODE $EMIT $SAMPLER $KERNELS $PREPROC $PREFILL $GRIDS $S2TMERGE $COMPILE"

# refuse to co-locate on a busy GPU (comparability rule)
IFS=',' read -ra GL <<< "$GPUS"
for g in "${GL[@]}"; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g"|tr -d ' ');
  [ "${m:-9999}" -lt 1000 ] || { echo "GPU $g BUSY ($m MiB) — ABORT (do not co-locate)"; exit 1; }; done

echo "Booting SINGLE config on GPUs $GPUS port $PORT  (config: qwen3omni_2gpu_dpenc.yaml)"
cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPUS PATH="$SVENV/bin:/usr/local/cuda/bin:$PATH" \
  CUDA_HOME=/usr/local/cuda PYTHONPATH="$REPO" HF_HOME="$HF_HOME" $FLAGS \
  timeout 14400 "$SVENV/bin/python" -m mstar.cli.main serve qwen3_omni \
    --config configs/qwen3omni_2gpu_dpenc.yaml --host 0.0.0.0 --port "$PORT" \
    --tensor-comm-protocol SHM --socket-path-prefix "$SOCK" --log-level WARNING

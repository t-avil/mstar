#!/usr/bin/env bash
# Launch instrumented M* Qwen3-Omni server for TTFT decomposition.
# Hard timeout, own process group, unique socket prefix, logs to file.
set -uo pipefail

WT=/home/tim/ttft-wt
VENV=/home/tim/mstar-encoders/.venv
SCRATCH=/m-coriander/coriander/tmp/claude-1072/-home-tim/62167ff1-f44b-495d-8500-0e89b3623c0a/scratchpad/ttft
LOG="$SCRATCH/server.log"
PORT=8140
SOCK="/home/tim/.ttftsock/"   # short path: ZMQ ipc has a 107-char sun_path limit
mkdir -p "$SOCK"

export PYTHONPATH="$WT"
export PATH="$VENV/bin:$PATH"   # flashinfer JIT shells out to `ninja` for prefill graph capture
export HF_HOME="${HF_HOME:-/m-coriander/coriander/hf}"
export HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/m-coriander/coriander/tmp}"
export CUDA_VISIBLE_DEVICES=0,1
# native encoders (M*-new); enable per-stage TTFT instrumentation
export MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=1
export MSTAR_QWEN3_NATIVE_VISION_ENCODER=1
export MSTAR_NODE_TIMING=10

cd "$WT"
echo "PORT=$PORT SOCK=$SOCK LOG=$LOG GPUS=$CUDA_VISIBLE_DEVICES" > "$LOG"
# 25 min hard ceiling on the whole server lifetime (covers warmup+graph capture+runs)
exec timeout --signal=TERM --kill-after=30 1500 \
  "$VENV/bin/python" -m mstar.cli.main serve qwen3_omni \
  --config configs/qwen3omni_2gpu.yaml \
  --gpus 0,1 \
  --tensor-comm-protocol SHM \
  --cache-dir "$HF_HOME/hub" \
  --socket-path-prefix "$SOCK" \
  --port "$PORT" >> "$LOG" 2>&1

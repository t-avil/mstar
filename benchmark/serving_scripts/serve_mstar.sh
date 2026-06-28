#!/usr/bin/env bash
# Launch the M* Qwen3-Omni server through the UNIFIED optimization switchboard
# (#131). Prod and benchmark ablations are the SAME code path: this script only
# ever sets the MSTAR_* env vars that mstar/model/qwen3_omni/config.py
# (OptimizationConfig) resolves -- it never patches code or YAML. Env wins over
# the YAML `model_kwargs.optim` block, so an ablation is one preset away.
#
# Pick a named PRESET (sets all wins at once) and/or override individual flags:
#
#   PRESET=baseline bash serve_mstar.sh   # parity baseline: every win OFF
#   PRESET=full     bash serve_mstar.sh   # all wins ON (native+gpu_mel+gpu_img
#                                         #   +prompt_layout+codec15)
#   PRESET=native   bash serve_mstar.sh   # M*-new encoders only (default)
#   PRESET=hf       bash serve_mstar.sh   # M*-old HF-wrapper encoders
#
#   # ablate a single win on top of a preset (any MSTAR_* flag passes through):
#   PRESET=full MSTAR_GPU_MEL=0 bash serve_mstar.sh
#
# All host paths are env-overridable (defaults match the original bench node).
set -uo pipefail

MSTAR_REPO="${MSTAR_REPO:-/home/timchick/mstar}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
CONFIG="${CONFIG:-configs/qwen3omni_thinker_tp2.yaml}"
GPUS="${GPUS:-0,2,3}"
PORT="${PORT:-8011}"
# PRESET is the unified selector. VARIANT is kept as a back-compat alias
# (native|hf) for the older scripts/runbooks.
PRESET="${PRESET:-${VARIANT:-native}}"

cd "$MSTAR_REPO" || { echo "MSTAR_REPO=$MSTAR_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/tmp/mstar_jit}"; mkdir -p "$TMPDIR"

# preset -> unified OptimizationConfig flags. We only ever EXPORT MSTAR_* here;
# the model resolves them in one place. A flag already present in the
# environment is preserved (so PRESET + a single override composes).
set_default() { local v="$1"; shift; for k in "$@"; do [ -z "${!k:-}" ] && export "$k=$v"; done; }
case "$PRESET" in
  baseline)
    set_default 0 MSTAR_QWEN3_NATIVE_AUDIO_ENCODER MSTAR_QWEN3_NATIVE_VISION_ENCODER \
                  MSTAR_GPU_MEL MSTAR_GPU_IMAGE_PREPROCESS \
                  MSTAR_VLLM_PROMPT_LAYOUT MSTAR_VLLM_AUDIO_SENTINELS ;;
  full)
    set_default 1 MSTAR_QWEN3_NATIVE_AUDIO_ENCODER MSTAR_QWEN3_NATIVE_VISION_ENCODER \
                  MSTAR_GPU_MEL MSTAR_GPU_IMAGE_PREPROCESS \
                  MSTAR_VLLM_PROMPT_LAYOUT ;;
  native)
    set_default 1 MSTAR_QWEN3_NATIVE_AUDIO_ENCODER MSTAR_QWEN3_NATIVE_VISION_ENCODER ;;
  hf)
    set_default 0 MSTAR_QWEN3_NATIVE_AUDIO_ENCODER MSTAR_QWEN3_NATIVE_VISION_ENCODER ;;
  *) echo "PRESET must be baseline|full|native|hf (got '$PRESET')"; exit 2 ;;
esac

echo ">>> serving M* preset=$PRESET  config=$CONFIG gpus=$GPUS port=$PORT"
echo "    optim env:"
env | grep -E '^MSTAR_(QWEN3_NATIVE_|GPU_MEL|GPU_IMAGE_PREPROCESS|VLLM_PROMPT_LAYOUT|VLLM_AUDIO_SENTINELS|CODEC_CHUNK_FRAMES)' | sort | sed 's/^/      /'
exec "${MSTAR_BIN:-mstar}" serve "${MSTAR_MODEL:-qwen3_omni}" \
  --config "$CONFIG" \
  --gpus "$GPUS" \
  --tensor-comm-protocol SHM \
  --cache-dir "$HF_HOME/hub" \
  --port "$PORT"

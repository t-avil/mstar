#!/usr/bin/env bash
# Launch the M* Qwen3-Omni server in one of two encoder variants so the
# M*-old (HF-wrapper encoders) vs M*-new (native encoders) comparison can be
# produced for the figs-5/6 + I2T/S2T charts.
#
#   VARIANT=native bash serve_mstar.sh   # M*-new (default): native encoders
#   VARIANT=hf     bash serve_mstar.sh   # M*-old: HF-wrapper encoders
#
# All host paths are env-overridable (defaults match the original bench node).
set -uo pipefail

MSTAR_REPO="${MSTAR_REPO:-/home/timchick/mstar}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
CONFIG="${CONFIG:-configs/qwen3omni_thinker_tp2.yaml}"
GPUS="${GPUS:-0,2,3}"
PORT="${PORT:-8011}"
VARIANT="${VARIANT:-native}"

cd "$MSTAR_REPO" || { echo "MSTAR_REPO=$MSTAR_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/tmp/mstar_jit}"; mkdir -p "$TMPDIR"

case "$VARIANT" in
  native) export MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=1 MSTAR_QWEN3_NATIVE_VISION_ENCODER=1 ;;
  hf)     export MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=0 MSTAR_QWEN3_NATIVE_VISION_ENCODER=0 ;;
  *) echo "VARIANT must be 'native' or 'hf' (got '$VARIANT')"; exit 2 ;;
esac

echo ">>> serving M*-$VARIANT  config=$CONFIG gpus=$GPUS port=$PORT"
echo "    native_audio=$MSTAR_QWEN3_NATIVE_AUDIO_ENCODER native_vision=$MSTAR_QWEN3_NATIVE_VISION_ENCODER"
exec "${MSTAR_BIN:-mstar}" serve "${MSTAR_MODEL:-qwen3_omni}" \
  --config "$CONFIG" \
  --gpus "$GPUS" \
  --tensor-comm-protocol SHM \
  --cache-dir "$HF_HOME/hub" \
  --port "$PORT"

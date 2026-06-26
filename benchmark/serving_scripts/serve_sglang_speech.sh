#!/usr/bin/env bash
# Launch SGLang-Omni speech server (I2S/T2S) for the Qwen3-Omni comparison.
# All host paths env-overridable (defaults match the original bench node).
set -uo pipefail
SGLANG_REPO="${SGLANG_REPO:-/mnt/storage/timchick/sglang-omni}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
SGLANG_PY="${SGLANG_PY:-$BENCH_ROOT/venvs/sglang-omni/bin/python}"
GPUS="${GPUS:-0,2,3}"; PORT="${PORT:-8092}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
# Per-component GPU placement (indices are into CUDA_VISIBLE_DEVICES=$GPUS).
GPU_THINKER="${GPU_THINKER:-0}"; GPU_TALKER="${GPU_TALKER:-1}"
GPU_AUDIO_ENC="${GPU_AUDIO_ENC:-2}"; GPU_IMAGE_ENC="${GPU_IMAGE_ENC:-2}"

cd "$SGLANG_REPO" || { echo "SGLANG_REPO=$SGLANG_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" TMPDIR="${TMPDIR:-/tmp/mstar_jit}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHINFER_USE_CUDA_NORM=1
exec env CUDA_VISIBLE_DEVICES="$GPUS" "$SGLANG_PY" \
  examples/run_qwen3_omni_speech_server.py --model-path "$MODEL" \
  --gpu-thinker "$GPU_THINKER" --gpu-talker "$GPU_TALKER" \
  --gpu-audio-encoder "$GPU_AUDIO_ENC" --gpu-image-encoder "$GPU_IMAGE_ENC" --port "$PORT"

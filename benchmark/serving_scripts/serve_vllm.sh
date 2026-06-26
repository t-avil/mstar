#!/usr/bin/env bash
# Launch vLLM-Omni for the Qwen3-Omni cross-framework comparison.
# All host paths env-overridable (defaults match the original bench node).
set -uo pipefail
VLLM_REPO="${VLLM_REPO:-/mnt/storage/timchick/vllm-omni}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
VLLM_BIN="${VLLM_BIN:-$BENCH_ROOT/venvs/vllm-omni/bin/vllm}"
GPUS="${GPUS:-0,2}"; PORT="${PORT:-8091}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
STAGE_CFG="${STAGE_CFG:-vllm_omni/deploy/qwen3_omni_moe.yaml}"

cd "$VLLM_REPO" || { echo "VLLM_REPO=$VLLM_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" TMPDIR="${TMPDIR:-$BENCH_ROOT/tmp}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec env CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM_BIN" serve \
  "$MODEL" --omni --port "$PORT" --stage-configs-path "$STAGE_CFG"

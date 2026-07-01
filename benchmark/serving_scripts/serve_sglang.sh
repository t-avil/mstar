#!/usr/bin/env bash
# Launch SGLang-Omni (text paths: I2T/S2T) for the Qwen3-Omni comparison.
# All host paths env-overridable (defaults match the original bench node).
set -uo pipefail
SGLANG_REPO="${SGLANG_REPO:-/mnt/storage/timchick/sglang-omni}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
SGLANG_PY="${SGLANG_PY:-$BENCH_ROOT/venvs/sglang-omni/bin/python}"
GPUS="${GPUS:-0,2}"; PORT="${PORT:-8092}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

cd "$SGLANG_REPO" || { echo "SGLANG_REPO=$SGLANG_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" TMPDIR="${TMPDIR:-$BENCH_ROOT/tmp}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Force the CUDA rmsnorm: flashinfer's cutlass-dsl rmsnorm JIT is SM100+ only and
# DSLRuntimeError's on H100/SM90 (the import try/except doesn't catch it).
export FLASHINFER_USE_CUDA_NORM=1
exec env CUDA_VISIBLE_DEVICES="$GPUS" "$SGLANG_PY" \
  examples/run_qwen3_omni_server.py --model-path "$MODEL" --port "$PORT"

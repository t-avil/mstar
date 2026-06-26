#!/usr/bin/env bash
# Benchmark ONE serving system on the text paths (I2T/S2T) for the TTFT/ITL
# charts, swept over concurrency. SYSTEM= and URL= required.
#   SYSTEM=vllm_omni URL=http://localhost:8091 bash bench_system.sh
# For M*, use SYSTEM=ours with OUT_SUFFIX=_native|_hf to separate the variants.
set -uo pipefail

MSTAR_REPO="${MSTAR_REPO:-/home/timchick/mstar}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
PY="${PY:-$BENCH_ROOT/venvs/mstar/bin/python}"
SYSTEM="${SYSTEM:?set SYSTEM=ours|vllm_omni|sglang_omni}"; URL="${URL:?set URL}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/bench_artifacts}"
OUT="$OUT_ROOT/${SYSTEM}${OUT_SUFFIX:-}"
# TTFT/ITL are reported per concurrency level; default sweeps a few.
BATCHES="${BATCHES:-1 4 8}"
NREQ="${NREQ:-32}"; NWARM="${NWARM:-2}"

cd "$MSTAR_REPO" || { echo "MSTAR_REPO=$MSTAR_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BENCH_ROOT/bench_cache/hf_datasets}"
export TMPDIR="${TMPDIR:-$BENCH_ROOT/tmp}" HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

declare -A DS=( [image_to_text]=food101 [audio_to_text]=libri )
for path in image_to_text audio_to_text; do
  for bs in $BATCHES; do
    od="$OUT/$path/bs$bs"; mkdir -p "$od"
    echo ">>> $SYSTEM${OUT_SUFFIX:-} $path bs=$bs $(date +%H:%M:%S)"
    timeout "${TIMEOUT:-900}" "$PY" -m benchmark.runner \
      --url "$URL" --model qwen3omni --inference-system "$SYSTEM" \
      --request-type "$path" --dataset "${DS[$path]}" \
      --num-requests "$NREQ" --batch-size "$bs" --num-warmup "$NWARM" \
      --profiling-type closed_loop --max-concurrency "$bs" \
      --local-cache "$BENCH_ROOT/bench_cache" \
      --output-dir "$od" > "$od/stdout.txt" 2>&1
    if [ -f "$od/results.json" ]; then echo "    OK $path bs=$bs"; else echo "    FAIL $path bs=$bs (tail:)"; tail -3 "$od/stdout.txt"; \
      "$PY" -m benchmark.serving_scripts.bench_record fail --dir "$od" --system "${SYSTEM}${OUT_SUFFIX:-}" --path "$path" --bs "$bs" --log "$od/stdout.txt"; fi
  done
done
echo "DONE_${SYSTEM}${OUT_SUFFIX:-}"

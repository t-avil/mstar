#!/usr/bin/env bash
# Speech paths (T2S, I2S) with RTF + audio throughput, swept over batch size
# for the figs-5/6 analog. SYSTEM= and URL= required.
#   SYSTEM=vllm_omni URL=http://localhost:8091 bash bench_speech.sh
#   SYSTEM=sglang_omni URL=http://localhost:8092 bash bench_speech.sh
# For M*, use SYSTEM=ours and label via OUT_SUFFIX=_native|_hf so the two
# encoder variants land in distinct dirs.
set -uo pipefail

MSTAR_REPO="${MSTAR_REPO:-/home/timchick/mstar}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
PY="${PY:-$BENCH_ROOT/venvs/mstar/bin/python}"
SYSTEM="${SYSTEM:?set SYSTEM=ours|vllm_omni|sglang_omni}"; URL="${URL:?set URL}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/bench_artifacts}"
OUT="$OUT_ROOT/${SYSTEM}${OUT_SUFFIX:-}"
BATCHES="${BATCHES:-1 4 8 16 32}"
NREQ="${NREQ:-32}"; NWARM="${NWARM:-2}"

cd "$MSTAR_REPO" || { echo "MSTAR_REPO=$MSTAR_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BENCH_ROOT/bench_cache/hf_datasets}"
export TMPDIR="${TMPDIR:-/tmp/mstar_jit}" HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

declare -A DS=( [text_to_speech]=text [image_to_speech]=food101 )
for path in text_to_speech image_to_speech; do
  for bs in $BATCHES; do
    od="$OUT/$path/bs$bs"; mkdir -p "$od"
    echo ">>> $SYSTEM${OUT_SUFFIX:-} $path bs=$bs $(date +%H:%M:%S)"
    timeout "${TIMEOUT:-1200}" "$PY" -m benchmark.runner \
      --url "$URL" --model qwen3omni --inference-system "$SYSTEM" \
      --request-type "$path" --dataset "${DS[$path]}" \
      --num-requests "$NREQ" --batch-size "$bs" --num-warmup "$NWARM" \
      --profiling-type closed_loop --max-concurrency "$bs" \
      --local-cache "$BENCH_ROOT/bench_cache" \
      --output-dir "$od" > "$od/stdout.txt" 2>&1
    [ -f "$od/results.json" ] && echo "    OK $path bs=$bs" || { echo "    FAIL $path bs=$bs"; tail -3 "$od/stdout.txt" | cut -c1-100; \
      "$PY" -m benchmark.serving_scripts.bench_record fail --dir "$od" --system "${SYSTEM}${OUT_SUFFIX:-}" --path "$path" --bs "$bs" --log "$od/stdout.txt"; }
  done
done
echo "SPEECH_DONE_${SYSTEM}${OUT_SUFFIX:-}"

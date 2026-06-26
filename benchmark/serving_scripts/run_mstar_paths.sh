#!/usr/bin/env bash
# Drive the M* server (already running) across I2T / S2T / I2S and a batch
# sweep, writing one results.json per (variant, path, batch) for plot_serving.py.
#
# Run once per encoder variant (the server must be launched with the matching
# serve_mstar.sh VARIANT):
#   VARIANT=native bash run_mstar_paths.sh    # -> $OUT_ROOT/ours_native/...
#   VARIANT=hf     bash run_mstar_paths.sh    # -> $OUT_ROOT/ours_hf/...
#
# All host paths are env-overridable (defaults match the original bench node).
set -uo pipefail

MSTAR_REPO="${MSTAR_REPO:-/home/timchick/mstar}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/storage/timchick}"
PY="${PY:-$BENCH_ROOT/venvs/mstar/bin/python}"
URL="${URL:-http://localhost:8011}"
VARIANT="${VARIANT:-native}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/bench_artifacts}"
OUT="$OUT_ROOT/ours_$VARIANT"
# Paper figs 5/6 sweep B in {1,4,8,16,32}; override with BATCHES="1 4 8".
BATCHES="${BATCHES:-1 4 8 16 32}"
NREQ="${NREQ:-32}"
NWARM="${NWARM:-4}"

cd "$MSTAR_REPO" || { echo "MSTAR_REPO=$MSTAR_REPO not found"; exit 1; }
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BENCH_ROOT/bench_cache/hf_datasets}"
export TMPDIR="${TMPDIR:-$BENCH_ROOT/tmp}" TEMP="$TMPDIR" TMP="$TMPDIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error HF_HUB_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

declare -A DS=( [image_to_text]=food101 [audio_to_text]=libri [image_to_speech]=food101 [text_to_speech]=text )
PATHS="${PATHS:-image_to_text audio_to_text image_to_speech text_to_speech}"
for path in $PATHS; do
  for bs in $BATCHES; do
    od="$OUT/$path/bs$bs"; mkdir -p "$od"
    echo ">>> M*-$VARIANT $path bs=$bs $(date +%H:%M:%S)"
    timeout "${TIMEOUT:-1200}" "$PY" -m benchmark.runner \
      --url "$URL" --model qwen3omni --inference-system ours \
      --request-type "$path" --dataset "${DS[$path]}" \
      --num-requests "$NREQ" --batch-size "$bs" --num-warmup "$NWARM" \
      --profiling-type closed_loop --max-concurrency "$bs" \
      --local-cache "$BENCH_ROOT/bench_cache" \
      --output-dir "$od" > "$od/run.log" 2>&1 \
      && echo "    OK $path bs=$bs" \
      || { echo "    FAIL $path bs=$bs (see $od/run.log)"; \
           [ -f "$od/results.json" ] || "$PY" -m benchmark.serving_scripts.bench_record fail \
             --dir "$od" --system "ours_$VARIANT" --path "$path" --bs "$bs" --log "$od/run.log"; }
  done
done
echo "ALL_MSTAR_PATHS_DONE variant=$VARIANT"

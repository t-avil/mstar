#!/usr/bin/env bash
# Speech paths (T2S, I2S) with RTF + audio throughput. SYSTEM=, URL= required.
set -uo pipefail
cd /home/timchick/mstar
export HF_HOME=/mnt/storage/timchick/hf_cache HF_DATASETS_CACHE=/mnt/storage/timchick/bench_cache/hf_datasets
export TMPDIR=/tmp/mstar_jit HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
PY=/mnt/storage/timchick/venvs/mstar/bin/python
SYSTEM="${SYSTEM:?}"; URL="${URL:?}"; NREQ="${NREQ:-6}"
OUT=/mnt/storage/timchick/bench_artifacts/$SYSTEM
declare -A DS=( [text_to_speech]=text [image_to_speech]=food101 )
for path in text_to_speech image_to_speech; do
  od="$OUT/$path"; mkdir -p "$od"
  echo ">>> $SYSTEM $path $(date +%H:%M:%S)"
  timeout 700 "$PY" -m benchmark.runner \
    --url "$URL" --model qwen3omni --inference-system "$SYSTEM" \
    --request-type "$path" --dataset "${DS[$path]}" \
    --num-requests "$NREQ" --batch-size 1 --num-warmup 1 \
    --profiling-type closed_loop --max-concurrency 1 \
    --local-cache /mnt/storage/timchick/bench_cache \
    --output-dir "$od" > "$od/stdout.txt" 2>&1
  [ -f "$od/results.json" ] && echo "    OK $path" || { echo "    FAIL $path"; tail -3 "$od/stdout.txt" | cut -c1-100; }
done
echo "SPEECH_DONE_$SYSTEM"

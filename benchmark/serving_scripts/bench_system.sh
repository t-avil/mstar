#!/usr/bin/env bash
# Benchmark ONE serving system on I2T/S2T/I2S using the repo harness + parse_i2t_table.
# Usage: SYSTEM=vllm_omni URL=http://localhost:8091 bash bench_system.sh
set -uo pipefail
cd /home/timchick/mstar
export HF_HOME=/mnt/storage/timchick/hf_cache
export HF_DATASETS_CACHE=/mnt/storage/timchick/bench_cache/hf_datasets
export TMPDIR=/mnt/storage/timchick/tmp HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
PY=/mnt/storage/timchick/venvs/mstar/bin/python
SYSTEM="${SYSTEM:?}"; URL="${URL:?}"
NREQ="${NREQ:-8}"
OUT=/mnt/storage/timchick/bench_artifacts/$SYSTEM
declare -A DS=( [image_to_text]=food101 [audio_to_text]=libri [image_to_speech]=food101 )
for path in image_to_text audio_to_text image_to_speech; do
  od="$OUT/$path"; mkdir -p "$od"
  echo ">>> $SYSTEM $path $(date +%H:%M:%S)"
  timeout 600 "$PY" -m benchmark.runner \
    --url "$URL" --model qwen3omni --inference-system "$SYSTEM" \
    --request-type "$path" --dataset "${DS[$path]}" \
    --num-requests "$NREQ" --batch-size 1 --num-warmup 1 \
    --profiling-type closed_loop --max-concurrency 1 \
    --local-cache /mnt/storage/timchick/bench_cache \
    --output-dir "$od" > "$od/stdout.txt" 2>&1
  if [ -f "$od/results.json" ]; then echo "    OK $path"; else echo "    FAIL $path (tail:)"; tail -3 "$od/stdout.txt"; fi
done
echo "DONE_$SYSTEM"

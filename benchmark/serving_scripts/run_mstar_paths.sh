#!/usr/bin/env bash
cd /home/timchick/mstar
export HF_HOME=/mnt/storage/timchick/hf_cache
export HF_DATASETS_CACHE=/mnt/storage/timchick/bench_cache/hf_datasets
export TMPDIR=/mnt/storage/timchick/tmp TEMP=/mnt/storage/timchick/tmp TMP=/mnt/storage/timchick/tmp
export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 DATASETS_VERBOSITY=error HF_HUB_VERBOSITY=error
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
PY=/mnt/storage/timchick/venvs/mstar/bin/python
OUT=/mnt/storage/timchick/bench_artifacts/ours
declare -A DS=( [image_to_text]=food101 [audio_to_text]=libri [image_to_speech]=food101 )
for path in image_to_text audio_to_text image_to_speech; do
  for bs in 1 2 4; do
    od="$OUT/$path/bs$bs"; mkdir -p "$od"
    echo ">>> $path bs=$bs $(date +%H:%M:%S)"
    timeout 900 "$PY" -m benchmark.runner \
      --url http://localhost:8011 --model qwen3omni --inference-system ours \
      --request-type "$path" --dataset "${DS[$path]}" \
      --num-requests 12 --batch-size "$bs" --num-warmup 2 \
      --profiling-type closed_loop --max-concurrency "$bs" \
      --local-cache /mnt/storage/timchick/bench_cache \
      --output-dir "$od" > "$od/run.log" 2>&1 \
      && echo "    OK $path bs=$bs" || echo "    FAIL $path bs=$bs (see $od/run.log)"
  done
done
echo "ALL_MSTAR_PATHS_DONE"

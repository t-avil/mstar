#!/usr/bin/env bash
#
# Drive the existing benchmark harness (benchmark/runner.py) across the
# I2T / S2T / I2S paths for ONE serving system, capturing TTFT / ITL / RTF /
# throughput into per-path results.json artifacts.
#
# The harness already speaks each system's API via --inference-system, so the
# only per-system inputs are the system tag and the server URL.
#
# Usage:
#   SYSTEM=ours      URL=http://0.0.0.0:8000 benchmark/run_omni_paths.sh
#   SYSTEM=vllm_omni URL=http://0.0.0.0:8091 benchmark/run_omni_paths.sh
#   SYSTEM=sglang_omni URL=http://0.0.0.0:8000 benchmark/run_omni_paths.sh
#
# Env knobs: NUM_REQUESTS (default 20), BATCH_SIZES (default "1 2 4"),
#            OUT (default benchmark/artifacts/serving/<system>).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SYSTEM="${SYSTEM:?set SYSTEM=ours|vllm_omni|sglang_omni}"
URL="${URL:?set URL=http://host:port}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"
OUT="${OUT:-benchmark/artifacts/serving/${SYSTEM}}"
PY="${PY:-/mnt/storage/timchick/venvs/mstar/bin/python}"

# path -> dataset (images for I2x, speech for A2T)
declare -A DATASET=( [image_to_text]=vbench [audio_to_text]=libri [image_to_speech]=vbench )

for path in image_to_text audio_to_text image_to_speech; do
  for bs in ${BATCH_SIZES}; do
    odir="${OUT}/${path}/bs${bs}"
    echo "=== ${SYSTEM} ${path} bs=${bs} -> ${odir} ==="
    mkdir -p "${odir}"
    "${PY}" -m benchmark.runner \
      --url "${URL}" --model qwen3omni --inference-system "${SYSTEM}" \
      --request-type "${path}" --dataset "${DATASET[$path]}" \
      --num-requests "${NUM_REQUESTS}" --batch-size "${bs}" \
      --profiling-type closed_loop --max-concurrency "${bs}" \
      --output-dir "${odir}" 2>&1 | tee "${odir}/run.log" || echo "  (path ${path} bs${bs} failed — logged)"
  done
done
echo "DONE ${SYSTEM} -> ${OUT}"

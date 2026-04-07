#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../test/bagel/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
CACHE_DIR=${CACHE_DIR:-/mnt/storage/${WHO:-$USER}/vbench}

python -m benchmark.runner \
    --url "${URL:-http://${HOST}:${PORT}}" \
    --model "${MODEL:-bagel}" \
    --dataset ${DATASET:-vbench} \
    --profiling-type "${PROF_TYPE:-offline}" \
    --request-type "${TASK:-text_to_image}" \
    --vbench-cache-dir "$CACHE_DIR" \
    --num-requests "${NUM_REQUESTS:-10}" \
    --inference-system "${INF_SYS:-ours}" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${RATE:+--rate "$RATE"}

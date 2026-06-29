#!/usr/bin/env bash
# run_benchmark.sh — A/B encoder-coalesce benchmark: control vs experiment
# S2T (audio_to_text) at B=1, B=4, B=8. Each: 5 warmup + 40 measured.
# Sequential: control B=1, exp B=1, control B=4, exp B=4, control B=8, exp B=8.
set -euo pipefail

BENCH_DIR="/home/tim/bench-coalesce-wt/benchmarks/encoder-coalesce"
VENV="/home/tim/mstar/.venv/bin/activate"
CONTROL_URL="http://127.0.0.1:8103"
EXP_URL="http://127.0.0.1:8104"

cd /home/tim/bench-coalesce-wt
source "$VENV"
export HF_HOME=/m-coriander/coriander/hf
export HF_DATASETS_CACHE=/home/tim/hf_datasets

run_one() {
    local label="$1" url="$2" numa="$3" concurrency="$4"
    local outdir="$BENCH_DIR/raw_${label}_s2t_b${concurrency}"
    echo "=== $(date -u +%Y%m%dT%H%M%SZ) Running $label B=$concurrency ==="
    numactl --cpunodebind="$numa" --membind="$numa" \
        timeout 1800 python -m benchmark.runner \
        --url "$url" \
        --model qwen3omni \
        --request-type audio_to_text \
        --dataset libri \
        --profiling-type closed_loop \
        --max-concurrency "$concurrency" \
        --num-requests 40 \
        --num-warmup 5 \
        --inference-system ours \
        --local-cache /home/tim/tmp/libri_wavs \
        --output-dir "$outdir"
    echo "=== $(date -u +%Y%m%dT%H%M%SZ) Done $label B=$concurrency ==="
}

# Sequential runs: control B=1, exp B=1, control B=4, exp B=4, control B=8, exp B=8
for B in 1 4 8; do
    run_one "control" "$CONTROL_URL" 0 "$B"
    run_one "exp"     "$EXP_URL"     1 "$B"
done

echo "ALL BENCHMARK RUNS COMPLETE"

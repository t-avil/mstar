#!/usr/bin/env bash
# Drive the M* benchmark runner for S2T (libri) and I2T (food101), B=1, closed-loop.
# 5 warmup + 25 measured each. Server must already be up on :8140.
set -uo pipefail
ENC=/home/tim/mstar-encoders
VENV=$ENC/.venv
BD=/home/tim/ttft-wt/benchmarks/ttft-decompose
URL=http://0.0.0.0:8140
export HF_HOME=/m-coriander/coriander/hf
# Shared dataset cache dirs are owned by other users (read-only locks); use a
# writable mirror (real dirs + symlinked arrow files) so `datasets` can lock.
export HF_DATASETS_CACHE=/m-coriander/coriander/tmp/claude-1072/-home-tim/62167ff1-f44b-495d-8500-0e89b3623c0a/scratchpad/ttft/hfds
export HF_HUB_OFFLINE=1
export PATH="$VENV/bin:$PATH"
cd "$ENC"

run_path () {
  local rtype="$1" ds="$2" odir="$3"
  mkdir -p "$odir"
  echo "=== $rtype ($ds) -> $odir ==="
  timeout 420 "$VENV/bin/python" -m benchmark.runner \
    --url "$URL" --model qwen3omni --inference-system ours \
    --request-type "$rtype" --dataset "$ds" \
    --num-requests 25 --num-warmup 5 --batch-size 1 \
    --profiling-type closed_loop --max-concurrency 1 \
    --output-dir "$odir" 2>&1 | tee "$odir/run.log"
}

run_path audio_to_text libri    "$BD/s2t"
echo "##### S2T_DONE marker #####"
run_path image_to_text food101  "$BD/i2t"
echo "##### I2T_DONE marker #####"
echo "ALL CLIENT RUNS DONE"

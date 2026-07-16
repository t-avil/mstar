#!/usr/bin/env bash
# bench_cell.sh — one matched-protocol benchmark cell against a running M* server.
# Matched to committed vLLM-Omni 0.22 protocol: closed_loop, natural EOS, greedy.
# Usage: bench_cell.sh <port> <path:i2t|s2t> <batch> <n> <warmup> <outdir> [EXTRA_RUNNER_ARGS...]
set -uo pipefail
PORT="${1:?port}"; PTH="${2:?i2t|s2t}"; B="${3:?batch}"; N="${4:?n}"; W="${5:?warmup}"; OUT="${6:?outdir}"; shift 6
EXTRA="${*:-}"
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
case "$PTH" in
  i2t) REQTYPE=image_to_text; DS=food101; CACHE=/m-coriander/coriander/hf ;;
  s2t) REQTYPE=audio_to_text; DS=libri;   CACHE=/home/tim/tmp/libri_wavs ;;
  *) echo "bad path $PTH"; exit 2 ;;
esac
mkdir -p "$OUT"
# GPUs 6,7 => NUMA node 1
env MSTAR_BENCH_GREEDY=1 HF_DATASETS_CACHE=/m-coriander/coriander/hf PYTHONPATH="$BENCH" \
  numactl --cpunodebind=1 --membind=1 \
  "$CVENV/bin/python" -m benchmark.runner \
    --url "http://127.0.0.1:$PORT" --model qwen3omni \
    --request-type "$REQTYPE" --dataset "$DS" \
    --profiling-type closed_loop \
    --max-concurrency "$B" --num-requests "$N" --num-warmup "$W" \
    --inference-system ours --local-cache "$CACHE" \
    --output-dir "$OUT" $EXTRA > "$OUT/cell.log" 2>&1
rc=$?
python3 /m-coriander/coriander/tim/campaign_i2t/analyze.py "$OUT/results.json" "$PTH" "$B" 2>/dev/null || echo "NO_RESULTS (rc=$rc) see $OUT/cell.log"
exit $rc

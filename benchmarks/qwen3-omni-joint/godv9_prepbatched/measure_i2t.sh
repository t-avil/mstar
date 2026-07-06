#!/usr/bin/env bash
# Honest i2t measurement of the losing/tie cells against a warm server.
# Usage: measure_i2t.sh <port> <outdir> <cells: "B:N,B:N,...">
set -uo pipefail
PORT=$1; OUT=$2; CELLS=$3
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
NUMAPFX=""
[ -f "$(dirname "$OUT")/numa_node" ] && NUMAPFX="numactl --cpunodebind=$(cat "$(dirname "$OUT")/numa_node") --membind=$(cat "$(dirname "$OUT")/numa_node")"
mkdir -p "$OUT"
IFS=',' read -ra CL <<< "$CELLS"
for c in "${CL[@]}"; do
  b=${c%%:*}; n=${c##*:}; od="$OUT/i2t_B${b}"; mkdir -p "$od"
  echo "[$(date -u +%H:%M:%S)] cell i2t B$b n=$n ..."
  PYTHONPATH=$BENCH timeout 1200 $NUMAPFX "$CVENV/bin/python" -m benchmark.runner \
    --url "http://127.0.0.1:$PORT" --model qwen3omni --request-type image_to_text \
    --dataset food101 --profiling-type closed_loop --max-concurrency "$b" \
    --num-requests "$n" --num-warmup 4 --inference-system ours \
    --local-cache /m-coriander/coriander/hf --output-dir "$od" > "$od/run.log" 2>&1
  "$CVENV/bin/python" - "$od/results.json" "$b" <<'PY' 2>/dev/null || echo "  B$b FAIL (see $od/run.log)"
import json,sys
d=json.load(open(sys.argv[1])); b=sys.argv[2]
def g(x,k):
    v=(x or {}).get(k); return v if v is not None else float('nan')
tt=d.get('ttft') or {}; it=d.get('itl') or {}
print("  RESULT i2t B%s | req/s %.3f | tok/s %.1f | ttft_ms mean %.1f p50 %.1f p95 %.1f | itl_ms mean %.2f p50 %.2f p95 %.2f | jct_ms %.0f | n=%s"%(
  b, d.get('request_throughput') or 0, d.get('text_token_throughput') or 0,
  g(tt,'mean'),g(tt,'p50'),g(tt,'p95'), g(it,'mean'),g(it,'p50'),g(it,'p95'),
  d.get('jct_mean_ms') or 0, d.get('completed') or d.get('num_requests') or '?'))
PY
done
echo "[$(date -u +%H:%M:%S)] MEASURE_DONE $OUT"

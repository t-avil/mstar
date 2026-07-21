#!/usr/bin/env bash
# t2s (text->speech) sweep, one server, natEOS closed-loop, speech n-cadence.
# Usage: t2s_sweep.sh <inference-system> <port> <label> [batches]
set -uo pipefail
SYS=${1:?system}; PORT=${2:?port}; LABEL=${3:?label}; BATCHES=${4:-"1 2 4 8 16 32"}
URL=http://127.0.0.1:$PORT
CVENV=/home/tim/mstar-encoders/.venv; BENCH=/m-coriander/coriander/tim/bench-v2
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
export TMPDIR=/m-coriander/coriander/tim/tmp; mkdir -p "$TMPDIR" /m-coriander/coriander/tim/hf_datasets
OUT=/m-coriander/coriander/tim/exp_nateos_out/$LABEL; mkdir -p "$OUT"
curl -sf "$URL/v1/models" >/dev/null 2>&1 || { echo "SERVER NOT UP on $PORT"; exit 20; }
declare -A NS=( [1]=12 [2]=20 [4]=24 [8]=40 [16]=64 [32]=96 )
for B in $BATCHES; do
  d="$OUT/text_to_speech_b${B}"; mkdir -p "$d"
  cd "$BENCH" && PYTHONPATH=$BENCH timeout 2400 "$CVENV/bin/python" -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type text_to_speech --dataset text --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests "${NS[$B]}" --num-warmup 5 --inference-system "$SYS" --local-cache /home/tim/tmp/libri_wavs \
    --output-dir "$d" > "$d/stdout.txt" 2>&1
  a=$(grep -oE '"audio_seconds_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  r=$(grep -oE '"request_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  c=$(grep -oE '"completed": *[0-9]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9]+'|head -1)
  echo "[$(date -u +%H:%M:%S)] $LABEL t2s B$B comp=$c audio_s/s=${a:--} req/s=${r:--}"
done
echo "T2S_${LABEL}_DONE"

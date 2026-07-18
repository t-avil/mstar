#!/usr/bin/env bash
# Full NATURAL-EOS sweep (the regime where BOTH user targets live):
#   text i2t/s2t tok/s (vs committed older-M* natural-EOS)
#   speech i2s/s2s audio_s/s + req/s + rtf (vs committed vLLM 0.22 natural-EOS)
# Reports the key scalar per cell. Usage: exp_full_nateos.sh <port> <label> [batches]
set -uo pipefail
PORT=${1:-8321}; LABEL=${2:-encoff}; BATCHES=${3:-"1 2 4 8 16 32"}
URL=http://127.0.0.1:$PORT
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/hf
export TMPDIR=/m-coriander/coriander/tim/tmp; mkdir -p "$TMPDIR"
OUT=/m-coriander/coriander/tim/exp_nateos_out/$LABEL; mkdir -p "$OUT"
curl -sf "$URL/v1/models" >/dev/null 2>&1 || { echo "SERVER NOT UP on $PORT"; exit 20; }
# per-batch request counts: text stable so mid-n; speech slow so small n (match committed)
declare -A NT=( [1]=64 [2]=64 [4]=96 [8]=96 [16]=96 [32]=128 )
declare -A NS=( [1]=12 [2]=20 [4]=24 [8]=40 [16]=64 [32]=96 )

textcell(){ local rt=$1 ds=$2 B=$3; local d="$OUT/${rt}_b${B}"; mkdir -p "$d"
  PYTHONPATH=$BENCH timeout 1800 "$CVENV/bin/python" -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type "$rt" --dataset "$ds" --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests "${NT[$B]}" --num-warmup 8 --inference-system ours --local-cache /home/tim/tmp/libri_wavs \
    --output-dir "$d" > "$d/stdout.txt" 2>&1
  local rc=$?; local tok=$(grep -oE '"text_token_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local rps=$(grep -oE '"request_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local comp=$(grep -oE '"completed": *[0-9]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9]+'|head -1)
  echo "[$(date -u +%H:%M:%S)] $LABEL $rt B$B rc=$rc comp=$comp tok/s=$tok req/s=$rps"; }

speechcell(){ local rt=$1 ds=$2 B=$3; local d="$OUT/${rt}_b${B}"; mkdir -p "$d"
  PYTHONPATH=$BENCH timeout 2400 "$CVENV/bin/python" -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type "$rt" --dataset "$ds" --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests "${NS[$B]}" --num-warmup 5 --inference-system ours --local-cache /home/tim/tmp/libri_wavs \
    --output-dir "$d" > "$d/stdout.txt" 2>&1
  local rc=$?; local aud=$(grep -oE '"audio_seconds_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local rps=$(grep -oE '"request_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local comp=$(grep -oE '"completed": *[0-9]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9]+'|head -1)
  echo "[$(date -u +%H:%M:%S)] $LABEL $rt B$B rc=$rc comp=$comp audio_s/s=$aud req/s=$rps"; }

echo "=== TEXT natural-EOS ==="
for B in $BATCHES; do textcell image_to_text food101 "$B"; done
for B in $BATCHES; do textcell audio_to_text libri "$B"; done
echo "=== SPEECH natural-EOS ==="
for B in $BATCHES; do speechcell image_to_speech food101 "$B"; done
for B in $BATCHES; do speechcell audio_to_speech libri "$B"; done
echo "FULL_NATEOS_DONE $LABEL"

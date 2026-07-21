#!/usr/bin/env bash
# Sweep ALL 5 paths against the ONE single-config server, natural-EOS, closed-loop.
# Writes results.json per cell under submission/raw/dpenc/<rt>_b<B>/.
#
#   Usage: run_sweep.sh <port> [batches]
#
# Requires the benchmark client venv (CVENV) + the bench harness (BENCH, the
# `benchmark` package in this repo) + libri_wavs local cache. Adjust paths at top.
set -uo pipefail
PORT=${1:?port}; BATCHES=${2:-"1 2 4 8 16 32"}
URL=http://127.0.0.1:$PORT
CVENV=${CVENV:-/home/tim/mstar-encoders/.venv}                 # client venv
BENCH=${BENCH:-/m-coriander/coriander/tim/bench-v2}            # dir containing the `benchmark` pkg
LIBRI=${LIBRI:-/home/tim/tmp/libri_wavs}                       # cached speech inputs
export HF_HOME=${HF_HOME:-/m-coriander/coriander/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/m-coriander/coriander/tim/hf_datasets}
export TMPDIR=${TMPDIR:-/m-coriander/coriander/tim/tmp}; mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
OUT=$(cd "$(dirname "$0")/../raw/dpenc" && pwd); mkdir -p "$OUT"
curl -sf "$URL/v1/models" >/dev/null 2>&1 || { echo "SERVER NOT UP on $PORT"; exit 20; }

# committed n-cadence (text heavier than speech; B32 text n=128, s2t validated at n=256)
declare -A NT=( [1]=64 [2]=64 [4]=96 [8]=96 [16]=96 [32]=128 )
declare -A NS=( [1]=12 [2]=20 [4]=24 [8]=40 [16]=64 [32]=96 )

cell(){ local rt=$1 ds=$2 B=$3 n=$4 warm=$5 tout=$6; local d="$OUT/${rt}_b${B}"; mkdir -p "$d"
  cd "$BENCH" && PYTHONPATH="$BENCH" timeout "$tout" "$CVENV/bin/python" -m benchmark.runner \
    --url "$URL" --model qwen3omni --request-type "$rt" --dataset "$ds" \
    --profiling-type closed_loop --max-concurrency "$B" --num-requests "$n" --num-warmup "$warm" \
    --inference-system ours --local-cache "$LIBRI" --output-dir "$d" > "$d/stdout.txt" 2>&1
  local tok=$(grep -oE '"text_token_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local rps=$(grep -oE '"request_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9.]+'|head -1)
  local comp=$(grep -oE '"completed": *[0-9]+' "$d/results.json" 2>/dev/null|grep -oE '[0-9]+'|head -1)
  echo "[$(date -u +%H:%M:%S)] $rt B$B comp=$comp tok/s=${tok:--} req/s=${rps:--}"; }

for B in $BATCHES; do cell image_to_text   food101 "$B" "${NT[$B]}" 8 1800; done  # i2t
for B in $BATCHES; do cell audio_to_text   libri   "$B" "${NT[$B]}" 8 1800; done  # s2t
for B in $BATCHES; do cell image_to_speech food101 "$B" "${NS[$B]}" 5 2400; done  # i2s
for B in $BATCHES; do cell audio_to_speech libri   "$B" "${NS[$B]}" 5 2400; done  # s2s
for B in $BATCHES; do cell text_to_speech  text    "$B" "${NS[$B]}" 5 2400; done  # t2s
echo "SWEEP_DONE -> $OUT"

#!/usr/bin/env bash
# GATING EXPERIMENT #1: is the speech "1.5x drop" a measurement artifact?
# Same live build (godv9 encoff on :8321), same batch, speech i2s+s2s,
# natural-EOS vs fixed-256. If natural-EOS ~2-3x vLLM(committed) while
# fixed-256 ~1.5x -> artifact. Compare each regime to its OWN committed vLLM ref.
set -uo pipefail
PORT=${1:-8321}; B=${2:-8}; N=${3:-32}
URL=http://127.0.0.1:$PORT
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/hf
OUT=/m-coriander/coriander/tim/exp_gate_out; mkdir -p "$OUT"
export TMPDIR=/m-coriander/coriander/tim/tmp; mkdir -p "$TMPDIR"

curl -sf "$URL/v1/models" >/dev/null 2>&1 || { echo "SERVER NOT UP on $PORT"; exit 20; }

cell(){ local tag=$1 rt=$2 ds=$3; shift 3; local d="$OUT/$tag"; mkdir -p "$d"
  PYTHONPATH=$BENCH timeout 1200 "$CVENV/bin/python" -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type "$rt" --dataset "$ds" --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests "$N" --num-warmup 3 --inference-system ours --local-cache /home/tim/tmp/libri_wavs \
    "$@" --output-dir "$d" > "$d/stdout.txt" 2>&1
  local rc=$?
  local aud=$(grep -oE '"audio_seconds_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+')
  local rps=$(grep -oE '"request_throughput": *[0-9.]+' "$d/results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+')
  local adur=$(grep -oE '"audio_duration_mean_s": *[0-9.]+' "$d/results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+')
  local comp=$(grep -oE '"completed": *[0-9]+' "$d/results.json" 2>/dev/null | head -1 | grep -oE '[0-9]+')
  echo "[$(date -u +%H:%M:%S)] $tag rc=$rc comp=$comp audio_s/s=$aud req/s=$rps mean_audio_dur_s=$adur"
}

echo "=== i2s B$B ==="
cell i2s_natEOS  image_to_speech food101
cell i2s_fixed256 image_to_speech food101 --ignore-eos --output-len-min 256 --output-len-max 256
echo "=== s2s B$B ==="
cell s2s_natEOS  audio_to_speech libri
cell s2s_fixed256 audio_to_speech libri --ignore-eos --output-len-min 256 --output-len-max 256
echo "GATE_DONE"

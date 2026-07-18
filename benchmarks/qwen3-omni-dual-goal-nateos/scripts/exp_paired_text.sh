#!/usr/bin/env bash
# Clean paired text A/B: encoff(:8323, node1) vs base(:8322, node1), interleaved
# per cell under identical wall-clock load -> isolates the pure TOPOLOGY effect
# on text tok/s (no cross-time load confound). High-batch cells matter most.
set -uo pipefail
PE=${1:-8323}; PB=${2:-8322}; ROUNDS=${3:-2}
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/hf
export TMPDIR=/m-coriander/coriander/tim/tmp; mkdir -p "$TMPDIR"
OUT=/m-coriander/coriander/tim/exp_paired_out; mkdir -p "$OUT"
for p in $PE $PB; do curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 || { echo "server $p not up"; exit 20; }; done

cell(){ local port=$1 rt=$2 ds=$3 B=$4 tag=$5; local d="$OUT/${tag}_${rt}_b${B}"; mkdir -p "$d"
  PYTHONPATH=$BENCH timeout 900 "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$port" --model qwen3omni \
    --request-type "$rt" --dataset "$ds" --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests 96 --num-warmup 6 --inference-system ours --local-cache /home/tim/tmp/libri_wavs \
    --output-dir "$d" > "$d/stdout.txt" 2>&1
  # parse tok/s from authoritative stdout ("Throughput: 874.12 text tok/s")
  grep -oE '[0-9.]+ text tok/s' "$d/stdout.txt" 2>/dev/null | grep -oE '^[0-9.]+' | head -1; }

declare -A ES BS   # accumulate for median-ish (just print each round)
for r in $(seq 1 $ROUNDS); do
 for rt_ds in "image_to_text food101" "audio_to_text libri"; do
  set -- $rt_ds; rt=$1; ds=$2
  for B in 8 16 32; do
    # alternate which topology goes first each round to balance ordering
    if [ $((r % 2)) -eq 1 ]; then
      e=$(cell $PE $rt $ds $B "encoff_r$r"); b=$(cell $PB $rt $ds $B "base_r$r")
    else
      b=$(cell $PB $rt $ds $B "base_r$r"); e=$(cell $PE $rt $ds $B "encoff_r$r")
    fi
    ratio=$(awk -v a="$e" -v b="$b" 'BEGIN{if(b>0)printf "%.3f", a/b; else print "na"}')
    echo "[$(date -u +%H:%M:%S)] r$r $rt B$B  encoff=$e  base=$b  encoff/base=${ratio}x  load=$(cut -d' ' -f1 /proc/loadavg)"
  done
 done
done
echo "PAIRED_TEXT_DONE"

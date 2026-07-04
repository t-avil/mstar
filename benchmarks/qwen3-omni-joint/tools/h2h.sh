#!/usr/bin/env bash
# Live head-to-head: M* (canonical 6,7:8321) vs vLLM-Omni 0.22 (2,3:8093).
# Alternating per-cell back-to-back — same window, contention cancels.
set -uo pipefail
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
OUT=/m-coriander/coriander/tim/h2h_out; mkdir -p "$OUT"
declare -A P=( [i2t]=image_to_text [s2t]=audio_to_text )
declare -A DS=( [i2t]=food101 [s2t]=libri )
declare -A CACHE=( [i2t]=/m-coriander/coriander/hf [s2t]=/home/tim/tmp/libri_wavs )
nfor(){ case $1 in 1) echo 12;; 8) echo 48;; 32) echo 96;; *) echo $((3*$1));; esac; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

run_one(){ # $1 sys(mstar|vllm) $2 path $3 B $4 idx
  local sys=$1 s=$2 b=$3 i=$4 n url isys node
  n=$(nfor $b)
  if [ "$sys" = mstar ]; then url=http://127.0.0.1:8321; isys=ours; node=1; else url=http://127.0.0.1:8093; isys=vllm_omni; node=0; fi
  local od="$OUT/${sys}_${s}_B${b}_$i"; mkdir -p "$od"
  PYTHONPATH=$BENCH timeout 1800 numactl --cpunodebind=$node --membind=$node \
    "$CVENV/bin/python" -m benchmark.runner --url "$url" --model qwen3omni \
    --request-type "${P[$s]}" --dataset "${DS[$s]}" --profiling-type closed_loop \
    --max-concurrency "$b" --num-requests "$n" --num-warmup 4 \
    --inference-system "$isys" --local-cache "${CACHE[$s]}" \
    --output-dir "$od" > "$od/run.log" 2>&1
  "$CVENV/bin/python" -c "import json;d=json.load(open('$od/results.json'));print('H2H $sys $s B$b #$i req/s %.3f tok/s %.1f jct %.0f'%(d['request_throughput'],d.get('text_token_throughput') or 0,d['jct_mean_ms']))" 2>/dev/null || echo "H2H $sys $s B$b #$i FAIL"
}

i=0
for c in i2t:32 i2t:32 i2t:8 s2t:8; do
  i=$((i+1)); s=${c%%:*}; b=${c##*:}
  run_one mstar $s $b $i
  run_one vllm  $s $b $i
done
log H2H_DONE

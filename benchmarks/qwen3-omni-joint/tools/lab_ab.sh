#!/usr/bin/env bash
# Fire an interleaved A/B against a RUNNING lab server (see lab_server.sh).
# Cost = cell time only (~1-3 min per cell): this is the 5-minute signal loop.
#
# Usage: lab_ab.sh <lab_name> <exp_name> '<flagsA_json>' '<flagsB_json>' [port] [cells] [rounds] [n_override]
#   cells: path:batch list (default i2t:32). n_override: reqs per cell
#   (default full-n 96 at B32; use 48 for smoke).
set -uo pipefail
LAB=$1; EXP=$2; FA=$3; FB=$4; PORT=${5:-8299}; CELLS=${6:-i2t:32}; ROUNDS=${7:-2}; NOVR=${8:-}
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
LABDIR=/m-coriander/coriander/tim/lab_$LAB
FLAGS_FILE="$LABDIR/dynflags.json"
[ -f "$FLAGS_FILE" ] || { echo "no lab '$LAB' (missing $FLAGS_FILE)"; exit 1; }
# dynflags applies ONLY keys present in the JSON — a '{}' side would leave the
# other side's values sticky (this bug invalidated a night of decompositions).
# Require identical key sets in both sides, with explicit values everywhere.
KEYSA=$(python3 -c "import json,sys;print(sorted(json.loads(sys.argv[1]).keys()))" "$FA" 2>/dev/null)
KEYSB=$(python3 -c "import json,sys;print(sorted(json.loads(sys.argv[1]).keys()))" "$FB" 2>/dev/null)
[ -n "$KEYSA" ] && [ "$KEYSA" = "$KEYSB" ] || { echo "REFUSED: flagsA/flagsB must be valid JSON with IDENTICAL key sets (explicit 0/1 per key; no '{}' vs keyed). A=$KEYSA B=$KEYSB"; exit 1; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "lab server not responding on $PORT"; exit 1; }
OUT="$LABDIR/ab_$EXP"; mkdir -p "$OUT"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
declare -A P=( [i2t]=image_to_text [s2t]=audio_to_text [i2s]=image_to_speech [s2s]=audio_to_speech )
declare -A DS=( [i2t]=food101 [s2t]=libri [i2s]=food101 [s2s]=libri )
declare -A CACHE=( [i2t]=/m-coriander/coriander/hf [s2t]=/home/tim/tmp/libri_wavs [i2s]=/m-coriander/coriander/hf [s2s]=/home/tim/tmp/libri_wavs )
nfor(){ [ -n "$NOVR" ] && { echo $NOVR; return; }; case $1 in 1) echo 12;; 8) echo 48;; 32) echo 96;; *) echo $((3*$1));; esac; }

# Pin the bench client to the lab server's NUMA node (unpinned clients float
# across nodes and add cross-node noise when multiple labs run concurrently).
NUMAPFX=""
if [ -f "$LABDIR/numa_node" ]; then NUMAPFX="numactl --cpunodebind=$(cat "$LABDIR/numa_node") --membind=$(cat "$LABDIR/numa_node")"; fi

run_cell(){ local side=$1 s=$2 b=$3 n; n=$(nfor $3)
  local od="$OUT/${side}_${s}_B${b}_r${ROUND}"; mkdir -p "$od"
  PYTHONPATH=$BENCH timeout 1200 $NUMAPFX "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$PORT" \
    --model qwen3omni --request-type "${P[$s]}" --dataset "${DS[$s]}" --profiling-type closed_loop \
    --max-concurrency "$b" --num-requests "$n" --num-warmup 2 --inference-system ours \
    --local-cache "${CACHE[$s]}" --output-dir "$od" > "$od/run.log" 2>&1
  "$CVENV/bin/python" -c "import json;d=json.load(open('$od/results.json'));print('LAB $LAB/$EXP $side $s B$b r$ROUND req/s %.3f tok/s %.1f jct %.0f'%(d['request_throughput'],d.get('text_token_throughput') or 0,d['jct_mean_ms']))" 2>/dev/null || echo "LAB $LAB/$EXP $side $s B$b r$ROUND FAIL"
}
for ROUND in $(seq 1 $ROUNDS); do
  IFS=',' read -ra CL <<< "$CELLS"
  for c in "${CL[@]}"; do
    s=${c%%:*}; b=${c##*:}
    echo "$FA" > "$FLAGS_FILE"; sleep 2
    run_cell A $s $b
    echo "$FB" > "$FLAGS_FILE"; sleep 2
    run_cell B $s $b
  done
done
echo "$FA" > "$FLAGS_FILE"   # leave the lab in the A (baseline) state
log "LAB_AB_DONE $LAB/$EXP"

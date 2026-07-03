#!/usr/bin/env bash
# Interleaved A/B on ONE server via MSTAR_DYNFLAGS: no per-config restart, and
# A/B cells are adjacent in time so box noise cancels. Triage only — commit
# numbers still come from static-env runs.
#
# Usage: dyn_ab.sh <name> <worktree> "<static_flags>" "<flagsA_json>" "<flagsB_json>" \
#                  [config] [gpus] [port] [cells] [rounds]
#   cells: comma list path:batch (default i2t:32,s2t:8); each round runs every
#          cell under A then under B (A,B adjacent per cell).
set -uo pipefail
NAME=$1; WT=$2; STATIC=$3; FA=$4; FB=$5
CFG=${6:-configs/qwen3omni_2gpu_encoff.yaml}
GPUS=${7:-6,7}; PORT=${8:-8299}; CELLS=${9:-i2t:32,s2t:8}; ROUNDS=${10:-2}
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
SVENV=/m-coriander/coriander/tim/mstar-new/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
OUT=/m-coriander/coriander/tim/dyn_$NAME; mkdir -p "$OUT"
SOCK=/home/tim/tmp/sk_dyn_$PORT; rm -rf "$SOCK"; mkdir -p "$SOCK"
FLAGS_FILE="$OUT/dynflags.json"
echo "$FA" > "$FLAGS_FILE"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
declare -A P=( [i2t]=image_to_text [s2t]=audio_to_text [i2s]=image_to_speech [s2s]=audio_to_speech )
declare -A DS=( [i2t]=food101 [s2t]=libri [i2s]=food101 [s2s]=libri )
declare -A CACHE=( [i2t]=/m-coriander/coriander/hf [s2t]=/home/tim/tmp/libri_wavs [i2s]=/m-coriander/coriander/hf [s2s]=/home/tim/tmp/libri_wavs )
nfor(){ case $1 in 1) echo 12;; 8) echo 48;; 32) echo 96;; *) echo $((3*$1));; esac; }

IFS=',' read -ra GL <<< "$GPUS"
for g in "${GL[@]}"; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g|tr -d ' '); [ "${m:-9999}" -gt 200 ] && { echo "ABORT gpu $g busy"; exit 1; }; done

# NUMA node derived from the FIRST GPU's PCI bus (was hardcoded 1 — wrong for
# GPUs 0-3 on this box). Override with DYN_NUMA_NODE.
if [ -z "${DYN_NUMA_NODE:-}" ]; then
  _bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i ${GL[0]} | tr -d ' ')
  DYN_NUMA_NODE=$(cat "/sys/bus/pci/devices/0000:$(echo "${_bus#*:}" | tr 'A-F' 'a-f')/numa_node" 2>/dev/null)
  case "$DYN_NUMA_NODE" in ''|-1) DYN_NUMA_NODE=1;; esac
fi

export CUDA_VISIBLE_DEVICES=$GPUS
log "DYN $NAME: launch $WT ($STATIC) dynflags=$FLAGS_FILE on $GPUS:$PORT numa=$DYN_NUMA_NODE"
setsid bash -c "cd $WT && PATH='$SVENV/bin:/usr/local/cuda/bin:'\$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=$WT HF_HOME=$HF_HOME $STATIC MSTAR_DYNFLAGS=$FLAGS_FILE \
  timeout 7200 numactl --cpunodebind=$DYN_NUMA_NODE --membind=$DYN_NUMA_NODE $SVENV/bin/python -m mstar.cli.main serve qwen3_omni \
  --config $CFG --host 0.0.0.0 --port $PORT --tensor-comm-protocol SHM \
  --socket-path-prefix $SOCK --log-level WARNING" </dev/null > "$OUT/server.log" 2>&1 &
SPID=$!
cleanup(){ kill -- -$SPID 2>/dev/null||true; sleep 3; for g in "${GL[@]}"; do nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null|while read -r p; do [ -n "$p" ]&&kill "$p" 2>/dev/null||true; done; done; }
trap cleanup EXIT INT TERM
for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 && { log "ready ~$((i*3))s"; break; }; kill -0 $SPID 2>/dev/null||{ echo SERVER_DIED; tail -25 "$OUT/server.log"; exit 1; }; sleep 3; done
curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 || { echo NEVER_READY; exit 1; }

CVENV=/home/tim/mstar-encoders/.venv
run_cell(){ # $1 side(A|B) $2 path $3 batch
  local side=$1 s=$2 b=$3 n; n=$(nfor $3)
  local od="$OUT/${side}_${s}_B${b}_r${ROUND}"; mkdir -p "$od"
  PYTHONPATH=$BENCH timeout 1200 "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$PORT" \
    --model qwen3omni --request-type "${P[$s]}" --dataset "${DS[$s]}" --profiling-type closed_loop \
    --max-concurrency "$b" --num-requests "$n" --num-warmup 4 --inference-system ours \
    --local-cache "${CACHE[$s]}" --output-dir "$od" > "$od/run.log" 2>&1
  "$CVENV/bin/python" -c "import json;d=json.load(open('$od/results.json'));print('DYN $NAME $side $s B$b r$ROUND req/s %.3f tok/s %.1f jct %.0f'%(d['request_throughput'],d.get('text_token_throughput') or 0,d['jct_mean_ms']))" 2>/dev/null || echo "DYN $NAME $side $s B$b r$ROUND FAIL"
}

for ROUND in $(seq 1 $ROUNDS); do
  IFS=',' read -ra CL <<< "$CELLS"
  for c in "${CL[@]}"; do
    s=${c%%:*}; b=${c##*:}
    echo "$FA" > "$FLAGS_FILE"; sleep 2   # settle: pollers pick up within ~50 steps
    run_cell A $s $b
    echo "$FB" > "$FLAGS_FILE"; sleep 2
    run_cell B $s $b
  done
done
log "DYN_DONE $NAME"

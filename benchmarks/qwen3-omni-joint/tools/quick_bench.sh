#!/usr/bin/env bash
# FAST triage bench: one server, small cells, ~10-14 min end-to-end.
# Usage: quick_bench.sh <name> <worktree> "<flags>" [config] [gpus] [port] [cells]
#   cells: comma list of path:batch (default i2t:1,i2t:8,i2t:32,s2t:8)
# Emits QB rows; compare against a reference run of the current best config on
# the SAME gpu pair, same session (triage only — confirm winners with the
# interleaved A/B before believing them).
set -uo pipefail
NAME=$1; WT=$2; FLAGS=$3; CFG=${4:-configs/qwen3omni_2gpu_encoff.yaml}
GPUS=${5:-4,5}; PORT=${6:-8260}; CELLS=${7:-i2t:1,i2t:8,i2t:32,s2t:8}
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
SVENV=/m-coriander/coriander/tim/mstar-new/.venv
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
OUT=/m-coriander/coriander/tim/qb_$NAME; mkdir -p "$OUT"
SOCK=/home/tim/tmp/sk_qb_$PORT; rm -rf "$SOCK"; mkdir -p "$SOCK"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
declare -A P=( [i2t]=image_to_text [s2t]=audio_to_text [i2s]=image_to_speech [s2s]=audio_to_speech )
declare -A DS=( [i2t]=food101 [s2t]=libri [i2s]=food101 [s2s]=libri )
declare -A CACHE=( [i2t]=/m-coriander/coriander/hf [s2t]=/home/tim/tmp/libri_wavs [i2s]=/m-coriander/coriander/hf [s2s]=/home/tim/tmp/libri_wavs )
# QB_FAST=1: half-size cells for triage-speed iteration (48 reqs at B32).
if [ "${QB_FAST:-0}" = "1" ]; then
nfor(){ case $1 in 1) echo 8;; 8) echo 32;; 32) echo 48;; *) echo $((2*$1));; esac; }
else
nfor(){ case $1 in 1) echo 12;; 8) echo 48;; 32) echo 96;; *) echo $((3*$1));; esac; }
fi

IFS=',' read -ra GL <<< "$GPUS"
for g in "${GL[@]}"; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g|tr -d ' '); [ "${m:-9999}" -gt 200 ] && { echo "ABORT gpu $g busy"; exit 1; }; done

# NUMA node derived from the FIRST GPU's PCI bus (was hardcoded 1 — wrong for
# GPUs 0-3 on this box). Override with QB_NUMA_NODE.
if [ -z "${QB_NUMA_NODE:-}" ]; then
  _bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i ${GL[0]} | tr -d ' ')
  QB_NUMA_NODE=$(cat "/sys/bus/pci/devices/0000:$(echo "${_bus#*:}" | tr 'A-F' 'a-f')/numa_node" 2>/dev/null)
  case "$QB_NUMA_NODE" in ''|-1) QB_NUMA_NODE=1;; esac
fi
log "QB $NAME: numa node $QB_NUMA_NODE for gpus $GPUS"

export CUDA_VISIBLE_DEVICES=$GPUS
log "QB $NAME: launch $WT ($FLAGS) on $GPUS:$PORT"
setsid bash -c "cd $WT && PATH='$SVENV/bin:/usr/local/cuda/bin:'\$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=$WT HF_HOME=$HF_HOME $FLAGS \
  timeout 5400 numactl --cpunodebind=$QB_NUMA_NODE --membind=$QB_NUMA_NODE $SVENV/bin/python -m mstar.cli.main serve qwen3_omni \
  --config $CFG --host 0.0.0.0 --port $PORT --tensor-comm-protocol SHM \
  --socket-path-prefix $SOCK --log-level WARNING" </dev/null > "$OUT/server.log" 2>&1 &
SPID=$!
cleanup(){ kill -- -$SPID 2>/dev/null||true; sleep 3; for g in "${GL[@]}"; do nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null|while read -r p; do [ -n "$p" ]&&kill "$p" 2>/dev/null||true; done; done; }
trap cleanup EXIT INT TERM
for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 && { log "ready ~$((i*3))s"; break; }; kill -0 $SPID 2>/dev/null||{ echo SERVER_DIED; tail -25 "$OUT/server.log"; exit 1; }; sleep 3; done
curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 || { echo NEVER_READY; exit 1; }

# Server warm-up cells (discarded): the first measured cell after server-ready
# reads ~30% low (measured 2026-07-03: 5.04 vs 7.2 steady on identical config —
# JIT/caches/allocator maturity spans ~100+ requests, far beyond --num-warmup).
# Run one small throwaway cell per distinct path before measuring. Opt out
# with QB_NO_WARMCELL=1.
if [ "${QB_NO_WARMCELL:-0}" != "1" ]; then
  IFS=',' read -ra CL <<< "$CELLS"
  seen_paths=""
  for c in "${CL[@]}"; do
    s=${c%%:*}
    case " $seen_paths " in *" $s "*) continue;; esac
    seen_paths="$seen_paths $s"
    wd="$OUT/warmcell_$s"; mkdir -p "$wd"
    log "warm cell $s (discarded)"
    PYTHONPATH=$BENCH timeout 900 "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$PORT" \
      --model qwen3omni --request-type "${P[$s]}" --dataset "${DS[$s]}" --profiling-type closed_loop \
      --max-concurrency 32 --num-requests 64 --num-warmup 0 --inference-system ours \
      --local-cache "${CACHE[$s]}" --output-dir "$wd" > "$wd/run.log" 2>&1 || true
  done
fi

IFS=',' read -ra CL <<< "$CELLS"
for c in "${CL[@]}"; do
  s=${c%%:*}; b=${c##*:}; n=$(nfor $b); od="$OUT/${s}_B$b"; mkdir -p "$od"
  PYTHONPATH=$BENCH timeout 1200 "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$PORT" \
    --model qwen3omni --request-type "${P[$s]}" --dataset "${DS[$s]}" --profiling-type closed_loop \
    --max-concurrency "$b" --num-requests "$n" --num-warmup 4 --inference-system ours \
    --local-cache "${CACHE[$s]}" --output-dir "$od" > "$od/run.log" 2>&1
  "$CVENV/bin/python" -c "import json;d=json.load(open('$od/results.json'));print('QB $NAME $s B$b req/s %.3f tok/s %.1f audio_s/s %.2f jct %.0f'%(d['request_throughput'],d.get('text_token_throughput') or 0,d.get('audio_seconds_throughput') or 0,d['jct_mean_ms']))" 2>/dev/null || echo "QB $NAME $s B$b FAIL"
done
log "QB_DONE $NAME"

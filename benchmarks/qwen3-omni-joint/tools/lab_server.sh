#!/usr/bin/env bash
# Persistent warm "lab" server for fast flag iteration. Boot ONCE (~7-10 min
# incl. one discarded warm cell), then fire lab_ab.sh against it repeatedly —
# each A/B pair costs only cell time (~2-6 min), no reboot, no cold cells.
#
# Usage: lab_server.sh <name> <worktree> "<static_flags>" [config] [gpus] [port] [max_hours]
#   Dynflags file: /m-coriander/coriander/tim/lab_<name>/dynflags.json
#   Triage-fast boot: prepend MSTAR_DECODE_BUCKETS/MSTAR_PREFILL_BUCKETS trims
#   to static_flags (capture-grid trim; TRIAGE ONLY — never for ship numbers).
set -uo pipefail
NAME=$1; WT=$2; FLAGS=$3; CFG=${4:-configs/qwen3omni_2gpu_encoff.yaml}
GPUS=${5:-6,7}; PORT=${6:-8299}; HOURS=${7:-4}
export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets
SVENV=/m-coriander/coriander/tim/mstar-new/.venv
CVENV=/home/tim/mstar-encoders/.venv
BENCH=/m-coriander/coriander/tim/bench-v2
OUT=/m-coriander/coriander/tim/lab_$NAME; mkdir -p "$OUT"
SOCK=/home/tim/tmp/sk_lab_$PORT; rm -rf "$SOCK"; mkdir -p "$SOCK"
FLAGS_FILE="$OUT/dynflags.json"; echo '{}' > "$FLAGS_FILE"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

IFS=',' read -ra GL <<< "$GPUS"
for g in "${GL[@]}"; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g|tr -d ' '); [ "${m:-9999}" -gt 200 ] && { echo "ABORT gpu $g busy"; exit 1; }; done

# NUMA node from first GPU's PCI bus (override: LAB_NUMA_NODE)
if [ -z "${LAB_NUMA_NODE:-}" ]; then
  _bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i ${GL[0]} | tr -d ' ')
  LAB_NUMA_NODE=$(cat "/sys/bus/pci/devices/0000:$(echo "${_bus#*:}" | tr 'A-F' 'a-f')/numa_node" 2>/dev/null)
  case "$LAB_NUMA_NODE" in ''|-1) LAB_NUMA_NODE=1;; esac
fi

export CUDA_VISIBLE_DEVICES=$GPUS
log "LAB $NAME: boot $WT on $GPUS:$PORT numa=$LAB_NUMA_NODE ttl=${HOURS}h dynflags=$FLAGS_FILE"
setsid bash -c "cd $WT && PATH='$SVENV/bin:/usr/local/cuda/bin:'\$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=$WT HF_HOME=$HF_HOME $FLAGS MSTAR_DYNFLAGS=$FLAGS_FILE \
  timeout $((HOURS*3600)) numactl --cpunodebind=$LAB_NUMA_NODE --membind=$LAB_NUMA_NODE $SVENV/bin/python -m mstar.cli.main serve qwen3_omni \
  --config $CFG --host 0.0.0.0 --port $PORT --tensor-comm-protocol SHM \
  --socket-path-prefix $SOCK --log-level WARNING" </dev/null > "$OUT/server.log" 2>&1 &
SPID=$!
echo "$SPID" > "$OUT/server.pgid"
cleanup(){ kill -- -$SPID 2>/dev/null||true; sleep 3; for g in "${GL[@]}"; do nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null|while read -r p; do [ -n "$p" ]&&kill "$p" 2>/dev/null||true; done; done; }
trap cleanup INT TERM   # NOT EXIT: server outlives this script by design; TTL timeout + lab_kill.sh are the reapers
for i in $(seq 1 400); do curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 && { log "ready ~$((i*3))s"; break; }; kill -0 $SPID 2>/dev/null||{ echo SERVER_DIED; tail -25 "$OUT/server.log"; exit 1; }; sleep 3; done
curl -sf "http://127.0.0.1:$PORT/v1/models">/dev/null 2>&1 || { echo NEVER_READY; exit 1; }

# One discarded warm cell (server steady-state: first cell reads ~30% low)
wd="$OUT/warmcell"; mkdir -p "$wd"
log "warm cell (discarded)"
PYTHONPATH=$BENCH timeout 900 "$CVENV/bin/python" -m benchmark.runner --url "http://127.0.0.1:$PORT" \
  --model qwen3omni --request-type image_to_text --dataset food101 --profiling-type closed_loop \
  --max-concurrency 32 --num-requests 96 --num-warmup 0 --inference-system ours \
  --local-cache /m-coriander/coriander/hf --output-dir "$wd" > "$wd/run.log" 2>&1 || true
log "LAB $NAME WARM+READY on port $PORT (ttl ${HOURS}h). Fire lab_ab.sh $NAME ... ; kill: lab_kill.sh $NAME"

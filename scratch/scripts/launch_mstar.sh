#!/usr/bin/env bash
# launch_mstar.sh <repo_dir> <gpus> <numa> <port> <sockname> <logfile> [extra env KEY=VAL ...]
# Daemonizes mstar-serve via setsid so it survives background-task reaping.
REPO="$1"; GPUS="$2"; NUMA="$3"; PORT="$4"; SOCKN="$5"; LOG="$6"; shift 6
SOCK="/home/tim/tmp/$SOCKN"; rm -rf "$SOCK"; mkdir -p "$SOCK"
ENVEXP=""; for kv in "$@"; do ENVEXP+="export $kv; "; done
setsid bash -c "cd $REPO && source .venv/bin/activate && \
  export CUDA_VISIBLE_DEVICES=$GPUS HF_HOME=/m-coriander/coriander/hf TMPDIR=/home/tim/tmp/launch-tmp; $ENVEXP \
  numactl --cpunodebind=$NUMA --membind=$NUMA mstar-serve --config configs/qwen3omni_2gpu.yaml \
    --host 0.0.0.0 --port $PORT --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
  > $LOG 2>&1" </dev/null >/dev/null 2>&1 &
disown
echo "launched detached on GPUs $GPUS port $PORT (log $LOG)"

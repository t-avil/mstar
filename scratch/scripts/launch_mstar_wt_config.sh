#!/usr/bin/env bash
# launch_mstar_wt_config.sh <worktree> <gpus> <numa> <port> <sockname> <logfile> <config> [ENV=VAL ...]
# Same as launch_mstar_wt.sh but takes an explicit config path as the 7th argument.
WT="$1"; GPUS="$2"; NUMA="$3"; PORT="$4"; SOCKN="$5"; LOG="$6"; CONFIG="$7"; shift 7
SHARED=/home/tim/mstar-encoders/.venv
OV="$WT/.edit_override"
if [ ! -f "$OV/__editable___mstar_0_1_0_finder.py" ]; then
  mkdir -p "$OV"
  sed "s#/home/tim/vllm-layout-wt#$WT#g" \
    "$SHARED/lib/python3.12/site-packages/__editable___mstar_0_1_0_finder.py" \
    > "$OV/__editable___mstar_0_1_0_finder.py"
fi
SOCK="/home/tim/tmp/$SOCKN"; rm -rf "$SOCK"; mkdir -p "$SOCK"
ENVEXP=""; for kv in "$@"; do ENVEXP+="export $kv; "; done
setsid bash -c "cd $WT && source $SHARED/bin/activate && \
  export PYTHONPATH=$OV CUDA_VISIBLE_DEVICES=$GPUS HF_HOME=/m-coriander/coriander/hf TMPDIR=/home/tim/tmp/launch-tmp; $ENVEXP \
  numactl --cpunodebind=$NUMA --membind=$NUMA python -m mstar.cli.main serve qwen3_omni \
    --config $CONFIG --host 0.0.0.0 --port $PORT \
    --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
  > $LOG 2>&1" </dev/null >/dev/null 2>&1 &
disown
echo "launched $WT mstar on GPUs $GPUS numa$NUMA port $PORT config=$CONFIG (override=$OV log=$LOG)"

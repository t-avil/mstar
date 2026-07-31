#!/usr/bin/env bash
# launch_mstar_wt.sh <worktree> <gpus> <numa> <port> <sockname> <logfile> [ENV=VAL ...]
# Serves a worktree's OWN mstar code using the shared venv (/home/tim/mstar-encoders/.venv)
# + a per-worktree editable-finder override (<worktree>/.edit_override) so no reinstall is needed.
# Daemonizes via setsid so it survives background-task reaping. Pins CUDA_VISIBLE_DEVICES + NUMA.
WT="$1"; GPUS="$2"; NUMA="$3"; PORT="$4"; SOCKN="$5"; LOG="$6"; shift 6
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
    --config configs/qwen3omni_2gpu.yaml --host 0.0.0.0 --port $PORT \
    --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
  > $LOG 2>&1" </dev/null >/dev/null 2>&1 &
disown
echo "launched $WT mstar on GPUs $GPUS numa$NUMA port $PORT (override=$OV log=$LOG)"
echo "verify code path: PYTHONPATH=$OV $SHARED/bin/python -c 'import mstar;print(mstar.__file__)'"

#!/usr/bin/env bash
# launch_server.sh <variant: off|on> <port> <gpus>
set -uo pipefail
variant="$1"; port="$2"; gpus="$3"
SCRATCH=/home/tim/merge-prefill-wt/benchmark-personal/merge-prefill-walks
cd /home/tim/merge-prefill-wt
source /home/tim/mstar-encoders/.venv/bin/activate
export CUDA_VISIBLE_DEVICES="$gpus"
export HF_HOME=/m-coriander/coriander/hf
export HF_HUB_OFFLINE=1
export TMPDIR=/m-coriander/coriander/tmp
if [ "$variant" = "on" ]; then export MSTAR_MERGE_PREFILL_WALKS=1; else unset MSTAR_MERGE_PREFILL_WALKS; fi
exec mstar-serve --config configs/qwen3omni_2gpu.yaml \
  --host 0.0.0.0 --port "$port" \
  --socket-path-prefix "/tmp/mstar_merge_${variant}_$$" \
  --tensor-comm-protocol SHM \
  --cache-dir "$HF_HOME/hub"

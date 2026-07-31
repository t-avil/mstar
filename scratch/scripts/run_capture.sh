#!/usr/bin/env bash
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for M*-new (8094) + vLLM (8093) to come up"
for i in $(seq 1 90); do
  up94=$( (exec 3<>/dev/tcp/127.0.0.1/8094) 2>/dev/null && echo 1 || echo 0)
  up93=$( (exec 3<>/dev/tcp/127.0.0.1/8093) 2>/dev/null && echo 1 || echo 0)
  [ "$up94" = 1 ] && [ "$up93" = 1 ] && { echo "[$(ts)] both UP"; break; }
  sleep 10
done
sleep 8
export PYTHONPATH=/home/tim/mstar HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
source /home/tim/mstar/.venv/bin/activate
echo "[$(ts)] === THINKER TEXT: M*-new vs vLLM (same clips) ==="
timeout 240 python /home/tim/text_divergence.py 2>&1 | grep -viE 'GLIBCXX|Warning|warn|trust_remote|loading script|Parquet'
echo "[$(ts)] CAPTURE DONE"

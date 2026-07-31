#!/usr/bin/env bash
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for M* 8103"
for i in $(seq 1 90); do (exec 3<>/dev/tcp/127.0.0.1/8103) 2>/dev/null && { echo "UP"; break; }; sleep 10; done
(exec 3<>/dev/tcp/127.0.0.1/8103) 2>/dev/null || { echo "TIMEOUT"; exit 2; }
sleep 5
export PYTHONPATH=/home/tim/mstar HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
source /home/tim/mstar/.venv/bin/activate
echo "[$(ts)] === M* clip4: transcribe vs answer ==="
timeout 300 python /home/tim/mstar_answer_test.py 2>&1 | grep -viE 'GLIBCXX|Warning|warn|trust_remote|loading script|Parquet'
echo "[$(ts)] ANSWER TEST DONE"

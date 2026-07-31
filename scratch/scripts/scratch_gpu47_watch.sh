#!/usr/bin/env bash
# Exit (re-invoking the agent) as soon as GPUs 4-7 are all essentially idle.
for i in $(seq 1 720); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' '$1>=4 && $2>2000 {c++} END{print c+0}')
  if [ "${busy:-9}" -eq 0 ]; then
    echo "GPUs 4-7 are now IDLE at $(date -u +%Y%m%dT%H%M%SZ)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 0
  fi
  sleep 120
done
echo "watcher timed out after ~24h with 4-7 still busy"

#!/usr/bin/env bash
# Kill a lab server + free its GPUs. Usage: lab_kill.sh <name>
set -uo pipefail
OUT=/m-coriander/coriander/tim/lab_$1
[ -f "$OUT/server.pgid" ] || { echo "no lab '$1'"; exit 1; }
kill -- -$(cat "$OUT/server.pgid") 2>/dev/null || true
sleep 3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "lab $1 killed"

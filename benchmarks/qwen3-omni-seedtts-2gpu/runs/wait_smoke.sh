#!/usr/bin/env bash
# wait_smoke.sh <system> <url>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf
cd /home/tim/mstar; source .venv/bin/activate
SYS="$1"; URL="$2"; PORT="${URL##*:}"
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for $SYS port $PORT (up to 25 min)"
for i in $(seq 1 300); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "[$(ts)] PORT OPEN ~$((i*5))s"; break; }
  sleep 5
done
(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "[$(ts)] TIMEOUT"; exit 2; }
echo "[$(ts)] SMOKE 1 req for $SYS"
timeout 400 python -m benchmark.runner --url "$URL" --model qwen3omni \
  --request-type text_to_speech --dataset seed_tts \
  --seed-tts-dir /home/tim/seedtts-cache/seedtts_testset --seed-tts-locale en \
  --profiling-type offline --batch-size 1 --num-requests 1 --num-warmup 0 \
  --inference-system "$SYS" --local-cache /home/tim/seedtts-cache \
  --output-dir "$D/runs/smoke_${SYS}" 2>&1 | tail -22
echo "[$(ts)] SMOKE DONE $SYS"

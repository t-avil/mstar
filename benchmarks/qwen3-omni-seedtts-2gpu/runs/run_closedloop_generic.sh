#!/usr/bin/env bash
# run_closedloop_generic.sh <system> <url> <port>  — closed-loop max-con=32 datapoint
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf
cd /home/tim/mstar; source .venv/bin/activate
SYS="$1"; URL="$2"; PORT="$3"
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
OUT="$D/runs/out_${SYS}_closedloop/B32"; mkdir -p "$OUT"
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for $SYS port $PORT (up to 30 min)"
for i in $(seq 1 360); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "[$(ts)] PORT OPEN ~$((i*5))s"; break; }
  sleep 5
done
(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "TIMEOUT"; exit 2; }
echo "[$(ts)] RUN $SYS closed-loop max-concurrency=32, warmup=5, num_requests=160"
timeout 2700 python -m benchmark.runner \
  --url "$URL" --model qwen3omni \
  --request-type text_to_speech --dataset seed_tts \
  --seed-tts-dir /home/tim/seedtts-cache/seedtts_testset --seed-tts-locale en \
  --profiling-type closed_loop --max-concurrency 32 \
  --num-requests 160 --num-warmup 5 \
  --inference-system "$SYS" --local-cache /home/tim/seedtts-cache \
  --output-dir "$OUT" > "$OUT/stdout.txt" 2>&1
echo "[$(ts)] rc=$?"
echo "===== RESULT ($SYS closed-loop B=32) ====="
grep -E '^RTF|audio sec/s|Requests :|Total wall' "$OUT/stdout.txt" | tail -5
echo "[$(ts)] DONE"

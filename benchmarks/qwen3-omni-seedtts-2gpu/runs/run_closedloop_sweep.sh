#!/usr/bin/env bash
# Full closed-loop sweep matching the fork scripts + reference CSV.
# B(max-concurrency) in {1,2,4,8,16,32}; warmup=5; num_requests per CSV.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf
SYS="$1"; URL="$2"; PORT="${URL##*:}"
cd /home/tim/mstar; source .venv/bin/activate
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
SEED=/home/tim/seedtts-cache/seedtts_testset
ts(){ date -u +%Y%m%dT%H%M%SZ; }
declare -A NREQ=( [1]=12 [2]=20 [4]=24 [8]=40 [16]=80 [32]=160 )
echo "[$(ts)] waiting $SYS port $PORT"; for i in $(seq 1 360); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
for B in 1 2 4 8 16 32; do
  N=${NREQ[$B]}; OUT="$D/runs/cl/out_${SYS}/B${B}"; mkdir -p "$OUT"
  echo "[$(ts)] $SYS closed_loop max-con=$B reqs=$N warmup=5"
  timeout 3000 python -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type text_to_speech --dataset seed_tts --seed-tts-dir "$SEED" --seed-tts-locale en \
    --profiling-type closed_loop --max-concurrency "$B" --num-requests "$N" --num-warmup 5 \
    --inference-system "$SYS" --local-cache /home/tim/seedtts-cache --output-dir "$OUT" > "$OUT/stdout.txt" 2>&1
  rc=$?; echo "[$(ts)] B=$B rc=$rc"; grep -E '^RTF|audio sec/s|Requests :' "$OUT/stdout.txt" | tail -3
  [ $rc -ne 0 ] && { echo "FAILED B=$B"; exit $rc; }
done
echo "[$(ts)] SWEEP DONE $SYS"

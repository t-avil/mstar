#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf
cd /home/tim/mstar; source .venv/bin/activate
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for M* port 8000 (up to 20 min)"
for i in $(seq 1 240); do
  (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && { echo "[$(ts)] PORT OPEN after ~$((i*5))s"; break; }
  pgrep -f mstar-serve >/dev/null || { echo "[$(ts)] SERVER DIED"; tail -20 "$D/runs/server_ours.log"; exit 1; }
  sleep 5
done
(exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null || { echo "[$(ts)] TIMEOUT no port"; exit 2; }
echo "[$(ts)] SMOKE: 1 TTS request via runner (ours)"
timeout 300 python -m benchmark.runner --url http://127.0.0.1:8000 --model qwen3omni \
  --request-type text_to_speech --dataset seed_tts \
  --seed-tts-dir /home/tim/seedtts-cache/seedtts_testset --seed-tts-locale en \
  --profiling-type offline --batch-size 1 --num-requests 1 --num-warmup 0 \
  --inference-system ours --local-cache /home/tim/seedtts-cache \
  --output-dir "$D/runs/smoke_ours" 2>&1 | tail -25
echo "[$(ts)] smoke results.json:"; cat "$D/runs/smoke_ours/results.json" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('completed',d.get('completed'),'failed',d.get('failed'),'wall',d.get('wall_time_s'));[print('  req',r['request_id'],'jct_ms',round(r['jct_ms'],1),'audio_bytes',r.get('output_bytes',{}).get('audio')) for r in d.get('per_request',[])]" 2>/dev/null || echo "(no results.json)"
echo "[$(ts)] SMOKE DONE"

#!/usr/bin/env bash
# recheck_i2s.sh <tag> <port> <reps>  — repeated I2S B1/B2 + I2T B1 to resolve variance vs regression.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
export BENCH_SPEECH_THINKER_TEMPERATURE=0.7 OPENAI_API_KEY=EMPTY
cd /home/tim/vllm-layout-wt && source .venv/bin/activate
TAG=$1; URL="http://127.0.0.1:$2"; REPS=${3:-3}; R=/home/tim/exp_rebench/recheck/$TAG
ts(){ date -u +%H:%M:%SZ; }
run(){ # path B rep
  local path=$1; local B=$2; local rep=$3; local O="$R/${path}_B${B}_r${rep}"; mkdir -p "$O"
  numactl --cpunodebind=1 --membind=1 timeout 900 python -m benchmark.runner --url "$URL" --model qwen3omni \
    --request-type "$path" --dataset food101 --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests $(( B*50<60?60:B*50 )) --num-warmup 5 --inference-system ours \
    --local-cache /home/tim/tmp/libri_wavs --output-dir "$O" > "$O/out.txt" 2>&1
  local line=$(grep -E 'audio sec/s|req/s|^RTF' "$O/out.txt" | tr '\n' ' ')
  echo "[$(ts)] $TAG $path B=$B rep=$rep | $line"
  rm -f "$O"/*.wav 2>/dev/null
}
for rep in $(seq 1 $REPS); do
  run image_to_speech 1 $rep
  run image_to_speech 2 $rep
  run image_to_text   1 $rep
done
echo "[$(ts)] $TAG RECHECK DONE"

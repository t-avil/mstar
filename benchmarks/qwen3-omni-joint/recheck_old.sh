#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
export BENCH_SPEECH_THINKER_TEMPERATURE=0.7 OPENAI_API_KEY=EMPTY
cd /home/tim/vllm-layout-wt && source .venv/bin/activate
R=/home/tim/exp_rebench/recheck/mstar_old; ts(){ date -u +%H:%M:%SZ; }
runold(){ local path=$1; local B=$2; local rep=$3; local n=$4; local O="$R/${path}_B${B}_r${rep}"; mkdir -p "$O"
  numactl --cpunodebind=1 --membind=1 timeout 600 python -m benchmark.runner --url http://127.0.0.1:8176 --model qwen3omni \
    --request-type "$path" --dataset food101 --profiling-type closed_loop --max-concurrency "$B" \
    --num-requests "$n" --num-warmup 5 --inference-system ours --local-cache /home/tim/tmp/libri_wavs --output-dir "$O" > "$O/out.txt" 2>&1
  echo "[$(ts)] OLD $path B=$B r=$rep | $(grep -E 'audio sec/s|req/s|^RTF' "$O/out.txt" | tr '\n' ' ')"; rm -f "$O"/*.wav 2>/dev/null; }
for rep in 1 2; do runold image_to_speech 1 $rep 30; runold image_to_speech 2 $rep 40; done
runold image_to_text 1 1 30
echo "[$(ts)] OLD RECHECK DONE"

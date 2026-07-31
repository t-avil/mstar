#!/usr/bin/env bash
# M*-new (native encoder, current code) full closed-loop sweep on 6,7 (port 8094).
# #131 acceptance #2 baseline: native vs HF across batch. seed=42, 10 wavs/batch.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
cd /home/tim/mstar; source .venv/bin/activate
ROOT=/home/tim/exp_mstarnew_sweep; URL=http://127.0.0.1:8094
ts(){ date -u +%Y%m%dT%H%M%SZ; }
declare -A NREQ=( [1]=12 [2]=20 [4]=24 [8]=40 [16]=80 [32]=160 )
declare -A DS=( [image_to_speech]=food101 [audio_to_speech]=libri ); KEEP=10
echo "[$(ts)] wait 8094"; for i in $(seq 1 60); do (exec 3<>/dev/tcp/127.0.0.1/8094) 2>/dev/null && break; sleep 5; done
for path in audio_to_speech image_to_speech; do
  for B in 1 2 4 8 16 32; do
    N=${NREQ[$B]}; OUT="$ROOT/out_mstar_new/$path/B${B}"; mkdir -p "$OUT"
    echo "[$(ts)] M*-new $path B=$B reqs=$N"
    numactl --cpunodebind=1 --membind=1 timeout 5400 python -m benchmark.runner --url "$URL" --model qwen3omni \
      --request-type "$path" --dataset "${DS[$path]}" --profiling-type closed_loop --max-concurrency "$B" \
      --num-requests "$N" --num-warmup 5 --inference-system ours --local-cache /home/tim/tmp/libri_wavs \
      --output-dir "$OUT" > "$OUT/stdout.txt" 2>&1
    echo "[$(ts)] $path B=$B rc=$?"; grep -E '^RTF|audio sec/s' "$OUT/stdout.txt" | tail -2
    mkdir -p "$OUT/samples"; ls "$OUT"/*.wav 2>/dev/null | sort -V | head -n $KEEP | while read -r f; do mv "$f" "$OUT/samples/"; done
    rm -f "$OUT"/*.wav "$OUT"/req_*.txt 2>/dev/null
  done
done
echo "[$(ts)] MSTARNEW SWEEP DONE"

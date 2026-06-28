#!/usr/bin/env bash
# run_sweep.sh <system> <url> — drive the Seed-TTS batch sweep for one system.
# system in {ours,sglang_omni,vllm_omni} (must match benchmark.runner --inference-system).
# Figure 5 protocol (paper Appendix I): offline waves of B, B in {1,4,8,16,32},
#   num_requests = max(10, 5*B), num_warmup=3, max_tokens=256 (baked into harness),
#   greedy thinker, Qwen3-Omni system prompt. RTF (lower) + audio throughput (higher).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/m-coriander/coriander/hf
SYS="$1"; URL="$2"
cd /home/tim/mstar
source .venv/bin/activate
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
SEED_DIR=/home/tim/seedtts-cache/seedtts_testset
OUTROOT="$D/runs/out_${SYS}"
mkdir -p "$OUTROOT"
ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] SWEEP system=$SYS url=$URL"

for B in 1 4 8 16 32; do
  N=$(( B*5 )); [ "$N" -lt 10 ] && N=10
  OUT="$OUTROOT/B${B}"; mkdir -p "$OUT"
  echo "[$(ts)] --- B=$B  num_requests=$N  warmup=3 ---"
  # Hard per-batch ceiling so a hung server can't stall the sweep (CLAUDE.md hygiene).
  timeout 2700 python -m benchmark.runner \
      --url "$URL" \
      --model qwen3omni \
      --request-type text_to_speech \
      --dataset seed_tts \
      --seed-tts-dir "$SEED_DIR" \
      --seed-tts-locale en \
      --profiling-type offline \
      --batch-size "$B" \
      --num-requests "$N" \
      --num-warmup 3 \
      --inference-system "$SYS" \
      --local-cache /home/tim/seedtts-cache \
      --output-dir "$OUT" \
      > "$OUT/stdout.txt" 2>&1
  rc=$?
  echo "[$(ts)] B=$B rc=$rc" | tee -a "$OUTROOT/sweep.log"
  echo "----- B=$B tail -----"; tail -20 "$OUT/stdout.txt"
  if [ $rc -ne 0 ]; then echo "[$(ts)] B=$B FAILED (rc=$rc) — stopping sweep for $SYS"; exit $rc; fi
  sleep 2
done
echo "[$(ts)] SWEEP DONE system=$SYS"

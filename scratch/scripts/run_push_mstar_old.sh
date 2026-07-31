#!/usr/bin/env bash
# Push M*-old (upstream main, HF encoder) I2S + S2S as two CLAUDE.md bench
# branches from main, merge each to benchmarks. Mirrors the vLLM push.
set -uo pipefail
cd /home/tim/mstar
source .venv/bin/activate
SRC=/home/tim/exp_mstar_oldsweep/out_mstar_old
TS=$(date -u +%Y%m%dT%H%M%SZ)
declare -A PATHMAP=( [i2s]=image_to_speech [s2s]=audio_to_speech )
declare -A DSMAP=( [i2s]=food101 [s2s]=libri )

build_dir(){
  local tag=$1 path=${PATHMAP[$1]} ds=${DSMAP[$1]}
  local dst=benchmarks/qwen3-omni-${tag}-mstar-old
  rm -rf "$dst"; mkdir -p "$dst/runs/out_mstar_old"
  for B in 1 2 4 8 16 32; do
    local s="$SRC/$path/B$B" d="$dst/runs/out_mstar_old/B$B"
    mkdir -p "$d/samples"
    cp "$s/results.json" "$s/stdout.txt" "$d/" 2>/dev/null
    cp "$s/samples/"*.wav "$d/samples/" 2>/dev/null
  done
  cp /home/tim/bench_analyze.py "$dst/analyze.py"
  cat > "$dst/command.txt" <<EOF
# M*-old (upstream main, HF-wrapper encoder) Qwen3-Omni ${tag^^} (${path})
# closed-loop max-concurrency sweep, seed=42, 2-GPU H200.
# Server: mstar upstream main @ae7d173, configs/qwen3omni_2gpu.yaml
#   (Thinker on one GPU; Talker+Code2Wav on the other), --tensor-comm-protocol SHM
# Client (this repo) benchmark.runner, per batch B in {1,2,4,8,16,32}:
python -m benchmark.runner --url http://127.0.0.1:8097 --model qwen3omni \\
  --request-type ${path} --dataset ${ds} \\
  --profiling-type closed_loop --max-concurrency B --num-requests N --num-warmup 5 \\
  --inference-system ours --local-cache /home/tim/tmp/libri_wavs
# num_requests N per B: 1->12 2->20 4->24 8->40 16->80 32->160
EOF
  bash benchmarks/qwen3-omni-seedtts-2gpu/capture_env.sh "$dst" >/dev/null 2>&1 || true
  /home/tim/mstar/.venv/bin/python -m pip freeze > "$dst/requirements.txt" 2>/dev/null || true
}

push_branch(){
  local tag=$1
  local br=bench/qwen3-omni-${tag}-mstar-old
  local dir=benchmarks/qwen3-omni-${tag}-mstar-old
  echo "########## $br ##########"
  git checkout main || return 1
  git checkout -B "$br" main || return 1
  git checkout benchmarks -- benchmarks/chartstyle.mplstyle
  build_dir "$tag"
  RUN_TS_UTC=$TS python /home/tim/bench_analyze.py "$dir" || return 1
  git add "$dir" benchmarks/chartstyle.mplstyle
  git commit -q -m "bench(${tag}-mstar-old): Qwen3-Omni ${tag^^} M*-old (HF encoder) closed-loop sweep B=1..32

Upstream main M* (HF-wrapper audio/vision encoder), closed-loop max-concurrency,
seed=42, 2-GPU H200. raw.json (every datapoint) + RTF/throughput charts + 10
audio samples/batch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013C22imPWJiPxcHoRZZWqbi" || return 1
  git push -u origin "$br" || { echo "PUSH FAILED $br"; return 1; }
  git checkout benchmarks || return 1
  git merge --no-ff "$br" -m "merge ${br}: M*-old ${tag^^} sweep into aggregation branch" || return 1
  git push origin benchmarks || { echo "PUSH benchmarks FAILED"; return 1; }
  git checkout main
  echo "OK $br + benchmarks pushed"
}

push_branch i2s && push_branch s2s
echo "=== branches now ==="; git branch | grep -E 'mstar-old|vllm'
echo "DONE-PUSH-MSTAR-OLD"

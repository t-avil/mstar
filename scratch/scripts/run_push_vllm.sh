#!/usr/bin/env bash
# Push vLLM I2S + S2S as two CLAUDE.md bench branches (from main), each with only
# its path's data + raw.json + charts + 10 audio samples/batch; merge each to the
# benchmarks aggregation branch. Never commits benchmark artifacts to main.
set -uo pipefail
cd /home/tim/mstar
source .venv/bin/activate
SRC=/home/tim/exp_vllm_i2s_s2s/out_vllm_omni
TS=$(date -u +%Y%m%dT%H%M%SZ)
declare -A PATHMAP=( [i2s]=image_to_speech [s2s]=audio_to_speech )
declare -A DSMAP=( [i2s]=food101 [s2s]=libri )

build_dir(){
  local tag=$1 path=${PATHMAP[$1]} ds=${DSMAP[$1]}
  local dst=benchmarks/qwen3-omni-${tag}-vllm
  rm -rf "$dst"; mkdir -p "$dst/runs/out_vllm_omni"
  for B in 1 2 4 8 16 32; do
    local s="$SRC/$path/B$B" d="$dst/runs/out_vllm_omni/B$B"
    mkdir -p "$d/samples"
    cp "$s/results.json" "$s/stdout.txt" "$d/" 2>/dev/null
    cp "$s/samples/"*.wav "$d/samples/" 2>/dev/null
  done
  cp /home/tim/bench_analyze.py "$dst/analyze.py"
  cat > "$dst/command.txt" <<EOF
# vLLM-Omni Qwen3-Omni ${tag^^} (${path}) — closed-loop max-concurrency sweep, seed=42
# Server: vLLM-Omni @60c15004 (0.21.0 line), vllm 0.21.0+cu129, 2-GPU
#   deploy qwen3_omni_moe.yaml: stage0/thinker->cuda:0 ; stage1/talker + stage2/code2wav->cuda:1 ; seed:42/stage
# Client (this repo) benchmark.runner, per batch B in {1,2,4,8,16,32}:
python -m benchmark.runner --url http://127.0.0.1:8093 --model qwen3omni \\
  --request-type ${path} --dataset ${ds} \\
  --profiling-type closed_loop --max-concurrency B --num-requests N --num-warmup 5 \\
  --inference-system vllm_omni --local-cache /home/tim/tmp/libri_wavs
# num_requests N per B: 1->12 2->20 4->24 8->40 16->80 32->160
EOF
  bash benchmarks/qwen3-omni-seedtts-2gpu/capture_env.sh "$dst" >/dev/null 2>&1 || true
  /home/tim/baselines/vllm-omni/.venv/bin/python -m pip freeze > "$dst/requirements.txt" 2>/dev/null || true
}

push_branch(){
  local tag=$1
  local br=bench/qwen3-omni-${tag}-vllm
  local dir=benchmarks/qwen3-omni-${tag}-vllm
  echo "########## $br ##########"
  git checkout main || return 1
  git checkout -B "$br" main || return 1
  git checkout benchmarks -- benchmarks/chartstyle.mplstyle
  build_dir "$tag"
  RUN_TS_UTC=$TS python /home/tim/bench_analyze.py "$dir" || return 1
  git add "$dir" benchmarks/chartstyle.mplstyle
  git commit -q -m "bench(${tag}-vllm): Qwen3-Omni ${tag^^} vLLM-Omni closed-loop sweep B=1..32

Closed-loop max-concurrency continuous batching, seed=42, 2-GPU H200. raw.json
(every per-request datapoint) + RTF/throughput charts + 10 audio samples/batch
for reproducibility verification. vLLM-Omni @60c15004 (vllm 0.21.0+cu129).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013C22imPWJiPxcHoRZZWqbi" || return 1
  git push -u origin "$br" || { echo "PUSH FAILED $br"; return 1; }
  echo ">>> merge $br -> benchmarks"
  git checkout benchmarks || return 1
  git merge --no-ff "$br" -m "merge ${br}: vLLM ${tag^^} sweep into aggregation branch" || return 1
  git push origin benchmarks || { echo "PUSH benchmarks FAILED"; return 1; }
  git checkout main
  echo "OK $br + benchmarks pushed"
}

push_branch i2s && push_branch s2s
echo "=== final branch state ==="
git branch -a | grep -E 'vllm|benchmarks' | grep -v seedtts
echo "DONE-PUSH-VLLM"

#!/usr/bin/env bash
# launch_server.sh <system> — bring up one system's Qwen3-Omni server on GPUs 6,7.
# 2-GPU disaggregation per the paper (Thinker on one GPU; Talker+Code2Wav on the other).
#   ours   : mstar-serve --config configs/qwen3omni_2gpu.yaml          (port 8000, /generate)
#   sglang : sgl-omni serve (default = disaggregated tp1, 2 GPUs)      (port 8000, /v1/chat/completions)
#   vllm   : vllm serve --omni --deploy-config qwen3_omni_moe.yaml     (port 8091, /v1/chat/completions)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/m-coriander/coriander/hf
export CUDA_VISIBLE_DEVICES=6,7
# Private TMPDIR: the shared TMPDIR (/m-coriander/coriander/tmp) has stale lock
# files owned by other users (e.g. sglang_omni_gpu_*_startup.lock) that block us.
export TMPDIR=/home/tim/tmp/launch-tmp
mkdir -p "$TMPDIR"
SYS="$1"
D=/home/tim/mstar/benchmarks/qwen3-omni-seedtts-2gpu
LOG="$D/runs/server_${SYS}.log"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
ts(){ date -u +%Y%m%dT%H%M%SZ; }

# CLAUDE.md hygiene: confirm GPUs 6,7 idle before launch (no co-location).
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==6||$1==7) && $2>500{print $1":"$2}')
if [ -n "$busy" ]; then echo "[$(ts)] ABORT: GPU(s) not idle: $busy" | tee -a "$LOG"; exit 9; fi
echo "[$(ts)] launching $SYS on GPUs 6,7 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" | tee -a "$LOG"

case "$SYS" in
  ours)
    cd /home/tim/mstar && source .venv/bin/activate
    exec setsid timeout 7200 mstar-serve --config configs/qwen3omni_2gpu.yaml \
        --host 0.0.0.0 --port 8000 --tensor-comm-protocol SHM --log-level INFO >>"$LOG" 2>&1
    ;;
  sglang)
    cd /home/tim/baselines/sglang-omni && source .venv/bin/activate
    export SGLANG_OMNI_STARTUP_TIMEOUT=2400   # talker AR graph capture can be slow
    # Use exported config with endpoints.base_path repointed to a writable dir
    # (default /tmp/sglang_omni is owned by another user). 2-GPU disaggregated.
    exec setsid timeout 7200 sgl-omni serve \
        --config "$D/runs/sglang_qwen3omni_2gpu.yaml" \
        --host 0.0.0.0 --port 8000 >>"$LOG" 2>&1
    ;;
  sglang_v0)
    # SGLang-Omni V0 (the architecture the M* team's instructions reference:
    # `sglang_omni.cli.cli serve`). V1's talker stage deadlocks; V0 uses the
    # documented explicit-placement speech-server script. 2-GPU disaggregation:
    # thinker->GPU6 (cuda:0), talker+code-predictor+code2wav->GPU7 (cuda:1).
    cd /home/tim/baselines/sglang-omni-v0 && source .venv/bin/activate
    export SGLANG_OMNI_STARTUP_TIMEOUT=2400
    exec setsid timeout 7200 python examples/run_qwen3_omni_speech_server.py \
        --model-path "$MODEL" \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code-predictor 1 --gpu-code2wav 1 \
        --host 0.0.0.0 --port 8000 >>"$LOG" 2>&1
    ;;
  vllm)
    cd /home/tim/baselines/vllm-omni && source .venv/bin/activate
    # Raise init timeouts: the 30B MoE multi-stage torch.compile+capture takes
    # >300s, exceeding the default stage_init_timeout=300 / init_timeout=600.
    exec setsid timeout 7200 vllm serve "$MODEL" --omni \
        --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
        --stage-init-timeout 1800 --init-timeout 3600 \
        --host 0.0.0.0 --port 8091 >>"$LOG" 2>&1
    ;;
  *) echo "unknown system $SYS"; exit 2;;
esac

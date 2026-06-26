#!/usr/bin/env bash
cd /mnt/storage/timchick/vllm-omni
export HF_HOME=/mnt/storage/timchick/hf_cache TMPDIR=/mnt/storage/timchick/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec env CUDA_VISIBLE_DEVICES=0,2 /mnt/storage/timchick/venvs/vllm-omni/bin/vllm serve \
  Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --stage-configs-path vllm_omni/deploy/qwen3_omni_moe.yaml

#!/usr/bin/env bash
cd /mnt/storage/timchick/sglang-omni
export HF_HOME=/mnt/storage/timchick/hf_cache TMPDIR=/tmp/mstar_jit
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHINFER_USE_CUDA_NORM=1
exec env CUDA_VISIBLE_DEVICES=0,2,3 /mnt/storage/timchick/venvs/sglang-omni/bin/python \
  examples/run_qwen3_omni_speech_server.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 --gpu-talker 1 --gpu-audio-encoder 2 --gpu-image-encoder 2 --port 8092

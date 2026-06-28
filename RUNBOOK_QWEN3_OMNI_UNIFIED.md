# Qwen3-Omni unified build — RUNBOOK

One serving entrypoint, one switchboard. Every optimization (#131) is gated by a
single config block, `OptimizationConfig`, in
`mstar/model/qwen3_omni/config.py`. You drive it two equivalent ways:

* **prod**: YAML `model_kwargs.optim` (see `configs/qwen3omni_2gpu.yaml`)
* **ablation**: `MSTAR_*` env vars (env wins over YAML)

| Config key (`optim.*`) | Env var | Default |
|---|---|---|
| `native_audio_encoder` | `MSTAR_QWEN3_NATIVE_AUDIO_ENCODER` | `1` |
| `native_vision_encoder` | `MSTAR_QWEN3_NATIVE_VISION_ENCODER` | `1` |
| `gpu_mel` | `MSTAR_GPU_MEL` | `0` |
| `gpu_image_preprocess` | `MSTAR_GPU_IMAGE_PREPROCESS` | `0` |
| `vllm_prompt_layout` | `MSTAR_VLLM_PROMPT_LAYOUT` | `0` |
| `vllm_audio_sentinels` | `MSTAR_VLLM_AUDIO_SENTINELS` | `0` |
| `codec_chunk_frames` | `MSTAR_CODEC_CHUNK_FRAMES` | `15` |

Defaults = byte-identical baseline where parity matters; the validated wins
(native encoders, codec 15) ship ON. Same-audio-vs-vLLM needs
`vllm_prompt_layout=1`.

## Serve (one GPU pair)

```bash
# Confirm the pair is idle first: nvidia-smi
GPUS=1,0 PORT=8100 PRESET=full bash benchmark/serving_scripts/serve_mstar.sh
#   PRESET = baseline | native | full   (or set MSTAR_* directly)
#   compose: PRESET=full MSTAR_GPU_MEL=0 ... serve_mstar.sh
```

Lower level (this workspace's detached launcher, pins CUDA_VISIBLE_DEVICES+NUMA):

```bash
# launch_mstar_wt.sh <worktree> <gpus> <numa> <port> <sockname> <log> [ENV=VAL...]
/home/tim/launch_mstar_wt.sh /home/tim/qwen3-omni-unified-wt 1,0 0 8100 uni-full \
  /home/tim/tmp/uni.log MSTAR_GPU_MEL=1 MSTAR_VLLM_PROMPT_LAYOUT=1
```

## Run each path (client)

`--request-type` selects the path; nothing else changes between paths:

| Path | `--request-type` | `--dataset` |
|---|---|---|
| **S2T** speech→text | `audio_to_text` | `libri` |
| **S2S** speech→speech | `audio_to_speech` | `libri` |
| **I2T** image→text | `image_to_text` | `food101` |
| **I2S** image→speech | `image_to_speech` | `food101` |

```bash
source /home/tim/vllm-layout-wt/.venv/bin/activate
# speech-out paths also need:  export BENCH_SPEECH_THINKER_TEMPERATURE=0.7
python -m benchmark.runner --url http://127.0.0.1:8100 --model qwen3omni \
  --inference-system ours --request-type audio_to_text --dataset libri \
  --num-warmup 5 --num-requests 10 --batch-size 1 \
  --profiling-type closed_loop --max-concurrency 1 \
  --local-cache /home/tim/tmp/libri_wavs --output-dir out/s2t/bs1
```

Swap `audio_to_text`→`audio_to_speech`/`image_to_text`/`image_to_speech` (+ the
matching `--dataset`) for S2S / I2T / I2S.

## Re-run everything (fan out across all idle pairs)

```bash
# auto-discovers idle H200 pairs, one isolated preset-server per pair, parallel,
# 5 warmup + 10 measured per (path,batch), auto-records env+command, resumable.
bash benchmark/serving_scripts/orchestrate_rerun.sh
# preview the plan without launching:
DRY_RUN=1 bash benchmark/serving_scripts/orchestrate_rerun.sh
# pin pairs / presets:
GPU_PAIRS="1,0 3,2" PRESETS="baseline native full" \
  bash benchmark/serving_scripts/orchestrate_rerun.sh
```

Artifacts land under `OUT_ROOT/<preset>/<path>/bs<N>/` (`results.json` + raw
datapoints + `command.txt`; `env.txt`/`requirements.txt` per preset). Re-running
skips any `(path,batch)` whose `results.json` already exists.

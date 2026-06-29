# Encoder Placement Reshuffle: Encoders to Rank 0

## Summary

Move `audio_encoder` and `vision_encoder` from Rank 1 (Thinker's GPU) to Rank 0
(Talker + Code2Wav's GPU). This is a **config-only change** -- no code
modifications required.

## Motivation

In the baseline 2-GPU config (`qwen3omni_2gpu.yaml`), during a speech-input
prefill, Rank 0 (Talker + Code2Wav) sits completely idle while the encoder +
Thinker prefill runs on Rank 1. Moving encoders to Rank 0 uses the idle GPU for
encoder compute. This is a prerequisite for overlapping encoder compute with
Thinker decode across requests.

## Why It Works as Config-Only

1. **`_divide_into_worker_graphs` handles cross-rank Sequentials**: When
   `prefill_audio = Sequential([audio_encoder, Thinker])` and the two nodes are
   on different ranks, the function (in `mstar/model/base.py:109-136`) checks
   `_group_id` of consecutive sections. Different group IDs produce separate
   WorkerGraphs. The encoder gets its own WorkerGraph on Rank 0; Thinker keeps
   its own on Rank 1.

2. **Tensor transport handles cross-rank transfer transparently**: The
   `audio_embeds` tensor produced by the encoder on GPU-0 is registered for
   send (RDMA memory registration or SHM file write). The conductor routes the
   `GraphEdge(next_node="Thinker", name="audio_embeds")` to the Thinker's
   worker on Rank 1. The Thinker's worker allocates a receive buffer on GPU-1
   and performs an async RDMA read. This is the exact same mechanism already
   used for Thinker->Talker streaming (`thinker_states`).

3. **No device-coupling in encoder submodules**: `AudioEncoderSubmodule` and
   `VisionEncoderSubmodule` are self-contained -- they reference only their own
   `nn.Module` weights. No shared buffers or references to Thinker parameters.

4. **ThinkerSubmodule has defensive `.to(device)` calls**: In
   `ThinkerSubmodule.prepare_inputs()`, encoder outputs are moved to the
   Thinker's device: `audio_embeds = inputs["audio_embeds"][0].to(device)`.
   When encoder and Thinker share a GPU, this is a no-op. When they are on
   different GPUs, the tensor was already placed on the Thinker's GPU by the
   tensor transport layer (RDMA/SHM allocates the receive buffer on the
   destination worker's `self.device`), so this is also a no-op.

5. **Conductor input routing already handles multi-node targets**: For
   `prefill_vision`, some edges target `vision_encoder` (pixel_values) and
   some target `Thinker` (image_grid_thw, video_second_per_grid). The
   conductor's `_split_inputs_to_workers` routes each edge to the correct
   worker based on `next_node`.

## Expected Mechanism

```
GPU-0 (Rank 0): Talker + Code2Wav + audio_encoder + vision_encoder
GPU-1 (Rank 1): Thinker (alone)

prefill_audio walk:
  1. Conductor sends audio_features -> audio_encoder on Rank 0
  2. audio_encoder runs on GPU-0, produces audio_embeds
  3. audio_embeds registered for RDMA send on GPU-0
  4. Conductor routes audio_embeds -> Thinker on Rank 1
  5. Rank 1 worker does async RDMA read of audio_embeds into GPU-1 buffer
  6. Thinker prefills with audio_embeds on GPU-1
```

## Risks

- **Memory pressure on Rank 0**: Now holds Talker (~3B params) + Code2Wav +
  audio_encoder (Whisper-style, ~600M params) + vision_encoder (SigLIP2 ViT,
  ~400M params). In bf16 this adds ~2 GB of encoder weights to GPU-0. Talker +
  Code2Wav already use ~8 GB, so total is ~10 GB. On 80 GB GPUs this is fine.
  On 40 GB GPUs, check that Talker KV cache + encoder weights + Code2Wav fit.

- **RDMA latency for encoder outputs**: `audio_embeds` is typically ~1500 tokens
  x 3584 hidden dim x 2 bytes (bf16) = ~10 MB. RDMA transfer at ~12 GB/s
  (PCIe / NVLink) takes <1 ms. Negligible compared to the Thinker prefill time.

- **No overlap yet**: This config alone does not overlap encoder compute with
  anything -- it just moves where the encoder runs. Overlap with Thinker decode
  (across requests) requires scheduler changes to dispatch encoder prefill for
  request N+1 while Thinker decodes request N.

## Config Diff

```yaml
# BEFORE (qwen3omni_2gpu.yaml):
node_groups:
  - node_names: [audio_encoder, vision_encoder]
    ranks: [1]            # <-- encoders on Thinker's GPU
  - node_names: [Thinker]
    ranks: [1]

# AFTER (qwen3omni_2gpu_enc_rank0.yaml):
node_groups:
  - node_names: [audio_encoder, vision_encoder]
    ranks: [0]            # <-- encoders on Talker's GPU
  - node_names: [Thinker]
    ranks: [1]
```

## Test Command

```bash
/home/tim/launch_mstar_wt.sh /home/tim/enc-placement-wt 0,1 0 8103 enc_place_sock /home/tim/logs/enc_place.log \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -- --config configs/qwen3omni_2gpu_enc_rank0.yaml
```

## A/B Test

Run the same request with both configs and compare:
- TTFT (time to first token) for audio and vision inputs
- GPU utilization on both ranks during prefill
- Memory usage on Rank 0

Baseline:
```bash
/home/tim/launch_mstar_wt.sh /home/tim/enc-placement-wt 0,1 0 8103 baseline_sock /home/tim/logs/baseline.log \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -- --config configs/qwen3omni_2gpu.yaml
```

Experiment:
```bash
/home/tim/launch_mstar_wt.sh /home/tim/enc-placement-wt 0,1 0 8103 enc_place_sock /home/tim/logs/enc_place.log \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -- --config configs/qwen3omni_2gpu_enc_rank0.yaml
```

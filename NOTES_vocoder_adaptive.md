# Vocoder Adaptive Chunk Policy

## Mechanism

The Talker->Code2Wav streaming edge uses a `LeftContextChunkPolicy` with a fixed
chunk size (default 25 frames). This is optimal for latency at B=1, but at high
batch the vocoder could process larger chunks more efficiently (fewer kernel
launches, better GPU utilization).

`AdaptiveChunkPolicy` (in `mstar/streaming/chunk_policy.py`) dynamically scales
the chunk size based on the current batch level on the Code2Wav worker:

```
effective_chunk = min(min_chunk * max(1, batch_level // 2), max_chunk)

B=1-3  -> chunk = 25   (same as fixed policy -- preserves pipeline overlap)
B=4-5  -> chunk = 50
B=6-7  -> chunk = 75
B>=8   -> chunk = 100
```

`left_context` stays fixed at 25: the vocoder's causal ConvNet receptive field
doesn't change with batch size.

## Batch level flow

1. Worker's `_poll_stream_buffers()` runs every poll cycle
2. For each streaming connection, it counts how many active requests have a
   StreamBuffer for that edge name -- this is the per-partition batch level
3. It calls `policy.set_batch_level(batch_level)` on every `AdaptiveChunkPolicy`
   instance for that edge
4. The policy updates its internal `_chunk` and `_window` fields
5. The next `is_ready()` / `next_chunk_size()` / `window_size()` calls use the
   updated chunk size

The batch level is per-edge (not global), so the Code2Wav chunk policy sees only
the number of requests actively flowing through the Talker->Code2Wav edge.

## Env vars

- `MSTAR_VOCODER_ADAPTIVE_CHUNK=1` -- enable adaptive chunking (default OFF)
- `MSTAR_VOCODER_ADAPTIVE_MAX_CHUNK=100` -- upper bound on chunk size (default 100)

When OFF, the code path is byte-identical to the existing fixed
`LeftContextChunkPolicy(chunk=25, left_context=25)`.

## Files changed

- `mstar/streaming/chunk_policy.py` -- new `AdaptiveChunkPolicy` class
- `mstar/model/qwen3_omni/qwen3_omni_model.py` -- env-gated policy factory in
  `get_partition_topology()`
- `mstar/worker/worker.py` -- batch-level update in `_poll_stream_buffers()`

## Expected win

- B=1: no change (chunk stays at 25, identical to current behavior)
- B>=4: fewer vocoder kernel launches per unit time, better batched utilization.
  The codec-chunk experiment showed +5-7% throughput on long audio (I2S) with
  static chunk=50. Adaptive chunking should capture that win at high batch
  without the -18% S2S regression (since short-audio requests at low batch keep
  chunk=25).

## Test command (A/B)

```bash
# Baseline (adaptive OFF -- existing behavior)
MSTAR_VOCODER_ADAPTIVE_CHUNK=0 python -m mstar.serve \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --cuda-devices 6,7 \
  ...

# Experiment (adaptive ON)
MSTAR_VOCODER_ADAPTIVE_CHUNK=1 python -m mstar.serve \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --cuda-devices 6,7 \
  ...

# Run closed-loop load test at varying batch levels:
# B=1 (latency parity check), B=4, B=8 (throughput win check)
```

## Composability with codec-chunk-frames knob

The `MSTAR_CODEC_CHUNK_FRAMES` / `MSTAR_CODEC_LEFT_CONTEXT_FRAMES` env vars from
the `codec-chunk` branch set `config.code2wav.codec_chunk_frames` and
`codec_left_context_frames` in `Code2WavConfig.__post_init__`. The adaptive
policy reads those same config values as its `min_chunk` and `left_context`
defaults, so the two knobs compose:

- `MSTAR_CODEC_CHUNK_FRAMES=30` + `MSTAR_VOCODER_ADAPTIVE_CHUNK=1` would use
  min_chunk=30 (scaling to 60/90/100 at higher batch)
- Without the adaptive flag, `MSTAR_CODEC_CHUNK_FRAMES` still works as a static
  override (existing behavior from codec-chunk branch)

The max_chunk is independently controlled by `MSTAR_VOCODER_ADAPTIVE_MAX_CHUNK`.

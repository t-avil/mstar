# M\* Qwen3-Omni I2T / S2T / I2S — existing-technique optimization review

Goal: per the ask, "optimize" = **apply M\*'s already-existing techniques** to the
I2T / S2T / I2S paths (TTS already wins; these text/other paths are less tuned).
This is the code-review half; the benchmark half measures where M\* actually
trails vllm-omni / sglang-omni and confirms which gap matters.

## Path → component → optimization status

Engine types (`qwen3_omni_model.py:get_node_engine_types`): encoders = STATELESS
(`enc_dec` flavor), Thinker/Talker = KV_CACHE, Code2Wav = STATELESS (`audio_codec`).

| Path | Components (in order) | TTFT driver | ITL/RTF driver |
|---|---|---|---|
| **I2T** image→text | vision_encoder → Thinker(prefill_vision → decode) | encoder + prefill_vision | thinker_decode |
| **S2T** audio→text | audio_encoder → Thinker(prefill_audio → decode) | encoder + prefill_audio | thinker_decode |
| **I2S** image→speech | vision_encoder → Thinker → Talker → Code2Wav | encoder + prefills | talker_decode + code2wav |

## Which existing techniques are already applied

| Component | torch.compile | CUDA graph | cross-req batching | Verdict |
|---|---|---|---|---|
| Thinker `thinker_decode` | ✅ | ✅ `BasicBatchedCudaGraphConfig` bs 1–32 | ✅ | fully optimized (ITL) |
| Thinker `prefill_text`/`prefill_audio` | ✅ | ✅ `FlashInferPackedCudaGraphConfig` (aliased) | ✅ | fully optimized (TTFT) |
| Thinker `prefill_vision` | ✅ | ✅ separate `FlashInferPackedCudaGraphConfig` | ✅ | fully optimized (TTFT) |
| Talker decode/prefill | ✅ | ✅ | ✅ | fully optimized (I2S) |
| Code2Wav | ✗ (fp32, by design) | ✅ `BasicBatchedCudaGraphConfig` | — | optimized (I2S RTF) |
| **vision/audio encoders** | ✅ (enc_dec `apply_torch_compile=True`) | ❌ **none** | ✅ (native, issue #131) | **partial** |

**Conclusion:** the Thinker — the bulk of I2T/S2T compute — is already maximally
optimized with M\*'s graph+compile machinery for *every* relevant walk. The one
under-optimized surface on these paths is the **encoder**, which is exactly the
TTFT contributor for I2T/S2T. Issue #131 (native, batched encoders — now default)
removed the dominant cost (vision Conv3d 3.5 s → ~13 ms). What remains:

## The one unapplied existing technique: PiecewiseCudaGraphRunner on the encoder

- The `enc_dec` engine already sets `enable_piecewise_runner=True`
  (`stateless_engine.py:79`). It activates **only** for submodules that implement
  `get_piecewise_runner_config()`.
- That technique exists and is in production use by **`vjepa2`**
  (`mstar/model/vjepa2/submodules.py:492`, `components/predictor.py`): it captures
  a *variable-length transformer block-loop* as piecewise CUDA graphs (preamble →
  captured layer-loop → postamble), sidestepping the "encoders have varying
  shapes so a single CUDA graph won't fit" gotcha from issue #131.
- The native qwen3 vision/audio encoders have exactly that structure
  (`for blk in self.blocks` / `for layer in self.layers`, variable token count)
  but **do not** implement `get_piecewise_runner_config()`, so they run
  eager+compiled — paying per-block kernel-launch overhead (27 vision blocks /
  32 audio layers) on every prefill.
- **This is the highest-leverage "use a technique that's already there" change for
  I2T/S2T TTFT**, and it mirrors vjepa2 almost 1:1. Apply only if the benchmark
  shows the encoder is still a material fraction of M\* TTFT after the native swap
  (the native encoder is ~7–13 ms/img; whether that matters depends on the
  measured prefill TTFT — hence benchmark-gated).

## Not gaps (don't "optimize" these)
- Thinker/Talker/Code2Wav: already graph+compile captured — nothing to add.
- Encoder full CUDA graph (non-piecewise): rejected by issue #131 (varying shapes);
  the piecewise runner is the correct existing tool, not a monolithic capture.
- Encoder fp32: N/A (runs bf16 autocast by design).

## Plan
1. Benchmark M\* vs vllm-omni vs sglang-omni on I2T/S2T/I2S (TTFT/ITL/RTF/throughput).
2. If M\* TTFT on I2T/S2T is encoder-bound, implement `get_piecewise_runner_config()`
   on the native encoder submodules (copy the vjepa2 pattern) — an *existing*
   technique, minimal new surface.
3. Re-benchmark to confirm the win.

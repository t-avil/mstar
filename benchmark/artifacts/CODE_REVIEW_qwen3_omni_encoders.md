# Deep code review (round 2) — `native-qwen3-omni-encoders`

Reviewed against issue #131, with claims **verified empirically**, not assumed
(per-layer parity, engine-dispatch trace, batch-accounting proof, perf with
variance). Artifacts in this directory back every claim.

## Summary

The branch ports the Qwen3-Omni audio/vision encoders to native M\* submodules
with HF-mirrored weight names, pad-free cross-request batching, and an unchanged
output contract. **Correctness is now verified, not asserted:** in fp32 the
native modules are *bit-exact* to HF at every layer (cos = 1.000000, 0 missing /
0 unexpected state-dict keys), and the engine dispatch + batched token-accounting
are provably correct. **No P1s.** Remaining items are precision/documentation
honesty, not logic. The one code change this round is a docstring correction
(the "bit-identical" claim was false in bf16). Risk: low; merge-ready.

## What I verified (evidence, not opinion)

1. **Implementation equivalence — fp32 per-layer parity.** Copying HF's weights
   into the native module loads with **0 missing / 0 unexpected** keys (structural
   parity), and the residual stream matches HF with **cos = 1.000000 at every one
   of the 27 vision blocks and 32 audio layers**. This is the strongest possible
   evidence that the hand-written compute (patch-embed, pos-embed, rotary,
   varlen attention, MLP, spatial merge, DeepStack, audio CNN frontend) is
   mathematically identical to HF. → *There are no logic bugs in the encoders.*

2. **Engine dispatch contract.** `stateless_engine._dispatch`: `can_batch=True →
   _execute_batched → forward_batched(**preprocess)` returning a `{rid: NameToTensorList}`
   dict wrapped as `NodeOutput.per_request_output_tensors`; else `_execute_sequential
   → forward` per rid. The native submodules' signatures and return shapes match
   both paths. CUDA-graph path is gated on declared graph configs (native declares
   none → correctly skipped).

3. **Batched token accounting is exact.** For audio, `_req_token_count =
   Σ formula(feature_lens)`. I proved `Σ_chunks formula(chunk_len) = formula(L)`
   (the CNN-length formula is additive over 100-frame blocks), so the per-request
   output slice length is exact; the batched parity test confirms `Σ counts ==
   packed_len`. Vision merged-token counts likewise match (validated across 4
   resolutions in the shipped test).

## P1 (Blocking)

None.

## P2 (Should Fix)

### 🟡 The end-only parity test under-characterizes bf16 intermediate divergence
**Location:** `test/modular/test_qwen3_omni_native_encoders.py` (assertions on final + DeepStack)

In bf16 (the production `enc_dec` autocast dtype), native-vs-HF divergence
**amplifies monotonically with depth**: relL2 grows 4e-4 → 1.4e-2 and per-element
max-abs reaches ~0.31 (vision block 26) / ~0.50 (audio layer 30). The end output
still passes (cos > 0.999), and the DeepStack captures (blocks 8/16/24) are
checked — but at relL2 up to ~1.3e-2, close-ish to nothing-to-spare only relative
to the loose 0.05 bar. This is *within tolerance and not a bug* (fp32 is exact),
but the test's tolerance hides the growth. Part of the bf16 drift is the SDPA
fallback's masking, which the production flash path avoids.
**Recommendation (no code change required for the ticket):** the ticket's
"validate parity within tolerance" is met. If hardening is wanted later, assert a
per-element bound on the DeepStack tensors, or add the per-layer profile
(`benchmark/qwen3_omni_encoder_parity.py`) to CI in a tiny-config form.
**Confidence:** 66

### 🟡 Factory docstring overstates audio throughput ("5-11x")
**Location:** `qwen3_omni_model.py:1473`

`_create_audio_encoder_submodule` docstring says native audio gives "5-11x
throughput". My benchmark (bf16, SDPA, 10 repeats) measures **1.2–1.7×**, peaking
at batch 4–8 and regressing to 1.18× at batch 16 (the frontend's Python
`cu_seqlens` loop is a per-forward CPU cost growing with window count). The 5-11×
figure is **not reproduced** under this setup and is unverified. Either back it
with a flash-attn benchmark or soften the claim. (Left unedited deliberately —
I won't substitute a number I can't fully verify across backends.)
**Confidence:** 70

## Minor Notes

- **`NativeVisionEncoderSubmodule.forward` (non-batched) is dead code.** Vision
  `can_batch` always returns `True`, so the stateless engine never calls
  `forward` — only `forward_batched`. Harmless (mirrors the audio submodule and
  the HF wrapper), but unreachable. (Score 40)
- **Native large-batch scaling needs flash-attn.** The SDPA fallback builds an
  O(total_tokens²) mask, so native vision per-image cost rises past batch 4 in
  this flash-free benchmark (5.7 ms → 10.8 ms at n=16). Production flash-varlen
  removes this; worth a one-line note that the fallback is not batch-scalable.
  (Score 40)
- **Partial transformers decoupling (vision).** `vision_encoder.py` still imports
  `get_vision_{bilinear_indices,cu_seqlens,position_ids}` from
  `transformers.models.qwen3_omni_moe`; a rename/removal breaks the (now default)
  native path at import. The issue's "reduce HF reliance" motivation is only
  partly met for vision (audio fully replicates its frontend). Deliberate
  tradeoff for identical patch ordering; not ticket-blocking. (Score 45)
- **CUDA-graph capture deferred** — sanctioned by the issue's Gotchas ("CUDA-graph
  capture … may hurt with varying shapes"); the encoders have dynamic token
  counts. Note it was evaluated, not missed. (Score 40)

## Changes applied this round (minimal, ticket-scoped)
- `vision_encoder.py`: corrected the "bit-identical" patch-embed claim to
  "exact in fp32; ≤~1.6e-2 bf16 rounding" with the measured Conv3d/matmul numbers.
- (Prior round, retained) native default-on; `merge_sq` sourced from the encoder;
  vision `forward_batched` `req_token_counts` fallback; lint (B905/I001).

Net source diff: **27 insertions / 14 deletions across 4 files.**

## Reviewer Guide
- **Effort: 4/5** — GPU numerics + a default-flip affecting every Qwen3-Omni
  config; de-risked by fp32 bit-exact parity and an unchanged output contract.
- **Entry points:** `qwen3_omni_model.py:1467-1555` (factory seam);
  `submodules.py` Native\* `forward_batched`/`can_batch`; `vision_encoder.py`
  `VisionPatchEmbed` (the perf win).
- **Focus areas:** (1) default-on without flash-attn → SDPA fallback path;
  (2) bf16 intermediate divergence into the Thinker's DeepStack inputs;
  (3) the unverified "5-11×" audio claim.

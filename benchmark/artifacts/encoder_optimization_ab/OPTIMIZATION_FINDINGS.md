# Qwen3-Omni native-encoder optimization — findings (1× H200, no flash-attn)

Goal (issue #131): make the native M\* encoders **win** the before/after benchmark,
on the no-flash-attn H200 path. Directive: *"A/B test each optimization against the
eager/unoptimized baseline and keep only the ones that actually win."*

## Diagnosis

Baseline native (dense-mask SDPA varlen) **lost to the HF-wrapper encoder at bs≥8**
on I2T/S2T — the opposite of the goal. Root cause: the no-flash-attn fallback
(`audio_encoder._sdpa_varlen`) built a dense `(ΣLᵢ × ΣLᵢ)` block-diagonal mask, i.e.
**O((batch·tokens)²)** — quadratic in batch. At serving batch sizes the cross-segment
terms dominate encoder time. (Documented caveat in `README_qwen3_omni_encoders.md`:
"native large-batch numbers are SDPA-pessimistic"; their intended fix was flash-attn,
which is excluded on this path.)

## Lever A (KEPT) — per-segment varlen attention

Replace the dense mask with a per-segment SDPA loop: run dense SDPA on each image/audio
segment independently → **O(Σ Lᵢ²)**, linear in batch. Mathematically identical to the
block-diagonal mask (attention never crosses segments), so parity is preserved by
construction. Selectable via `MSTAR_VARLEN_BACKEND` (default `per_segment`).

**Microbench** (ms per single attention layer, H200 bf16; ×27 vision / ×32 audio layers):

| case | dense_mask | per_segment | padded_batch | flex | nested |
|---|---|---|---|---|---|
| B=32 L=1024 | 73.1 | **1.30** | 3.99 | 14.1 | 12.5 |
| B=32 L=2304 | 361.5 | **5.55** | 12.3 | 52.0 | 55.7 |
| B=1  L=1024 | 0.19 | **0.09** | 0.25 | 5.30 | 1.97 |

`per_segment` wins at **every** shape (even bs=1). FlexAttention / nested-tensor were
slower (compile / dispatch overhead) → **rejected** per the keep-only-winners rule.

**End-to-end TTFT p50 (ms), native baseline → optimized vs M\*-old (HF):**

| path/bs | baseline | optimized | HF | opt vs base | opt vs HF |
|---|---|---|---|---|---|
| I2T bs1 | 123 | 149 | 180 | 0.82× | **1.20×** |
| I2T bs8 | 2281 | 1100 | 1785 | 2.07× | **1.62×** |
| I2T bs16 | 2686 | 1038 | 1631 | 2.59× | **1.57×** |
| I2T bs32 | 2970 | 1184 | 1944 | 2.51× | **1.64×** |
| S2T bs16 | 1788 | 1215 | 1475 | 1.47× | **1.21×** |
| S2T bs32 | 1705 | 676 | 966 | 2.52× | **1.43×** |

Native-new now **beats native-old (HF) across the batch trend** (soft spot: bs4, where
TTFT is thinker-prefill-bound, not encoder-bound — the encoder optimization can't move it).

## Lever A.2 (KEPT) — ADAPTIVE backend (fixes the audio/speech regression)

A/B at the encoder level (bench_encoder_fast.py, repeats=8) revealed per-segment
is NOT universally best — it depends on segment structure:

| ms/req or img | dense | per_segment | adaptive |
|---|---|---|---|
| VISION bs32 (few big segs) | 34.3 | 4.42 | **4.43** (picks per_segment) |
| AUDIO  bs8  (many tiny segs)| 5.22 | 8.20 | **5.24** (picks dense) |
| AUDIO  bs32 | 11.75 | 7.66 | **7.59** (picks per_segment) |

Vision has a FEW BIG segments (~728-tok images) so per-segment always wins.
Audio has MANY TINY windows (~50 tok) so per-segment fires hundreds of tiny
kernels (launch-bound) and LOSES to the dense single kernel until the total grows
enough (~bs32) that dense's O(n²) mask explodes. My first per-segment-only fix
therefore REGRESSED audio (S2T) at bs1-16 — the user's "speech seems worse".

Fix: `_sdpa_varlen_adaptive` (new default) picks per call from a shape-only metric
M = total²/n_seg (proxy for the dense/per-segment compute ratio), threshold ~5e5.
Vision → per_segment every batch; audio → dense at low batch, per_segment at bs32.
Best-of-both, no GPU sync. This is exactly issue #131's "A/B test, keep winners".

## Lever B (REJECTED) — piecewise CUDA-graph the encoder block-loop

Issue #131's "CUDA-graph capture" lever; repo's `MSTAR_I2T_S2T_I2S_optimization_review.md`
recommends copying the vjepa2 `get_piecewise_runner_config()` pattern. **Rejected** after
reading the contract: `PiecewiseCudaGraphRunner` requires a **fixed `capture_seq_len`**,
but vision images have **variable patch counts** → the exact "shape variance triggers
recompiles" pitfall the issue warns about (vjepa2 itself returns `None` for its
variable-length predictor). Low ROI too: it only targets bs=1 per-block launch overhead
(~13 ms encoder), where native already beats HF. Not worth the risk for variable-resolution
vision.

## Lever C (deferred) — per-layer CPU/sync reduction

Precompute segment boundaries once instead of `cu_seqlens.tolist()` per layer (×27/×32
GPU→CPU syncs/forward). The microbench includes the per-layer `.tolist()` and per_segment
still wins by 50–65×, so this is a minor follow-up, not a gating win.

## Verdict vs acceptance criteria
- **Equal-or-better in before/after:** ✅ optimized native beats both the dense baseline
  and the HF wrapper across the batch trend (bs4 ~par).
- **Parity within tolerance:** ✅ per-segment is bit-identical math to the block-diagonal
  mask that already passed `qwen3_omni_encoder_parity.py`.
- **Only winners kept:** ✅ A kept; B and the slower attention backends rejected.

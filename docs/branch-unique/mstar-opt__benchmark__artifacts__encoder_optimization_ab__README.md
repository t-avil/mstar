# Qwen3-Omni encoder optimization — A/B evidence

Measured on 1× H200, bf16, **no flash-attn** (SDPA path). Per issue #131's
directive: A/B each optimization against the eager/unoptimized baseline, keep
only winners.

## Charts (measured)
- `vision_encoder_ab.png` — varlen attention backends on the native VISION
  encoder: dense (old baseline) vs per-segment vs padded. per-segment is flat &
  **7.6× faster than dense at bs32**.
- `encoder_adaptive_ab.png` — vision + audio, dense vs per-segment vs **adaptive**.
  Vision → per-segment; audio → dense at low batch, per-segment at high batch.
  Adaptive = best-of-both (fixes the audio low-batch regression per-segment caused).
- `torch_compile_ab.png` — eager vs `torch.compile(dynamic=True)`. **Does NOT win**:
  vision 0.63× @bs1 / ~1.0× mid; audio 0.51–0.97× (loses everywhere); 14–16
  recompiles across shapes (the #131 shape-variance pitfall).
- `cudagraph_ab.png` — eager vs CUDA-graph replay of the encoder block-loop
  (fixed shape). **2.27× at bs1** (launch-bound), **~1.0× at bs≥4** (compute-bound).
  Idealized best case (mask precomputed in a static buffer since capture forbids
  the real data-dependent build; math SDPA for capture-safety).

## Data
- `varlen_attn_microbench.txt` — per-layer attention kernel timings (dense /
  per_segment / padded / flex / nested), the basis for the backend choice.
- `cudagraph_ab.json` — raw eager vs graph block-loop timings.

## Reproduce (scripts)
Run from the repo root with the mstar venv (HF cache + `flash_attn` blocked):
- `bench_varlen_attn.py` — attention-kernel microbench (no model load).
- `bench_encoder_fast.py` — native vision/audio encoder forward, per backend.
- `bench_compile.py` — eager vs torch.compile(dynamic=True), with recompile count.
- `bench_cudagraph.py` — eager vs CUDA-graph replay of the block-loop.
- `plot_*.py` — render the corresponding charts.

## Conclusion
torch.compile and CUDA-graph capture do **not** win for these variable-shape
encoders on the no-flash SDPA path. The kept optimization is the **adaptive
varlen SDPA**. The recommended next step is migrating encoder attention to
**FlashInfer** (already powers M*'s Thinker/Talker with CUDA-graphs on this box):
its plan/run split makes varlen attention graph-capturable, which would unlock
the measured ~2× bs1 CUDA-graph win cleanly. See `OPTIMIZATION_FINDINGS.md`.

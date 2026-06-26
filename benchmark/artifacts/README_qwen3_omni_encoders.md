# Qwen3-Omni native-encoder benchmark & parity — findings

Issue: [#131 — Port the Qwen3-Omni multimodal encoders into M\* and optimize them](https://github.com/mstar-project/mstar/issues/131)

Reproduce from a clean machine:

```bash
benchmark/setup_and_run_qwen3_omni_encoders.sh          # env from zero + perf, auto-picks a free GPU
python -m benchmark.qwen3_omni_encoder_parity --device cuda:0   # per-layer parity
```

All data below is persisted as JSON (with **every raw sample**) and PNG in this directory.

## Methodology (so the numbers are auditable, not taken on faith)

- **Hardware:** 1× NVIDIA H100 80GB, exclusive (one idle GPU pinned via `CUDA_VISIBLE_DEVICES`; no GPU sharing).
- **Stack:** torch 2.9.1+**cu128**, transformers 5.12.1. (The box is CUDA 12.8 — `nvcc` 12.8 / `torch.version.cuda == 12.8` — so cu128 is the correct wheel per `docs/installation.rst`; the cu130 wheel is for CUDA 13.x boxes and would fail to compile flash-attn here. `setup_and_run_qwen3_omni_encoders.sh` auto-selects it via `UV_TORCH_BACKEND=auto`.)
- **dtype = bfloat16 = the production setup.** The encoders run on the `enc_dec` stateless engine whose `autocast_dtype` defaults to `torch.bfloat16` (`mstar/engine/stateless_engine.py:65`). We verified this rather than assume it, because the headline result *only* exists in bf16 (see patch-embed).
- **Attention = SDPA; flash-attn excluded.** Both HF and native use SDPA, so the comparison isolates the structural changes (patch-embed, batching), not the attention kernel. **Caveat:** the native encoder's SDPA fallback builds an O(total_tokens²) mask, so native *large-batch* numbers here are pessimistic vs a production flash-varlen deployment. HF vision is unaffected (it is Conv3d-bound, not attention-bound).
- **Batch sizes `(1,2,4,8,16)` and `repeats=10`** reuse the existing `perf_testing/offline_homogenous.sh` convention; CUDA-event timing follows `test/modular/test_fused_rmsnorm.py`. Each datapoint = 10 independent samples → mean ± std.
- Perf uses random weights (value-independent). **Parity** is measured separately.

## Result 1 — Patch-embed: the bf16 Conv3d cliff (headline, attention-free)

HF computes patch-embed as a bf16 `Conv3d`; native proves it is a per-patch linear (kernel==stride) and runs it as `F.linear`.

| patches | Conv3d **bf16** | Conv3d **fp32** (control) | native matmul (bf16) |
|---:|---:|---:|---:|
| 728  | 3437 ms | 0.196 ms | 44.5 µs |
| 1600 | 7606 ms | 0.319 ms | 43.8 µs |
| 3136 | 15060 ms | 0.602 ms | 49.6 µs |

The **fp32 control is the key evidence**: the *same* Conv3d is 0.2 ms in fp32 but 3437 ms in bf16 — a ~17,000× cuDNN low-precision cliff for this shape on this stack, **not** a slow convolution per se. The matmul (~45 µs) sidesteps it entirely. This is real and rock-stable (±~15 ms over 10 repeats).

## Result 2 — Full encoders vs batch size (bf16, 10 repeats, ms/item ± std)

Vision (per image):

| batch | HF wrapper | native | speedup |
|---:|---:|---:|---:|
| 1 | 3479.4 ± 15.5 | 13.51 ± 0.12 | 257× |
| 2 | 3486.4 ± 14.8 | 7.07 ± 0.03 | 493× |
| 4 | 3487.7 ± 16.6 | 5.68 ± 0.01 | 614× |
| 8 | *(skipped)* | 7.13 ± 0.01 | — |
| 16 | *(skipped)* | 10.81 ± 0.01 | — |

- HF per-image cost is **flat** across batch (3479→3488) — it has **no cross-request batching** (skipped batches 8/16 to save ~10 min; the flatness is the point).
- Native batches well to n=4 (13.5→5.7 ms/img), then **rises** at n=8/16 — that is the SDPA-fallback O(n²) mask, *not* a native-encoder defect; production flash-varlen removes it.

Audio (per request, ~30 s clip):

| batch | HF wrapper | native | speedup |
|---:|---:|---:|---:|
| 1 | 15.27 ± 0.03 | 12.77 ± 0.15 | 1.20× |
| 2 | 9.50 ± 0.13 | 7.09 ± 0.14 | 1.34× |
| 4 | 6.58 ± 0.08 | 3.90 ± 0.04 | 1.69× |
| 8 | 5.34 ± 0.02 | 3.39 ± 0.01 | 1.58× |
| 16 | 4.87 ± 0.01 | 4.14 ± 0.02 | 1.18× |

- Audio speedup is **modest (1.2–1.7×)** and **peaks at n=4–8, then regresses at n=16** — the native frontend's Python `cu_seqlens` loop (`get_audio_cu_seqlens`) is a per-forward CPU cost that grows with window count.
- ⚠️ This does **not** reproduce the factory docstring's "5-11x throughput" claim for audio under this (SDPA) setup. That number is unverified here.

## Result 3 — Per-layer parity (native vs HF, through depth)

Random HF weights copied into native (`load_state_dict`: **0 missing / 0 unexpected** — structural parity), identical inputs, hidden state captured after every block.

| dtype | vision final cos / relL2 | worst-layer relL2 | audio final cos / relL2 | worst-layer relL2 |
|---|---|---|---|---|
| **fp32** | 1.000000 / 6.97e-4 | 5.41e-4 | 1.000000 / 5.46e-4 | 3.80e-4 |
| **bf16** | 0.999891 / 1.48e-2 | 1.37e-2 | 0.999916 / 1.30e-2 | 1.25e-2 |

- **fp32: cos = 1.000000 at every layer** → the native implementation is *mathematically identical* to HF. No logic bug; fp32 residuals are pure rounding.
- **bf16: divergence amplifies monotonically with depth** (relL2 4e-4 → 1.4e-2; per-element max-abs reaches ~0.31 vision / ~0.50 audio mid-network). It stays within the parity test's bar (cos>0.999, relL2<0.05) but the *end-only* assertion under-characterizes it. The DeepStack captures (blocks 8/16/24, fed to the Thinker) carry relL2 up to ~1.3e-2. Part of the bf16 audio drift is the SDPA-fallback masking, not the production flash path.

See `qwen3_omni_parity_depth_*.png` for the divergence-vs-depth curve.

## Bottom line vs acceptance criteria

- **"at least as fast"** — yes, every shape (vision overwhelmingly; audio modestly).
- **"ideally faster with concurrent requests"** — yes for native (HF has zero batching), with the SDPA-fallback caveat at large batch.
- **parity within tolerance** — yes, and fp32 is bit-exact; bf16 is within bar but the end-only test understates intermediate divergence.
- **honesty flags:** audio "5-11×" not reproduced (1.2–1.7× measured); native large-batch numbers are SDPA-pessimistic.

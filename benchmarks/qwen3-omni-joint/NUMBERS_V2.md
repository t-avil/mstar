# NUMBERS_V2.md — M*-v2 (opt/decode-v2 1e171e1 (+FAST_POSTPROC), fp8 MoE + fused topk +
# worker CPU fixes + inline/batch emit + encoders-on-rank-0) vs recorded baselines

Sweep: 2026-07-02, GPUs 6,7, WARMUP=5, N=max(50,10B), closed loop.
Baselines (mstar_new/mstar_old/vllm) are the committed v2 rebenchmark
aggregates — NOT re-run. Metric: req/s primary (cross-system;
tok/s embeds output-length skew: vLLM generates ~20% longer text).

**Note on the W1 re-sweep (final numbers above):** the s2t/i2t cells are from
the 1e171e1 build (adds MSTAR_FAST_POSTPROC, validated +6.4% i2t B32 /
+13.5% s2t B8 in a contention-robust interleaved A/B, ab_w1v2/). Sweep-scale
deltas vs the d04dcb4 sweep are within sweep-to-sweep variance (~±5%) — the
interleaved A/Bs carry the per-feature evidence; single sweeps bound it.

## s2t — audio_to_text

| B | M*-v2 req/s | M*-new | M*-old | vLLM | v2/vLLM | v2/new |
|---|---|---|---|---|---|---|
| 1 | 4.360 | 4.032 | 2.036 | 3.831 | 1.14x | 1.08x |
| 2 | 7.546 | 5.848 | 4.106 | 9.143 | 0.83x | 1.29x |
| 4 | 11.801 | 8.188 | 4.460 | 13.225 | 0.89x | 1.44x |
| 8 | 17.197 | 10.066 | 4.440 | 15.791 | 1.09x | 1.71x |
| 16 | 24.935 | 12.572 | 6.641 | 19.798 | 1.26x | 1.98x |
| 32 | 31.565 | 19.087 | 6.724 | 30.850 | 1.02x | 1.65x |

## i2t — image_to_text

| B | M*-v2 req/s | M*-new | M*-old | vLLM | v2/vLLM | v2/new |
|---|---|---|---|---|---|---|
| 1 | 0.800 | 0.686 | 0.675 | 0.898 | 0.89x | 1.17x |
| 2 | 1.294 | 1.125 | 1.011 | 1.556 | 0.83x | 1.15x |
| 4 | 2.138 | 1.719 | 1.526 | 2.426 | 0.88x | 1.24x |
| 8 | 3.361 | 2.401 | 2.069 | 3.455 | 0.97x | 1.40x |
| 16 | 4.707 | 3.451 | 2.852 | 5.112 | 0.92x | 1.36x |
| 32 | 6.299 | 4.394 | 3.363 | 8.210 | 0.77x | 1.43x |

## s2s — audio_to_speech

| B | M*-v2 req/s | M*-new | M*-old | vLLM | v2/vLLM | v2/new |
|---|---|---|---|---|---|---|
| 1 | 2.367 | 2.152 | 1.695 | 0.899 | 2.63x | 1.10x |
| 2 | 3.668 | 3.365 | 3.019 | 1.385 | 2.65x | 1.09x |
| 4 | 5.479 | 5.273 | 4.411 | 2.229 | 2.46x | 1.04x |
| 8 | 7.853 | 6.922 | 5.228 | 3.542 | 2.22x | 1.13x |
| 16 | 10.860 | 10.763 | 5.164 | 4.370 | 2.49x | 1.01x |
| 32 | 12.441 | 13.096 | 5.170 | 5.516 | 2.26x | 0.95x |

| B | M*-v2 audio_s/s | vLLM audio_s/s | v2/vLLM | M*-v2 RTF p50 | vLLM RTF p50 |
|---|---|---|---|---|---|
| 1 | 10.46 | 6.39 | 1.64x | 0.097 | 0.159 |
| 2 | 16.35 | 10.10 | 1.62x | 0.121 | 0.209 |
| 4 | 24.18 | 15.35 | 1.58x | 0.169 | 0.252 |
| 8 | 37.37 | 23.00 | 1.62x | 0.216 | 0.321 |
| 16 | 50.55 | 28.98 | 1.74x | 0.314 | 0.451 |
| 32 | 59.26 | 37.54 | 1.58x | 0.543 | 0.694 |

## i2s — image_to_speech

| B | M*-v2 req/s | M*-new | M*-old | vLLM | v2/vLLM | v2/new |
|---|---|---|---|---|---|---|
| 1 | 0.287 | 0.255 | 0.250 | 0.116 | 2.47x | 1.13x |
| 2 | 0.446 | 0.403 | 0.403 | 0.175 | 2.54x | 1.11x |
| 4 | 0.827 | 0.794 | 0.734 | 0.290 | 2.85x | 1.04x |
| 8 | 1.258 | 1.216 | 1.116 | 0.433 | 2.91x | 1.03x |
| 16 | 1.815 | 1.869 | 1.667 | 0.648 | 2.80x | 0.97x |
| 32 | 2.082 | 2.311 | 2.054 | 0.871 | 2.39x | 0.90x |

| B | M*-v2 audio_s/s | vLLM audio_s/s | v2/vLLM | M*-v2 RTF p50 | vLLM RTF p50 |
|---|---|---|---|---|---|
| 1 | 12.71 | 7.10 | 1.79x | 0.079 | 0.141 |
| 2 | 20.03 | 10.71 | 1.87x | 0.098 | 0.187 |
| 4 | 35.71 | 17.56 | 2.03x | 0.111 | 0.223 |
| 8 | 57.28 | 27.22 | 2.10x | 0.133 | 0.287 |
| 16 | 80.64 | 41.28 | 1.95x | 0.194 | 0.382 |
| 32 | 93.14 | 54.75 | 1.70x | 0.337 | 0.575 |

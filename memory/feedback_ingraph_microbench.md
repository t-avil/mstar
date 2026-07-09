---
name: ingraph-microbench-required
description: Decode-size GPU kernel microbenchmarks are invalid outside CUDA graphs — capture a loop in a graph and time replays
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e01b1ff5-c6b6-4e43-a2ed-40135d9fe641
---

Out-of-graph CUDA-event timing of decode-sized kernels (M≈1–64) on H200 is dominated by launch gaps and clock ramp and gives WRONG rankings. Example (Qwen3-Omni MoE, real weights): out-of-graph said fp8 w8a8 LOSES at M=32 (0.85×); in-graph (24 calls captured in one graph, timed over replays) it WINS 1.61×. The production decode step runs inside CUDA graphs, so in-graph is the only faithful setting.

**Why:** This exact artifact misled the 2026-06 optimization agent into rejecting fp8 MoE ("gemm 2× but end-to-end loses") and nearly misled again in July.

**How to apply:** For any decode-path kernel A/B: capture `for _ in range(24): fn()` in one `torch.cuda.CUDAGraph`, warm up, time `g.replay()` with CUDA events, divide. Cross-check totals against nsys `--cuda-graph-trace=node` kernel sums from a live server (see [[mstar-decode-bottleneck-2026-07]]).

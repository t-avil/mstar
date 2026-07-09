---
name: project_moe_bakeoff_24
description: M* campaign
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

Campaign #24 (MoE grouped-GEMM kernel bake-off) closed 2026-07-03: keep the
shipped Triton block-fp8 w8a8 grouped GEMM. No available challenger clears the
+8% decode gate.

**Result:** in-graph (E1), Qwen3-Omni thinker MoE (E=128, top_k=8, H=2048,
I=768), Triton wins at every decode M ∈ {8..128}. DeepGEMM masked
(`fp8_m_grouped_gemm_nt_masked`) — the canonical modern grouped-GEMM challenger,
same 128×128 blockwise-fp8 scheme — REGRESSES 10-20% (dg/tri 1.10–1.20×).

**Why (structural, not tuning):** DeepGEMM masked has a block_m=64 floor per
*active* expert (verified: batched slot T<64 → garbage, T≥64 correct; needs
`disable_ue8m0_cast=True` on sm90/H200). At decode nearly all 128 experts are
active but see only ~2-8 real tokens each, so DeepGEMM pays ~E_active×64-row
tiles vs Triton's token-sorted ~M×8 padded rows. It's an EP/large-batch kernel,
wrong for dense low-M decode. DeepGEMM contiguous is worse (128-token/expert
align). The two grouped GEMMs are ~90% of the MoE forward.

**Tile sweep** at new W7 buckets M=24/28: shipped BLOCK_M=16 is already optimal
(M=28 best outright; M=24 only 1.7% from num_warps 4→8, sub-gate). fp8 pins
BLOCK_N=BLOCK_K=128; Triton needs power-of-2 BLOCK_M. Keep BLOCK_M=16.

**Not benchmarked (available, full-fused, different layouts):** flashinfer
`trtllm_fp8_block_scale_moe`, sgl `fp8_blockwise_scaled_grouped_mm`/`cutlass_moe`,
vllm fused_moe. trtllm low-latency MoE is the one follow-up worth trying (fuses
launches, could win on overhead despite the tile floor).

Harness + raw numbers: /m-coriander/coriander/tim/mb_moe_bakeoff/ (bench.py,
tile_sweep.py, raw.json, command.txt). Related: [[project_mstar_decode_bottleneck]]
(MoE fp8 = 1.13× e2e; this confirms the fp8 grouped GEMM itself is near-optimal).

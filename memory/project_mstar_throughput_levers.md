---
name: project_mstar_throughput_levers
description: "M* i2t throughput lever results 2026-07 — MoE-autotune washes, naive multi-step regresses; TTFT dominates B32 JCT"
metadata: 
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

Empirical results chasing i2t decode throughput vs vLLM (measured ignore_eos len=256,
warmed, node-isolated). Goal: beat vLLM on tok/s+req/s all batches.

- **MoE autotune (MSTAR_MOE_AUTOTUNE, opt/moe-autotune @50e1abd2):** per-M-bucket tuned
  Triton tiles (mostly num_warps=8/num_stages=4 which the hardcoded config never set) →
  **+4-12% on the isolated MoE GEMM** (moe_microbench.py) but **e2e WASH** (B8 -0.5%, B16
  -1.7%, B32 +4% within noise). Confirms **decode is CPU-floor-bound, GPU ~50% idle → GPU
  kernel wins don't move decode throughput.** Committed, bit-identical, helps only GPU-bound
  prefill. Not a decode lever.
- **Multi-step decode replay (MSTAR_DECODE_MULTISTEP, opt/decode-multistep):** replay captured
  decode graph n times/scheduler pass to amortize the 26ms Python. Greedy output BYTE-IDENTICAL
  n=1/2/4 (correct!). But **B32 REGRESSES 3.5-4.8x** (n=1 5.66 req/s → n=2 1.61 → n=4 1.18,
  ITL 7→19ms). Root: the n-deep FlashInfer pre-plan is INFEASIBLE (one wrapper, plan()
  overwrites in place, captured graph binds its addresses), so the agent re-plans INLINE per
  micro-step on the GPU thread AND loses the speculative overlap → per-step plan cost >> Python
  saved. To win needs cheap plan-advance or n captured graphs (the empty exp/mixed-cg-* wall).
  Don't scale as-is.
- **SCOREBOARD REFRAME 2026-07-07 (mstar-new live @8402, all opt flags ON, vs committed
  vLLM 0.22 h2h):** M* ALREADY BEATS vLLM on **ITL every batch** (B32 9.3 vs 15.4ms) and
  **req/s B1-B16** (B8 4.20>3.67, B16 5.89>5.62). At B32 len512 (decode-dominated) M*
  **1771 tok/s vs vLLM 1769, ITL 10 vs 15.4** → decode at parity-or-better. **The ONE i2t
  loss is B32 req/s (7.42 vs 8.32), caused 100% by TTFT: ~2400ms vs vLLM FLAT 179ms (14x)
  = prefill serialization at high concurrency.** The already-ON mixed-prefill flags
  (MIXED_BATCH/MERGED_PREFILL/CHUNKED_PREFILL_V2) only nudged 2532→2369. **⇒ The decode
  refactors (R1-R10 in AR_LOOP_10_REFACTORS.md) optimize ITL/tok-s = axes we already win;
  they CANNOT fix the loss. Best-ROI target = B32 prefill admission/chunking scheduling
  (run a waiting req's first prefill chunk promptly, vLLM flat-TTFT behavior). Parity: pure
  chunking of ONE prefill is parity-safe; prefill+decode CO-BATCHING is the parity risk.**
  IMPORTANT: mstar-new is a DIFFERENT/newer build than the on-disk 2434-line worker.py
  checkout (running worker.py is 4600+ lines) — [[feedback_mstar_worktree_pythonpath]].
- **KEY REFRAME:** at B32, TTFT (~2987ms) > whole decode phase (256tok×7ms≈1792ms). In
  closed-loop req/s = C/JCT, JCT = TTFT + decode. So **B32 req/s is gated by TTFT (prefill
  serialization), NOT decode throughput.** Cutting TTFT 2987→~250ms lifts B32 req/s ~2.8x —
  the single biggest throughput lever AND priority #2. Fix = stall-free mixed prefill+decode /
  encoder side-stream overlap (WS-B). SIDE_PREFILL is broken (qo_indptr race). s2t already wins
  (36 vs 27 req/s B32). See [[project_mstar_decode_bottleneck]], [[project_mstar_vllm_mrv2]],
  BEAT_VLLM_MASTERPLAN.md.

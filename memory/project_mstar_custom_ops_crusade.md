---
name: mstar-custom-ops-crusade
description: "opt/custom-ops branch — torch.library custom ops removed the compiled-thinker graph breaks (816→~165), MSTAR_CUSTOM_OPS=1; boot-time cache fix + residual walls"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

M* compiled-thinker "unbroken graph" campaign (2026-07-04). Branch `opt/custom-ops`
(pushed to fork), worktree `/m-coriander/coriander/tim/mstar-crusade`, forked from
opt/integration-v4. All gated by `MSTAR_CUSTOM_OPS=1` (default-off; off-path
byte-identical to baseline). Supersedes the "Compile round-2 — CLOSED" dead-end in
[[mstar-decode-bottleneck]] / [[mstar-24option-board]].

**The route that works: torch.library custom ops.** A `@torch.compiler.disable`
call from compiled code forces a graph break (fusion boundary). A registered custom
op is an opaque in-graph node dynamo never traces into — same kernels, no break.
The op can't take the stateful BatchedCacheManager as an arg (dynamo would trace in),
so state comes from a forward-context global (`compile_ops.set_active_manager`),
published by the non-compiled driver in `cache_manager.plan_attention` +
`cuda_graph_runner` capture loop. Exactly vLLM's unified_attention pattern. New module
`mstar/engine/compile_ops.py`.

**Four steps (SHAs):** step1 `b7fb94e` mstar::run_attention (Thinker) + drop
set_layer_idx (thread layer_idx explicitly); step2 `f152e3b` extend to Talker (both
construction sites + both loops); step3 `be16eeb` mstar::fused_experts_fp8 +
`prequantize_fp8_experts` hoist (quantize experts BEFORE compile so the mutating
lazy-quant never traces — this is what E1's disable was guarding; forward then reads
cached fp8 weights + calls the op, guard folds to constant); step4 `12ce776`
mstar::apply_rope (base _apply_rope, Talker/code-pred; op clones q/k to avoid
input-aliasing). Census 1617→816(norm fix)→438→312→**41 header / ~165 attributed**
(<200 bar crushed). Every per-layer class zeroed: run_attention, set_layer_idx, fp8
MoE, talker apply_rope.

**Boot-time tension (shippability fix):** fewer breaks = bigger fused regions =
longer Inductor max-autotune (steps 3+ pushed boot past ~40min, missed lab's 20min
readiness window). Fix, verified: `TORCHINDUCTOR_FX_GRAPH_CACHE=1` +
`TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache` (on pool, 616M/
70852 files after one boot) = one-time cost. `TORCHDYNAMO_CACHE_SIZE_LIMIT=128`
killed a frame-[7/84] recompile-limit eager fallback (0 hits after). Always boot the
custom-ops stack with these three envs.

**Validated:** every step CPU/meta-proven pre-GPU (disabled=1break/2graphs →
op=0/1graph; register_fake meta-traces; eager output identical). Steps 1-3 GPU
end-to-end correct + fast (6.325 req/s i2t B32, contended box). Full-stack census
GPU-banked.

**Residual ~165 = head/tail boundaries, none per-layer:** advance_seq_lens
(talker.py:173 + thinker.py:261, end-of-forward, ~0 fusion payoff — low value);
injected_sampler.sample (submodules.py:2309 — WALL by design, vLLM keeps sampling
out too); get_qo_indptr_buf (submodules.py:1809, prefill branch); decode_attn_nhd
(talker.py:547, code-predictor dense-cache attn — real step-5 candidate if talker
must be fully clean); code2wav (compile=False by design, out of scope). Compiled
forward is now ~one fused graph embed→layer-stack, same shape as vLLM.

**DEFERRED (clean window):** ON/OFF perf A/B — MSTAR_CUSTOM_OPS is compile-time so
needs two sequential same-pair boots (on vs off); tonight's box too contended
(identical-arm cells 3.0–6.3). The fewer-breaks→bigger-fusion SPEEDUP is unquantified;
this + the cache-proof warm-time are the first clean-window items. See
[[feedback-benchmark-metrics]] for the A/B protocol.

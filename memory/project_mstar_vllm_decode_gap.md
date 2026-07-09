---
name: project_mstar_vllm_decode_gap
description: "Root-cause decomposition of why M* text-decode throughput trails vLLM 0.22, and which levers actually help"
metadata: 
  node_type: memory
  type: project
  originSessionId: d5aa69ca-2a9c-4695-942a-6593b1ee2874
---

Measured (committed benchmarks branch, 2gpu 6,7): on S2T text decode M* is
**0.62–0.64× vLLM** at B≥2 and scales worse with batch (M* 4.7× vs vLLM 8.1×
over B=1→32). M* *wins* at B=1 (1.05×). At B=32 M*'s inter-token latency is
**2.1× vLLM's** and the penalty grows with batch — so the gap is in the decode
STEP, not prefill bubbles.

**There is no single 2× silver bullet.** Profiled decode-step decomposition
(`--log-stats`, thinker_decode B=32): fwd(GPU) 16.6ms + pre(plan) 3.19ms +
post(overlapped) 5.23ms + per-token SHM comm ~4.8ms (2.26ms tx + 2.52ms rx to
ship an 88-byte token). MoE is only ~46% of fwd (~7.6ms; microbench).

Levers, by ROI×safety:
1. `MSTAR_NUM_SLOTS=3` — free runtime flag, may hide the 3.19ms un-overlapped plan.
2. Tuned MoE Triton tiles (H200) — free, numerically identical, but only ~1.02–1.18× kernel / ~1–2% e2e. The static 16×32×64 was already near-optimal (decode MoE is memory-BW-bound; see [[project_mstar_moe_tuned_tiles]]).
3. fp8 MoE (w8a8) — vLLM's actual #2 opt (DeepGEMM). ~2× the ~7.6ms MoE slice → ~1.1–1.15× decode. M*'s Triton kernel is a sglang port with the fp8 branch STRIPPED (kernels.py:5-8) — re-add it. Numerics change (validate cos≥0.99). Caveat: the vLLM we benchmarked was bf16 too, so fp8 isn't apples-to-apples.
4. **Continuous batching (`MSTAR_MIXED_WALK`) — the right lever, but the EAGER impl LOSES.** On exp/combined-coalesce-piggyback, the eager mixed prefill+decode path (`kv_cache_engine._execute_mixed_eager`, `mstar/engine/mixed_walk.py`) runs but is NOT graphed. A time-separated sequential A/B showed S2T 1.26–1.66× — but that was a **CONTENTION ARTIFACT**. The **clean interleaved confirm (both servers side-by-side) OVERTURNED it**: S2T B8 = 0.78×, S2T B32 = **0.17×**, i2t B32 = 0.24× (318/320 succeeded, genuine ~6× per-step slowdown, not errors). Root cause VERIFIED: the mixed forward has no CUDA graph → every mixed step runs the 30B Thinker EAGER; at B8 the eager penalty already exceeds the piggyback benefit, at B32 nearly all steps are mixed → collapse. **The stubbed graphed mixed replay is REQUIRED, not optional** — continuous batching can't beat baseline until the mixed step is graphed. Ported to 4c33b33 (opt/mixed-walk, imports+logic OK); graphed replay design in DESIGN_mixed_walk_graph.md. Lesson: always interleave A/B arms on this contended box — time-separated ratios are unreliable ([[feedback_benchmark_metrics]]).

Config-level + fp8 both proven negative. Continuous batching is the right direction (vLLM's is graphed) BUT unproven until graphed on M*. Next: implement graphed mixed capture/replay, then re-run the interleaved A/B; only then is there a validated win.

vLLM's #1 decode win (full-step decode CUDA graph, FULL_AND_PIECEWISE) M* ALREADY has (bs 1–32). Attention is FlashInfer parity. Work lives on branch `opt/vllm-parity` (off 4c33b33) in worktree /m-coriander/coriander/tim/mstar-opt.

**UPDATE 2026-07-02:** Superseded by [[mstar-decode-bottleneck-2026-07]]. The gap decomposition here was incomplete: profiling showed the B32 bottleneck is the per-step Python/GIL floor (~26ms) plus phased-batching prefill serialization, not the forward. fp8 MoE was never actually wired on opt/vllm-parity; it is now implemented and validated (+13% e2e at batch) on branch `opt/decode-v2`.

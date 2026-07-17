# Autonomous Campaign — Experiment Queue (ranked, living doc)
# Adheres to /remote-control spec: max parallelism, no idle GPUs, planning rounds,
# graveyard never pruned, median-of-repeats, matched greedy protocol, parity gate, push branches.
# Status: QUEUED | RUNNING | DONE(result) | DEAD(cause)

## Current best (baseline of record)
- i2t: wt-boot-cache@2b43f06e + PD(qwen3omni_2gpu_pd.yaml) + preproc pool. WINS all metrics all
  batches (B32 TTFT ~tie 200 vs 179). Pushed bench/i2t-preproc-pd.
- s2t: same build. WINS req/s+tok/s+ITL(4x); LOSES TTFT (structural). Pushed bench/s2t-pd.

## OPEN PROBLEMS (targets)
1. i2t B32 TTFT: ~tie (200 vs 179), want clean win. Structural = single-GPU prefill.
2. s2t TTFT: loses all batches (66-217 vs 77-499). Structural = single-GPU audio prefill.
Both root: vLLM TP-prefill across 2 GPUs; M* TP regresses host-bound decode.

## RANKED QUEUE
### Tier 1 — cheap, high-confidence, untested (do first)
- [RUNNING] E1 eiv2 build (mstar-eiv2 @5e8ce330 moonshot/decode-composed) PD+preproc baseline:
  is the newer eiv2 build better than wt-boot-cache for i2t/s2t? Isolates build. Bakes eiv2-PD cache.
- [QUEUED] E2 eiv2 + MSTAR_FUSED_KV_HANDOFF=1 (cert'd i2t B32 10.06rps prior): i2t/s2t A/B vs E1.
- [QUEUED] E3 eiv2 + MSTAR_DECODE_SYNCFREE=1 (targets worker.py:3193 per-step sync, 36% blocked):
  ITL/tok-s at all batches. Parity gate (greedy determinism + coherence).
- [QUEUED] E4 eiv2 + MSTAR_FULLSTEP_DECODE=1 (+in-graph greedy): decode host-floor.
- [QUEUED] E5 eiv2 + MSTAR_CONDUCTOR_EVENT_WAIT=1 (~2ms off every TTFT path): TTFT all paths.

### Tier 2 — s2t TTFT attack (the structural deficit)
- [QUEUED] E6 s2t colocated topology (qwen3omni_colocated.yaml / 2gpu.yaml): does removing PD
  KV-handoff lower s2t TTFT? (trade vs throughput). Measure s2t sweep.
- [QUEUED] E7 s2t + MSTAR_PREFILL_CHUNK_TOKENS / MIXED_BUDGET raise: chunk audio prefill.
- [QUEUED] E8 s2t audio-encoder overlap variants (ENC_OVERLAP_V2 gated, batch-gated N<=2):
  low-batch s2t TTFT without high-batch throughput regression.

### Tier 3 — i2t B32 TTFT polish + structural
- [QUEUED] E9 i2t MSTAR_PREPROC_PROCS 8->16 + MIXED_BUDGET 4096->8192: co-admit B32 prefill 1 wave.
- [QUEUED] E10 mixed prefill-decode graph capture (vLLM-style co-admission) — the big structural
  lever; scope feasibility from code (micro_scheduler one-walk-per-batch relax).
- [QUEUED] E11 TP-prefill-only feasibility scoping (weights sharded-for-prefill, replicated-decode).

### Tier 4 — validation / rigor
- [QUEUED] E12 s2t full sweep at n=256 (wave-lottery) x3 repeats -> solid s2t medians.
- [QUEUED] E13 i2t parity vs frozen bf16 base reference (exact-token) — close the fp8 caveat.
- [QUEUED] E14 cross-check best build stability: re-measure i2t/s2t best at low host load x5.

## PLANNING ROUNDS
- Round 1: 2026-07-16 ~21:50 — 4 agents (Idiot/Research/Code/Propose) refining queue + novel ideas.

## Planning Round 1 — Research correction (IMPORTANT)
- vLLM Thinker is TP=1 SINGLE-GPU (not 2-GPU TP). Flat TTFT = co-admit decode+prefill in ONE
  forward (unified 32k token budget) + async sched + MRV2 zero-sync. => DROP E11 (TP-prefill moot).
- M* can't co-admit: whole-step fixed-shape CUDA graphs, one graph_walk/batch (micro_scheduler.py:97-105);
  mixed step matches no captured graph = 20x collapse (MSTAR_MIXED_WALK). Real structural fix =
  record-around-attention substrate for mixed prefill+decode (LARGE rewrite; E10 was NEUTRAL prior).
- NEW LEVER: MSTAR_ENCODER_COALESCE measured s2t B8 -32% TTFT (different from ENCODER_ASYNC which
  regressed s2t). => E8b PRIORITY: test ENCODER_COALESCE for s2t TTFT (+ ENCODER_ASYNC hardening).
- Idiot: E5(2ms) too small; E9 budget capped at 2048 bucket (modest); E10/E11 low-payoff.

## Planning Round 1 — CONSOLIDATED execution order (Propose + Idiot + Research)
Run as config-only reboots on the baking eiv2 build (mega-cache now warming). Each boot: i2t B32 +
s2t B8/B32 cells (the two open TTFT gaps), parity-gate (greedy determinism 16/16) FIRST for
sync-free/in-graph flags.
1. E1  eiv2 baseline (all new flags OFF) — is eiv2 build > wt-boot-cache? [RUNNING boot]
2. E2  +MSTAR_FUSED_KV_HANDOFF=1 — highest leverage; hits BOTH TTFT gaps (shared PD handoff edge). Prior cert i2t B32 10.06 rps.
3. E3  +MSTAR_DECODE_SYNCFREE=1 — worker.py:3193 36%-blocked sync; PARITY-CRITICAL (determinism gate first).
4. E4  +MSTAR_FULLSTEP_DECODE=1 (in-graph greedy) — decode host floor; parity gate.
5. E5  +MSTAR_CONDUCTOR_EVENT_WAIT=1 — ~2ms all paths (small; bundle, don't spend a solo boot).
6. E6  s2t colocated topology (qwen3omni_colocated.yaml) — s2t low-batch TTFT (trade throughput).
7. E12 s2t n=256 x3 — confirm s2t TTFT deficit is real not ramp-artifact.
NOVEL (scope/code, later): N1 dual-stream prefill co-admission (cheap structural B32 lever;
relax micro_scheduler one-walk-per-batch via 2nd CUDA stream); N7 audio-encoder-only TP for s2t.
NOTE: MSTAR_ENCODER_COALESCE not found in eiv2/wt-boot-cache code grep — confirm existence before E8b.
DROP: E11 (vLLM is single-GPU, TP-prefill moot). DEAD list unchanged.

## Planning Round 1 — Code agent corrections (eiv2 build = moonshot/decode-composed)
- E5 CONDUCTOR_EVENT_WAIT + chunk/budget flags DO NOT EXIST in eiv2 (bucketed capture instead). DROP E5.
- Present & boot-static: FUSED_KV_HANDOFF (kv_store.py:329), DECODE_SYNCFREE (cuda_graph_runner.py:216,
  worker.py:328), FULLSTEP_DECODE (submodules.py:406). => E2/E3/E4 valid; each needs own reboot.
- Co-admission (vLLM-style) = CODE change at micro_scheduler._select_node_priority:97-105 + token-budget
  accumulator at :276 (replaces flat max_batch_size cut). = NOVEL N1 real hook.
- CHEAP: widen MSTAR_PREFILL_BUCKETS / MSTAR_PREFILL_BATCH_SIZES (submodules.py:1163,1234) to co-admit
  more prefill tokens (captured-bucket ceiling is real; above top bucket = eager). => new E7b.
- s2t TTFT localization: MSTAR_PHASE_TIMING is decode-loop-only (worker.py:3300); add _phase_record
  hooks around audio_encoder exec (qwen3_omni_model.py:526) + Thinker prefill = small additive code.
  => E-DIAG: instrument s2t prefill to split audio-encode vs Thinker-prefill vs KV-handoff. HIGH VALUE.
- audio_encoder is STATELESS separate node (qwen3_omni_model.py:468,526) — the s2t prefill lever.
ROUND-1 DONE. Execution starts on eiv2 readiness: E1 baseline -> E2 fused-KV -> E3 syncfree -> E4 fullstep
-> E7b widen buckets -> E6 s2t colocated -> E-DIAG instrument. Parity-gate E3/E4 (determinism 16/16 first).

## Planning Round 2 — co-admission is a FLAG, not a rewrite (BIG)
- E-SPLIT: MSTAR_MIXED_SPLIT_ATTN=1 (+MIXED_BATCH +MIXED_CHUNK_SIZES=256,512) on champion. The real
  co-admission lever, committed + default-off + UNBENCHMARKED. Targets i2t B32 TTFT AND s2t TTFT.
  PARITY-GATE FIRST (mixed fp8/split-KV nondeterminism risk). Expected: possibly NEUTRAL (B32 CPU-floor)
  but must test. TOP PRIORITY next boot.
- If E-SPLIT neutral -> residual is CPU-floor (postprocess exile / syncfree checkstop) = harder.
- Rigor still queued: s2t n=256 x3, i2t stability, parity-vs-bf16.

## PHASE 3 (user GREENLIT 2026-07-17): decode-CPU-floor rewrite
Goal: cut the ~26ms/step single-GIL decode floor (postprocess 23% + check_stop 18% + zmq 13% + sync 8%)
that bounds i2t B32 TTFT + s2t TTFT. Approach: FRESH on champion (opt/decode-cpu-floor off infra/boot-cache,
worktree mstar-decodefloor), incremental, PARITY-GATED each step. Do NOT reuse broken eiv2 syncfree.
R1: exile _postprocess_batch off the per-step critical path (double-buffer / defer to a worker thread
    that can't hold the decode GIL). R2: capturable plan-advance. Each: determinism 32/32 + coherence gate,
    then i2t B32 + s2t TTFT/ITL A/B vs champion. Ship only if parity-clean AND beats baseline.

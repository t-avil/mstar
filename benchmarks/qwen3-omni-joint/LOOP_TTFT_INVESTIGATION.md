# LOOP STATE: i2t batch-TTFT regression investigation (started 2026-07-12 ~04:10Z)

MISSION (user directive, no questions until DONE):
- i2t TTFT exploded somewhere in the 92-commit lineage 4c33b33 (encoders,
  B32 TTFT ~576ms) -> modern builds (2300-3400ms). BISECT git history to the
  exact feature/commit. Profile M* during the TTFT explosion + why tok/s
  trails vLLM. Profile vLLM-Omni (user authorized in-session, overriding the
  owner rule for this investigation). Agent-research vLLM-Omni impl + our own
  code. Produce 20 fixes. Fix the regression. Scrappy benchmarks first,
  re-benchmark clean when scrappy looks good. Learnings -> markdown.
- DONE criteria: flat i2t TTFT better than current AND tok/s beats 0.22.
- GPU policy: ONE free pair only, never co-schedule (kanzhu on 0-3 now; use
  6,7; fallback 4,5). Never let the pair sit idle between loop steps.
- Loop: wake every ~10 min, drive the state machine below, min 8 hours.

## STATE MACHINE
PHASE = A-BISECT (current)
Bisect lineage (first-parent milestones, FEATURES_SINCE_ENCODERS.md):
  4c33b33 encoders [REF, expect B16~405/B32~576]
  -> 2239005 decode-v2 -> 74a985c sched-pack -> 01856e8 speech-floor
  -> 56e65f8 v2-policy -> 41200ec prefill-merge -> 12ce776 custom-ops
  -> ded928d cfgcache-v2 -> 7ef5150 stack-n2 -> 620de91 prep-h2d [BAD]
Protocol per point: worktree /m-coriander/coriander/tim/mstar-bisect
(detached checkout), lab_server boot NO MSTAR flags (each build's DEFAULT
path — the regression exists flags-off per the lean table), config:
qwen3omni_2gpu_encoff.yaml if present else qwen3omni_2gpu.yaml, GPUs 6,7,
port 8340+step. Cells (scrappy): i2t B16 n=32 w=3, i2t B32 n=48 w=3.
Metric: TTFT p50. Verdict: point GOOD if B32 TTFT <1000ms, BAD if >1800ms;
between = repeat once. Record every point below. Then narrow bisect within
the guilty milestone's commits.

## RESULTS TABLE (append per point)
| commit | label | B16 p50 | B32 p50 | B32 rps | load | verdict |
|---|---|---|---|---|---|---|

## PHASES
- A-BISECT: in progress. First boot: 4c33b33 reference.
- B-PROFILE-MSTAR: py-spy + GPU-util sampling during B32 TTFT window on the
  first BAD point + on godv9. PHASE_TIMING if available.
- C-PROFILE-VLLM: boot vLLM-Omni (user launch scripts in
  /m-coriander/coriander/tim/baselines/), i2t B16/B32 TTFT curve + util
  pattern. Their 0.22 wheel/scripts; do NOT commit vLLM numbers as h2h.
- D-RESEARCH: agents (running in background): vllm-omni impl reading
  (encoder/prefill scheduling at batch), our own current code re-read.
- E-SYNTHESIS: 20-fix list, fix the bisected regression, scrappy rebench,
  clean rebench, learnings markdown (this file + LEARNINGS_TTFT.md).

## LOG (append per wakeup: time, what ran, what next)
- 04:10 setup: state file created, bisect worktree + first boot launching,
  research agents spawning.
| 4c33b33 | encoders-REF | 1.7s | 2.764s | 4.22 | 46.62 | BAD |
- 06:32 wakeup1: point0 encoders-REF came back BAD (B32 2.76s) at load 46 —
  but the ORIGINAL encoders sweeps (n=320 x3) really did measure 532-717ms.
  And modern bs16v4 measured 629ms QUIET. => HYPOTHESIS SHIFT: there may be
  NO flags-off code regression; the lean-table 576-vs-2532 compared
  quiet-encoders vs loaded-modern. The true enemies: (1) extreme HOST-LOAD
  sensitivity of M* TTFT (~5x at load 40-70 vs quiet — attackable: prefill
  host-work, priority/pinning), (2) quiet-quiet structural gap 630 vs vLLM
  ~170 (single-rank prefill compute). Bisect protocol patched: load-gate <25
  + B16 n=48 / B32 n=96. RE-RUNNING point 0 gated (port 8341). If it comes
  back ~600ms => replace mid-lineage bisect with a 3-point quiet A/A
  (encoders vs godv9-default vs godv9-ship) to settle "did we regress" —
  then pivot the fix hunt to load-sensitivity + vLLM flatness profiling.
- 06:40 research(vllm-impl) landed -> RESEARCH_VLLM_OMNI_TTFT.md. KEY
  CORRECTION: vLLM Thinker is TP=1 SINGLE-GPU on CUDA (no all-reduce, no
  TP prefill; i2t is stage-0-only, talker idle). Flat-TTFT recipe = (1) ONE
  varlen-packed eager ViT forward for all scheduled images (budget 32768
  embed-tokens/step), (2) unified step co-scheduling full prefill + 31
  decodes (stall bounded ~1 ITL), (3) async scheduling (no CPU bubble),
  (4) greedy batched argmax + detok in separate API process. => our
  quiet 630ms vs their 170ms is ORCHESTRATION, not GPU compute; and their
  single-process async design explains load-robustness. Fix list backbone.
- 06:38 research(mstar-code) landed -> RESEARCH_MSTAR_TTFT_PATH.md. HEADLINE:
  flags-off path BYTE-IDENTICAL 4c33b33..HEAD except TWO default-ON flags.
  #1 suspect MSTAR_CONDUCTOR_POLL=1 (sleep(1ms) -> wait_for_work(50ms);
  wedged v1, re-smoke never done; 32 serialized vision walks x ~50ms missed
  wakeups ~= the whole 576->2300 delta). #2 FUSED_TOPK (microseconds).
  DECISIVE TEST queued: godv9 default-flags, POLL=0 vs POLL=1, gated B16/B32
  (bisect_step.sh now takes worktree+extra-flags). Milestone bisect PAUSED
  pending this A/B — it likely replaces the whole lineage search.
- 06:45 wakeup2: REF2 script had been corrupted by a mid-flight edit (bash
  incremental read — LESSON: never edit a script that a live bash is
  executing; copy-then-edit). Server survived; loop_step2.sh now runs REF2
  cells against the live 8341 encoders server, then chains DIRECTLY into
  the decisive godv9 MSTAR_CONDUCTOR_POLL=0 step (port 8342). POLL=1
  comparison after that.
| 4c33b33 | encoders-REF2-gated | .447s | .84s | 4.34 | 47.98 | ref |
| 1e7f10da | godv9-POLL0 | .428s | .671s | 5.27 | 58.54 | GOOD |
- 07:07 wakeup3: ★★★ REGRESSION FOUND. godv9-POLL0 (default flags +
  MSTAR_CONDUCTOR_POLL=0): B16 428ms / B32 671ms GOOD at load 58 — matches
  encoders ref (447/840). The conductor wait_for_work(timeout_ms=50)
  (default ON since b1c1ff18, re-smoke never done) is THE batch-TTFT
  regression: missed wakeups on per-hop conductor waits. POLL=0 restores
  sleep(1ms) polling => flat AND load-robust. POLL1 twin launching (8343)
  for the paired proof; then: flip default to 0 (+ roadmap task to fix
  wait_for_work wakeup sources properly), then SHIP-config + POLL0 scrappy
  rebench for the DONE criteria (TTFT flat-better ✓ expected; tok/s needs
  the full stack).
| 1e7f10da | godv9-POLL1 | .42s | .6s | 5.16 | 55.67 | GOOD |
- 07:22 wakeup4: ★ POLL HYPOTHESIS FALSIFIED — godv9-POLL1 (default flags,
  poll ON) ALSO GOOD: B16 420/B32 600ms @load56. => DEFAULT-FLAGS godv9 is
  fast AND load-robust; NO flags-off regression exists (matches byte-identical
  code study). THE REAL CULPRIT = SHIP FLAG STACK x HOST LOAD (ship: 629
  quiet but 1800-3600 @load40-75; default: ~600 even @load55). Also ship
  stack = +60% rps (7.7-8.5 vs 5.2). => FLAG-GROUP LADDER launched (one
  ship boot, port 8344): B32 cell per dynflag-group-off under CONTROLLED
  nice-19 spinner load (24 cores, lowest prio). Groups: L1 mixed/chunked off,
  L2 emit stack off, L3 sampler/checkstop caches off, L4 budgets 512,
  L0/L5 ship-full controls. Boot-static suspects (SIDECAR, CUSTOM_OPS,
  ORDERED, vis-batch) get follow-up boots if the ladder is flat.
| ladder | L0_ship_full | - | .932s | 5.09 | 88.86 | - |
| ladder | L1_no_mixedchunk | - | 6.14s | 4.91 | 103.46 | - |
| ladder | L2_no_emitstack | - | 6.174s | 4.92 | 116.24 | - |
| ladder | L3_no_samplercs | - | 3.224s | 5.26 | 120.97 | - |
| ladder | L4_budgets512 | - | .78s | 5.16 | 126.61 | - |
| ladder | L5_ship_full_again | - | .43s | 5.34 | 122.27 | - |
- 07:45 wakeup5: ladder round 1 CONFOUNDED by real external load mid-run but
  gold anyway: L0/L5 ship-full = 0.93/0.43s at nominal load 89-122 (nice-19
  spinners DON'T preempt M* -> load AVERAGE is not the trigger; NORMAL-
  priority neighbor CPU is). L1 (-mixed/chunk) 6.1s and L2 (-emit) 6.2s hit
  the real-load window — consistent with lean-history "machinery off = worse
  under load" but confounded. => (a) ship stack CAN do 0.4-0.9s B32 under
  high nominal load; (b) enemy = same-priority CPU preemption; fix directions:
  CPU-set pinning for M* procs, critical-path host-work cuts, and possibly
  keeping mixed/chunked ON as load armor. LADDER ROUND 2 launched on the
  live server for clean group attribution.
| ladder | L0_ship_full | - | 4.492s | 6.04 | 57.59 | - |
| ladder | L1_no_mixedchunk | - | 4.416s | 6.25 | 84.60 | - |
| ladder | L2_no_emitstack | - | .341s | 5.34 | 110.06 | - |
| ladder | L3_no_samplercs | - | 3.349s | 5.41 | 124.02 | - |
| ladder | L4_budgets512 | - | 4.864s | 5.89 | 120.66 | - |
| ladder | L5_ship_full_again | - | 1.173s | 7.94 | 105.21 | - |
- 07:56 wakeup6: LADDER ROUND 2 vs 1 = INCOHERENT (same flags 0.34<->6.2s
  across rounds; no group replicates). VERDICT: flag attribution CLOSED —
  the B32 TTFT explosion is STOCHASTIC under CPU contention, not a flag or
  code regression. Mechanism: M* B32 bursts ~21 cores (top: 2126% CPU);
  any same-or-similar-priority contention => admission-wave stalls,
  multi-second TTFT, bimodal. vLLM's single-process async design needs far
  less burst CPU => flat. FIX FAMILY = host-CPU demand reduction on the
  prefill/admission path + ops guidance (pinning/priority), NOT flag surgery.
  Ship rebench (natural conditions) launching on live 8344; LEARNINGS +
  20-fix list next.
| shipreb | i2t_B1_r1 | - | .269s | 0.92 | 24.74 | tok=172.12 |
| shipreb | i2t_B16_r1 | - | .76s | 6.14 | 61.12 | tok=1085.94 |
| shipreb | i2t_B32_r1 | - | 1.44s | 8.68 | 79.79 | tok=1536.13 |
| shipreb | i2t_B32_len212 | - | 3.21s | 6.67 | 78.92 | tok=1189.38 |
| shipreb | s2t_B32_r1 | - | .284s | 40.27 | 55.60 | tok=843.43 |
| shipreb | i2t_B32_r2 | - | 3.43s | 6.93 | 67.24 | tok=1202.10 |

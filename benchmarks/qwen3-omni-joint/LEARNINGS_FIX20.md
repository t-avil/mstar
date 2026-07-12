# LEARNINGS: the FIX-20 implementation campaign (2026-07-12 evening loop)

Goal: implement/bench/integrate all 20 fixes from LEARNINGS_TTFT.md. Branch
`opt/fix20` (worktree mstar-fix20, base 1e7f10da). All flags default-off,
byte-identical off, dynflag-toggleable unless noted. Every A/B tonight ran under
neighbor load 65-168 — see "contention lottery" caveat before trusting any pair.

## Scoreboard (fix -> status)

| # | fix | code | verdict tonight |
|---|---|---|---|
| 8/9/18 | emit rid-index + inline fastpath + greedy argmax | bc00dee6 | **WINNER**: 2/2 pairs, +4-5% tok/s, TTFT -4..-7% at load 123-168; B1 token-identity PASS. Promoted into bench baseline. |
| 13 | off-process detok child | 1c4c25c3 | **POSITIVE-LEAN**: boot-pair cells 0.70s/922tok + 4.93s/1046tok vs same-load prior-boot 7.41s/732tok; s2t guard PASS (0.42s/34.6rps/720tok). Needs a load-matched boot-pair repeat. |
| 4 | admit fastpath (arrival-triggered yield) | bc00dee6 | weak TTFT win (2/2 TTFT −1%/−6%, rps/tok wash) at load 117-158; retest quiet. |
| 2 | encoder step budget | bc00dee6 | inconclusive 1/2 (one +5.9%rps/+4.5%tok pair, one wash). KEY FINDING below. |
| 1 | co-admission (COADMIT) | ee37885a | wash at load 100-142 (7.42 vs 7.41). KEY FINDING below. Retest quiet. |
| 6 | producer emit seqnums | 08f57777 | unresolved — pair polluted by lottery (same-config sibling cells 0.367s vs 7.41s). |
| 14 | CPU-set pinning (LAB_CPUSET) | lab_server.sh | base cells wash vs unpinned boot at load 100-156. |
| 7 | plan reuse | NO CODE | **NO-GO**: plans are per-request distinct (seq_len/pages/pos differ); the "one plan per wave" already exists = BATCH_VISION_PREFILL's single vectorized plan call. Reuse across walks would corrupt FlashInfer/RoPE state. |
| 3 | async scheduling | NO CODE | **NO-GO**: already shipped — spec batch for N+1 built pre-sync, FlashInfer pre-planned on plan_stream (MSTAR_PRE_PLAN_SPEC=1 default), WGD pre-sync. Only remaining deferral = the V1 non-identical failure. |
| 5/15/20 | PD topology / bench protocol / keep mixed on | prior work | already validated, unchanged. |
| 16/17/19 | length-matched cells / greedy default / DP long-output cert | none | not reached tonight (bench-only; queue for a quiet session). |
| 10/11/12 | shared sample+unpack / batched WGD msgspec / sidecar i2t | none | not reached tonight (wave-2 remainder). |

## The three load-bearing findings (they change the roadmap)

1. **The captured-chunk 512 ceiling (#1):** the V2 mixed budget NEVER binds —
   n_decode+C <= 544 always (chunk grid {256,288,512}, B<=32), so ship budget
   4096 had 7.5x slack and "raise to 32k" does nothing. vLLM folds a full 32k
   prefill only because its prefill is EAGER. Under captured CUDA graphs, full
   one-step co-admission is impossible until the captured mixed bucket grows at
   boot. The real lever list is: bigger captured mixed buckets (boot-time),
   vision-mix at boot, or an eager fold fallback.
2. **The encoder wave was never grid-capped (#2):** eager encoder waves already
   run unbounded in one varlen forward; MSTAR_VIS_BATCH_SIZES only shapes the
   THINKER prefill captures. So "batch all pending encodes" was already true,
   and #2 is an OOM-safety ceiling, not an unlock. The encoders theory of the
   TTFT gap is dead; the host-side burst + prefill-freezes-decode remain.
3. **Async input-prep is already shipped (#3):** N+1 batch built pre-sync,
   attention pre-planned on a side stream (double-buffered slots), feeds on-GPU
   under DIRECT_FEED. What's left after the sync is sample-dependent by
   construction. The V1 deferred-postprocess design remains the only untapped
   deferral and it is known-broken (non-identical outputs).

## A/B tables (all cells i2t B32 food101 closed-loop unless noted)

w1a (unpinned boot, emit+argmax OFF/ON, n=48): off 6.80/4.27/740 @123; on
6.33/4.49/774 @145; off 6.47/4.33/739 @158; on 6.21/4.45/777 @168.
w1b: identity B1x8 sequential PASS; f4 off/on 6.17->6.10, 6.79->6.40 (TTFT only);
f2 off/on 6.22->6.16 +5.9%rps then 6.22->6.22 wash.
w3 (CPUSET boot): cs_base 6.23/4.24/728 @142, 7.82/3.89/675 @156; seq_on 3.98/6.47/1100 @123;
seq_off 0.367/5.52/974 @110; coad on/off 7.42/4.39/733 vs 7.41/4.30/732 @100-105; coad_on2 6.41/4.95/839 @142.
w4 (DETOK+CPUSET boot): dt_base 0.70/5.48/922 @98; 4.93/6.02/1046 @91; s2t B32
n=128 guard 0.42s/34.61rps/720tok @83.

## Protocol lessons (hard-won tonight)

- **Contention lottery pollutes high-load pairs**: same-config same-boot cells
  spanned 0.367s..7.41s TTFT at load 100-110. A 2-cell pair at load>60 can lie
  in either direction; only consistent multi-pair trends (like w1a's 2/2 with
  load rising against the winner) are meaningful. Quiet-window retests are
  mandatory before shipping any of tonight's non-winners.
- **`${VAR:-default}` expands to the VALUE when set** — my LAB_CPUSET guard fed
  numactl the cpuset twice and it tried to exec "64-87,192-215". Compose
  optional CLI args in explicit if-statements, not clever expansions.
- **Self-match footgun (x2 tonight, x3 this week)**: any kill/pgrep compound that
  contains the target name in its own cmdline kills itself (exit 144). Use
  pgrep -f "name[x]" bracket patterns and separate the kill from any command
  mentioning the name.
- **RAM-gate boots on a shared box**: node1 had 19G free at one point (neighbor
  700G-RSS jobs); strict membind boots died mid-compile with no kernel OOM
  record. Gate on `numactl -H` free + `free -g` avail; fall back to
  --preferred=NODE instead of --membind when the local node is tight.
- **lab_server wrapper watchdog vs slow compiles**: fresh inductor cache under
  load 100+ takes >20 min; the wrapper exits NEVER_READY but the serve tree
  lives on. Check serve procs before declaring death; chain benches with their
  own ready-waits.
- **Idle-guard pattern** (now standing infra, idle_guard.sh): poll 60s; our
  serve holding GPUs with util==0 and no benchmark client for 10 min -> warn,
  15 min -> kill our own server; orphan reaper every 30 min (ppid==1, tim,
  >2h, known leak patterns). ~115 orphans from Jun29-Jul11 sessions reaped.
- Boot-time vs dynflag flags: DETOK_PROC (process topology) and vision-mix /
  capture grids are boot-time; everything else tonight toggles via dynflags
  (worker refreshes ~50 iters; data_worker flags engage — proven by w1a deltas).

## Recommended next session (quiet box)

1. Load-gated (<25) paired retests: #13 boot-pair, #4, #1, #6, #2 (n>=96).
2. If #13 confirms: promote MSTAR_DETOK_PROC to ship set; it composes with
   SEQNUMS by construction and its failure mode is inline fallback.
3. Grow captured mixed bucket (P3 from #1's report) + vision-mix boot flag —
   the only path to real one-step co-admission.
4. #16/#17/#19 bench-only certifications + remaining wave-2 (#10-12).

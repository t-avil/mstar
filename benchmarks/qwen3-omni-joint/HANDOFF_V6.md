# HANDOFF_V6 — M* vs vLLM-Omni campaign (state as of 2026-07-04 ~23:30 UTC)

Supersedes HANDOFF_V5.md. Single entry point for the next agent. Read this, then
EXPERIMENTS.md (pool root, ~70 entries — THE knowledge base) for any referenced
entry, and GOAL_MATRIX.md (the endgame driver, one row per cell). Trust nothing
that isn't a live adjacent-pair A/B graded by ab_verdict.py on this box; when a
mechanism-certain win doesn't convert, check the mechanism is alive (WALK_STATS
counters at WARNING) before doubting the idea.

Big change since V5: the campaign moved from a single build chasing i2t B32 to a
**two-config, full-matrix live-graded scoreboard**. The war shrank to three i2t
cells. All numbers below are live, ab_verdict-gated (soft-cell + tok/req-parity
gates), n=3 unless noted.

## 1. THE SCOREBOARD (GOAL_MATRIX.md rev-4; live, grade-A unless noted)

**21/24 cells GREEN** (≥1.05× req/s live). Raw committed under h2h_out* /
h2h_smallbatch_final, verdicts via `ab_verdict.py`.

- **Speech: 12/12 GREEN** (s2s + i2s, 2.1–2.9×, committed sweep grade D, huge margin).
- **s2t: 6/6 GREEN** — B1 1.14 (D), **B2 3.062 / B4 2.180** (A, [TQ✓]),
  B8 1.401 (B), B16 1.354 (B, wants r3), **B32 1.349** (A, proof-grade n=3).
  The s2t small-batch ratios are large because vLLM ANSWER-MODES on interrogative
  audio (see §Reliability + the [TQ] sweep); M* transcripts are at parity.
- **i2t: 3/6 GREEN, 3 red** — B8 1.221 (B), **B16 1.096** (A), **B4 1.1051** (A,
  n=7, 95%LB 1.0663 — WON, pooled h2h_smallbatch_final; the ab_verdict SUSPECT flag
  is the stale-B32-band false positive, adjudicated: B4 caption parity 0/12);
  RED = **B1 0.989 [0.971–1.009] n=5 (+6%), B2 0.942 [0.910–0.976] n=5 (+11%),
  B32 0.918 [0.889–0.948] n=3 (+14%)**. All honest reads; the small-batch tok/req
  (~189–200) is a batch-dependent length profile common to BOTH systems (vLLM
  208–220 at B1), not a defect — the 172–179 identity band is B32-calibrated.

**vLLM reliability ledger — FIVE distinct failure events on 2026-07-04, one box,
one window (full detail + evidence paths in VLLM_RELIABILITY.md, committed on the
docs branch):**
- **A** 17:40 — stage-0+stage-1 subprocess death, pre-race, cause undetermined.
- **B** 18:24–18:25 — stage-1 then stage-2 died **mid-race under load**; cost
  h2h_v2 pairs #6–7 (M* healthy the same rounds). Committed race_ext.log.
- **C** 20:16 — stage-0 death → full self-teardown while **IDLE, no load**.
- **D** ~20:46–21:12 — **ZOMBIE**: EngineCore silently dead while the front-end API
  still returned 200; discovered only when a race hit it (HTTP 500 on 12/12
  requests × 5 cells). Defeats a naive `/v1/models` liveness check.
- **E** ~22:18 — **mid-race collapse**, 16 cells 500'd; vLLM's OWN orchestrator
  SIGTERM'd the wedged engine (timeline-confirmed internal, ≥1 min before any
  operator kill) and the teardown raised a real `RuntimeError: release unlocked
  lock` — a lock-safety defect on the shutdown path.
MTBF ~30–60 min under our race cadence; 3 of 5 mid-race, 2 off-load; the zombie
and lock-bug are failure MODES beyond a plain crash. **M*: six boots tonight under
HEAVIER churn, zero crash signatures, served every cell** — in the event-E race M*
served 18/18 cells error-free while vLLM lost 16. CAVEATS (do not over-claim): not
root-caused; event A undetermined; the E lock-bug is a real secondary teardown
defect but not proven the spontaneous first cause; **0.23 may fix all of it** —
re-test before the reliability claim survives. Plus the **answer-mode correctness
failures** (13/13 s2t cells, one interrogative clip — [TQ] sweep). "Wins on
reliability + correctness" is earned on this window's logs; keep them.

Campaign trajectory at i2t B32: 0.53× (v0.22) → 0.77 → 0.83 → 0.88 (sidecar) →
0.883 shipping live → **0.918 merge live** (+4% from the merge config).

## 2. THE WINNING CONFIGS (plural — this is what flipped vs V5)

V5 shipped ONE build. V6 ships a **primary + one documented per-workload config**,
per the GOAL build-rule. Encoder placement + prefill merge are the axes.

Both are the V5 custom-ops winner with the prefill-merge swap; flags below are the
verbatim boot commands.

**PRIMARY candidate = `imerge` (encoff + vision merge)** — best all-round i2t, audio-neutral.
- worktree `mstar-crusade` (opt/custom-ops); config `configs/qwen3omni_2gpu_encoff.yaml`;
  lab imerge, GPUs 6,7:8321.
- Verbatim:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache
TORCHDYNAMO_CACHE_SIZE_LIMIT=128 MSTAR_CUSTOM_OPS=1 MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1
MSTAR_FAST_POSTPROC=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1
MSTAR_PREFILL_CHUNK_TOKENS=512 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1
MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1
MSTAR_EMIT_SIDECAR=1 MSTAR_MIXED_BUDGET_TOKENS=512 MSTAR_FAST_CHECKSTOP_TALKER=1
MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_MERGED_PREFILL=1 MSTAR_WALK_STATS=1
```
  = the V5 winning set **minus** `MSTAR_CHUNKED_PREFILL_V2_VISION` and
  `MSTAR_MIXED_BATCH_VISION`, **plus** `MSTAR_MERGED_PREFILL=1`. (`MSTAR_WALK_STATS=1`
  is diagnostic — optional for shipping.) Merge collapses
  prefill_text+prefill_vision into one prefill step/admission (within-request
  coalescing — the only surviving coalescing form, §8): +4% at B32 over shipping,
  and it touches ONLY vision flags ⇒ byte-identical to shipping on audio paths ⇒
  zero audio risk. Race: h2h_out_imergecol2/.

**s2t-small config = `arm3` (default co-located + vision merge + audio merge):**
- worktree `mstar-audiomerge` (opt/prefill-merge-audio @ a2d2f47); config
  `configs/qwen3omni_2gpu.yaml` (default, co-located); lab arm3, was GPUs 4,5:8305.
- **Identical flag set to imerge, PLUS `MSTAR_MERGED_PREFILL_AUDIO=1`** (the audio
  twin: merges prefill_text+prefill_audio; counter `merged_prefill_audio_walks`).
- Why: +11–13% at s2t B2/B4. WALK_STATS confirm merged_prefill_walks +
  prefill_multimodal live (lab_arm3/server.log). arm3 i2t is not preferred
  (co-location adds nothing for i2t; the batch-length profile is identical).

Boot ~12.5 min (Inductor cache; first-ever boot pays ~40 min autotune once).

## 3. BRANCH MAP (fork t-avil/mstar; all pushed)

Carried from V5: **opt/custom-ops** (THE base build) + opt/integration-v4
(6 composed wins) + opt/v2-policy, opt/speech-floor, opt/prefill-merge,
opt/sched-pack (REJECTED), opt/async-sched (V1 PARKED). New/changed tonight:
- **opt/sidecar-checkstop** @ 5414a5a — Stage-2 check_stop offload: BUILT +
  CPU-tested, **PARKED BEFORE A/B** (profile gate shows main-thread < GPU
  post-custom-ops ⇒ valve-dead; also shares V1's overrun-drift risk). Substrate
  kept; re-open only if a future main-thread-WORK cut flips the profile.
- **opt/prefill-merge-audio** @ a2d2f47 (worktree mstar-audiomerge) — the audio
  twin: **SHIPPED into the arm3 config** (+11–13% s2t B2/B4, mechanism-alive counters).
- **opt/admit-jitter** — MSTAR_ADMIT_JITTER_MS: BUILT, **PARKED-INERT** (guard
  self-suppresses at the closed-loop trough; admit_jitter_held 3/480).
- **opt/prefill-gather** — MSTAR_PREFILL_GATHER_MS + BATCH_VISION_PREFILL:
  **CLOSED-DEAD** (readiness serializes; prefill_packed_bs1=2038 vs bs2=6).
- **opt/w2-retest** — W2 postprocess memoization: **IN VERIFY** (boot running; the
  last small-batch lever, §4).

## 4. THE REMAINING-GAP PLAN (in expected-value order)

**i2t B4 is WON** (1.1051, n=7). The war is down to **three i2t cells**: B1 (+6%,
the nearest miss), B2 (+11%), B32 (+14%, flagship). This is a defensible stopping
point — the next agent's job is to PROVE the 21/24, not to force B1/B2/B32.

1. **Proof sweep — the primary next action.** `proof_sweep.sh --vllm-relaunch`,
   then `ab_verdict.py` per PROOF_SWEEP_PROTOCOL: ×5 clean adjacent pairs on the
   flagship, ×3 elsewhere, mandatory soft-cell rejection gate (variance is bimodal;
   the gate is load-bearing, not more rounds). Solo-box window. Commit raw +
   NUMBERS_V4 + the HEADTOHEAD method. Also grabs s2t B16 round 3 (n=2→3).
2. **W2 retest — un-parked, viable but LOW EV (optional, solo-boot).** Code is
   EXONERATED (the OOM was a /dev/shm double-boot, not W2 — see §7). Postprocess
   memoization, honest +2–5% at B1–B4. **B1 needs only +6%**, so W2 (or any small
   new host-side cut) could just reach it; won't cover B2 or B32. Run it on a
   SOLO-IDLE boot (verify-gate = shadow mode ZERO mismatches, THEN A/B at B1/B2).
3. **i2t B32 = Tier-S3 ceiling — state it, don't chase it.** 0.918, grade A. Every
   host-side lever falsified/parked (§8); merge added the last +4%. **~0.92× is
   plausibly at/near the structural ceiling** vs their fresh-boot 8.4–8.5; a clean
   1.05× needs an EngineCore-class scheduler rewrite (week+), not a flag.
   GOAL_MATRIX.md has the paragraph to quote.
4. **Record hygiene:** s2t B16 round 3 (n=2→3); i2t caption parity is DONE (PASS —
   B4 0/12, B32 19/96 divergent-by-verbosity-only, quality-equivalent, zero
   truncations).

## 5. HARNESS TOOLBOX (all in /m-coriander/coriander/tim/)

Carried: **lab_server.sh** (setsid the wrapper!), **lab_ab.sh** (refuses
mismatched key sets), **lab_kill.sh**, quick_bench.sh/dyn_ab.sh (NUMA-fixed).
New/updated tonight:
- **ab_verdict.py `<dir> [--threshold X]`** — THE verdict gate. Auto-detects h2h
  ({mstar,vllm}_...) vs lab_ab ({A,B}/{base,v2}_...). Per-cell pooled geomean + SE
  + 95% LB + WIN/LOSS/WASH. **Soft-cell gate = per-arm robust-z low-outlier on
  req/s (two-threshold: >15% below arm median, OR >8% + robust-z>3), NOT intra-cell
  JCT skew.** tok/req parity gate (intra-M*: >5% arm gap ⇒ length-confounded ⇒
  verdict on tok/s). Identity band 172–179 (i2t, B32-calibrated). Validated on
  h2h_out/, h2h_out_p2verify/, h2h_out_smallbatch2/, h2h_out_imergecol2/.
- **h2h_cells.sh** — live race driver, now takes an `MSTAR_URL` arg (race any
  booted M* config vs vLLM).
- **ttft_probe.py** — streaming-mode TTFT capture (the closed-loop harness records
  TTFT=None; needed to decompose prompt-path vs decode at small batch).
- **proof_sweep.sh --vllm-relaunch** — the Stage-2 proof-sweep driver.
- **profile_gate.sh** — py-spy + nvidia-smi dmon; the main-thread-%-vs-GPU-busy
  gate that killed the checkstop/V1 levers (main 32% vs GPU 64.8% at B32).
- vLLM boot: /home/tim/baselines/launch_vllm_2_3_numa0.sh (port 8093);
  runner arg `--inference-system vllm_omni`.

## 6. LAWS (violate = wasted time; each earned an entry tonight)

Carried V5 laws 1–8 (in-graph microbench only; remove-work-not-waits GIL-valve;
box lies at cell granularity / adjacent pairs or ≥3 rounds; verify mechanism ALIVE
via WALK_STATS; req/s for cross-system; A-B-A bracketing; effect-size gate <2% →
microbench; re-decompose after structural changes). NEW tonight:
9. **Anti-zombie engine probe before racing** — confirm the engine is actually
   serving (a warm non-zero cell) before trusting any A/B; dead/degraded servers
   return 0.000 or crawl and poison the arm.
10. **Soft-cell rejection = per-arm robust-z on throughput, NOT intra-cell JCT
    skew.** The known-soft 0.723 h2h cell had the TIGHTEST intra-cell JCT of its
    arm; the signal is that it's a low outlier vs its own arm's other rounds.
11. **tok/req parity gate for intra-M* A/Bs.** n=12 small-batch length variance
    (±11%) can manufacture a double-digit fake req/s delta; if two same-system arms
    differ >5% in tok/req, judge on tok/s. (Law 5 fires INSIDE an A/B, not just
    cross-system.)
12. **Max 2 M* servers + vLLM; never boot during a race (host-RAM / /dev/shm rule).**
    Only ~300GB host RAM is free for us — Ray plasma owns ~290GB of /dev/shm. Each
    boot spikes RAM; multi-server windows exceed the headroom → OOM kills healthy
    engines by clean SIGTERM (the "graceful death" we misread as wrapper-reaping).
    `free -g` / `df -h /dev/shm` before every boot; boot spikes also contaminate
    running cells ~−25% (§7). This root-caused the W2 "OOM" — W2 is EXONERATED.
13. **Readiness-serialization kills cross-request coalescing.** Closed-loop
    admissions cluster in TIME but prefill READINESS serializes through the
    KV-read/encode pipeline, so a gather/jitter window never sees ≥2 ready. Only
    WITHIN-request coalescing (merged walk) works at B32 food101.

## 7. OPS HAZARDS (all cost us time; confirmed tonight)

Carried V5 hazards (wrapper reaping → setsid the wrapper; one owner per GPU pair;
/tmp on rootfs → TMPDIR to pool; create log dir before redirect; never remove
@torch.compiler.disable; GPU-7 fell off the bus once; idle-gate before every boot).
Reconfirmed / new tonight:
- **pkill/pgrep self-match (exit 144) bit us AGAIN** — patterns containing the
  command string match your own shell. Use `pgrep -f benchmark.runner`; kill by
  pgid file, never by broad pattern.
- **Boot contamination −25% same-node** — booting a second engine on a node while
  cells run on it depresses the running arm ~25%; never boot on an active bench
  node (and see Law 12's RAM cap).
- **/dev/shm host-RAM pressure = the graceful-death root cause** — ~300GB free for
  us, Ray plasma owns ~290GB of /dev/shm; boot spikes in multi-server windows
  OOM-killed crusade, imerge, AND vLLM by clean SIGTERM (same signature we chased
  as wrapper-reaping). It also produced the **W2 false-fail** (env double-boot on
  4,5, not a W2 defect — build EXONERATED, CUDA-inert to set_device). Rule: **max 2
  M* + vLLM, never boot during a race**; check `free -g` / `df -h /dev/shm` first.

## 8. WHAT'S PROVEN DEAD (don't re-litigate without new conditions)

Carried V5 dead list (SCHED_PACK peek-backoff; chunk 128/768; single-chunk folding
at any batch; GC-tune/jemalloc at B32; DeepGEMM/trtllm/FA3; NUM_SLOTS=3; GIL
interval; V1-as-built identity fail; disable-removal compile edits). NEW tonight —
the whole coalescing/host-side frontier for i2t B32 food101:
- **Fold/smoothing family CLOSED at short spans** — 100% of i2t B32 food101
  prefills are short unchunked spans; _fold_ok/_mix_opp/budget_folds all 0, so V2
  budget, adaptive floor, and single-chunk have nothing to fold.
- **Admission jitter — mechanism-dead** (self-suppresses at the closed-loop trough).
- **Prefill gather — mechanism-dead** (readiness serializes; bs=1 essentially always).
- **Checkstop offload at the current regime — valve-dead** (profile gate: main-thread
  < GPU post-custom-ops; wait-removal converts only while main > GPU).
- **V1-family (async-sched, deferred postprocess) — self-cancelling** (the win needs
  the stop-deferral that causes the +9% length drift; keeping stop sync removes the
  mechanism).
Net: cross-request coalescing and every zero/low-code host lever are exhausted for
this cell; the only thing that moved it was WITHIN-request merge (+4%). B32 at
~0.92 is the honest ceiling (§4.2).

## 9. REPORTING FORMAT (user requirement, unchanged)

1. "We are losing at these paths" — compact ratio table, losing cells flagged
   (GOAL_MATRIX.md is that table now). 2. "What we tried" — ONE sentence. 3.
   "What's next" — ONE sentence. Sloppy-fast mode: 2-round/small-cell POCs,
   direction over precision, stack winners on the integration branch, full sweeps
   only for 30–50%-class accumulations. Document EVERY verdict (incl. failures) in
   EXPERIMENTS.md; commit+push docs branch + merge to benchmarks after every batch;
   keep GOAL_MATRIX.md + project memory current. **Never quote an un-graded or
   correctness-unresolved cell as acceptance-grade; projections never outrank
   measurements.**

## §4-ADDENDUM (2026-07-05 03:20) — the next agent's P0 queue, pre-registered
The B32 residual (0.918) is the ~35% GPU idle between decode steps (main 32% /
GPU 65% / gpu-thread 58%; admission path exonerated by the closed gather
family). The GPU-thread column re-opens under Law 8 — the record pre-registered
"defer-sample overlap should flip positive once the GIL shade is gone" and that
condition is now measured. Queue, in EV order:
1. profile_gate.sh on a warm merge-config server (imerge flags) — split the 35%
   idle into: sample-D2H block / submit-hop gap / plan-inline / kernel bubble.
2. DEFER-SAMPLE (the narrow V1): move sampled-token D2H/remap off the gpu-thread
   critical path (copy-stream + event, lazy placeholder repair); keep check_stop
   + penalty/RNG synchronous — dodges V1's identity trap (which came from
   deferring the STOP, not the transport). Historically the sampler was the
   single biggest gpu-thread cost (6 syncs killed → +10.5%).
3. W3 future-token run-ahead — revisit iff the idle is the submit hop.
4. Decode pre-plan (banked MIXED_PREPLAN substrate) — iff plan is inline.
OPS: pair 4,5 killed 4 straight boots (quarantine until verified); every M*
server death correlates with a concurrent boot RAM spike (max 2 M* + vLLM,
never boot during measurement); box was lost to foreign jobs on GPUs 2-6 at
~03:00 — verify solo before racing.

# HANDOFF_V7 — M* vs vLLM-Omni campaign (state as of 2026-07-05 ~02:00 UTC)

Supersedes HANDOFF_V6.md. Single entry point for the next agent. Read this, then
EXPERIMENTS.md (pool root, ~75 entries — THE knowledge base) for any referenced
entry, and GOAL_MATRIX.md rev-5 (the endgame driver, one row per cell). Trust
nothing that isn't a live adjacent-pair A/B graded by ab_verdict.py on this box;
when a mechanism-certain win doesn't convert, check the mechanism is alive
(WALK_STATS counters at WARNING) before doubting the idea.

Big changes since V6: (1) the **stack build** landed — merge + cfgv2 + checkstop —
moving i2t B32 into vLLM's live band (peaks ≥1.05); (2) the wall **re-flipped to
the host side** after cfgv2, which re-opened and converted the parked checkstop
lever; (3) final standing = **21/24 GREEN ≥1.05, the other 3 i2t cells at
parity-class 0.94–1.00 (not losses)**. The Tier-S3 "0.92 structural ceiling" claim
from V6 is RETRACTED — the ceiling was not reached. All numbers live,
ab_verdict-gated, n as noted.

## 1. THE SCOREBOARD (GOAL_MATRIX.md rev-5; live, grade as noted)

**21/24 cells GREEN ≥1.05×; the other 3 (all i2t) are PARITY-CLASS 0.94–1.00, not
losses — every cell of 24 is ≥0.94, with wins up to 3.06×.** Raw committed under
h2h_out* / h2h_smallbatch_final / flagship2 (stack), verdicts via `ab_verdict.py`.

- **Speech: 12/12 GREEN** (s2s + i2s, 2.1–2.9×, committed sweep grade D, huge margin).
- **s2t: 6/6 GREEN** — B1 1.14 (D), **B2 3.062 / B4 2.180** (A, [TQ✓]),
  B8 1.401 (B), B16 1.354 (B, wants r3), **B32 1.349** (A, proof-grade n=3).
  The s2t small-batch ratios are large because vLLM ANSWER-MODES on interrogative
  audio (see §Reliability + the [TQ] sweep); M* transcripts are at parity.
- **i2t: 3/6 GREEN, 3 parity** — B8 1.221 (B), **B16 1.096** (A), **B4 1.1051** (A,
  n=7, 95%LB 1.0663 — WON; the ab_verdict SUSPECT flag is the stale-B32-band false
  positive, adjudicated: B4 caption parity 0/12). PARITY (stack build opt/stack-n2)
  = **B1 0.972 [0.946–0.997] n=3 (+8%), B2 0.999 [0.940–1.060] n=3 WASH (+5%),
  B32 0.938 [0.885–0.994] n=6 (+12%), peaks 1.048/1.031**. At B32 the stack put M*
  absolutes 7.1–9.0 INSIDE vLLM's live band (8.2–8.6) for the first time — the
  remaining gap is cell VARIANCE, not compute. The small-batch tok/req (~189–200)
  is a batch-dependent length profile common to BOTH systems (172–179 band is
  B32-calibrated).

**vLLM reliability ledger — SEVEN distinct failure events across the campaign, one
box (full detail + evidence paths in VLLM_RELIABILITY.md, committed on the docs
branch):**
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
- **F/G** (07-05) — two further mid-race deaths, incl. the flagship2 stack race
  (r3–r5 zeroed). Event G is the SEVENTH. (See VLLM_RELIABILITY.md for the F/G rows.)
MTBF ~30–60 min under our race cadence; the majority mid-race; the zombie and
lock-bug are failure MODES beyond a plain crash. **M*: multiple boots under
HEAVIER churn, zero crash signatures, served every cell** — in the event-E race M*
served 18/18 cells error-free while vLLM lost 16. CAVEATS (do not over-claim): not
root-caused; event A undetermined; the E lock-bug is a real secondary teardown
defect but not proven the spontaneous first cause; **0.23 may fix all of it** —
re-test before the reliability claim survives. Plus the **answer-mode correctness
failures** (13/13 s2t cells, one interrogative clip — [TQ] sweep). "Wins on
reliability + correctness" is earned on this window's logs; keep them.

Campaign trajectory at i2t B32: 0.53× (v0.22) → 0.77 → 0.83 → 0.88 (sidecar) →
0.883 shipping → 0.918 merge → **0.938 pooled / peaks 1.048 (stack: +cfgv2 +5.2%,
+checkstop)** — M* now inside vLLM's live band; variance-bound, not ceilinged.

## 2. THE WINNING CONFIGS (primary STACK build + one s2t config)

Primary is the **stack build `opt/stack-n2`** (merge + cfgv2 + checkstop) — the V6
imerge config plus the two host-side wins that landed 07-05. Plus one documented
per-workload config for s2t small-batch. Flags below are the verbatim boot lines.

**PRIMARY = stack `opt/stack-n2` (encoff + vision merge + cfgv2 + checkstop):**
- worktree `mstar-crusade` lineage, branch **opt/stack-n2** (= opt/custom-ops +
  opt/cfgcache-v2 @ded928d + opt/sidecar-checkstop); config
  `configs/qwen3omni_2gpu_encoff.yaml`; lab imerge/flagship2, GPUs 6,7:8321.
- Verbatim (= the V6 imerge line **plus** the two stack flags):
```
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache
TORCHDYNAMO_CACHE_SIZE_LIMIT=128 MSTAR_CUSTOM_OPS=1 MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1
MSTAR_FAST_POSTPROC=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1
MSTAR_PREFILL_CHUNK_TOKENS=512 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1
MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1
MSTAR_EMIT_SIDECAR=1 MSTAR_MIXED_BUDGET_TOKENS=512 MSTAR_FAST_CHECKSTOP_TALKER=1
MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_MERGED_PREFILL=1 MSTAR_SAMPLER_CFG_CACHE_V2=1
MSTAR_SIDECAR_CHECKSTOP=1 MSTAR_WALK_STATS=1
```
  = V5 winner **minus** `MSTAR_CHUNKED_PREFILL_V2_VISION`/`MSTAR_MIXED_BATCH_VISION`,
  **plus** `MSTAR_MERGED_PREFILL=1` (within-request prefill coalescing, +4% B32,
  audio-neutral), **plus** `MSTAR_SAMPLER_CFG_CACHE_V2=1` (slot-tensor key — the old
  tuple-of-rids cache THRASHED under B32 churn and re-paid its own ~9ms six-sync;
  +5.2%) **plus** `MSTAR_SIDECAR_CHECKSTOP=1` (check_stop wait-removal — viable
  again because cfgv2 re-flipped the wall to the host side; shadow-gated zero
  mismatches). (`MSTAR_WALK_STATS=1` diagnostic, optional for shipping.) Race:
  flagship2/. NOTE: `MSTAR_SAMPLER_CFG_CACHE_V2` pays a one-time slot-init cost on
  first-pass cells — WARM before measuring or the cell reads low.

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

Carried: **opt/custom-ops** (base build) + opt/integration-v4 + opt/v2-policy,
opt/speech-floor, opt/prefill-merge, opt/sched-pack (REJECTED), opt/async-sched
(V1 PARKED). Current/new:
- **opt/stack-n2** — THE primary build = opt/custom-ops + opt/cfgcache-v2 +
  opt/sidecar-checkstop, on the encoff+merge config. The 21/24 + parity-B32 scoreboard.
- **opt/cfgcache-v2** @ ded928d — sampler config cache slot-tensor key: **WIN +5.2%
  i2t B32** (the old tuple-of-rids cache thrashed under churn). SHIPPED in the stack.
- **opt/sidecar-checkstop** @ 5414a5a — Stage-2 check_stop offload: **UN-PARKED,
  SHIPPED in the stack.** The V6 "valve-dead" parking is REVERSED — cfgv2 re-flipped
  the wall to the host side (main-thread 55% = wall again), so the wait-removal
  converts; shadow-gated zero mismatches. (Still carries V1's overrun-drift risk —
  the shadow-mode + tok/req gate is what made it safe.)
- **opt/prefill-merge-audio** @ a2d2f47 (worktree mstar-audiomerge) — the audio
  twin: **SHIPPED into the arm3 config** (+11–13% s2t B2/B4, mechanism-alive counters).
- **opt/admit-jitter** — **PARKED-INERT** (guard self-suppresses at the trough).
- **opt/prefill-gather** — **CLOSED-DEAD** (readiness serializes; bs1=2038 vs bs2=6).
- W2 postprocess memoization — **CLOSED**: 4/4 boot failures across two environments;
  cost exceeds its 2–5% EV (the earlier "OOM" was the /dev/shm double-boot, §7).

## 4. THE REMAINING-GAP PLAN (in expected-value order)

**21/24 GREEN; the 3 remaining i2t cells are parity-class (B1 0.972, B2 0.999 WASH,
B32 0.938 with peaks 1.048).** The stack put M* inside vLLM's B32 live band. The
next agent's #1 item is NOT a new lever — it is VARIANCE.

1. **B32 variance / soft-cell diagnosis — THE flagship item now.** The stack's B32
   cells spread 7.1–9.0 absolute (pooled 0.938, band 0.885–0.994) while peaks
   already read 1.048/1.031 ≥ bar. So the win is gated by cell-to-cell variance,
   not compute. Profile WHY the spread (admission-wave residual? allocator? foreign
   NUMA neighbor?); a variance/soft-cell fix converts the peaks into a pooled ≥1.05.
   This is the one thing that could flip B32 GREEN.
2. **Proof sweep** — `proof_sweep.sh --vllm-relaunch`, then `ab_verdict.py` per
   PROOF_SWEEP_PROTOCOL: ×5 clean adjacent pairs on the flagship, ×3 elsewhere,
   mandatory soft-cell rejection gate (variance is bimodal; the gate is load-bearing,
   not more rounds). Solo-box window. Commit raw + NUMBERS_V5 + HEADTOHEAD method.
   Grabs s2t B16 round 3 (n=2→3). This certifies the 21/24 for the record.
3. **i2t B1 (0.972, +8%)** rides the B32 variance work + any small host cut. B2
   (0.999) is already a wash at parity; no dedicated lever. (W2 is CLOSED, §3.)
4. **Record hygiene:** s2t B16 round 3; caption parity DONE (PASS — B4 0/12,
   B32 19/96 verbosity-only, zero truncations).

The V6 Tier-S3 "0.92 structural ceiling" is **RETRACTED** — two more host-side wins
(cfgv2, checkstop) landed after it and the gap is now variance, not a ceiling. Do
not repeat the ceiling claim; say "host-side, shrinking, inside their band,
variance-bound."

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
  gate. It read main 32% / GPU 64.8% post-custom-ops (parked checkstop), then
  main 55% / GPU 85.7% post-cfgv2 (re-opened it) — RE-RUN IT after every landed win
  (Law 14).
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
14. **The wall SEESAWS — re-profile after every landed win.** Each host-side fix
    un-shades whatever it was blocking and can flip which thread is the wall, which
    re-opens or re-closes wait-removal levers. Post-custom-ops main < GPU (checkstop
    dead); post-cfgv2 main > GPU again (checkstop LIVE, +converted). A lever's
    "dead" verdict is regime-scoped, not permanent — run profile_gate.sh after each
    win and re-decompose (this is Law 8 with teeth). Corollary: **a cache that can't
    HIT under the workload's churn is worse than no cache** — the cfgv2 story: the
    tuple-of-rids key missed every B32 step and re-paid the ~9ms sync it existed to
    remove; the whole "sampler-cache wash" saga was measuring a thrashing cache.

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
  as wrapper-reaping). Rule: **max 2 M* + vLLM, never boot during a race**; check
  `free -g` / `df -h /dev/shm` first.
- **A `*.json` .gitignore rule silently EMPTIED four raw-data commits** (h2h_v2/raw,
  p2verify, smallbatch2, imergecol) — the "committed" scoreboard raw was absent from
  git despite clean commit messages, so "recomputable from committed raw" was
  silently false for those cells. Repaired with `git add -f` (98 files). RULE:
  after committing benchmark raw, ALWAYS `git ls-files <dir>` to verify it entered
  the tree; add raw with `-f` or carve a `!benchmarks/**/*.json` exception.

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
- **W2 postprocess memoization — CLOSED** (4/4 boot failures across two envs; EV < cost).
- **V1-family (async-sched, deferred postprocess) — self-cancelling** (the win needs
  the stop-deferral that causes the +9% length drift; keeping stop sync removes the
  mechanism).
NOT dead (reversed from V6): **checkstop offload CONVERTED** once cfgv2 re-flipped
the wall to the host side — it is SHIPPED in the stack, not parked. This is Law 14
in action: "dead" is regime-scoped. **The V6 "B32 ~0.92 structural ceiling" claim
is RETRACTED** — cfgv2 (+5.2%) and checkstop landed after it, moving B32 to 0.938
pooled with peaks in vLLM's band. The remaining B32 gap is host-side VARIANCE, not
a ceiling; the next real lever is a variance/soft-cell fix (§4.1), and cross-request
coalescing / fold family remain dead (within-request merge is the only coalescing
that works).

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

## §2/§4 CORRECTION (2026-07-05 07:15, closes the night's measurement program)
Order-symmetric re-measurement (ABBA harness, lab_stackc/ab_cfgv2sym) supersedes
the stack magnitudes: **cfgv2 ≈ +2% ± 4%** (mechanism real — 22%-of-wall sync,
profiled — but mostly GIL-shade-overlapped; identity PASS; keep default-ON),
**checkstop +2.3% owes the same order-symmetric re-bank** (inside the legacy
harness's +5-12% B-favoring ordering-bias band). THE HEADLINE STANDS ON A
DIFFERENT LEG: the flagship is at **PARITY at true steady state** (warm cells
5-8: 8.29 req/s vs vLLM live 8.24-8.46); the 0.938 pooled race carried warm-in
tail + h2h alternation bias (asymmetric against M*), both now diagnosed and
harness-corrected (ABBA + criterion warm-in in lab_ab.sh; h2h mitigations
documented). ACCEPTANCE SWEEP REQUIREMENT: use the corrected protocol or it
inherits the bias. Next agent: (1) re-bank checkstop order-symmetric, (2)
steady-state flagship race with tail-amortizing n (≥384/cell) or keepalive,
(3) then the proof sweep.

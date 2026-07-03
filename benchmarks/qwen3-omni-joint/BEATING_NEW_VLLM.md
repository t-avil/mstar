# BEATING NEW vLLM — campaign handoff

Single entry point for the M* vs vLLM-Omni 0.22 performance campaign on
Qwen3-Omni-30B-A3B (2×H200). Read this, then EXPERIMENTS.md for any entry
referenced. State as of 2026-07-03 ~07:30.

## 0. Read order for onboarding
1. This file (state, laws, protocols).
2. `EXPERIMENTS.md` — one paragraph per experiment, verdict + data pointers.
   THE knowledge base. Do not trust any other markdown or code comments
   (docs drift; several were measurably wrong — read code).
3. `NUMBERS_V2.md` — the committed official sweep (2026-07-02, canonical
   pair, WARMUP=5, n≥50). `HEADTOHEAD.md` for the raw comparison method.
4. Workspace `CLAUDE.md` (repo root of the workspace, /home/tim) — GPU
   hygiene, git topology, chart rules. Non-negotiable conventions.
5. The option board + V1 design (bottom of EXPERIMENTS.md and task notes
   reproduced in §6 here).

## 1. Scoreboard (req/s, ÷ vLLM-Omni 0.22 recorded baselines)
Committed sweep (07-02, canonical pair 6,7): speech s2s/i2s 2.2–2.9×
everywhere (+RTF, +audio-s/s); s2t 0.83–1.26×; i2t 0.77–0.97× (the gap).
Final-stack session (07-03, pair 0,1 — CROSS-NUMA HANDICAPPED, add ~10%
for canonical): i2t B1 0.844 (0.94×), B8 3.936 (**1.14×, now winning**),
B32 best 6.894 (~0.92× projected canonical vs 8.210). s2t B8 18.47
(1.17×). No path regressed. vLLM's speech deficit is STRUCTURAL (three
engines joined by pickle+flock+/dev/shm per codec chunk — see the vLLM
source-dive entry): our lead there is durable.

## 2. The final stack (all validated by adjacent-pair A/B, byte-exact outputs)
Flags on top of `configs/qwen3omni_2gpu_encoff.yaml`:
```
MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1
MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_CHUNKED_PREFILL_V2_VISION=1
MSTAR_MIXED_BATCH=1 MSTAR_MIXED_BATCH_VISION=1 MSTAR_MIXED_SPEC=1
MSTAR_PREFILL_CHUNK_TOKENS=256
MSTAR_SLIM_EMIT=1 MSTAR_FAST_ROUTE=1
MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_FAST_CHECKSTOP=1
```
Night-of-07-03 additions and their measured deltas at i2t B32:
SLIM_EMIT +17–26% (skip per-rid GraphEdge pickle; api-server template
inflation); FAST_ROUTE +7–10% (memoized replicated fanout); N3 in code
(set_config change-detect — made the sampler cache real); CACHE+CHECKSTOP
+10.5% geomean. Validated opt-ins not in the default set:
MSTAR_MIXED_SPLIT_ATTN + MSTAR_MIXED_PREPLAN (mixed-step attention 3.3×
faster; e2e-neutral at current fold volume — flips positive if fold volume
rises). Audio-optimal alternative config: default yaml (see E4b).

## 3. Branch map (fork t-avil/mstar)
- `opt/decode-v2` = shipped v2 (committed sweep).
- `exp/mixed-batch-p2` = W5 mixed batching (merged into the stack's base).
- `exp/fold-rate` = single-chunk experiment (closed) + split-attn +
  WALK_STATS + dynflags.
- `exp/overlap-sched` = **the live branch**: everything in §2 plus E9
  direct-feed cherry-pick, N2 conductor poll (fixed, needs re-smoke).
- `encoders-implemeneted-benchmarked-mstar-v2` / `-v3` = docs + numbers.
- `encoders-implemeneted-benchmarked-v3-w5-datapoints` = every W5/preplan
  datapoint (READMEs inside).
Worktrees: /m-coriander/coriander/tim/mstar-p2 (live), mstar-v2 (shipped),
bench-merge (docs/numbers).

## 4. Hard-won laws (violate = wasted days; each has an EXPERIMENTS entry)
1. **In-graph microbench only.** Out-of-graph kernel timing inverted the
   fp8 verdict once. Capture loops, time replays. `mb_split_attn.py` is
   the pattern.
2. **Remove work, not waits (GIL-valve law).** Two GIL threads per worker:
   gpu-thread waits shelter main-thread Python. Removing a wait without
   removing work REGRESSES (sampler cache −7% pre-slim → +4-15% once the
   main thread got lighter).
3. **This box lies at cell granularity.** Foreign users swing identical
   configs 4.0–6.9 req/s. ONLY adjacent interleaved A/B pairs (or ≥3
   rounds) count. Every coverage run carries an i2t:32 sentinel cell;
   sentinel off-band ⇒ invalidate. Never co-locate (idle gate is in the
   harness; respect its ABORTs).
4. **Flags may be structurally dead.** The sampler cache "regressed" three
   times while a per-step set_config wiped it before every hit. When a
   mechanism-certain win doesn't convert, verify the mechanism is ALIVE in
   production (sync counts, WALK_STATS counters) before believing e2e.
5. **NUMA**: quick_bench binds cpunode 1 — correct for GPUs 4–7 only.
   Pair 0,1 numbers are ~10% understated. Canonical numbers: pair 6,7.
6. **Walk-gate every fast path** (unconditional decode fast paths taxed
   Talker steps 17% once). And every declared input edge must be emitted
   (empty tensor_info ok) or the graph hangs silently.
7. **Effect-size gate**: <2% predicted → microbench or arithmetic, not
   e2e cells. tok/req ≈177 at i2t is the correctness sentinel (drift =
   sampling/stop bug).

## 5. Harnesses (all in /m-coriander/coriander/tim/)
- `quick_bench.sh name worktree "FLAGS" config gpus port cells` — one
  server, cells like `i2t:32,s2t:8`; `QB_FAST=1` halves n (smoke only —
  ±25% spread). Full n: B32=96, B8=48, B1=12.
- `dyn_ab.sh` — ONE server, interleaved A/B via MSTAR_DYNFLAGS runtime
  flag file. As of fbe4b1c all four §2 winners are runtime-flippable →
  a full-stack A/B costs one 4-min startup. Prefer this.
- `MSTAR_WALK_STATS=1` — per-(node,walk) step counts + _ms + chain/fold/
  peek counters every 200 steps at WARNING. First tool for "why".
- `torch.cuda.set_sync_debug_mode('warn')` in a 60s repro — sync
  attribution (found the 6 hidden syncs/step).
- nsys profile scripts: prof_winner.sh pattern (nvtx_sum + cuda_api_sum +
  the indexed sqlite queries in the session log).
- ab_*.sh family — two-server interleave when both pairs are free.
- WARMUP (measured, INFO probe 2026-07-03): ready=204s = load/precompute
  ~116s + 34 Thinker captures 70s (the tail; Talker 24 caps + codec run
  parallel on rank 0). Triage trim: MSTAR_DECODE_BUCKETS=24,28,32
  MSTAR_PREFILL_BUCKETS=256,512 (~-50s; largest bucket mandatory, guard
  enforces; NEVER for committed sweeps). CUDA graphs CANNOT migrate across
  GPUs (device-bound cudaGraphExec) — pre-warming on another pair is
  physically out; the pattern is WARM-STANDBY: after claiming a pair,
  start one server and keep it for the whole session; every A/B flips via
  MSTAR_DYNFLAGS (all winning-stack flags runtime-refreshable @ fbe4b1c).

## 6. What's next (sized; full designs in EXPERIMENTS.md option board)
IMMEDIATE (blocked only on free GPUs):
- Canonical 6,7 sweep of §2 → official NUMBERS_V2 refresh.
- Mid-batch i2t B2/B4/B16 cells (last run invalidated by co-location).
- N2 re-smoke (conductor blocking poll; wedge root-caused to an unguarded
  event fd, fixed at b1c1ff1 — expect TTFT −6–10ms on chunked prompts).
THE LAST ~8% AT i2t B32 (pick one, in order of expected value):
- **V1 async scheduling** (task #33 has the full design): submit replay
  N+1 before host sample bookkeeping; sampler kernels enqueue on-stream
  (sync-free now), D2H on a copy stream + event consumed one step late;
  token feed via E9 direct-feed (already on the branch). vLLM's biggest
  structural edge. Est +5–10%.
- E10 two-step decode re-test on the new stack (old FLAT verdict predates
  every win; rebase recipe in the audit entry). Est +2–8% (+cache synergy).
- N1-full: register/route flag-caching beyond the landed checkstop-lean.
BIGGER SWINGS (vLLM source-dive entries): V2 budgeted chunked-prefill
interleave policy (our mixed graphs already exist — policy only), V3
persistent-batch diffs, V4 detok out-of-process, V5 encoder mm_hash cache,
V6 Gumbel sampler.

## 7. Protocols that keep the data honest
One branch per experiment; commit+push every checkpoint; every valid run's
dir committed on its bench branch and merged to `benchmarks` immediately;
never bench artifacts on main; record EVERY verdict (incl. failures — the
dead-cache saga and both wedges are entries) in EXPERIMENTS.md with the
datapoint paths. GPU etiquette per workspace CLAUDE.md: hard timeouts,
cleanup traps, no co-location, clock teardown when locked.

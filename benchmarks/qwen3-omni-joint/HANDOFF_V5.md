# HANDOFF_V5 — M* vs vLLM-Omni campaign (state as of 2026-07-04 ~18:45 UTC)

Single entry point for the next agent. Read this, then EXPERIMENTS.md (pool
root, ~60 entries — THE knowledge base; the repo copy under
benchmarks/qwen3-omni-joint/ is synced) for any referenced entry. Trust
nothing that isn't an adjacent-pair A/B on this box; when a mechanism-certain
win doesn't convert, check the mechanism is alive (counters) before doubting
the idea.

## 1. THE SCOREBOARD (live head-to-head, 2026-07-04 18:00-18:35, committed in
benchmarks/qwen3-omni-joint/h2h_v2/)

M* full stack (canonical 6,7) vs LIVE vLLM-Omni 0.22 (2,3), adjacent
per-cell pairs, same window:
- i2t B8: **1.22× — WON LIVE**
- s2t B8: **1.40× — WON LIVE**
- speech (s2s/i2s): 2.1–2.9× — won since campaign start
- i2t B32: **0.87–0.90× warm (pairs 0.723/0.869/0.897) — THE LAST STAND,
  losing by 10–13%**
- Reliability: vLLM-Omni CRASHED twice that day under load (incl. mid-race);
  M* zero self-inflicted deaths in 2 days of continuous benching.
- tok/req sanity: M* 172–177, vLLM 210–212 (they generate ~20% longer; use
  req/s for cross-system claims).

Campaign trajectory at i2t B32: 0.53× (v0.22 release) → 0.77× → 0.85× →
0.89× → 0.88× live. ~1.0 req/s remains.

## 2. THE WINNING BUILD (what ran in the race)

Worktree /m-coriander/coriander/tim/mstar-crusade, branch **opt/custom-ops**
(= opt/integration-v4 + 4 custom-op commits). Launch:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache
TORCHDYNAMO_CACHE_SIZE_LIMIT=128
MSTAR_CUSTOM_OPS=1
MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1
MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_CHUNKED_PREFILL_V2_VISION=1
MSTAR_MIXED_BATCH=1 MSTAR_MIXED_BATCH_VISION=1 MSTAR_MIXED_SPEC=1
MSTAR_PREFILL_CHUNK_TOKENS=512
MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1
MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_FAST_CHECKSTOP=1
MSTAR_FAST_SEND=1 MSTAR_EMIT_SIDECAR=1
MSTAR_MIXED_BUDGET_TOKENS=512
MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_CODEC_CHUNK_EMIT=1
```
config configs/qwen3omni_2gpu_encoff.yaml. Boot ~12.5 min (Inductor cache;
first-ever boot pays ~40 min autotune once). Alternative for
small-batch/latency deployments: drop CHUNKED_PREFILL_V2_VISION +
MIXED_BATCH_VISION, add MSTAR_MERGED_PREFILL=1 (+4–6% at B1/B2/B4; B32
impact unmeasured).

## 3. BRANCH MAP (fork t-avil/mstar; all pushed)
- **opt/custom-ops** = THE build (integration + compile work). 4 op commits:
  b7fb94e/f152e3b (mstar::run_attention thinker+talker), be16eeb
  (mstar::fused_experts_fp8 + pre-quant hoist), 12ce776 (mstar::apply_rope).
- **opt/integration-v4** = six composed wins: sidecar(+9% B32),
  norm-compile-fix(+4.5%), chunk-512(+3-5%), speech bundle(+1-3%),
  V2 budget(+7-11% B32), merged prefill(+4-6% B1-B4) + GC_TUNE (wash,
  default off).
- opt/v2-policy, opt/speech-floor, opt/prefill-merge, opt/sched-pack
  (REJECTED −5-7%, parked), opt/async-sched (V1 PARKED: +9% output-length
  identity fail + tok/s wash; substrate kept).
- Docs/numbers: encoders-implemeneted-benchmarked-mstar-v2 (+ benchmarks
  branch mirror). Worktrees under /m-coriander/coriander/tim/mstar-*.

## 4. THE B32 GAP PLAN (in expected-value order)
1. **Sidecar stage-2**: exile route_outputs + check_stop consumption to the
   sidecar process (stage-1 exiled emit/WGD, +9%; route ≈1.8ms +
   check_stop ≈1.1-2.1ms remain on the main thread — the same GIL-exile
   move, design notes in docs/SIDECAR_DESIGN.md on the branch).
2. **V1 revisit post-compile**: V1's precondition ("pays only when
   main-thread < GPU time") is now closer after custom-ops; its identity
   failure needs the stop-lag redesign — defer emit/transport only, keep
   stop-state sync (analysis in the V1 FINAL entry).
3. **Custom-ops step-5 + tail**: talker dense-attn op (9 breaks),
   advance_seq_lens tail (~52, low payoff), sampler = wall by design.
   Residual census 41 header-breaks (from 1617).
4. **Race variance**: M*'s soft cells cost ~0.05× of pooled ratio — more
   rounds tighten the true number; also profile WHY our B32 cells vary more
   than vLLM's (suspect: admission waves vs their every-step budgeting —
   possibly V2 budget tuning, MSTAR_MIXED_BUDGET_TOKENS sweep 256/512/1024).
5. Canonical warm re-baseline sweep for official NUMBERS_V4 (GPU 7 is back).

## 5. HARNESS TOOLBOX (all in /m-coriander/coriander/tim/)
- **lab_server.sh name worktree "FLAGS" config gpus port ttl_hours** — boots
  a server: auto-NUMA-bind from GPU PCI, TMPDIR to pool + stale-upload
  clean, discarded warm cell (first cell after ready reads −30% otherwise),
  writes lab_<name>/numa_node for client pinning. lab_kill.sh name kills.
- **lab_ab.sh lab exp '<flagsA>' '<flagsB>' port cells rounds** — interleaved
  dynflags A/B on a warm server; REFUSES mismatched key sets ('{}' vs keyed
  = the contamination bug); clients NUMA-pinned. ~5-min signal per pair.
- **h2h.sh** — the live race driver (M* vs vLLM alternating cells).
- quick_bench.sh / dyn_ab.sh — older harness, NUMA-fixed + warm-cells.
- vLLM boot: /home/tim/baselines/launch_vllm_2_3_numa0.sh (port 8093);
  runner arg --inference-system vllm_omni.
- Runtime-flippable flags: SLIM_EMIT(2), FAST_ROUTE(2), CFG_CACHE,
  CHECKSTOP(+TALKER), FAST_SEND, CODEC_CHUNK_EMIT, MIXED_BUDGET_TOKENS,
  DIRECT_FEED, SCHED_PACK, GC_TUNE. Capture/process-static (reboot per arm):
  CUSTOM_OPS, MERGED_PREFILL, SPLIT_ATTN/PREPLAN, EMIT_SIDECAR, buckets.

## 6. LAWS (violate = wasted days; each has an EXPERIMENTS entry)
1. In-graph microbench only; out-of-graph decode timing lies.
2. Remove work, not waits (GIL-valve) — gpu-thread wait removal pays only
   after main-thread Python shrinks. Empirically reconfirmed twice.
3. This box lies at cell granularity: adjacent interleaved pairs or ≥3
   rounds; same-config spread gate before believing anything; sawtooth
   regimes are valid for RATIOS only (consistent-slow is usable).
4. Verify the mechanism is ALIVE (WALK_STATS counters at WARNING level)
   before believing any e2e verdict. Gate log lines must be WARNING.
5. req/s for cross-system claims (tok/req differs 177 vs 212).
6. A-B-A bracketing under trending load; never two B32 cells concurrently
   on one NUMA node; absolutes need a solo box.
7. Effect-size gate: <2% predicted → microbench/arithmetic, not e2e cells.
8. Flags may be structurally dead or superseded — re-decompose after
   structural changes (the sidecar zeroed several old flag wins).

## 7. OPS HAZARDS (all cost us hours; don't repeat)
- **Wrapper reaping**: agent-launched lab wrappers die when the agent's
  shell tree is reaped between turns → cleanup trap kills the healthy
  server. Agents must `setsid bash lab_server.sh ... < /dev/null` or let
  the coordinator own all boots. Fire lab_ab immediately after WARM+READY.
- **pkill/pgrep self-match**: patterns containing the command string match
  your own shell (exit 144). pgrep -f benchmark.runner is safe; kill by
  pgid file.
- **One owner per GPU pair**; never fire on an agent's pair within ~2 min
  of ordering them to act (message-crossing double-boots happened twice).
- **/tmp**: 70G rootfs; uploads go to TMPDIR (pool) now, but watch rootfs.
- **Redirects**: create the log's directory BEFORE `cmd > dir/log &`.
- **Compile edits**: never remove @torch.compiler.disable from engine
  objects (wedge/retrace-storm); the custom-op route is the only safe path.
- GPU 7 fell off the bus once (15h dead; masqueraded as code wedges —
  check `nvidia-smi -i 7` when boots die with "invalid device ordinal").
- Foreign jobs come and go (Ray at 4×93GB, etc.) — idle-gate before every
  boot, never co-schedule, never touch their processes.

## 8. WHAT'S PROVEN DEAD (don't re-litigate without new conditions)
SCHED_PACK peek-backoff (fairness yields are load-bearing); chunk 128
(−12%) and 768 (=512 structurally); finer decode-bucket grids; single-chunk
folding at any batch; GC-tune/jemalloc at B32 (mechanism-proven wash);
DeepGEMM (structurally wrong at decode M) + trtllm MoE (Blackwell-only);
FA3; NUM_SLOTS=3; GIL interval; E9/E10/W2/W3 (flat, substrates kept);
V1-as-built (identity-vs-win tension: the win NEEDS deferred stop-checks,
deferred stop-checks CAUSE ~+9% output-length drift under sampling);
disable-removal compile edits.

## 9. REPORTING FORMAT (user requirement, every experiment)
1. "We are losing at these paths" — compact ratio table, losing cells
   flagged. 2. "What we tried" — ONE sentence. 3. "What's next" — ONE
   sentence. Sloppy-fast mode: 2-round/small-cell POCs, direction over
   precision, stack winners on the integration branch, full sweeps only for
   30-50%-class accumulations. Document EVERY verdict (incl. failures) in
   EXPERIMENTS.md; commit+push docs branch + merge to benchmarks after every
   batch; keep project memory current.

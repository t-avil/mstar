---
name: mstar-24option-board
description: 2026-07-03 re-research verdict on campaign docs + the 24-option board and phased plan (PLAN_BEAT_VLLM_V4.md)
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

Full re-research of the M* vs vLLM-Omni 0.22 campaign (2026-07-03): plan and
verdicts live in `/m-coriander/coriander/tim/PLAN_BEAT_VLLM_V4.md`.

Key durable facts (verified against git/raw JSON, not docs):
- Campaign docs substantially truthful; ALL committed numbers reproduce
  exactly. BUT two headline claims are unbacked by any results.json: "best
  cell 6.894" (real best 6.787) and "+10.5% cache+checkstop seal"
  (reproducible evidence ≈ +6.8%). Honest i2t B32 = 0.80× on pair 0,1
  (committed NUMBERS_V3), canonical pair never measured for the final stack.
- W7 denser decode buckets [24,28] SHIPPED (merged d197dbd into opt/decode-v2;
  env `MSTAR_DECODE_BUCKETS` on exp/overlap-sched) — EXPERIMENTS queue entry
  is stale. The real EXPERIMENTS.md (679 lines, pool root) is UNCOMMITTED;
  repo has a stale 72-line file under the same name.
- vLLM-Omni 8.21 req/s i2t B32 = torch.compile + FULL cuda graphs (thinker
  glue ≈ zero host Python) + v0.22 rebase async sampled-id D2H + MoE
  kernel-path fix. Kernels NOT better than M*'s fp8.
- vLLM speech deficit (msgpack+flock+/dev/shm per codec chunk) confirmed at
  HEAD but DEPRECIATING: inline small-payload fast-path exists in their tree,
  default-off (shm_connector.py:60-79). Speech moat is perishable.
- Mid-batch hole: i2t B2 0.75× / B4 0.83×; v3 s2t B2–B16 never measured.
  No option on any board owned this until Phase-0 diagnosis was added.
- quick_bench.sh/dyn_ab.sh hardcode numactl node 1 — wrong for GPUs 0–3;
  harness fix is Phase 0 of the plan.
- 12 NEW options (#13–24) added to the existing 12: SCHED_PACK ready-index,
  NUMA pin, speech host-floor bundle (+K-step talker escalation), Code2Wav-SP
  e2e retry, encoder-bundle retry, bucket-geometry retry, pinned-staging
  rewrite, transport pack (batched WGD+msgspec), fusion pack (gated), plan-
  reuse pack, per-request encoder routing, MoE grouped-GEMM bake-off.
  Ordering law: main-thread cuts BEFORE gpu-thread cuts (GIL shade budget).

Canonical-pair (6,7) results 2026-07-03 evening (committed, sweep_canon_20260703
on the docs branch): sidecar bundle (SLIM_EMIT2/FAST_ROUTE2/FAST_SEND/
EMIT_SIDECAR @ ab75034) +9.2% i2t B32 / +8.2% B1 in adjacent A/B — positive,
stays in stack. Canonical i2t B32 = 0.73–0.85× band (best adjacent 6.962);
+10% NUMA projection NOT confirmed. s2t B32 34.233 = 1.11× (best ever, from a
degraded window — likely understated). N2 re-opened (re-smoke wedged,
default reverted 4a19bbb).

Evening session 2026-07-03 (lab era): built lab_server.sh/lab_ab.sh/lab_kill.sh
(persistent warm server + dynflags A/B = ~7 min per experiment; boot ~3 min
warm-cache). Measured: cross-NUMA −15.5% B32 (harness now autobinds); cold
first cell −30% (warm cell added to quick_bench); warm steady i2t B32
7.0–7.4 = 0.86–0.90× (best honest). SCHED_PACK REJECTED (−5-7%, fairness
yields load-bearing). Direct-feed wash re-confirmed. DECOMPOSITION FINAL:
post-sidecar every flippable flag = 0–3% (cache+checkstop +7-10% claim
RETRACTED; FAST_ROUTE not cleanly converting pooled) — re-decompose flags
after structural changes. Two concurrent labs = ratios only, absolutes need
solo box. NEXT: V1 async scheduling (only resolvable lever left, +5-10% est),
then V2 chunk policy for B2-B4/TTFT. User wants per-experiment reports as
losing-paths table + tried/next one-liners (see [[experiment-report-format]]).

Night-2 close (2026-07-04 ~00:30): V1 async-sched PARKED after full falsification
(+9% output length = identity fail, tok/s wash; fundamental tension: the win
needs deferred check_stop, deferred check_stop causes the drift; branch
opt/async-sched @ 18b1244 kept as substrate). WINNERS BANKED: chunk-512
(+3-5% B32, B1 gate passed; code default was already 512 — canonical flag
set was overriding to 256!), speech bundle +1-3% (talker fast-checkstop +
codec chunk-emit; speech track CLOSED per user), norm compile fix
(opt/compile-fix 1733fab: pure-torch RMSNorm under compile, graph breaks
1617->816; throughput A/B STILL OPEN). torch.compile is ALREADY ON for all
text walks (compile-then-capture, cuda_graph_runner.py:609) — the compile
gap was only the breaks. Integration branch opt/integration-v4 (worktree
mstar-iv4) = sched-pack base + speech items; stacked test OPEN. ROOT CAUSE
of night's server deaths: /tmp (70G rootfs) hit 100% from
mstar_uploads_tim — lab_server.sh now sets TMPDIR to the pool + cleans at
boot; check rootfs free before believing degraded windows. NEXT SESSION
ORDER: (1) sequential norm-fix A/B, (2) stacked integration test, (3) V2
budgeted admission policy build (main open lever: B2-B4 + TTFT + unlocks
banked split-attn). Protocol: labs one-at-a-time for verdicts; two labs only
for independent ratio streams on different NUMA nodes; NEVER two B32 cells
concurrently on the same node.

NIGHT-2 FINAL (2026-07-04 01:05): norm compile fix WON +4.5% (opt/compile-fix
1733fab, pure-torch RMSNorm under compile, breaks 1617->816; cherry-picked
to integration). STACKED TEST +6.3%: opt/integration-v4 @ 7aebb1d (speech
bundle + norm fix) with PREFILL_CHUNK_TOKENS=512 + talker-checkstop +
codec-chunk-emit = i2t B32 7.340 mean / 7.593 best (0.89x/0.93x vs vLLM
8.210), s2t B8 19.49 = 1.23x (best 1.29x). Wins COMPOSE. Trajectory: 0.53 ->
0.77 -> 0.85 -> 0.89x. NEXT ORDER: (1) graph-break round 2 —
thinker.py:225 layer-loop ~48 breaks + moe.py:478; (2) V2 budgeted admission
policy (B2-B4 + TTFT + unlocks banked split-attn +4.4%); (3) canonical
warm+512 re-baseline sweep for official numbers. Protocol locked: solo
sequential labs for verdicts; /tmp TMPDIR hardening in lab_server.sh.

NIGHT-2 EXTENDED (04:00 close): V2 BUDGET POLICY WON — MSTAR_MIXED_BUDGET_TOKENS=512
= +7-11% i2t B32, +1-10% B8 (every-step folding kills phase drain; predicted
neutral, actually won); floor stays 24 (lowering = negative, occupancy
economics confirmed 3rd time); MERGED into opt/integration-v4 (contains:
speech bundle + norm compile fix + V2; run with PREFILL_CHUNK_TOKENS=512).
Compile round-2 CLOSED (both disable-removals wedge or storm; custom-op
route is the correct path; set_layer_idx revert may be false-negative —
GPU-7 died mid-test). HARDWARE: GPU 7 fell off the bus ~02:30 2026-07-04
("No devices were found") — canonical pair DEAD until admin reset; box has
7 GPUs now. B2/B4 remain open (fixed host cost + prompt-path latency; NOT
fold policy — three fold levers closed). NEXT: arm-3 (budget+split+preplan
— fold volume now high at B32, banked +4.4%/step should convert), stacked
re-test with V2 in, canonical re-baseline after GPU-7 repair. Best numbers:
stacked 7.34-7.59 pre-V2 (0.89-0.93x); V2 adds +7-11% on top at B32 →
projected AT/ABOVE vLLM parity, needs the clean stacked confirm.

SESSION FINAL (04:50 2026-07-04): MERGED PREFILL WON +5.6% at B2 AND B4 (the
losing cells — first direct hit; opt/prefill-merge 41200ec, one merged
prefill_multimodal walk reusing the prefill_vision capture, saves a
conductor round-trip per admission; B1 read -4.4% opposite-sign — question
for the confirm; flag default-OFF VALIDATED-OPT-IN). opt/integration-v4
final composition: speech bundle + norm compile fix + V2 budget + merged
prefill; run flags: final stack + PREFILL_CHUNK_TOKENS=512 +
FAST_CHECKSTOP_TALKER=1 + CODEC_CHUNK_EMIT=1 + MIXED_BUDGET_TOKENS=512
(+ MERGED_PREFILL=1 opt-in, needs vision-chunking OFF). CLEAN-WINDOW QUEUE:
(1) canonical re-baseline sweep of integration-v4 (needs GPU-7 admin
reset), (2) merged-prefill confirm incl. B1 sign, (3) W2-at-small-batch
theory test, (4) arm-3 budget+split+preplan. Harness lesson: stagger cell
orders across concurrent two-server streams (B32-vs-B32 collision).

CRUSADE RESULT (09:15 2026-07-04): graph breaks 1617 -> 41 header (~165
attributed) via 4 custom-op steps on opt/custom-ops @ 12ce776
(mstar::run_attention thinker+talker, mstar::fused_experts_fp8 + pre-quant
hoist, mstar::apply_rope; TORCHDYNAMO_CACHE_SIZE_LIMIT=128 fixes the
recompile fallback; TORCHINDUCTOR_FX_GRAPH_CACHE + persistent cache dir
/m-coriander/coriander/tim/inductor_cache solves the 40-min autotune boots
— cache-proof timing pending). DEFERRED: MSTAR_CUSTOM_OPS ON/OFF perf A/B
(clean window) + integration decision. Night-runner queue (GC/jemalloc,
merged-prefill confirm, W2, arm-3) still starved by the Ray job on 0/1/4/6.
OPS LESSON (twice now): message-crossing between main and agents causes
double-boots — before firing anything on an agent-owned pair, POLL for its
in-flight action first; never fire within 2 min of issuing an order.

DAY-2 AFTERNOON (16:10 2026-07-04): custom-ops perf A-B-A = POSITIVE
direction (ON 6.47 above both OFF brackets 5.77/4.75 on a sawtooth box;
magnitude +5-20% unresolved — canonical A/B quantifies). Cache double-proof
12m28s/12m44s boots. Merged prefill SHIPPING CONFIRM: +4-6% at B1/B2/B4 vs
the shipping config (same-pair sequential; B1 anomaly was pair-band); B32
read deferred. GC/jemalloc: wash at B32, cleanly falsified, zero-code levers
exhausted. W2 + arm-3: parked at noise gates. Box saga: Ray job came and
went (0/1/4/6), GPU 7 STILL DEAD, sawtooth load (60-370). STABLE-WINDOW
QUEUE: canonical re-baseline (needs GPU-7 admin reset — TELL THE USER),
custom-ops magnitude A/B, merge-config B32, arm-3, W2. Ship candidates
stack: opt/integration-v4 (+ opt/custom-ops pending magnitude).

HEADTOHEAD_V2 LIVE RACE (18:35 2026-07-04, committed in h2h_v2/): M* full
stack (custom-ops build) on canonical 6,7 vs LIVE vLLM-Omni 0.22 on 2,3:
i2t B8 1.22x WIN, s2t B8 1.40x WIN, i2t B32 pairs 0.723/0.869/0.897 (warm
~0.87-0.90x — still losing B32 by ~10-13%). vLLM DIED MID-RACE (2nd
collapse that day; M* zero self-deaths). GPU 7 returned (reset happened).
B32 gap-closers queue: sidecar stage-2 (route/check_stop exile), custom-ops
step-5 + tail, V1-post-compile revisit, more race rounds. Speech + B8 both
paths = BEATEN LIVE.

Related: [[mstar-decode-bottleneck]], [[mstar-bench-env]], [[ingraph-microbench]]

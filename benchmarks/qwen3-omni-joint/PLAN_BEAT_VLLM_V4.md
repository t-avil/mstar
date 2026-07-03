# PLAN_BEAT_VLLM_V4 — re-research, 24-option board, and execution plan

Produced 2026-07-03 by a full re-research pass under the instruction "assume
every word can be a lie": three parallel deep-reads (doc fact-check vs
git/raw-data; M* paper arXiv 2606.12688 + live code; vLLM-Omni 0.22 source),
followed by five adversarial validation rounds (academic vs senior-SWE), then
this plan. No benchmarks were run. GPU state at research time: pair 0,1 idle,
2–5 foreign, canonical 6,7 blocked (GPU 7 at 88%, 76 GB).

Companion docs: BEATING_NEW_VLLM.md (campaign map — corrections below),
EXPERIMENTS.md (pool root, 679-line knowledge base — UNCOMMITTED, fix that),
NUMBERS_V2/V3.md + HEADTOHEAD.md (branch encoders-implemeneted-benchmarked-mstar-v2).

---

## Part 1 — Verdict on BEATING_NEW_VLLM.md and EXPERIMENTS.md

### Overall
Both documents are **substantially truthful and unusually well-instrumented**.
Every one of the 46 cited commits exists and does what it claims. Every
final-stack flag exists and is read on `exp/overlap-sched` (verified at
file:line). Every *committed* number reproduces to the digit from raw JSON:
vLLM i2t 8.210/3.455/0.898, M*-v2 sweep 6.299/3.361/0.800, s2t 17.197/31.565,
speech 2.2–2.9×. The laws (§4 of BEATING) are internally consistent with the
experiment trail and the code confirms the mechanisms they rest on
(e.g. `sampling.py:213-220` documents the GIL-valve law in-source).

### Where the docs lie (or spin)
1. **"Best cell 6.894" does not exist on disk.** No results.json anywhere
   contains it. Real best slim-on cell: 6.787 (`qb_fnB2`). The single highest
   M* i2t B32 cell on disk is 7.606 — from a SLIM_EMIT-**off** run
   (`qb_slimoff1`), which alone shows the box's noise dwarfs single cells.
2. **"+10.5% sealed" (cache+checkstop) is not reproducible.** The cited cells
   (5.69/5.89/5.70 vs 6.57/6.57/5.97) appear in no results.json. The closest
   real A/B on disk (`qb_fnA1-3` vs `qb_fnB1-3`) gives **≈ +6.8%** geomean,
   with two near-flat pairs. The arithmetic in the doc is self-consistent but
   the inputs are unverifiable. Treat the true cache+checkstop win as +5–10%
   pending the Phase-0 re-seal.
3. **"~0.92× projected canonical" stacks two optimisms**: the unbacked best
   cell (6.894) × a +10% NUMA correction. The committed V3 preview says
   **i2t B32 = 6.559 = 0.80×**; the coverage-sweep line says 6.134. Three
   different numbers exist for the same cell; the scoreboard quotes the max.
4. **The +10% NUMA correction is circular**: derived from cross-campaign band
   comparison (pair-0,1 ~5.0–5.7 vs canonical 6.0–6.2), which is exactly the
   cross-time comparison Law 3 forbids. It is plausible but unmeasured. The
   harness fix in Phase 0 makes it measurable.
5. **"No path regressed"** contradicts the coverage sweep's own s2s B8 line
   (−5% req/s, −13% audio-s/s vs committed). Attributed to NUMA; unproven.
6. **E6 entry self-contradicts** (header "WASH, off in final config" vs body
   "KEPT ON"). Body is correct; BATCH_EMIT is in the stack and SLIM_EMIT
   requires it. Fix the header.
7. **Stale queue entries**: W1 (FAST_POSTPROC) and W7 (buckets [24,28]) are
   described as "PROMOTED / combo in flight" but both **already shipped** in
   `opt/decode-v2` (W7 merged via `d197dbd`, live at `submodules.py:1129`;
   env-tunable `MSTAR_DECODE_BUCKETS` on `exp/overlap-sched:1461`). The
   "W1+W7 combo interference" open item is moot.
8. **EXPERIMENTS.md vs EXPERIMENTS.md**: the real 679-line knowledge base at
   the pool root is uncommitted anywhere; the file committed in the repo under
   the same name is a stale 72-line plan from an earlier era. Anyone cloning
   the fork gets the wrong file. Also `sampling.py:207` comment says the cfg
   cache defaults ON; code says OFF.
9. **HEADTOHEAD vs EXPERIMENTS on baseline quality**: EXPERIMENTS claims "the
   committed vLLM B1/B8 baselines themselves proved understated"; HEADTOHEAD's
   own table shows live ≈ committed for i2t (0.886 vs 0.898 B1; 3.401 vs 3.455
   B8) and only s2t B1 understated (3.831 → 4.312). The good news: the
   headline **i2t B8 1.14× win survives against the live baseline too**.
10. **"vLLM's speech deficit is structural" — TRUE but decaying, and
    irrelevant to i2t.** Confirmed in vllm_omni source: three engine processes
    joined per codec chunk by msgpack (not pickle — minor doc error) +
    /dev/shm segment + fcntl.flock, with no zero-copy (`shm_connector.py:61-66,
    88-104`, `serialization.py:30-38` "TODO: Enable zero-copy"). BUT vLLM HEAD
    already contains an inline small-payload fast-path (`shm_connector.py:60-79`)
    that is merely default-off and blocked by the legacy async-chunk adapter.
    **One release could activate it.** The speech moat is real today and
    should be treated as depreciating. And none of this touches i2t (thinker
    only, no codec chunks) — the 8.21 baseline owes nothing to it.

### Does the BEATING narrative hold up logically?
Yes, with the above discounts. The causal story — gap = main-thread Python
floor + prefill serialization, not the forward; remove-work-not-waits;
mechanism-alive verification — is coherent, matches the profiling data, and
correctly predicted the two wins that converted (SLIM_EMIT, FAST_ROUTE). The
honest current scoreboard is: **i2t B32 ≈ 0.80× on the handicapped pair
(canonical unknown), i2t B8 ≈ 1.14× (real, survives live baseline), B1 0.94×,
mid-batch B2/B4 losing 0.75×/0.83× with no assigned owner, s2t ≥1.0× at
B8/B32, speech 2.2–2.9× but perishable.** The single biggest measurement debt
is that no final-stack number exists on the canonical pair.

---

## Part 2 — How vLLM-Omni 0.22 got to 8.21 req/s at i2t B32

(Full citations in the dive; venv confirmed vllm 0.22.0+cu129 + vllm_omni
0.22.1.dev61.)

The i2t path is **thinker-only** — no talker, no code2wav, no shm handoffs.
The speed comes from core-vLLM V1 machinery finally running clean under the
omni layer:

1. **torch.compile + FULL CUDA graphs on the whole thinker decoder**
   (`@support_torch_compile` at `qwen3_omni_moe_thinker.py:512`, run under
   `CUDAGraphMode.FULL`), kept alive by the deepstack-under-compile fix
   (`bb9f21d0`). Their host-side Python between steps is near zero because
   everything from embed to logits is one replay + compiled glue.
2. **The v0.22 rebase (`f92d84fc`)** activating async scheduling: sampled ids
   copy D2H on a side stream with an event, host syncs only when needed
   (`vllm_omni/worker/gpu_model_runner.py:249-266`) — decode never stalls on
   readback; plus batch-level routed-experts bookkeeping with non_blocking D2H.
3. **MoE kernel-path fix (`3a7c7f14`)** — force-disables a slow FlashInfer
   fp16 MoE path for Qwen3-Omni (`stage_init_utils.py:485-491`).
4. **Sync-stall removal (`26967510`)** — skip redundant D2H when tensors are
   already on CPU.

Structurally, their edge over M* at B32 is: **(a)** compiled/graph-captured
host glue vs M*'s ~13 ms/step of graph-walk Python, and **(b)** async sampled-id
readback vs M*'s (now cache-mitigated) sampler syncs. Their kernels are not
better — M*'s fp8 MoE beats their bf16/fp16 path. This is why V1 (async
scheduling) remains the headliner on the existing board, and why the new
options below are dominated by host-floor removal.

---

## Part 3 — The 24-option board

### Existing 12 (option board of 2026-07-03, unchanged numbering)
N1 check_stop/register fast-path (partially landed as FAST_CHECKSTOP — the
"N1-full" remainder stays queued) · N2 kill sleep-quantized hops (fix landed,
re-smoke pending; see the durable-fix note in #13) · N3 set_config
change-detect (LANDED) · R1 sampler-cache redo (LANDED as final stack) ·
R2 E10 two-step decode + E9 rebase re-test · R3 GIL-interval/NUM_SLOTS sweep ·
V1 async scheduling / GPU-resident sampled ids (**headliner**) · V2 budgeted
chunked-prefill interleave policy (**now carries a mandatory rider**, see
change C14) · V3 persistent batch + diffs · V4 detok out-of-process ·
V5 encoder mm_hash cache + per-step budget · V6 Gumbel sync-free sampler.

### NEW 12 (#13–#24), final post-debate versions

**M\*-derived (paper + code deep-read):**

**#13 MSTAR_SCHED_PACK — event-driven ready-set index + duplicate-work cuts.**
`get_next_batch` rebuilds the ready set from scratch 3–5× per decode step
(nested scans, `micro_scheduler.py:617-646`, accessors
`node_manager_utils.py:293-311`; call sites `worker.py:3242,2138,3669`).
Replace with an incrementally-maintained `(node,walk)→set(rid)` index updated
where the ready set already mutates (`node_manager_utils.py:126-146`), plus:
fairness-peek exponential backoff (`worker.py:3239-3246`, reusing the
mixed-peek backoff at `worker.py:3295-3318`), compute `_inline_emit_uuids`
once not twice (`worker.py:1133,1239`), drop the `sum(dict.values(),start=[])`
(`worker.py:1100-1107`), reuse check_stop's `flat.tolist()` for
`prem_per_request` (`worker.py:2547` vs `worker.py:2441-2443`), and delete the
redundant completion-event sync at `worker.py:2517` (superseded by
`_prematerialize_for_check_stop`'s wait at `worker.py:2538`). One flag, one
WALK_STATS counter per sub-cut (law 4). Est. **1.5–3.5 ms off the 13.2 ms
main thread**; every ms here also widens the sampler-cache shade budget.
Size: ~2–3 days. Risk: index invalidation — ship with a shadow-verify mode
asserting index == scan (the W2 pattern, zero mismatches expected).

**#14 NUMA/CPU affinity — harness fix first, then in-process pinning.**
(a) `quick_bench.sh:34` and `dyn_ab.sh:33` hardcode
`numactl --cpunodebind=1 --membind=1` — wrong for GPUs 0–3 and the reason
every pair-0,1 number carries an unquantified handicap. Derive the node from
the GPU indices (`nvidia-smi topo -m` or sysfs) — **half-day, do before any
other measurement**, it converts pair 0,1 into a fully valid bench pair and
makes the "+10%" claim measurable (numactl A/B on the same pair).
(b) In-process: the repo has zero affinity code. Pin each worker process to
its GPU's NUMA node and give main/gpu/plan threads distinct cores via
`ThreadPoolExecutor(initializer=...)` (`worker.py:3046,3069,3096`).
Est. few-hundred-µs–2 ms/step + tail tightening on the GIL ping-pong.
Size: (a) 0.5 d, (b) 1–2 d. Risk: low; per-core pinning opt-in.

**#15 Speech host-floor bundle (escalation: K-step talker graph).**
Three small cuts on the RTF-dominant path, then one big one if they convert:
(a) extend FAST_CHECKSTOP to `talker_decode` — the Talker still does a
per-request `.item()` host sync every AR step (`submodules.py:2360`; the gate
at `worker.py:2539` is hardcoded to thinker_decode); (b) **colocate
Talker+Code2Wav on one worker** — the paper (§4.2) promises this kills the
per-frame IPC, but NO shipped yaml does it; the `codec_tokens`
StreamingGraphEdge (`qwen3_omni_model.py:929-933`) crosses a worker boundary
every frame; (c) batch the codec edge hand-off to the chunk the consumer
actually needs (`LeftContextChunkPolicy(chunk=25)`,
`qwen3_omni_model.py:1025-1028`; per-frame `put`s at
`stream_buffer.py:51-63,136-143`) — ~25× fewer transfers. Escalation once
(a–c) are in: K-frame-unrolled talker_decode capture (K=5 with EOS-trim; the
whole per-frame forward incl. MTP is already one graph,
`submodules.py:2405-2417`) collapsing K host rounds into one.
Est. bundle: direct attack on the B=1 host floor that is ~98.6% of I2S/S2S
RTF; sizes 0.5–1 d each, K-step 3–5 d. Why now: the vLLM speech moat is
depreciating (their inline fast-path exists, default-off) — this is the
durable version of the speech lead.

**Retries (tried before, execution was flawed):**

**#16 Code2Wav sequence parallelism — run the serving A/B that was never run.**
Branch `code2wav-sp` (`c2ec12a` + parity `ce359b9`): frame-dim vocoder
sharding, halo=32, **bit-exact parity committed (cos=1.0000)** — then
rejected on an *isolated conv-stack microbench* (0.46–0.62×,
uncommitted `exp_code2wav/`) that structurally could not win: it dropped CUDA
graphs and added a cross-device copy to an unamortized standalone stack, i.e.
it measured the wrong thing (in-graph law violation, out-of-context law
violation). The commit itself says "INTERRUPTED before the parity gate and
the I2S/S2S A/B". Correct retry: e2e serving A/B, **B1/B2 long-audio cells
only** (at B≥8 both GPUs are already busy; SP is a latency lever, not a
throughput lever), measure RTF p50/p95 + TTFA, SP on/off.

**#17 Encoder-side bundle — three implemented-then-abandoned changes, zero
fair tests among them.** (a) `exp/async-audio-pipeline` (`4a2a3e1`,
MSTAR_ENCODE_PREFILL_OVERLAP): its own DESIGN doc says "has not been run on
GPU" — blocked by an unfixed EOS `torch.cat` crash. Fix the crash, test under
concurrent load. (b) `audio-encoder-opt` GPU log-mel (`d2ef987`,
MSTAR_GPU_MEL) and (c) adaptive varlen-selector recalibration (`796c2f3`):
microbench-only, and (c) silently overrode a previously-validated selector
without re-running that A/B. Correct retry: paired serving A/Bs on s2t/s2s
cells, encoders under the co-located contention they actually serve in.

**#18 Bucket geometry — W8 re-run + grid sweep.** W8 denser prefill buckets
(`exp/prefill-buckets`, `08e7123`) was "inconclusive standalone" because a
bursty foreign user forced its quick-bench onto a different GPU pair than its
reference — cross-pair drift made it unresolvable *by the docs' own account*,
and it was never re-run after the TTFT signal (B1 p50 −22%) looked real.
Correct retry: same-pair `dyn_ab` (buckets are capture-time, so two-server
`ab_*` alternation instead), i2t B1/B8/B32 + TTFT percentiles. Piggyback: the
live branch now exposes `MSTAR_DECODE_BUCKETS` (`submodules.py:1461`) — sweep
decode grid {[...,16,24,28,32]} vs {+20} vs {+26,30} and the mixed-bucket grid
(288/320/544) against live fold-size telemetry (WALK_STATS) rather than
guessing.

**vLLM-derived (source dive, beyond V1–V6):**

**#19 Pinned-staging + vectorized assembly rewrite (CpuGpuBuffer pattern).**
vLLM's `CpuGpuBuffer` (`v1/utils.py:109-137`): persistent pinned-CPU tensor +
GPU twin + `.numpy()` alias; per-step input prep is numpy slice-assigns into
pinned memory + one `copy_(non_blocking=True)`; token gather via
`torch.index_select` into the pinned buffer; block-table H2D kicked first to
overlap the CPU math (`gpu_model_runner.py:1882-1927`). M* has
`prepare_inputs_batched` (E3) but still materializes fresh tensors per step.
This is the concrete "how" for V3's "what" — implement as the staging layer
under prepare_inputs + sampling metadata + check_stop. Est. **0.5–1.5 ms/step
main-thread + removes residual H2D staging syncs**. Size: 2–4 d. Risk: low —
mechanical, shadow-comparable.

**#20 Control-plane transport pack.** SLIM/BATCH_EMIT coalesced api_server
token emits, but per-rid `WORKER_GRAPHS_DONE` to the conductor is still one
pickled ZMQ PUSH per rid (~32/step at boundary-heavy steps; construction at
`worker.py:1350-1369`, serial dispatch `conductor.py:1114`) — add a
`wgd_collector` mirroring `batch_collector` (`worker.py:2761`) + a batch
branch looping through `_process_worker_graphs_done` (`conductor.py:840`).
Swap conductor/api control messages from pickle to msgspec (vLLM
`serial_utils.py` pattern; ZMQ IO threads release the GIL). Dropped from
scope after debate: zero-copy tensor frames (audio already rides SHM;
SLIM_EMIT already killed the token-path pickle). Est. **~1–1.5 ms of the
3.4 ms send at prefill/loop boundaries + conductor CPU**. Size: 1–2 d.
Risk: message ordering — preserve per-rid order and `partition_done`
(`conductor.py:1124-1128`).

**#21 In-graph fusion pack — microbench-gated.** M* runs RMSNorm and residual
adds as separate ops (`components/thinker.py:118-131`) and unfused rope
application; vLLM uses `fused_add_rms_norm`, in-place `rotary_embedding`,
`silu_and_mul` (`_custom_ops.py:323,412-428`, `activation.py:117-148`). QKV is
already merged in M* (`distributed/attention.py:102`) — that part is done.
Debate arithmetic says decode-graph saving is small (~0.2–0.5 ms/step; tensors
are tiny at B32, and graphs already amortize launches) but prefill/mixed
buckets move real bytes. **Gate: mb_split_attn-pattern in-graph microbench
must show ≥0.4 ms/step on the decode graph or measurable prefill-bucket wins,
else drop without an e2e cell** (law 7). Size: 1–2 d incl. microbench.

**#22 FlashInfer plan-reuse + GPU block table + spec-drop pre-plan coverage.**
Three related cuts on the gpu-thread's plan path: reuse the FlashInfer decode
plan between page boundaries instead of replanning per step
(`cache_manager.py:337`; ~0.75 ms/step); keep an append-only persistent block
table instead of per-step Python page-index rebuild
(`cache_manager.py:287-303,323-326` — vLLM keeps it GPU-resident with a
Triton slot-mapping kernel, `v1/worker/block_table.py:141-167,326-380`);
guarantee pre-plan coverage when speculation drops via a self-speculate
identity next-batch (`worker.py:3410,3422`) killing the occasional ~1 ms
inline-plan stall on the critical path. Est. **~0.5–1.5 ms/step gpu-thread**
— note gpu-thread cuts convert only after main-thread work shrinks (law 2),
so schedule after #13/#19/#20. Size: 2–3 d.

**#23 Per-request encoder routing to the idler GPU.** The measured placement
tension is fundamental: encoff wins s2t +31%/i2t +9% but costs s2s −22%/i2s
−4%; no static yaml wins all four paths, and EXPERIMENTS' own deployment
guidance names per-request routing as the fix that "would subsume both" —
never queued. vLLM sidesteps it by budgeting encoder work per-step inside one
engine (EncoderCacheManager); M*'s multi-worker layout can instead route each
request's encoder node to whichever GPU is idler at admission (scheduler-level
placement decision; encoder nodes are stateless per request). Win: keep the
encoff text numbers AND the default-yaml speech numbers in ONE config —
worth ~+9–31% on whichever paths the shipped static config currently
sacrifices. Size: 3–5 d (dynamic node→worker binding at walk instantiation).
Risk: medium — placement must be sticky per request; VRAM headroom on both.

**#24 MoE grouped-GEMM bake-off (in-graph).** Decode MoE = 76% of the 12.8 ms
B32 graph. The current Triton w8a8 block-fp8 kernel won its slot against
bf16 and w8a16 — but never against the current best grouped-GEMM
implementations (DeepGEMM/CUTLASS-grouped fp8, SGLang's kernels; vLLM's
modular stack at `fused_moe/modular_kernel.py:1265,1353` +
`moe_align_block_size` shows the no-host-sort/no-sync dispatch shape). A
10% kernel win = ~1 ms/step = ~5% e2e at B32, multiplied across every path.
Pure in-graph microbench first (bench_fp8_moe.py exists as the harness
skeleton), real layer-10 weights, decode M ∈ {8..128}: only a ≥8% in-graph
win earns an e2e cell. Size: 2–4 d exploration. Risk: low (kernel swap behind
the existing MOE_FP8 seam).

---

## Part 4 — The five validation rounds (what changed and why)

Format: each round is an argued pass over the then-current slate; ≥2
substantial changes per round, all grounded in evidence gathered during the
round. 15 changes total.

**Round 1 — fact-grounding (academic demands evidence; SWE runs git).**
- C1: KILLED miner's top retry "re-apply W7 buckets" — `git branch --contains`
  proves 845faff shipped in `opt/decode-v2` (merge `d197dbd`) and is
  default-on env-tunable on the live branch; the miner had misread the stale
  code snapshot on the *docs* branch (`c865770`) as a shipped engine branch.
  Replaced by #18 (W8 prefill buckets — the bucket item that really was lost
  to bad measurement).
- C2: KILLED "sampler all-greedy/no-penalty host gating" (vLLM candidate) —
  M*'s fused sampler kernel already one-hots greedy rows in-kernel
  (`sampling.py:58-102`), and the benchmark workload samples at temp>0, so
  the batch-level gate would never fire. Slot refilled by the control-plane
  transport idea (→ #20 after C9).
- C3: KILLED "guard-free torch.compile of the backbone" — M*'s decode,
  prefill, and mixed steps are all already CUDA-graph-captured; the compile
  win vLLM gets is glue-code elimination M* must win by removing graph-walk
  Python instead (that IS #13/#19/#20); encoders are <1% of B=1 e2e per
  FINDINGS. Slot refilled by per-request encoder routing (#23).

**Round 2 — overlap purity & scope (SWE attacks duplication with the old board).**
- C4: KILLED retry "E8 captured side-stream prefill" — it targets the same
  prefill-stall as V2 (budgeted interleave policy, already board-resident)
  with strictly more risk (cuda_graph_runner surgery, per-slot static
  buffers) and its predecessor E7 collapsed for reasons (eager+GIL) that V2's
  approach doesn't share. W5's captured-mixed machinery + V2 policy is the
  right vehicle. Slot refilled by the encoder-side retry bundle (#17).
- C5: MERGED the two vLLM host-staging candidates (pinned CpuGpuBuffer +
  vectorized numpy assembly) into one rewrite (#19) — they are one change at
  implementation level. Freed slot → MoE grouped-GEMM bake-off (#24), which
  attacks the single largest GPU-time block (76% of the decode graph) and was
  missing from both boards.
- C6: RESCOPED K-step talker unroll from K=25 (chunk-sized, high EOS waste)
  to K=5 with EOS-trim as the entry point.

**Round 3 — mechanism & measurement (academic: "which measured milliseconds
does this remove?").**
- C7: RESCOPED "GPU-resident block table" — standalone it doesn't hit a
  measured cost; merged with FlashInfer plan-reuse + spec-drop pre-plan
  coverage into #22, which maps 1:1 onto the measured plan/prepare slice of
  the gpu-thread's 11.1 ms.
- C8: RESCOPED Code2Wav SP retry (#16) to B1/B2 long-audio latency cells
  only — at B≥8 both GPUs are saturated; frame-dim sharding cannot win
  req/s there and pretending otherwise would burn the A/B on a doomed cell.
- C9: CONCRETIZED the transport option to batched-WGD + msgspec (#20) and
  DROPPED zero-copy ZMQ tensor frames from its scope — token emits already
  ride SLIM_EMIT and bulk tensors ride the SHM protocol; the remaining
  measurable cost is message *count* and pickle CPU, not copies.

**Round 4 — effect-size law & sequencing (SWE: "none of these survive a lone
e2e cell").**
- C10: RESTRUCTURED the scheduler micro-cuts (ready index, peek backoff,
  dup-work cuts, completion-sync deletion) into ONE flag-gated pack (#13)
  with per-cut WALK_STATS counters — individually each is <2% (law 7:
  microbench or arithmetic, never lone e2e cells); as a pack it is one
  A/B-able ~1.5–3.5 ms claim, and the counters keep per-cut attribution.
- C11: SPLIT NUMA (#14) into harness-fix-first then in-process pinning, and
  PROMOTED the harness fix to Phase 0 — it de-risks every subsequent
  measurement on pair 0,1 and finally quantifies the "+10%" folklore.
- C12: DEMOTED the fusion pack (#21) to microbench-gated — round arithmetic
  (48 layers × tiny decode tensors, launches already graph-amortized) puts
  the decode-graph ceiling at ~0.2–0.5 ms/step; it earns an e2e cell only if
  the in-graph microbench clears 0.4 ms/step or shows real prefill-bucket wins.

**Round 5 — portfolio balance & durability (academic: "the queue over-fits
i2t B32 today; what about B2/B4, and what survives vLLM 0.23?").**
- C13: SWAPPED the K-step talker graph out of the top slots for the speech
  host-floor bundle (#15, with K-step as escalation) — three ~1-day cuts with
  independent WALK_STATS-verifiable mechanisms beat one 5-day graph surgery
  as the first strike on the same host floor; and the speech moat is
  depreciating (vLLM's inline fast-path exists, default-off), so cheap-now
  beats big-later.
- C14: ATTACHED a mandatory rider to V2 (existing board): when the budgeted
  interleave policy lands and fold volume rises, re-test
  MSTAR_MIXED_SPLIT_ATTN + MSTAR_MIXED_PREPLAN as default-on — their parked
  verdict was explicitly fold-volume-limited ("flips positive if fold volume
  rises"); the +4.4% mixed-step win is otherwise left on the table.
- C15: ADDED mid-batch diagnosis as a REQUIRED Phase-0 deliverable — the
  committed data shows i2t B2 0.75× / B4 0.83× and v3's s2t B2–B16 cells were
  never measured; no option on either board owns this region. Diagnose before
  assigning (suspects: admission/spec-chain behavior at tiny concurrency,
  bucket padding at bs 2–4, per-step fixed host cost amortization).

---

## Part 5 — Execution plan

### Principles (inherited, non-negotiable)
Adjacent interleaved A/B pairs or ≥3 rounds only; i2t:32 sentinel cell in
every coverage run; tok/req ≈ 177 correctness sentinel; mechanism-alive
verification (WALK_STATS/sync counts) before believing any e2e verdict;
effect-size gate (<2% predicted → microbench/arithmetic only); in-graph
microbenches only; idle-gate + hard timeouts + cleanup traps per workspace
CLAUDE.md; one branch per experiment, commit every valid run on its bench
branch, merge to `benchmarks` immediately.

### Harness upgrades (Phase 0, before any measurement)
1. **NUMA-correct binding** (#14a): `quick_bench.sh` / `dyn_ab.sh` derive
   `--cpunodebind/--membind` from the GPU indices instead of hardcoding
   node 1. Then a one-off numactl-A/B on pair 0,1 (same server, node 0 vs
   node 1 binding, i2t:32 ×3 rounds) finally *measures* the cross-NUMA
   handicap instead of folklore-correcting by +10%.
2. **dyn_ab-first discipline**: every new flag in #13–#24 must register in
   the MSTAR_DYNFLAGS refresh path on day one (the fbe4b1c pattern), so its
   A/B costs one server start. Capture-time knobs (buckets, colocation,
   routing) are exempt → two-server `ab_*` alternation.
3. **Per-option WALK_STATS counters**: each pack sub-item ships with a
   counter proving the mechanism fired (folds, cache hits, batched-WGD count,
   pinned-buffer reuse count). No counter, no verdict.
4. **make_numbers.py stays the aggregation path**; every claimed number in
   docs must come from a committed results.json (the 6.894 lesson).

### Phase 0 — measurement debt + doc hygiene (0 new perf code, ~1–2 days)
Gate: canonical numbers exist; docs stop lying.
- Canonical 6,7 sweep of the final stack when the pair frees (the single
  most important missing number: i2t B32 canonical). Until then, pair 0,1
  with corrected NUMA binding.
- Re-seal the cache+checkstop delta (validator: reproducible evidence says
  ~+6.8%, doc says +10.5%) — 3 adjacent pairs, dyn_ab, i2t:32.
- Mid-batch cells: i2t B2/B4/B16 + s2t B2/B4/B16 on the final stack (v3 has
  holes there) + WALK_STATS/nsys diagnosis of the B2/B4 loss (C15).
- N2 re-smoke (conductor poll fix b1c1ff1) — TTFT-focused cells; while there,
  apply the durable fix (move `wait_for_work` at `conductor.py:1162-1163`
  inside the try/except rather than relying on the fd guard).
- Doc fixes: commit the real EXPERIMENTS.md + BEATING_NEW_VLLM.md to the docs
  branch; fix E6 header, the 6.894/"+10.5%" scoreboard lines (quote committed
  numbers, mark projections as such), the stale W1/W7 queue entries, the
  `sampling.py:207` comment; note the s2s B8 coverage regression as open.

### Phase 1 — main-thread floor, cheap wins (~1 week)
Rationale: law 2 — every main-thread ms both adds throughput AND widens the
shade budget that V1 and the gpu-thread cuts need.
1. **#13 SCHED_PACK** (slice order: peek-backoff + dup-work cuts first — ~1 d,
   dyn_ab triage; full ready-index behind shadow-verify — +2 d).
2. **#20 transport pack** (batched WGD first; msgspec second).
3. **#14b thread pinning** (after the Phase-0 harness fix; A/B is one dyn_ab
   flag flip since pinning can be runtime-applied at flag refresh).
4. **N1-full remainder** (existing board) rides along with #13's counters.
POC per item: QB_FAST dyn_ab smoke (i2t:32,s2t:8) → full-n dyn_ab ×3 rounds →
promote into the stack only on geomean ≥+2% with sentinel in-band.
Expected compound: main thread 13.2 → ~9–10 ms/step; i2t B32 +8–15% on top of
current — approaching/at vLLM parity on the honest 0.80× baseline.

### Phase 2 — the structural swing (~1–2 weeks)
1. **V1 async scheduling** (existing board, full design in task #33): submit
   replay N+1 before host bookkeeping; sampler already sync-free post-N3;
   D2H on a copy stream consumed one step late; E9 direct-feed already on the
   branch. This is vLLM's remaining structural edge (their
   `async_copy_ready_event` pattern). Est +5–10% standalone, more as
   main-thread shrinks.
2. **#19 staging rewrite** — feeds V3 and de-risks V1 (fewer per-step
   allocations to reason about when steps overlap).
3. **#22 plan-reuse pack** — schedule strictly after Phase 1 (its wins live
   on the gpu-thread; law 2 ordering).
POC: V1 lands behind a flag with a 3-arm dyn_ab (off / V1 / V1+cache) — the
R2 lesson (E10 was tested pre-every-win and self-handicapped) says re-test
the arm combinations, not just the new flag.

### Phase 3 — GPU-time and geometry (parallel with Phase 2 on free days)
1. **#24 MoE bake-off** — in-graph microbench only until a kernel clears +8%;
   then one dyn_ab-able swap.
2. **#21 fusion pack** — run its gate microbench in the same session as #24
   (same harness setup); drop without ceremony if <0.4 ms/step.
3. **#18 bucket geometry** — two-server ab_* alternation (capture-time), one
   afternoon per grid point; use live fold/batch-size telemetry to pick grid
   candidates instead of sweeping blindly.
4. **R2/R3** (existing board re-tests) fill scheduler gaps between cells.

### Phase 4 — speech durability + placement endgame (~1–2 weeks, interleaved)
Motivation: the speech moat is real but perishable (vLLM's inline fast-path
is one config flip away), and speech is where M* ships value today.
1. **#15 speech bundle**: talker fast-checkstop → colocation yaml →
   chunk-batched codec edge; A/B on s2s:8, i2s:8 + RTF/TTFA; escalate to
   K-step talker graph only if the bundle converts and the residual per-frame
   host floor still dominates.
2. **#16 Code2Wav SP** serving A/B (B1/B2 long-audio, RTF p50/p95 + TTFA).
3. **#23 per-request encoder routing** — the config-unification endgame;
   ship criterion: one config ≥ max(encoff, default) on all four paths.
4. **#17 encoder retries** — opportunistic, lowest priority in the phase.

### Prioritization logic (why this order)
1. Phase 0 first because every later verdict inherits its trustworthiness —
   and because two advertised numbers (canonical B32, +10.5%) are currently
   unbacked; a campaign claiming rigor cannot carry those.
2. Main-thread before gpu-thread (law 2: shade budget) — #13/#20/#19 before
   V1/#22/#24 conversions are believable.
3. Cheap-reversible before structural: everything in Phase 1 is
   dynflags-flippable; V1 is the only item whose failure costs >3 days.
4. Speech last not because it matters less but because it's currently WON —
   the marginal req/s per engineering-day is higher on i2t, and Phase 4's
   items are latency/durability plays whose value doesn't decay in a week.
5. The 12 old + 12 new options are a menu, not a contract: every phase gate
   re-ranks the remainder against the freshest profile (the night of 07-03
   proved the profile moves — sampler-cache flipped from regression to win
   when the main thread lightened).

### Sizing summary
| # | Option | Size | Predicted effect | Kill criterion |
|---|---|---|---|---|
| 13 | SCHED_PACK | 2–3 d | 1.5–3.5 ms main-thread | shadow-verify mismatch, or pack <+2% e2e |
| 14 | NUMA fix+pin | 0.5+1.5 d | harness validity + µs–2 ms | pin A/B flat ×3 rounds |
| 15 | speech bundle | 2–3 d (+3–5 K-step) | RTF/TTFA at B1–B8 | per-cut counter shows no fires |
| 16 | Code2Wav SP A/B | 0.5 d (code exists) | B1/B2 RTF p50 | <5% RTF at B1 long-audio |
| 17 | encoder retries | 2–3 d | TTFT s2t/s2s | crash unfixable in 1 d → drop (a) |
| 18 | bucket geometry | 1–2 d | TTFT + B32 padding | no grid beats current ×2 rounds |
| 19 | staging rewrite | 2–4 d | 0.5–1.5 ms main-thread | counters show reuse but e2e flat after Phase-1 landed |
| 20 | transport pack | 1–2 d | ~1 ms boundary steps | WGD count unchanged (dead flag) |
| 21 | fusion pack | 1–2 d | 0.2–0.5 ms (gate) | microbench <0.4 ms/step |
| 22 | plan-reuse pack | 2–3 d | 0.5–1.5 ms gpu-thread | flat with Phase 1 landed (re-park for post-V1) |
| 23 | encoder routing | 3–5 d | unify +9–31% config split | any path < static best ×3 rounds |
| 24 | MoE bake-off | 2–4 d | ~5% e2e if kernel +10% | no kernel ≥+8% in-graph |

### GPU logistics
Pairs 0,1 usable immediately (idle now) once #14a lands; canonical 6,7 for
committed numbers when the foreign job clears; never co-locate (idle gate
ABORTs are final); hard `timeout` on every server (5400 s pattern in
quick_bench); cleanup trap + compute-apps verification after every run; no
clock locking on this shared box (record unlocked in env.txt). All artifacts:
one dir per benchmark under the existing layout, commit on the bench branch,
immediate merge to `benchmarks`, never on main.

### Git topology for the new work
- One branch per option off the live base: `opt/sched-pack`, `opt/cpu-pin`,
  `opt/speech-floor`, `retry/code2wav-sp-e2e`, `retry/encoder-bundle`,
  `retry/bucket-geometry`, `opt/staging`, `opt/transport`, `opt/fusion-pack`,
  `opt/plan-reuse`, `opt/encoder-routing`, `opt/moe-bakeoff`.
- Winners merge into the shipping candidate branch after their full-n A/B;
  the stack re-freezes and NUMBERS_V4 sweeps on the canonical pair.
- Docs (EXPERIMENTS.md updates per verdict — including failures) commit on
  the docs branch same-day as the verdict. The knowledge base being
  uncommitted was this campaign's one process failure; don't repeat it.

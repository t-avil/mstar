# Current Test Roadmap

## Top 6 levers for text-generation (decode) throughput

Leaving out the docs' own #1 flagged priority (async encoder-prefill for i2t B32
TTFT) since that's a vision-encoder/TTFT fix, not a decode-throughput one. Ranked by
strength of evidence and how load-bearing our own notes call them.

### 1. Batched in-graph decode loop — fix `MSTAR_MULTISTEP_DECODE`
Run multiple decode steps inside one captured graph instead of one CPU dispatch per
step. The first attempt regressed B32 ~3.5× from inline re-plan cost, but output
stayed greedy-identical, so the mechanism is sound — the re-plan just landed on the
critical path. Our docs call a correct version of this "the load-bearing path to
further B32 gains."
**Test:** move the re-plan cost off the per-step path, rerun the B32 sweep.

### 2. Async sampled-token materialization — `DEFER-SAMPLE` [IN-FLIGHT]
Overlap the sampled-token-ID D2H copy with the next step's compute.
**Caveat:** double-check what #4476 actually does before finishing this (see watch
list) — the citation may point to a different mechanism than the D2H overlap
described here.

### 3. Decouple per-step output/payload construction from the decode step
Same bottleneck class as #2, called out separately because vLLM-Omni just published
hard numbers on it: moving payload assembly off the decode path shrank their Talker
inter-step gap from ~2.8ms to ~41µs, and that change was their **second-largest
throughput lever in the whole stack**, behind only CUDA graph capture. Different
pipeline (their number is for Talker/speech, not Thinker text-gen), but the mechanism
transfers directly, and it's basically what our Tier-S "FlashInfer attention-plan
double-buffering" idea is reaching for.
**Test:** profile how much of the decode-step wall time is currently non-GPU
(plan-build, payload construction, emit) before implementing, so we know the ceiling
is worth it.

### 4. CUDA-graph-captured side-stream prefill — fix `MSTAR_SIDE_PREFILL`
Same idea already tried, graph-captured instead of eager. The eager version won at B1
(1.055×) before collapsing at B32 (0.098×): GIL plus eager kernel launches starved
the decode chain. The win is real; the eager implementation is what killed it at
scale.

### 5. DP-replica Thinker with dedicated encode/decode routing
Round-robin DP replicas already win decode/tok-s big (ITL 3–6ms, +16% tok/s at
len-512); the problem is encode and decode contending for the same 2 GPUs, tanking
short-output req/s. This is a validated pattern, not a guess: vLLM-Omni's own scaling
approach adds replica capacity only to the stages that saturate rather than the whole
pipeline, and that pushed their throughput to the highest point in the sweep, with the
margin widening as concurrency climbs. Dedicating replicas to encode vs. decode
instead of round-robin should get the same effect.

### 6. Per-batch routed-expert bookkeeping
Named as one of 5 specific reasons vLLM hits its 8.21 req/s number — a known,
identified gap.
**Note:** only 1 of the 5 reasons is captured here; if the other four are written
down elsewhere, pull them in too.

---

**Grouping / sequencing notes**
- **2 and 3** are the same bottleneck class (host-side work blocking the next decode
  step) — implement/test together.
- **1** has the biggest ceiling but also the trickiest history.
- **5 and 6** are the most contained if you want a fast isolated win first.

---

## Progress log (autonomous loop)

**Target (committed):** only losing text-gen cell = **i2t B32** (M* 7.42 vs vLLM 8.32 req/s, −11%);
s2t B2 marginal (−4%, noise). Live best-build baseline this session: i2t B32 = **7.77** (need 8.32).
Live i2t B32 breakdown: JCT 3735ms, **TTFT 2333ms = 62% of JCT**, ITL 7.93ms → cell is PREFILL-bound.
Win math: need JCT ≤3488 (−6.6%) = ~10% TTFT cut OR ~17% ITL cut.

**Lever #1 (multistep decode) — TESTED, NEGATIVE for i2t B32.** Same-server dynflag A/B on
mstar-multistep (re-plan fix already on branch): OFF avg 5.50 vs ON avg 5.10 req/s (wash/slight
regress). Multistep amortizes decode CPU floor, but i2t B32 is prefill-bound, so no help. Skip
porting to best build. (Would only help decode-bound cells.)

**KEY PROFILE (best build godv9, i2t B32 decode step, MSTAR_PHASE_TIMING):**
iter_total ~8.4ms mean, but **await_gpu ~0.54ms (GPU idle ~94% of the decode step)**;
submit_spec ~2.44ms + ~5.3ms unaccounted (postprocess/sample/emit). Decode is ~94% HOST-bound
even on the fully-optimized build → validates roadmap thesis; big headroom for #2/#3 (strip
host work off per-token path). Same saturated GIL thread also can't service decodes during the
prefill mega-walk → explains the 2333ms i2t B32 TTFT. Next: implement #2 (defer sample D2H) +
#3 (decouple payload construction), A/B at i2t B32 (need JCT 4038->~3488, ~17% ITL or ~10% TTFT).

**Lever #2/#3 investigation — decode host floor is SCHEDULER-CRITICAL, not freely decouplable.**
worker.py:3355-3427 route/register (mark_node_complete, process_node_outputs, _register_outputs,
store_and_populate_graph_edges_fast) drives per-rid walk advancement the NEXT step reads → can't
thread it off. Client emit is ALREADY collapsed to one sidecar send (MSTAR_EMIT_SIDECAR). So #3's
clean decouple captures ~nothing on godv9; the ~5.3ms is irreducible per-rid (x32) scheduler
bookkeeping. Only MULTISTEP (one postprocess per N GPU steps) amortizes it. Lever #2 (defer
sample D2H) = small ceiling (already batched/deferred) + identity risk (same-step stop). 
=> Re-testing multistep MECHANISM on a decode-bound cell (forced long output) to decide if
completing steps A/B (hoist per-micro-step Python) + porting to godv9 is worth it.

**Lever #1 step-A (hoist invariant per-micro-step Python) — IMPLEMENTED (mstar-multistep
cuda_graph_runner.py, byte-identical penalty-off), but STILL negative:** decode-bound B8/512-tok
OFF 3.19 req/s ITL 9.45ms vs ON 2.90 req/s ITL 10.25ms. Hoist saved ~0.25ms; dominant per-micro-step
cost is preprocess(plan_rope CPU rebuild)+embed-pack+static-copies+serial GPU replays. Real ceiling
needs step C (full in-graph N-step capture) = major refactor. DEPRIORITIZED: even perfect multistep
only touches i2t B32's 38% decode share.

**DECISION: decode levers (#1-3) blocked/exhausted for i2t B32 (the only losing cell).** #1 negative,
#2/#3 can't decouple (emit collapsed, host floor scheduler-critical). i2t B32 is 62% TTFT-bound.
Pivot to the prefill mega-walk (one-walk-per-batch: 32 vision-encode+prefill run synchronously,
freezing decodes -> TTFT 2333ms vs vLLM 179). godv9 HAS MIXED_BATCH+CHUNKED_PREFILL_V2 but food101
prefills are <=256-tok (unchunkable) so they don't interleave. Investigating a scheduler change to
spread/interleave the 32 short prefills with decode (vLLM token-budget style) = real i2t B32 fix.

**Prefill-scheduler investigation (agent) — i2t B32 gap is STRUCTURAL, not a scheduler cap.**
Prefill already runs bs=1 (submodules.py:1264-1270); MIXED_BATCH bails (prefill_multimodal not a
mixable walk + short prompts carry no chunk metadata, micro_scheduler.py:250-261/281-283). Real
serializer = per-request ENCODE READINESS: ~1 prefill ready per cycle on single process/GPU. ALL
scheduler levers falsified (admit-jitter, prefill-gather, side-prefill, batch-vision, mixed-fold =
wash/negative). Flagship = documented local optimum; only un-falsified lever = more prefill
PARALLELISM. => Testing lever #5 DP=2 (two encode/prefill pipelines) as the direct fix for the
encode-readiness serialization; config-only, targets the exact bottleneck.

**Lever #5 (DP=2, dp2a Variant A) — TESTED, NEGATIVE for i2t B32.** 6.9 req/s (TTFT 3780ms) vs
single-topo baseline 7.77 (TTFT 2333) vs vLLM 8.32. DP improves ITL (5.6-6.5 vs 7.93ms, 2 decode
threads) but each replica's own-encoder-on-same-GPU-as-decode makes compute-bound vision-prefill
SLOWER per replica -> TTFT worse. Wins ITL/decode-bound, loses prefill-bound short-output i2t B32
(matches memory). DP inductor cache now warm at inductor_cache_dp.

## CONVERGENT CONCLUSION (all levers tested/reasoned)
i2t B32 is the ONLY losing text-gen cell (7.77 vs 8.32, -6.6%). It is 62% TTFT/prefill-bound.
- #1 multistep: negative (per-micro-step Python > amortization; step-A hoist didn't flip; needs
  step-C in-graph = major refactor; only touches 38% decode share).
- #2/#3 decode decouple: emit already collapsed (sidecar); host floor is scheduler-critical per-rid
  bookkeeping -> can't decouple. Nothing to gain on best build.
- #4 prefill interleave/side-prefill: prefill already bs=1; real serializer = per-request encode
  READINESS (~1/cycle). ALL scheduler levers falsified historically (admit-jitter, prefill-gather,
  side-prefill, batch-vision, mixed-fold). Flagship = documented local optimum.
- #5 DP: negative for i2t B32 (encode contends with decode per replica).
- #6 routed-expert: MoE/decode optimization -> wrong bottleneck for prefill-bound cell.
The ONLY un-falsified lever that increases prefill readiness = MSTAR_ENCODER_ASYNC (off-stack,
measured -30% TTFT BUT s2t -18%; user EXCLUDED it as out-of-scope). Gap (7.77 vs 8.32) is also
WITHIN the documented +-18% boot-lottery band. Decision needed: (A) authorize encoder-async +
mitigate/gate s2t, (B) accept i2t B32 as within-noise near-tie, (C) deeper arch change (multi-thread
the single GIL process). No in-scope decode-throughput lever can flip it.

## DECISION (user, this session): pursue the DEEPER ARCHITECTURAL CHANGE.
Root cause = single GIL-bound Thinker host thread: decode step 94% host-bound (submit_spec 2.44ms +
per-rid x32 route/register 5.3ms, GPU only 0.54ms); same thread serializes encode readiness -> i2t
B32 TTFT 2333ms. Python ThreadPool won't help (GIL). Real fixes: (A) GIL-releasing C++/nanobind
extension for the per-rid route/register hot loop, (B) multi-process workers per GPU (shared weights),
(C) separate encode-readiness pipeline off the decode GIL, (D) free-threaded Python. Next: architect
design of the minimal-viable first slice (risk/effort/gain), then staged prototype + A/B.

**ARCHITECTURE DESIGN (agent) — recommendation: (C) split ENCODE into its own process.**
Attacks the ONLY losing cell (i2t B32 TTFT) by de-serializing encode readiness; isolates (not
replicates) so avoids DP-worse; disaggregation routing already exists (worker.py:2373-2374,
rope.py:207-208). Ranked (C)>>(D)free-thread~(B)IPC-decode-split>>(A)native-ext. Per-rid postprocess
IS embarrassingly parallel (disjoint rid-keyed state, CUDA-free on hot path) but blocked by GIL +5
shared surfaces (container dicts, scheduler-readiness publish, ZMQ sockets, ref-counts, CUDA events).
MVP DISCRIMINATOR (1-day, zero/low code): "drain-all-ready-encoders-on-yield" + tune
MSTAR_MAX_CONSECUTIVE_SPEC_STEPS (default 1024) / SPEC_PEEK_FOR_FAIRNESS. If TTFT drops -> limiter is
scheduling POLICY (cheap win). If not -> true GIL saturation -> justifies (C) encode-split. Guardrail:
decode ITL must not regress.

**Spec-yield discriminator (MSTAR_MAX_CONSECUTIVE_SPEC_STEPS 1024/16/4, same-server dynflag A/B at
i2t B32) — NO systematic effect.** TTFT 1937/2788 (1024), 2958/2183 (16), 1818 (4): within-setting
variance swamps between-setting. => limiter is NOT scheduling policy; it's TRUE GIL SATURATION.
Confirms (C) encode-split is needed. (spec-yield knobs made dynflag-tunable in godv9 worker.py,
byte-identical.) NEXT: C1 = dedicated-encode CONFIG (encoders on own rank/process/GPU + own GIL,
Thinker-decode on the other) — cheapest form of the encode-split; A/B i2t B32 vs 7.77 baseline.

**REFINED FINDING — (C1) encode-split is ALREADY the shipping config.** The baseline `encoff` yaml
puts audio+vision encoders on rank0 (own process/GPU/GIL), Thinker on rank1. Encode ALREADY runs
parallel to decode. Yet i2t B32 TTFT=2333ms. Math proves encode is NOT the serializer: 32 serial
encodes @28ms = 896ms << 2333ms. The real serializer = **prefill_vision (first token) runs on
rank1's Thinker GIL, competing with the 94%-host-bound decode postprocess for the single GIL thread**
(~73ms effective prefill+queue/req). => The (C) "move encoder to its own process" fix is DONE and
INSUFFICIENT. The remaining architectural fix must reduce rank1's decode-postprocess GIL load (A:
native GIL-releasing ext for per-rid route/register, or D: free-threaded CPython) OR split prefill to
its own GIL sharing Thinker weights+KV via IPC (B-variant) — all MULTI-WEEK. User rejected async-encoder;
chose GIL architectural change. Needs scope confirmation before committing weeks to a native ext
(torch==2.9.1 pinned for sgl-kernel complicates fresh native builds).

## DECISION (user): build the NATIVE GIL-RELEASING EXTENSION (A).
Target: move the per-rid route/register loop (worker.py:3331-3399: store_and_populate_graph_edges_fast,
mark_node_complete, process_node_outputs, set_output_ref_counts) off the GIL so rank-1 can service
prefill_vision (first token) concurrently -> cuts i2t B32 TTFT + ITL. Approach: C++ persistently owns
per-rid POD state, parallel_for with GIL released, publish readiness back to Python atomically per step.
Milestones: M0 build-toolchain POC (nanobind + GIL-release parallel_for builds/imports vs torch 2.9.1)
-> M1 port memoized replay POD patch -> M2 own the containers -> M3 wire + A/B i2t B32.

**M0 (native-ext toolchain POC) — DONE/SUCCESS.** pybind11 3.0.4 + g++ 11.5 builds a native ext in
the mstar-new venv (torch 2.9.1 cxx11abi=True); GIL-released std::thread parallel_for = 7.91x on 8
threads. POC at /m-coriander/coriander/tim/gilext_poc/. KEY: marshaling (Py->C++ convert) is GIL-held
and dominated a memory-bound test -> design MUST minimize marshaling (extract POD once, work GIL-free).
Next M1: port memoized replay (_replay_route_plan/_replay_populate_plan, tensors.py:598-712 = POD
struct + int/ptr patch) to C++, measure marshaling vs compute; if marshaling-bound, must have C++ own
the containers persistently (M2).

**M1 finding — hot loop is Python-object-ENTANGLED, no quick POD slice.** _FastPopulatePlan
(communication/tensors.py:76-97) holds refs to live Python GraphEdge objects and mutates
edge.tensor_info every step (needs GIL). Route/register (node_manager_utils.py:451 process_node_outputs,
234/430 mark_node_complete, 691 _replay_route_plan) all manipulate GraphEdge/WorkerGraphIO Python
objects. => a partial port is GIL/marshaling-bound (per M0 lesson). A real win requires C++ to OWN the
tensor/edge state (per_req_tensors[rid][uuid]->tensor+refcnt is the most POD-like: TensorStore,
tensors.py:102-145). M2 = prototype C++ TensorStore (unordered_map<rid,unordered_map<uuid,TensorRef>>),
run put/get/refcount GIL-released, measure vs Python; then extend to the edge state. Multi-week; the
user committed to this path. NOTE: this is the crux — if even TensorStore-in-C++ is marshaling-bound,
the whole (A) approach is net-neutral (architect's warning) and we'd revisit.

**M2 (C++ TensorStore + GIL-released per-rid step) — DONE, mechanism VALIDATED.**
tstore_poc.cpp: state owned in C++ across steps, only new-tensors marshaled in / ready-rids out.
At 32 rids/step: C++ 1-thread 5.2x faster than Python (interpreter overhead), C++ 8-thread 17.6x
(GIL-release working, NOT marshaling-bound). Scaled to the real 5.3ms floor -> ~0.3ms. => native
ownership pays off IF the per-rid work is POD. CAVEAT (M1): real route/register mutates Python
GraphEdge objects; achievable win = the POD-portable FRACTION of the 5.3ms. 
M3 = measure that fraction: instrument the real _postprocess_batch route/register (worker.py:3331-3399)
to split TensorStore-ops (uuid->tensor bookkeeping, POD-portable) vs GraphEdge/WorkerGraphIO traversal
(needs owning graph state too). Then port the portable core to a C++ TensorStore replacing
communication/tensors.py behind a flag (MSTAR_NATIVE_TSTORE), wire into worker, A/B i2t B32 (TTFT+ITL).

**M3 (native TensorStore) — BUILT + interface PASS, but refcount ops are MARSHALING-BOUND (0.87x
8-thread).** native_tstore.cpp: full TensorStore interface (put/get/remove/refcount/metadata) + GIL-
released batch refcount. Refcounts are too TRIVIAL to parallelize — bucketing the batch dominates.
=> the cheap slice does NOT deliver. KEY REFINEMENT: the real 165us/rid cost is Python-INTERPRETER
overhead in the GraphEdge/WorkerGraphIO traversal (mark_node_complete node_manager_utils.py:234/430,
process_node_outputs :451, _replay_route_plan :691). M2 showed C++ eliminates ~5x of that (single-
thread, no parallelism needed) — but requires porting the graph-state MACHINE to C++ bit-identically
(the architect's "very large surface", 15-30d). This IS the crux/core, unavoidable for the win.
M4 = build C++ WorkerGraphIO/GraphEdge state model + port mark_node_complete/process_node_outputs;
validate bit-identical vs Python on recorded step traces; then wire behind MSTAR_NATIVE_GRAPH + A/B.

**M4 slice 1 (C++ GraphEdge + edge classification) — BUILT + BIT-IDENTICAL (2000 trials).**
native_graph.cpp: GEdge POD struct (name/next_node/flags/handle) + classify_edges (streaming/
non_streaming/to_conductor/new_token buckets = process_node_outputs lines 483-489). Payload stays
in NativeTensorStore (opaque handle). Validated bit-identical vs Python mirror.
REMAINING M4 core (multi-week): slice2 = sharding_config.fanout_graph_edges (TP fanout) in C++;
slice3 = WorkerGraphIO.process_new_inputs (per-rid queue ready/waiting state machine,
node_manager_utils.py) in C++; slice4 = mark_node_complete; then assemble the full
process_node_outputs, validate bit-identical on RECORDED step traces, wire behind MSTAR_NATIVE_GRAPH,
A/B i2t B32 (TTFT+ITL). Foundation proven; this is the bit-identical routing-machine port grind.
Modules so far: /m-coriander/coriander/tim/gilext_poc/{gilext_poc,tstore_poc,native_tstore,native_graph}.cpp

**M5 (libtorch C++ ext + _plan_matches port) — DONE, BREAKTHROUGH for fast results.** libtorch ext
builds via torch.utils.cpp_extension in 28s vs pinned torch 2.9.1 (the big build blocker CLEARED).
plan_matches (tensor shape/dtype/stride/device checks, tensors.py:643-672) = 4.62x faster in C++
(60->13ms/12800), NET-POSITIVE incl. marshaling (unlike trivial refcounts M3). torch_meta_poc.cpp.
=> REFINED FAST-RESULT PLAN (not the multi-week full GraphEdge rewrite): port the bounded, net-positive
store_and_populate_graph_edges_fast HOT REPLAY (_plan_matches + _replay_populate_plan tensor patching,
~128 tensor-ops/step) to a libtorch ext behind MSTAR_NATIVE_POPULATE; shadow-validate bit-identical on
live traffic; A/B i2t B32 (ITL/TTFT). This is a few-iteration MEASURABLE win, not weeks. Agent porting
the WorkerGraphIO queue state machine in parallel. Modules: gilext_poc/{...,torch_meta_poc}.cpp

**AGENT RESULT — WorkerGraphIO queue state machine ported to C++ (native_wgio.cpp), BIT-IDENTICAL
4000 trials/94,671 comparisons.** Micro-bench (32 rids x 8-layer AR graph): Python 2280ms -> C++
1-thread 36ms (63x interpreter-elim) -> C++ 8-thread 9.1ms (250x, near-linear GIL-released). Ported:
process_new_inputs, mark_node_complete, ingest/ready/loop/registry state (int node-id+edge-handle
keyed, POD, disjoint per-rid). Stays Python: tensor_info/refcount, loop output cache, sharding fanout.
Graph structure handed to C++ ctor once at admission. test_wgio.py validates. => the per-rid postproc
components ARE portable at huge speedup. Modules now: native_wgio (63-250x), torch_meta_poc (4.6x),
native_graph, native_tstore, tstore_poc, gilext_poc. REMAINING = INTEGRATION: wire native_wgio into
godv9 worker behind MSTAR_NATIVE_WGIO in SHADOW mode (run both, assert equal, use Python) -> flip ->
A/B i2t B32. This is the path to the measurable win.

**INTEGRATION (agent) — native_wgio wired into godv9 worker behind MSTAR_NATIVE_WGIO, shadow-mode.**
native_wgio_bridge.py + additive node_manager_utils.py refactor (12 methods -> dispatcher + verbatim
_py_* bodies; default-OFF = 1 bool check, byte-identical). Offline test_native_bridge.py: 10513
matches/0 mismatches incl. decode AR-loop. Flag: unset=off(no-op), shadow=run both+assert+use Python,
1=C++ for loopless walks. CRITICAL LIMITATION: native(=1) DECODE LOOP (thinker_decode) falls back to
Python (loop re-injects external-input GraphEdges w/ runtime tensor_info the int-keyed C++ can't yet
reconstruct) -> native currently only speeds LOOPLESS prefill walks. Decode-postproc speedup needs
loop-payload reconstruction (remaining hard part). NOW: shadow boot on 6,7 validating C++==Python on
real i2t B32 decode traffic (correctness milestone, must show mismatches=0). THEN: solve loop-payload
so decode goes native -> A/B i2t B32 = the measurable win.

**SHADOW VALIDATION (real i2t B32) — CAUGHT A REAL BUG (103,701 mismatches). Native flip BLOCKED (safe).**
C++ WGIO returns ready=set() where Python returns {'Thinker'} on the REAL thinker_decode self-loop
(completion/num_times_run MATCH; only looped-node RE-READINESS differs). 4000 synthetic trials missed
it — real decode graph's loop-back re-readiness differs from synthetic. FIX: C++ Loop.complete_iter
re-injection -> re-mark looped node ready, against the REAL graph. Then re-validate shadow=0 -> loop-
payload -> native -> A/B. Shadow mode makes this careful multi-week native-port debug SAFE.

**REGRESSION CHECK — default-off integration is a NO-OP (SAFE).** godv9 booted with MSTAR_NATIVE_WGIO
unset: i2t B32 = 7.691 req/s == 7.77 baseline. The additive node_manager_utils.py refactor (dispatcher
+ verbatim _py_* bodies) does not damage the shipping path. So the native-port integration can sit in
the tree safely default-off while the C++ decode-loop readiness bug is fixed (debug agent in flight).

**C++ WGIO BUG FIXED (debug agent).** Root cause: speculation-gated loop re-readiness. Worker sets
node._speculatively_scheduled=True out-of-band (worker.py:2726/2538/4513); Python gates the ready-queue
ADDS on it, C++ added unconditionally -> on decode loop-back during a spec chain, C++ wrongly re-marked
Thinker ready. Fix: C++ gates adds on per-node spec_scheduled flag (default 0 = byte-identical) + new
set_speculatively_scheduled(rid,node,val). Validated: test_wgio_real.py 8308->0 mismatches; synthetic
4000-trial + shadow + bridge tests still PASS. FOLLOW-ON (in flight): native_wgio_bridge.py must FORWARD
node._speculatively_scheduled -> cpp.set_speculatively_scheduled at top of each op (change-cached) for
production shadow parity. THEN: re-boot shadow -> confirm 0 mismatches on real i2t B32 -> loop-payload
reconstruction -> native flip -> A/B.

**★ SHADOW VALIDATION PASSES ON REAL TRAFFIC — ~2.4M comparisons, 0 mismatches.** godv9 booted
MSTAR_NATIVE_WGIO=shadow with bridge spec-forwarding; i2t B32 real decode traffic: matches=631k+1157k+632k
across all worker graphs, mismatches=0. The C++ WorkerGraphIO port is bit-identical to Python on the
LIVE decode loop (incl. spec-scheduled self-loop). This clears the correctness gate. (shadow req/s 8.06 =
double-work + good boot, not the result.) LAST BLOCKER for native decode speedup = loop-payload
reconstruction: in native(=1) mode, map C++ loop-back edge handles -> live Python GraphEdge tensor_info
for re-injection (bridge already has handle<->edge translation for loopless; extend to the loop path).
Then flip native, A/B i2t B32. Estimated win: WGIO is ~40% of the 5.3ms postproc, 63x native -> ~2ms
saved -> ITL ~6ms -> JCT ~3395 -> ~9.4 req/s > vLLM 8.32.

**★ SHADOW3 (loop-payload + native-recon cross-check) PASSES — 0 mismatches, ~1.7M comparisons.**
Native decode reconstruction (input/output tensor_info, slot lifecycle, iter counters) bit-identical on
live i2t B32. Correctness fully cleared. NOW flipping MSTAR_NATIVE_WGIO=1 for the A/B measurement.

**★★ NATIVE A/B RESULT — WASH, no win. Native offload does NOT flip i2t B32.**
MSTAR_NATIVE_WGIO=1 engaged on decode loop (native_capable=True incl loops=1, bit-identical ~4M live
compares). i2t B32: native 7.62-7.81 req/s == baseline 7.69-7.77 (ITL 8.6-8.8 slightly WORSE than
7.93). Decode-step phase timing UNCHANGED: iter_total p50 ~8.33-9.08ms vs ~8.4ms baseline. submit_spec
~2.4ms, await_gpu ~0.6ms. DIAGNOSIS: WGIO state machine = only ~1-2ms of the 8.4ms step; submit_spec +
store_and_populate + set_ref_counts + fanout + emit STAY PYTHON; the bridge's per-op Python translation
(spec sync, edge<->handle, payload recon, event replay) EATS the C++ savings -> marshaling/boundary-
bound (architect's #1 risk realized). To win needs C++ owning the ENTIRE per-rid path end-to-end (no
per-op boundary) = 15-30d full rewrite, NOT the incremental offload. VERDICT: i2t B32 remains ~7%
structural gap (within +-18% boot noise). Native WGIO port is correct+fast in isolation but net-neutral
integrated. Committed artifacts: gilext_poc/*.cpp (validated), native_wgio_bridge.py + node_manager_utils.py
(shadow-validated, default-off no-op). Remaining real options: (A) full C++-owns-everything rewrite
(15-30d, uncertain), (B) accept within-noise tie, (C) async-encoder (rejected).

**★★★ NATIVE-WGIO PATH DEFINITIVELY CLOSED (bridge agent, hard numbers).** Pure-Python WGIO cost =
0.405ms/step; bridge tax (pybind + per-rid payload stitching) = 0.502ms/step (optimized from 0.617,
-18%) -> net -0.097ms/step (native SLOWER). The 63x isolation speedup is on a 0.4ms fraction of the
8.4ms step behind a 0.5ms boundary tax that CANNOT cross zero on the per-rid architecture. Batching
can't remove the per-rid Python payload stitching without a worker-loop+payload-model rewrite (weeks).
Native GIL-offload = confirmed dead for i2t B32. bench_bridge_overhead.py has the proof.
=> PIVOT to prefill analysis. vLLM 0.22 prefill = token-budget continuous batching (prefill chunks
interleaved with decode in one varlen forward) + MRV2 zero-sync single-process. M* = one-walk-per-batch
+ multi-process GIL + bs=1 vision prefill. OPEN QUESTION deciding all options: is i2t B32 prefill
GPU-COMPUTE-bound (32x~60ms serialized = ~1920ms ~ 2333 TTFT -> only faster/batched kernel helps) or
HOST/OVERLAP-bound (-> mixed-batch/async-encoder helps)? PROFILING NOW via GPU-util during i2t B32.

**★★★ PREFILL PROFILE (i2t B32 sustained, GPU-util) — THE decisive redirect.** GPU7 (Thinker
prefill+decode) median 88% p90 99% (73% samples >80%) = near-compute-bound; GPU6 (encoders+Talker/
Code2Wav) median 7% = 93% IDLE. => i2t B32 is THINKER-GPU-COMPUTE-BOUND on GPU7, with a WASTED idle
GPU6. Encoder is NOT the bottleneck (7%) -> async/batched-encoder can't help (correctly deprioritized).
Native-GIL can't help (GPU7 compute-bound, not host-bound during prefill). WIN PATH = use idle GPU6 for
Thinker compute OR cheaper prefill kernel. TOP-3 TESTABLE IDEAS:
(1) TP=2 Thinker + MSTAR_SYMM_ALLREDUCE (qwen3omni_2gpu.yaml) — split Thinker across GPU6+7, halve
    prefill compute/GPU. Prior full-TP net-neg, but profile's 88%-vs-7% imbalance + compute-bound
    prefill justify re-test at i2t B32 with symm-mem.
(2) MSTAR_MIXED_SINGLE_CHUNK=1 — fold vision prefill chunk into the decode step (vLLM-style mixed
    forward). TEST#6 washed but re-test with compute-bound framing.
(3) fp8 Thinker PREFILL / better prefill-M MoE tiles — reduce GPU7 prefill compute (MOE_FP8 covers
    decode-M; check prefill path).

**REFINED TOP-3 (compute-bound GPU7 => only THINKER-COMPUTE-REDUCTION helps; scheduling/fold/async-enc
are OUT):** (1) TP=2 Thinker split across GPU6+7 (halve prefill compute/GPU) — full-TP was net-neg
(decode all-reduce penalty on host-bound decode > prefill gain) but RE-TEST at i2t B32 w/ symm-mem on
mstar-moe; the real variant is prefill-only TP (untested, needs mixed-parallelism code). (2) fp8/cheaper
Thinker PREFILL kernel (reduce GPU7 prefill FLOPs). (3) MSTAR_MIXED_SINGLE_CHUNK (low expectation on
compute-bound but cheap flag A/B). Testing (3) now + prepping TP config.

**Idea #3 (MSTAR_MIXED_SINGLE_CHUNK=1) — NEGATIVE.** i2t B32 6.97/7.53 req/s (avg ~7.25), TTFT
3041-3691ms vs baseline 7.7/2333ms. Fold reorganizes but can't cut compute-bound GPU7 + adds TTFT.
Confirms: scheduling levers dead on a compute-bound bottleneck. => Idea #1 (TP=2, actual compute split).

**Idea #1 (TP=2) — boot CRASHED (NCCL collective timeout, likely symm-mem rendezvous or config).**
COMPLETE DIAGNOSIS of the i2t B32 loss: M* decode is HOST/GIL-bound -> full-TP's per-layer all-reduce
penalizes host-bound decode more than it speeds prefill -> full-TP net-negative (memory confirms) -> M*
can't use TP -> GPU6 wasted (7%) -> Thinker prefill stuck single-GPU compute-bound (GPU7 88%) -> high
TTFT -> loses. vLLM decode is zero-sync (MRV2) so it TPs the Thinker across both GPUs FOR FREE -> 2x
prefill compute -> wins 8.32. The clean fix = PREFILL-ONLY TP (TP=2 for prefill walks, TP=1 for decode)
-> use both GPUs for the compute-bound prefill WITHOUT the host-bound-decode all-reduce penalty. Untested,
needs mixed-parallelism code. Retrying full-TP w/o symm-mem for a baseline number.

**★★★★ TERMINAL DIAGNOSIS — i2t B32 loss is architectural, all tractable levers exhausted.**
Top-3 results: #1 full-TP CRASHED/net-neg; #2 fp8-prefill = ALREADY fp8 (_ensure_fp8_experts frees bf16
so prefill MoE is fp8, moe.py:242/269); #3 mixed-single-chunk NEGATIVE (6.97-7.53 req/s). 
CHAIN: i2t B32 = Thinker-GPU-compute-bound (GPU7 88%, prefill MoE already fp8 -> FLOPs are fundamental)
with idle GPU6. Using GPU6 = TP, but TP SHARDS the Thinker weights -> BOTH prefill AND decode use the
sharded weights -> decode pays per-layer all-reduce. M* decode is HOST/GIL-bound, so the all-reduce
penalty > prefill compute gain -> full-TP net-negative. "Prefill-only TP" is INFEASIBLE (can't have
weights sharded-for-prefill + replicated-for-decode without 2x mem). => the ONLY fix is making M*
decode zero-sync (vLLM MRV2-style single-process, no GIL, no per-step host sync) so TP's decode
all-reduce is cheap -> then TP both GPUs -> 2x prefill -> win. That's the multi-week full rewrite; the
native-GIL incremental attempt was net-neutral (boundary-bound). i2t B32 stays ~7% within-boot-noise
gap (M* good-boot 8.06 overlaps vLLM band 8.03-8.50). No non-rewrite lever remains.

## LEAN A/B result (2026-07-10) — batch machinery ON vs OFF, GPUs 6,7

Booted opt/prep-pos-batched-v9 with batch machinery (CHUNKED_PREFILL_V2, MIXED_BATCH,
MIXED_SPEC, PREFILL_CHUNK_TOKENS, MIXED_BUDGET_TOKENS) REMOVED; kept all decode opts +
MERGED_PREFILL + PREP_DEVICE_POS.

              full(ON)   encoders   vLLM    LEAN(OFF)
 i2t B1 TTFT   ~230       ~230       -       233     unchanged
 i2t B16 TTFT  912        405        168     846     barely moved (still 2x enc)
 i2t B32 TTFT  2532       576        179     3735    WORSE than full
 s2t B1 TTFT   163        97         66      91      RECOVERED to encoders
 i2t B32 tok/s  -          -          -      1135    (req/s 6.58 vs vLLM 8.0-8.5)

CONCLUSIONS:
1. s2t B1 regression = batch machinery. Lean recovers 163->91ms (=encoders 97).
   FIX: concurrency==1 fast-path bypassing chunked/mixed scheduler. Parity-safe.
2. "machinery starves prefill at i2t B32" HYPOTHESIS FALSIFIED. Machinery OFF makes
   B32 WORSE (2532->3735). Chunked prefill was MITIGATING the explosion by slicing
   the giant B32 prefill walk.
3. Real i2t B16/B32 culprit = MERGED_PREFILL one-walk-per-batch encode+prefill freeze
   (+ host-sync stall). Common to full & lean, absent in encoders (which did 576).
   NEXT: A/B MERGED_PREFILL OFF (keep chunked prefill ON) at i2t B16/B32; and try
   harder prefill chunking (PREFILL_CHUNK_TOKENS 512->256). Real lever = ENCODER_ASYNC.

## MERGED_PREFILL OFF A/B result (2026-07-10) — full machinery, merged OFF, GPUs 6,7

              encoders  merged-OFF  full   lean   vLLM
 i2t B1 TTFT   ~230      232        ~230   233    -
 i2t B16 TTFT  405       647        912    846    168
 i2t B32 TTFT  576       2374       2532   3735   179
 s2t B1 TTFT   97        95         163    91     66
 i2t B32 tok/s  -        1232       -      1135   -

CONCLUSIONS (both structural hypotheses now FALSIFIED for i2t B32):
1. merged_prefill is NOT the i2t B32 killer: merged-OFF 2374 ~= full 2532 (B32 low
   variance, p50~=mean). Flipping it off changed nothing at B32.
2. Combined w/ lean (machinery-off made B32 WORSE), NEITHER batch machinery NOR
   merged_prefill causes the B32 TTFT explosion.
3. B32 killer = the encode+prefill HOST-SYNC STALL (~2s GPU-idle) inherent to the
   modern build, absent in encoders(576). Matches prior TTFT root-cause note. Fix is
   code-level (async encoder / kill host sync), NOT a flag. THIS is the real bug.
4. Side-wins: (a) s2t B1 bad(163) only when machinery AND merged BOTH on; off either
   -> ~92-95 (=encoders). (b) merged-OFF mildly beats full: B16 912->647, s2t 163->95,
   B32 flat, tok/s 1232. B16 noisy (mean 1060 vs p50 647) -> needs repeat.
NEXT: profile the i2t B32 TTFT window to locate the ~2s host-sync stall (py-spy the
   worker during prefill; look for .item()/.cpu()/synchronize in encode+prefill).

## *** BREAKTHROUGH: prefill token-budget throttle (2026-07-10) *** GPUs 6,7

Batch-scaling sweep (merged-off server) showed i2t TTFT is a STEP function, not smooth:
  B1=216 B4=378 B8=351 | B16=974 B24=1082 | B32=2079  (p50 ms)
  -> plateau-cliff-plateau-cliff = a BUCKETED token-budget throttle serializing
     vision prefills into groups (NOT O(B^2), NOT a fixed host-sync stall).

Root cause: MSTAR_PREFILL_CHUNK_TOKENS=512 + MSTAR_MIXED_BUDGET_TOKENS=512 too small
for vision (each food101 image = few hundred vision tokens) -> only ~1-2 prefills
batch per forward; the other 30 at B32 serialize. vLLM uses a large prefill budget
-> all 32 batch -> 179ms flat.

FIX (live dynflag, parity-documented byte-identical, NO reboot):
  PREFILL_CHUNK_TOKENS 512->2048, MIXED_BUDGET_TOKENS 512->4096
                    baseline(512)   raised(2048/4096)   encoders  vLLM
  i2t B16 TTFT p50   974            402                 405       168
  i2t B32 TTFT p50   2079           1006                576       179
  i2t B32 req/s      7.22           8.28  <-- IN vLLM BAND 8.03-8.50
  errors             -              0
Both knobs read live (mixed_budget re-read in _refresh_dynamic_flags; dynflags applies
to os.environ). Caveat: high tail (B32 mean 1643, p95 3251) - median great, tail slow.
TODO: (1) push budget higher (8192/16384) + capture bigger prefill bucket for tail;
      (2) VERIFY parity empirically (diff outputs budget-off vs on); (3) confirm on
      full/committed build (this was merged-off, CDT-off boot).

## CORRECTION (2026-07-10): budget win is REAL but MODEST (paired A/B), not 2x

My first budget cell (B32 1006ms/8.28rps) was a VARIANCE LOW, not the true effect.
Host CPU load swings 14->102 and M* TTFT is CPU-floor-sensitive -> 3x run-to-run
noise. Repeats of the SAME 4096 config gave p50 1958/2908/1805/2965. Single samples
are untrustworthy on this box.

PAIRED A/B (OFF 512 vs ON 2048/4096, back-to-back x4, cancels slow contention):
  pair  OFF_ttft ON_ttft  dTTFT | OFF_rps ON_rps
   1     2.507   2.039   -468ms |  7.44   7.63
   2     2.561   2.033   -528ms |  6.83   7.86
   3     2.963   2.351   -612ms |  7.03   6.88
   4     1.619   1.605   - 14ms |  7.51   7.76
  mean   2.41    2.01    -405ms(-17%) | 7.20  7.53 (+4.6%)
ON beats OFF on TTFT 4/4 pairs, req/s 3/4. REAL parity-safe win but ~17% TTFT/~5% rps,
NOT enough to pass vLLM (8.03-8.50 rps, 179ms TTFT). ON rps ~7.5 still trails.

Why capped: chunk maxes at largest CAPTURED bucket 2048 (~8 images) -> B32 still ~4
serial prefill groups. NEXT LEVER: reboot w/ expanded PREFILL_TOKEN_BUCKETS (4096/8192)
to batch all 32 images' prefill in one forward (vLLM-style). Needs capture-grid change.
Measurements need a QUIET host for ship-grade numbers.

## ROOT CAUSE + FIX: MSTAR_BATCH_VISION_PREFILL (2026-07-10) GPUs 6,7

Code root cause (submodules.py:1458-1471): prefill_vision ASSERTS one request per
step (PREFILL_VISION_CAPTURE_BATCH_SIZES=[1]); at B32 all 32 images encode+prefill
SERIALLY. This (not schedulers/merged/budget) is why i2t B32 TTFT explodes vs vLLM
(which batches vision). Fix flag exists: MSTAR_BATCH_VISION_PREFILL (batches vision
tower over concatenated images, captured [1,2,4]; parity-documented byte-identical).

Booted with it (base=merged-off). Structural sweep vs merged-off (p50 ms):
   B      1    4    8    16    24    32
 off     216  378  351  974   1082  2079
 on      155  284  427  467   1209  1653
-> serialization cliff MOVED RIGHT one step: low plateau now extends to B16 (467 vs
   974, -52%); B32 -20% (2079->1653). Real structural win, but capped at bs=4 so B32
   still ~8 groups -> 1653ms, still ~9x vLLM 179ms. Variance high (single runs).
Stackable parity-safe levers: BATCH_VISION_PREFILL (boot) + budget 2048/4096 (live).
To close the rest: expand PREFILL_VISION_BATCH_CAPTURE_BATCH_SIZES + PREFILL_CAPTURE_
BATCH_SIZES [1,2,4]->[..,8,16,32] (big capture cost + OOM risk); and single-GPU
prefill of 32 images is inherently heavier than vLLM's TP'd zero-sync path.

## NEXT STEPS PREPPED (2026-07-10) — blocked on GPUs (root claimed whole node)

User directed: do easy lever (verify+ship modest) THEN hard lever (bs=32 capture).
Both PREPPED and syntax-checked; waiting for GPUs 6,7 (root ray::StreamingM job took
all 8 GPUs ~18-20GB each right after teardown; my boot auto-aborted, no co-location).

CODE EDITS DONE (mstar-godv9 + bench-v2):
- bench-v2/benchmark/runner.py: env-gated greedy (MSTAR_BENCH_GREEDY=1 -> temperature/
  thinker_temperature=0) so outputs are deterministic & diffable for PARITY check.
  (server confirmed: temp==0 -> argmax greedy, sampling.py:60-63.)
- submodules.py: made both prefill capture caps env-overridable via new _env_batch_sizes:
  MSTAR_VIS_BATCH_SIZES (vision tower, line ~1602) + MSTAR_PREFILL_BATCH_SIZES (thinker
  prefill, line ~1658). Default unchanged [1,2,4]. Lets us push past bs=4 to batch more
  images/forward (OOM risk on 30B Thinker -> expand incrementally 8->16->32).
- bs8_boot.sh ready: BATCH_VISION_PREFILL=1 + VIS/PREFILL_BATCH_SIZES=1,2,4,8. Runs
  greedy B1x2+B32 (parity) then normal B16/B32 (TTFT vs bs4 visbatch 467/1653).

WHEN GPUs FREE: run bs8_boot.sh; check (1) parity: par_b1a==par_b1b (determinism) and
par_b1a==par_b32 (batched-vision byte-identical); (2) TTFT: does bs8 beat bs4 (1653)?
If OOM at boot -> drop to VIS_BATCH_SIZES=1,2,4,6 or limit vision token buckets. If bs8
helps + no OOM -> push 16, then 32. Measurement needs quieter host (load was 14-102).

## SESSION 2026-07-10b: three root causes found (bs8 ran, fixes in godv9 worktree)

**bs8 captured route (bs8_boot.sh ran, GPUs 6,7, load ~9):** boot OK (no OOM, 33GB).
i2t B16 TTFT p50 **288ms** (vs bs4 467, base 974; vLLM 168) — structural win, ladder
974->467->288 confirms per-group serialization model. B16 p95 2.0s tail. B32 p50 3018
(single sample, 4 serial groups at bs=8 — needs bs16/32 grids; noisy, don't conclude).

**(1) UNCAP eager-crash ROOT CAUSE = cross-thread active-manager race, NOT kernel scale.**
fp8 MoE exonerated by isolated repro (per_token_group_quant + fused_experts_fp8 PASS at
M=32768 contig+noncontig, CUDA_LAUNCH_BLOCKING=1; GEMM2-quant to 262k rows). The crash:
plan_executor's speculative pre-plan calls plan_attention -> compile_ops.set_active_manager
(cache_manager.py:213) from ITS thread; an in-flight EAGER packed-prefill forward on the
gpu thread resolves _ACTIVE_MANAGER per custom op (48 layers x run_attention/apply_rope)
-> mid-forward it picks up the decode slot's manager/plan -> IMA in FlashInfer attn, or
surfacing at the next launch (= the fp8.py:75 red herring). Captured replays never rerun
the op Python => invisible in shipping config. FIX (in worktree): plan_attention(...,
publish_manager=False) from pre_plan_for_batch (cuda_graph_runner.py:1145/1155).
NOTE: UNCAP=32 live-flip at i2t B32 on the RACY build did NOT crash and did NOT help —
encode readiness (~1-2 prefills ready/cycle) means the eager path never assembles bs>4
at i2t. UNCAP only matters where readiness pools (s2t audio — gated out anyway). The
i2t lever is the CAPTURED route (vis-batch pools readiness + big prefill graphs consume).

**(2) FIRST-TOKEN artifact ROOT CAUSE = api_server emit-channel reorder (SHIPPING BUG,
affects ALL concurrent i2t traffic, not just BATCH_VISION_PREFILL).** Greedy parity
(MSTAR_BENCH_GREEDY=1, b1a==b1b determinism 0/32 mism): at ANY concurrency>1 every
request's FIRST generated token is displaced mid-stream (b2/b4/b32: " on the...Based")
or past stream end (looks dropped). NOT numerics (continuation byte-identical). Killed
suspects: emit flags (but note MSTAR_BATCH_EMIT/_inline_emit are NOT dynflag-refreshed —
init-cached), mixed fold (MSTAR_MIXED_BATCH=0 still drops), vis-batch (only changes
displacement 1 vs 5-6 tokens), sidecar (i2t rids aren't sidecar-owned: prefill_vision
not in SIDECAR_WALKS). MECHANISM (api_server/data_worker.py): decode tokens ride INLINE
emit (values in message, emitted synchronously in arrival order); the prefill step's
first token FAILS inline condition (c) (its uuid also feeds the prefill->decode loop-back
edge => must keep SHM write) -> api_server _read_result_tensor only STARTS an async SHM
read; chunk emits on completion => displaced by fetch-latency/ITL tokens under load; B1
in-order (no pressure). Explains why prior "byte-identical" A/Bs missed it: both sides
of same-concurrency A/Bs displace identically; only vs-B1 ground truth exposes it.
FIX (in worktree): MSTAR_ORDERED_EMIT=1 — per-(rid,modality) arrival-order FIFO in
PreprocessWorkerThread; inline chunks emit only at FIFO head, SHM entries emit at head
on read completion. Default OFF byte-identical.

**(3) get_max_batch_size MSTAR_UNCAP_PREFILL + KeyError fix + env-overridable capture
grids (MSTAR_VIS_BATCH_SIZES/MSTAR_PREFILL_BATCH_SIZES/_env_buckets raise-ceiling)** —
uncommitted in mstar-godv9, all default-off.

RESULTS (2026-07-10b, boots bs16fix/bs16v2/bs16v3 on GPUs 6,7):

- **bs16fix (bs16 grids, sidecar OFF, fixes loaded, load ~9):** s2t B32 canary CLEAN
  37.45 req/s (race fix verified inert). Parity: reorder STILL present with sidecar
  off => sidecar definitively exonerated. i2t B16 p50 409ms (vs bs8 288, bs4 467 —
  single samples); B32 p50 3664/7.09 rps — bs16 grids do NOT move B32 (4 groups of 8
  vs 2 of 16 wash => B32 is compute-bound serialization, grouping can't move p50;
  terminal diagnosis stands).
- **ORDERED_EMIT v1 wedge:** first live deploy returned ALL-EMPTY responses.
  ORDEMIT trace: the FIRST emit item per rid is a SIGNAL-ONLY edge (uuids=[]) —
  entry was created not-ready with nothing to complete it => permanent FIFO head
  block => 15s TTL drop. Fixed (trivially-ready + immediate flush). Also made the
  uuid->entry map list-valued (aliased uuid arrivals).
- **bs16v2 (ship config + ORDERED_EMIT):** ★ PARITY GATE PASSED — greedy first
  tokens match B1 ground truth 16/16 at conc 2 AND 32 (was 0/16). Residual text
  diffs are batch-size-dependent greedy near-tie flips (numerics; same class as
  vLLM cross-batch nondeterminism), NOT ordering. s2t canary clean 34.7 rps.
  i2t B32 4.6-4.9 rps / TTFT p50 6.3-6.6s — WORSE than fix-boot, initially
  attributed to the first-token SHM fetch landing on the critical path (ordered
  emit stops hiding it), BUT SEE CONTAMINATION below.
- **MSTAR_INLINE_DUAL** (worker.py, default off): emit copies of dual-consumer
  prematerialized tokens ride inline while KEEPING the SHM write/registration for
  loop-back consumers (ref economy: emit ref locally released as pure-inline).
  bs16v3 (ORDERED+DUAL): parity still 16/16 first tokens at conc 2/32; s2t clean.
- **HOST-LOAD CONTAMINATION:** load average exploded 9 -> 213 (root jobs, 14-18
  cores each) across the v2/v3 window. ALL v2/v3 TTFT/req-s numbers are suspect —
  the v2 "ordered emit costs ~30% at B32" and the v3 B16 3.2s cells track the load
  ramp, not the flags. Parity verdicts are load-independent and stand. The
  ordered-emit perf cost and INLINE_DUAL's recovery need a QUIET-HOST paired A/B
  (ordered-off vs ordered vs ordered+dual, same server via dynflags — all three
  flags are dynflag-refreshable) before any ship decision.

## FINAL (bs16v4 boot, ordered+dual+prem, quiet-host window 10:49-10:54)

DUAL-without-prem was a NO-OP by construction: the prem dict was
thinker_decode-only (worker.py), so the prefill token was never an inline
candidate — measured wash/slight-negative 3/3 pairs on v3. Fixed (prem extended
to thinker prefill walks under the flag, commit f4038ead), then v4:

★★ QUIET-HOST CELLS (load ~28, the root jobs' gap): **i2t B32 = 8.14 req/s,
TTFT p50 629ms, 1443 tok/s** — IN the committed vLLM band (8.03-8.50) WITH
correct first tokens; vs the historical 2333-2532ms TTFT / 7.4-7.7 rps cells.
B16 431-539ms / 6.2-6.3 rps (ordered-off band was 409/5.98). s2t B32 37.7 rps.
Parity: determinism 0/16; first tokens vs B1 = 16/16 at conc 2 AND 32.
Load returned (58-98) for r2 + the A/B; r2 6.51/2103ms shows the recontamination.

DUAL on/off PAIRED A/B (prem engaged, v4 server, load 67-98): OFF (ordered-only)
ahead 3/3 pairs — ON 5.37/5.61/7.80 vs OFF 5.99/6.89/8.21 rps (~-7% for dual).
And dualOFF_3 = 8.21 rps @ load 97: ordered-ONLY also reaches vLLM-band.
=> the bs16v2 "ordered emit costs ~30% at B32" was ENTIRELY host-load artifact.

SHIP GUIDANCE:
- SHIP MSTAR_ORDERED_EMIT=1 — correctness fix for the shipping first-token bug,
  parity-certified, no measurable B32 cost (best ordered-only cell 8.21 rps).
- PARK MSTAR_INLINE_DUAL (+prem extension) — mechanism verified engaged but
  net -7% (3/3 pairs); the SHM fetch it hides is not on the quiet-host critical
  path. Keep the code; revisit only if TTFT profiling shows first-token fetch
  stalls on the certified config.
- SHIP the captured route: MSTAR_BATCH_VISION_PREFILL +
  MSTAR_VIS_BATCH_SIZES/MSTAR_PREFILL_BATCH_SIZES=1,2,4,8,16 — B16 TTFT
  974 -> ~430-540ms; B32 best cells land in the vLLM band.
- Committed h2h numbers were measured with the first-token bug; re-certify the
  flagship with ORDERED_EMIT on, quiet host, user's paired protocol.

Branch: all fixes on opt/prep-pos-batched-v9 (pushed t-avil/mstar), commits
0c1003b5 (race) c0b4ece0 (UNCAP) b0ee21e3 (grids) 94e440cb (native-WGIO shadow)
ebb066ca (ORDERED_EMIT) 94ebe978 (INLINE_DUAL) 42d3b6f0 (tp2 yaml)
f4038ead (prem extension). Parity harness: bench-v2 d7c0dfcf
(branch bench/greedy-parity-harness). Raw cells: /m-coriander/coriander/tim/
rm_out/{bs8,bs16fix,bs16v2,bs16v3,bs16v4}/ + v3/v4_dual_ab.run.log.

## SESSION 2026-07-11: cert grid + PD-DISAGGREGATION BREAKTHROUGH

**CERT1 (colocated winning config: ship stack + ORDERED_EMIT + bs16 grids +
budgets 2048/4096, static env, GPUs 6,7).** Four-stat verdicts vs committed
vLLM bands (cell / rps / tok/s / TTFT p50 / ITL p50):
  i2t B1  1.09 WIN / 198 WIN / 114ms band / 4.4ms LOSS(-1ms)
  i2t B2  1.76 WIN / 317 band / 176ms LOSS / 5.5ms WIN
  i2t B4  3.14 WIN / 543 WIN / 152ms band / 5.6ms WIN
  i2t B8  4.53 (no ref) / 817 / 218ms / 5.8ms
  i2t B16 6.48 WIN / 1151 ~ / 449ms LOSS / <1ms WIN     (load 55)
  i2t B32 8.23 band / 1445 LOSS / 1803ms LOSS / <1ms WIN (load 61)
  s2t B1  5.06 / 98 / 93ms / 4.4ms (no ref; old 163ms machinery-regression GONE)
  s2t B2  9.94 WIN(2.9x) / 189 ~ / 87ms LOSS(-15ms) / 4.8ms band
Pattern: rps+ITL won nearly everywhere; every LOSS is TTFT-driven (+1ms B1 ITL).

**s2t B32 "budget regression" FALSIFIED:** all 4 budget combos hit 39.5-39.8
in good cells; same-flags back-to-back gave 39.7 then 24.6 -> bimodal
admission-wave lottery at n=96 (whole cell = 3 waves, 2.4-3.9s wall; ITL
identical, only TTFT tails differ). n=256 wave-averaged truth: 29.59 rps
(vLLM 25.2-28), TTFT p50 221 WIN, ITL 12.3 band, 620 tok/s band.
=> s2t B32 cert MUST use n>=256. Budgets 2048/4096 ship globally.

**★★★ PD-DISAGGREGATION (configs/qwen3omni_2gpu_pd.yaml, commit 6c70f758):
prefill Thinker on rank0 + decode Thinker on rank1 — probe results:**
  i2t B1: 115ms/1.06rps == colocated (KV handoff free)
  i2t B16: 7.52 rps (+16% colo, +31% vLLM)
  i2t B32: 8.08 rps AT LOAD 79 (colo needs quiet host for that); ITL p99 29ms
  s2t B32 n=256: **48.14 rps** (+63% colo, ~+80% vLLM), TTFT p99 847 vs 2837
  i2s/s2s: functional; s2s audio ITL 301 vs 228 needs A/B (cross-rank states)
Decode isolation removes the prefill-freezes-decode structural defect entirely.
Costs: 2x Thinker weights (fits: 77/66GB), first-boot +24min autotune (cached
now), no mixed/chunked machinery on this topology.

NEXT: (1) PD as flagship candidate — full grid + speech A/B + i2t B2 TTFT
(admission timing) on PD; (2) B1 decode ITL -1ms (host floor micro-work);
(3) certify with user's paired protocol; s2t B32 at n>=256.

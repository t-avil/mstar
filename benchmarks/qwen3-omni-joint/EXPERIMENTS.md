# Qwen3-Omni optimization — experiment knowledge base

Maintained by the optimization session of 2026-07-02. One paragraph per
experiment: what was ACTUALLY implemented (not what docs claim), where it
lives, and the measured verdict with data pointers. All perf ratios are from
interleaved A/Bs (two preloaded servers, per-cell back-to-back runs, ratio
contention-robust) unless noted. Committed code: branch `opt/decode-v2`
(fork t-avil/mstar), base = encoders-implemeneted @ 4c33b33.

**Ground truth on the baseline gap** (do not trust older docs): committed raw
data (`benchmarks/qwen3-omni-joint/raw_image_to_text.json`, branch
`benchmarks`) shows M*-new i2t B32 at 4.394 req/s / 768.6 tok/s vs vLLM-0.22
at 8.210 req/s / 1723.2 tok/s → 0.45–0.53×. The NUMBERS.md claim of M* winning
1.48× at that cell came from a superseded 6-datapoint favorable-conditions run
(fixed 2026-07-02). Profiling (nsys `--cuda-graph-trace=node` + NVTX,
`/m-coriander/coriander/tim/prof_kern2/`, `prof_v2/`): at B32 the thinker GPU
was 49% idle on base (37% after fp8) — the gap is per-step Python
(~26ms/step across two GIL-sharing threads: postprocess 13.2ms + send 4.75ms +
GPU-thread prepare/plan) plus phased-batching prefill stalls (~8.7ms/step
amortized; encoder 17.6ms + thinker prefill 27.5ms per admission), NOT the
forward. Decode graph itself: 12.8ms at B32 (76% = MoE at the bf16 bandwidth
floor), 5.1ms at B1.

## E1 — block-fp8 w8a8 MoE (`MSTAR_MOE_FP8`) — WIN, kept
Commit ca97ba2 + fix 22b3cd6. Triton grouped-GEMM with per-(128,128)-block
weight scales + per-(token,128)-group activation quant (DeepSeek scheme);
weights quantized lazily at first forward under `@torch.compiler.disable`
(dynamo re-trace of a later capture bucket otherwise sees the freed bf16
param and the capture fails — that bug silently sent 29 buckets to eager in
the first A/B attempt). bf16 originals freed (−27GB VRAM). Decode tiles tuned
in-graph (BLOCK_M=16). Validated with REAL layer-10 checkpoint weights:
cos 0.998, worst-token 0.9977; in-graph kernel 1.44–1.61× for M=8..128.
E2E (ab_fp8/): i2t 1.00/1.13/1.13× at B1/8/32; s2t 1.00/1.11/1.05×. Outputs
text-identical to bf16. NOTE: the same kernel measured OUT-of-graph shows
0.75–0.85× — out-of-graph microbenchmarks at decode M are launch-noise
garbage; always graph-capture loops (this artifact misled the June agent
into rejecting fp8).

## E2 — fused router topk (`MSTAR_FUSED_TOPK`, default ON) — WIN, kept
Commit 856d103. `sgl_kernel.topk_softmax` replaces softmax→torch.topk→renorm
(gatherTopK+bitonicSortKVInPlace, 27.6µs/layer → 11.3µs in-graph). Expert ids
bit-identical, weights equal to 3e-8. ~0.5–0.8ms/step at B32. Measured only
as part of the round-2 stack (not isolated).

## E3 — worker CPU cuts (send prematerialized ints, batched prepare_inputs,
## batched check_stop D2H) — PARTIAL WIN, kept
Commits 9193cd7, de4a741. (a) `_send_outputs` now consumes check_stop's
side-stream D2H ints instead of 32 per-rid `get_tensor().cpu()` default-stream
syncs; (b) `prepare_inputs_batched` for thinker_decode: one token cat + ONE
embed_tokens + one pos H2D instead of 32 per-rid embeds; (c) check_stop D2H:
one flat cat + one pinned copy. Round-2 stack A/B (ab_stack/, includes E1+E2):
i2t 1.134/1.224/1.153×, s2t 1.057/1.165/1.201× vs base. Lesson from the v2
re-profile: send_outputs stayed ~5ms — the cost was per-rid ZMQ message
construction/pickling, not the D2H; (a) mostly mattered for unblocking overlap.

## E4b — encoder-placement tension + audio-path contamination fix (2026-07-02 late)
The first full sweep showed audio-output paths regressing vs M*-new at batch
(i2s B32 0.61×). Attribution probes (qb_queue1.log): (1) ~17% was
UNCONDITIONAL worker fast paths (prem-ints dict + batched check_stop probe)
running on Talker steps as pure overhead — fixed by gating both to the
thinker_decode walk (commit d04dcb4; flag-off parity with pristine 4c33b33
restored: i2s B32 2.071 vs 2.089). (2) The rest is a REAL placement tension:
with the fix, encoff gives s2t +31% and i2t +9% but costs s2s −22% and i2s
−4% (audio encoder on rank 0 collides with talker+code2wav). No static
placement wins all four paths. Resolution: encoff = primary config
(maximizes the minimum margin vs vLLM: s2t flips from 0.81× to 1.07×; s2s
keeps ≥1.78× over vLLM), default layout = documented audio-optimized
alternative with its own mini-sweep numbers.

## E4 — encoders on rank 0 (`configs/qwen3omni_2gpu_encoff.yaml`) — WIN for text, see E4b
Commit 9be08a2. One-line topology change: audio+vision encoders move to the
Talker GPU (idle on text paths), so encoder batches stop serializing against
thinker decode. Bundled with E5 in round-3 (ab_encoff/): i2t
1.044/1.014/1.090×, s2t 1.093/1.071/0.983× on top of the round-2 stack.
(Attribution between E4/E5 not isolated; s2t B32 −1.7% ≈ wash.)

## E5 — inline token emit (`MSTAR_INLINE_EMIT`) — WIN (bundled w/ E4), kept
Commit a06c08f. Qualifying integer new-token emit_to_client tensors ride the
result_tensors message metadata; no /dev/shm file per token per rid per step,
no data-worker fetch, no TENSOR_RECEIVED ack; producer releases the tensor
ref locally. Only uuids used exclusively by emit edges qualify; audio/
multimodal excluded. Ran clean under load (no leaks/errors in server logs).

## E6 — batched per-step emit (`MSTAR_BATCH_EMIT`, implies E5) — WASH, off in final config
Commit c2b9b5b. All qualifying inline emits of one decode step coalesce into
ONE result_tensors_batch APIServerMessage (was 32 messages/step at B32),
fanned out per-item on the api_server. Round-5 (ab_r5/, vs the E4+E5 config):
i2t 0.960/0.997/1.011×, s2t 1.017/1.008/**1.107×** — noise-band on i2t, but a
real win on s2t B32 (highest token-message rate: ~480 msg/s coalesced 32:1).
Conclusion: per-rid ZMQ pickling only matters at extreme message rates; the
residual i2t send cost is the per-rid python around it. KEPT ON in the final
config (harmless where it's a wash, +10% where messaging saturates).

## E7 — side-stream prefill overlap (`MSTAR_SIDE_PREFILL`) — REJECTED (catastrophic at B32)
Commits 31ea1bc (engine: eager-path gate via node_batch.metadata["side_stream"],
locks in kv_store.get_state / WorkspaceBufferManager.get) + b9de820 (worker:
PendingSide, side executor + side stream, main-thread-only scheduling/
postprocess, loop-stop snapshot/restore, KV visibility via side-stream
completion_event host-synced in postprocess before token routing). Design
constraint: side batches take the EAGER prefill path (captured prefill graph
is single-writer). Round-4 (ab_r4/): i2t B1 1.055×, B8 **0.820×**, B32
**0.098×** (0.545 req/s — collapse; run killed after this cell). The B8 cell
was clean data (uniform +23% JCT, no outliers, no foreign GPU processes); at
B32 the eager side prefill + GIL contention starves the decode chain almost
completely. Verdict: the eager-path side stream is not viable. A future E8
would need the captured prefill graph made side-thread-safe (per-slot static
buffers + locked next_slot) AND bounded side-thread Python; until then the
flag stays default-off and out of the final config.

## Rejected with data
- **FA3 decode attention** (sgl_kernel.flash_attn_with_kvcache): correct on
  M*'s paged layout (cos 0.9999) but SLOWER than FlashInfer on H200 GQA 28/4
  (14.2µs vs 7.6µs at bs=1; parity at bs=32). FA4 not compiled into
  sgl_kernel 0.3.21. Keep FlashInfer.
- **w8a16 weight-only fp8 MoE**: in-graph only 1.16–1.26× vs w8a8's
  1.44–1.61×. Kernel kept in fp8.py for reference.
- **Prior agent's claims** (audited): tuned bf16 MoE tiles + NUM_SLOTS=3 =
  wash (confirmed); MSTAR_MIXED_WALK eager loses at concurrency (confirmed,
  0.17× at B32); fp8 "validated cos 0.998 but loses e2e" — was NEVER wired
  into the model; the microbench used random weights.

## Queue (not yet run)
- E8: side-prefill v2 — replay captured prefill graph from the side thread
  via per-slot static buffers (needs cuda_graph_runner surgery), or
  decode-priority chunked prefill.
- E9: direct GPU token feed (`MSTAR_DIRECT_FEED`, commit 6644913 on
  exp/direct-feed) — implemented; SCOPE CORRECTION from implementation: the
  route/store block is load-bearing (mark_node_complete drives
  Loop.complete_iter; the stored loop-back edge feeds next-step readiness for
  the non-spec fallback), so it cannot be skipped and the spec path never
  fetched from the registry anyway. Standalone gain expected MARGINAL; value
  = the in-thread feed substrate for E10. Round-6 verdict (ab_r6/): WASH as
  predicted (i2t 0.960/1.018/0.964, s2t 0.993/0.995/1.067) with clean outputs
  and zero dropped-rid/traceback — the mechanism is correctness-validated for
  E10. Note: the s2t B32 cell shows ±7% round-to-round variance (0.983 r3,
  1.107 r5, 1.067 r6) — treat single-cell s2t B32 ratios as noisy.
- E10: two-step decode (`MSTAR_MULTISTEP_DECODE=2`, commit 6bf6136 on
  exp/two-step-decode, same-slot double-replay, requires DIRECT_FEED) —
  implemented + GPU-tested round-7 (ab_r7/): **FLAT** (i2t 0.969/0.995/0.998).
  Outputs correct, zero failures. Interpretation: with E9 also flat, the
  "main-loop cycle overhead" hypothesis is falsified at this granularity —
  the awaited/submit bookkeeping already overlaps GPU time; the true residual
  floor is the per-token route/store/emit Python that E9's analysis proved
  load-bearing (Loop semantics). Further B32 gains need batched loop
  bookkeeping across rids/steps — a deeper engine redesign (future work).
- E11: NUM_SLOTS=3 retest — DONE via quick-bench (qb_queue1.log): REGRESSION
  (i2t B32 0.81× vs ref; B1/B8 also down). Third graph slot hurts on the
  current stack. Rejected.
- W6: MSTAR_PY_SWITCH_INTERVAL_SEC=0.001 — DONE via quick-bench: REGRESSION
  (i2t B32 0.78×). Faster GIL switching adds context-switch overhead on the
  hot loops. Rejected.
- E12: lm_head fp8 (B1 lever: ~0.6GB weight read per step) — not implemented.
- W1: memoized decode postprocess (`MSTAR_FAST_POSTPROC`, commit 4ce0125 on
  exp/batched-postprocess) — caches the step-invariant parts of
  store_and_populate (sharding/tp lookups, TensorPointerInfo construction)
  per (rid,node,walk) with conservative invalidation; routing traversal +
  Loop.complete_iter proved load-bearing and untouched. Quick-bench: i2t
  1.031/1.062/**1.068×** vs ref. PROMOTED to combo confirmation.
- W7: denser decode graph buckets [+24,+28] (commit 845faff on exp/bucket24) —
  cuts round-up padding at churn (live bs 17-31 padded to 32). Quick-bench:
  i2t 1.051/1.075/**1.126×**. Strongest single result since fp8. PROMOTED.
- Combo W1+W7 (exp/combo-w1w7, c8cebc7) — triage in flight; if additive
  (~+18% at i2t B32) the final config is re-frozen and affected sweep cells
  re-run.
- vLLM-0.22 feature inventory vs M*: full-decode CUDA graphs (have),
  async scheduling (have, as speculation), chunked prefill (their prefill
  lever; our E8 alternative), FA3 (tested, loses on our shapes), bf16 triton
  MoE (we beat it with fp8), prefix caching (N/A for this benchmark's
  prompt distribution).

## Data-quality rules (hard-won)
- Only interleaved A/B ratios are trustworthy on this shared box.
- Check `nvidia-smi` idle before every run; never co-schedule with other
  users' jobs (GPUs 0–5 are often taken; 6,7 = project set).
- Inspect per-request JCT std/max in results.json for contention outliers
  before believing a cell.
- tok/s comparisons across systems embed output-length differences
  (vLLM ~217 tok/req vs M* ~176 at i2t B1); use req/s for cross-system
  ranking, tok/s for within-system deltas.
- Out-of-graph kernel timing at decode M: invalid (see E1 note).

## Learnings (2026-07-02 session wrap)

**Methodology (these paid for themselves repeatedly):**
1. In-graph kernel timing only — out-of-graph decode-M microbenchmarks
   inverted the fp8 verdict and had already misled a previous engineer.
2. Interleaved A/B or paired probes only; time-separated ratios lie on this
   box (the committed vLLM B1/B8 baselines themselves proved understated).
3. Attribute regressions with a pristine-base probe BEFORE blaming the
   obvious suspect: the "encoff audio regression" was mostly an
   unconditional-code bug (walk-gating fix d04dcb4), not placement.
4. req/s for cross-system claims; tok/s embeds output-length skew (vLLM
   generates ~20% longer text on identical inputs).
5. One branch per experiment + quick-bench triage (12 min) before any
   45-min interleaved A/B. Env knobs get triaged first — both "obvious"
   knobs (NUM_SLOTS=3, GIL interval) were −20%.
6. Fast paths in shared code MUST be walk/node-gated: thinker-decode
   optimizations silently taxed Talker steps 17%.
7. pgrep/kill by pattern self-matches your own shell; setsid survives
   TaskStop — kill by pgid, verify by GPU compute-apps list.

**Deployment guidance (per-workload configs endorsed):** the encoder
placement tradeoff is fundamental to the current pipeline (s2t and i2t want
encoders off the thinker GPU; s2s wants the audio encoder off the talker
GPU). Ship per-workload: text-heavy deployments run
`qwen3omni_2gpu_encoff.yaml`, speech-generation-heavy run the default yaml.
A future scheduler-level fix (per-request encoder routing to the idler GPU)
would subsume both.

**Open items (priority order):**
1. Speech generation at B16/B32 still ≤ M*-new even in the audio config
   (i2s B32 0.95×, s2s B16 1.00×): residual attribution in flight
   (suspects: fused talker topk, fp8 side effects on thinker_states timing,
   batch-emit; probes A1-A3).
2. i2t B32 vs vLLM (0.76×): per-token routing floor; W1 memoization (+7%
   triaged) needs rebase onto d04dcb4 + interleaved confirm; batched loop
   bookkeeping is the structural swing.
3. W1+W7 combo interference unexplained (combo < W7 alone) — rerun after
   W1 rebase.

## A1-A3 — i2s high-batch residual attribution (2026-07-02) — EXONERATED
Probes on equal quick-bench terms (default config, i2s B16/B32, GPUs 4,5):
fused-topk off 2.097, fp8 off 2.089, batch-emit off 2.095, all-flags 2.060,
no-flags 2.071, pristine 4c33b33 2.089 req/s — a ±1% band. No feature causes
the apparent i2s B32 −5% from the sweep table; that delta is cross-campaign
measurement noise (same class as the documented ±7% s2t B32 variance).
Speech-generation parity vs encoders-implemeneted original is confirmed in
the audio config; the encoff config's documented audio cost is placement
contention only.

## P-tiles — fp8 prefill tile tuning (commit 2239005) — kernel WIN, e2e neutral
In-graph sweep at M=2048/4096 (real weights): BLOCK_M=64/GROUP=8 = 1.63x over
the previous prefill config on the fp8 MoE GEMM. End-to-end TTFT unchanged
within noise (the prefill MoE slice is ~30ms of a ~200ms TTFT). Kept: free
kernel improvement, no regression (i2t B32 unchanged at 5.91 qb-scale).

## Deep-research-derived roadmap (2026-07-02, 103-agent verified sweep)
Full cited report: tasks/woiddyj7e.output. Key verified findings and the queue
they generate (ranked by win × feasibility at OUR bottlenecks):
- **W3 — future-token overlap scheduler** (from SGLang v0.4, pure Python,
  1.1× measured, default-on there): run scheduler/metadata prep for batch N+1
  BEFORE batch N's tokens exist, using placeholder "future token" tensors
  resolved on-GPU (CUDA-event ordered). Kills the remaining await→thread→
  submit dependency our speculation pipeline still has. Composes with W2.
  DIRECTLY refutes our "irreducible ~5ms/step Python" assumption.
- **W5 — token-budget mixed batching, done the way vLLM actually does it**:
  vLLM does NOT graph mixed batches with FlashInfer — FlashInfer is the
  backend that BLOCKS it; vLLM runs attention EAGER inside piecewise graphs
  for mixed/prefill steps, full graphs only for uniform decode. Our MIXED_WALK
  failed because the ENTIRE step went eager, not because mixing is wrong.
  Two viable shapes for us: (a) piecewise-style: graph the MoE/dense stack,
  eager FlashInfer prefill-wrapper attention for mixed buckets; (b) creative
  hybrid: FA3 varlen (validated correct on our KV layout, CG-ALWAYS capable)
  as the attention inside FULLY-graphed mixed buckets — FA3's per-kernel
  slowness vs FlashInfer matters little on occasional mixed steps replacing
  27.5ms serialized prefills. Also: vLLM V1 chunks VISION prefill via an
  encoder cache (EncoderCacheManager) — needed for chunked i2t prefill.
- **W4 — output-process isolation** (vLLM V1 EngineCore pattern): detok/
  stream/preprocess in a separate process. We're partway there (api_server
  is separate; inline/batch emit cut the transport); W2 txn covers most of
  the rest of the worker-side cost.
- **POD-Attention** (ASPLOS'25): fused prefill+decode attention kernel, up to
  +22% e2e — the kernel-level version of W5; revisit if W5(a/b) attention
  becomes the bottleneck.
- **Scoped out with evidence**: megakernels (batch-1/1B-model regime, no MoE
  support in MPK); free-threaded Python (not production-ready as of 11/2025);
  Rust/C++ scheduler rewrite (TRT-LLM's C++ executor — wrong cost/benefit for
  us given W3 exists in pure Python). Spec-decode for MoE/multimodal and MoE
  kernel claims did not survive adversarial verification — treat as unproven.

## W2 — decode-step transaction (`MSTAR_STEP_TXN`, commit b98de66 on exp/step-txn) — CORRECT but WASH
Full memoized replay of the per-rid graph-walk bookkeeping (3-4× fewer Python
ops on the routing slice), validated by a shadow-verify mode with ZERO
prediction mismatches over thousands of B8/B32 steps. Perf: i2t B32 0.995×,
B8 0.96×, s2t B8 1.00× — flat. Combined with E9/E10 this closes the case:
postprocess-side Python is NOT the binding serial chain at B32 (it overlaps
GPU); the residual is the submit gap (await→result→thread→submit, W3) and
prefill serialization (W5). NOT merged into the shipping branch — kept on
its branch as validated infrastructure if the balance shifts.

## W3 — future-token run-ahead — SKIPPED BY DESIGN ANALYSIS (do not implement)
The design audit (w3-design, 2026-07-02) corrected its own premises: the
decode GPU-thread path is ALREADY host-sync-free end-to-end (sampler is
deterministic flashinfer + fused triton, no .item()/sync; future.result()
returns near SUBMIT of N, not completion), so M* already holds the
SGLang-overlap win via speculation + double-buffer + pre-plan. The only
residual is the result→thread→submit inter-thread hop (≤~1ms/step);
projected ceiling 1.00-1.05× at B32, likely flat per E9/E10 precedent.
Not worth the implementation risk. All remaining B32 leverage concentrates
in W5 (prefill-stall elimination via token-budget mixed batching).

## W5 design (2026-07-02) — the B32 endgame, variant (c): CAPTURED mixed batches
Key code-verified finding (w5-design): M*'s prefill attention is FlashInfer's
PAGED-KV BatchPrefillWithPagedKVCacheWrapper, which already accepts a mixed
qo_indptr (N decode rows len-1 + one prefill-chunk row len-C); the split
between decode/prefill is literally one `all(sl==1)` check (cache_manager.py:
337). And M* already CUDA-graphs this wrapper via FlashInferPackedCudaGraph
machinery — the static-addressing problem that forces vLLM to run mixed
attention EAGER is already solved in our engine. So the mixed step can be
captured end-to-end: no eager (the June/E7 killer), no FA3 (fallback only),
no piecewise (the in-tree "PiecewiseCudaGraphRunner" is a V-JEPA2 helper,
not a GEMM/attention splitter). Causality is uniformly causal=True via
FlashInfer bottom-right alignment. Bucket grid bs×C ≈ 15 captures ≈ ~4GB
(affordable post-fp8). MRoPE resume = slice a staged (3,total) grid; vision
embeds staged once per request (~2MB). Phasing: P1 chunked-prefill-only
(~4-5d, expected 0.77→~0.85-0.90×, shippable alone) → P2 mixed captured
forward (~5-7d, →~0.90-0.95×+) → P3 grid/tuning (~3-4d). Risks table incl.
deferred mark_node_complete (Loop, E9), deepstack chunk alignment, MRoPE
pos_advance handling — all with token-identity gates.

## W8 — denser prefill token buckets (exp/prefill-buckets, 08e7123) — inconclusive standalone, adopted into W5-P1
Quick-bench on a different GPU pair than its reference (forced by a bursty
foreign user on 4,5): i2t B32 5.646 vs 5.910 ref — within cross-pair drift,
unresolvable at triage scale. TTFT signal positive: i2t B1 p50 0.187 vs
0.240 ref, B32 0.603 vs 0.645 (−22%/−7%, cross-pair caveat). Denser buckets'
real value is chunk-size granularity for W5-P1 (chunks are sized from
PREFILL_TOKEN_BUCKETS); the branch merges into exp/chunked-prefill-v2 when
P1 lands rather than shipping alone.

## W5-P1 — chunked prefill via step alternation — REJECTED as shipped form; plumbing validated for P2
Branch exp/chunked-prefill-v2 (76f3afe text, 6e39106 vision, 23d07ee readiness
fix — the silent hang was a missing empty-payload edge for a declared input).
Correctness: multi-chunk execution verified live (offsets 0/128/256), all
seq_len/position asserts green, outputs semantically identical to unchunked
(same dish IDs on every sample; bitwise divergence after a few tokens is
expected — chunked spans route through different fp8 tile configs and
FlashInfer split-KV schedules, so greedy amplifies ULP drift; same property
as vLLM's chunked prefill). Perf: i2t B1 0.82×, B8 0.90×, B32 0.82×,
s2t B8 0.73× — the per-chunk conductor round-trips (~3-4 per admission) cost
more than the stall they remove AT EVERY batch size. Verdict: alternation-
style chunking is structurally unprofitable in this engine; the validated
chunk cursor/staging/deferred-completion plumbing feeds W5-P2 (mixed
CAPTURED step: chunk rides the decode step, no extra round-trips), now the
sole path for the i2t B32 win.

## W5-P2 + P3-lite — CAPTURED mixed batches, implemented (2026-07-02 evening)
Branch exp/mixed-batch-p2 (on P1's 23d07ee): P2 = 1093527/b265cd1/7c09d10
(scheduler mixed assembly, per-request effective_walk routing — the batch-walk
routing variant of the silent-hang class was caught pre-GPU via the shared
trap warning; TP gate; thinker_mixed FlashInferPacked buckets 288/544 —
BOTH CAPTURED SUCCESSFULLY ON H200 at first boot, retiring the
capture-vs-replay risk's first half). P3-lite = 864d492/a599bb6/f45cbbd
(vision chunk rows in mixed steps: dense row-position-agnostic deepstack
packing with zero rows for decode + always-emit-under-flag to prevent
static-buffer bleed; per-request custom_pos_advance list for the two-part
MRoPE landing). Flags: MSTAR_MIXED_BATCH / MSTAR_MIXED_BATCH_VISION, default
OFF; CPU suites 16/16. Known perf wrinkle: vision-span tails (258=256+2)
miss buckets → eager fallback per admission; tail-merge planned if frequent.
GPU ladder: text-mixed smoke → vision-mixed smoke → B32 interleaved A/B vs
shipping config.

## W5-P2/P3 first integration — mixed steps WORK but chain-break wiring LOSES
Fix 0cc7c71 (chain-break-for-mixed) + tail-merge 8595fc4: mixed assembly
CONFIRMED LIVE (88 captured mixed steps at B16 probe, n_decode=13-15 C=256,
zero asserts, healthy outputs — the captured-mixed machinery is correct end
to end, retiring all remaining capture/numerics risks). But the A/B
(ab_p2/): i2t 0.959/0.956/0.912× — LOSES at every batch. Economics: each
admission's mixed step requires breaking the speculation chain (decode
nodes are invisible to the scheduler while _speculatively_scheduled — root
cause confirmed at base.py:729), costing ~2-3 non-overlapped pipeline steps
(~15-25ms) against ~9ms of stall saved. CONCLUSION: mixed batching can only
win inside the speculation chain (spec-capable thinker_mixed: _can_speculate
allowance + spec-path per-request routing for mixed batches) — the deferred
deep variant is now the required next step, sized ~1-2 days. All machinery
(capture, assembly, routing, chunk plumbing, tail-merge) is validated and
waiting for it.

## W5 spec-chain fold + bucket fix — FIRST NET-POSITIVE MIXED RESULT (+3.2% i2t B32)
Iteration trail (all measured, all committed on exp/mixed-batch-p2):
chain-break wiring 0.912× → spec-chain fold (505c833/3c42b5d/e31f39e:
mixed batches assembled INSIDE the speculation chain — 41 in-chain folds vs
3 breaks at B16 probe, B1/B8 penalties erased, B32 0.944×) → bucket fix
(62472cf: tail-merged 258-token chunks + 31 decodes = 289 tokens overflowed
the 288 bucket by ONE token → every fold padded to 544, ~88% waste; added
C=288 bucket → total 320) → **back-to-back B32: 6.179 vs 5.986 shipping =
+3.2%, the first positive mixed-batch number.** Remaining tuning levers:
pre-plan support for packed spec batches (mixed step still plans inline),
fold-rate telemetry, bucket grid. Full interleaved A/B running for the
definitive record.

## W5 FINAL (2026-07-02): spec-fold mixed batching = NET WIN, one lever from decisive
Definitive interleaved A/B (ab_p2spec/, bucket-fixed build 62472cf, full
flag stack vs shipping): i2t 1.027/1.017/0.969×, s2t 1.040/1.066/1.007× (complete 6-cell table; 5 positive). The i2t B32
cell straddles unity across measurements (interleaved 0.969, same-pair
back-to-back +3.2%) → true effect ≈ 1.00±0.03. Verdict: mixed batching is
positive at B1-B8 on both text paths, neutral at B32, fully captured, fully
validated (zero asserts/misses across all runs). NEXT LEVER (sized to the
residual): pre-plan support for FLASH_INFER_PACKED spec batches — the folded
mixed step still plans FlashInfer inline on the GPU thread (~1-3ms exposed
per fold) where decode steps enjoy pre-plan overlap. After that: fold-rate
telemetry + bucket grid. Ship posture: flags validated and available;
default-on recommended after the pre-plan increment clears B32 ≥1.03×.

## Packed pre-plan (MSTAR_MIXED_PREPLAN, commits 49d0204..824b84a) — stream-race theory RETRACTED; kept default-off (neutral)
Original verdict (off 6.141 / on+asserts 5.465 / on-clean **3.083**) blamed a
plan_stream concurrency defect. Follow-up (2026-07-02 22:15-23:00) REFUTED
that: (1) nsys-profiled on-clean run was healthy — 4.42 req/s WITH profiler
overhead, await_plan median 3.4µs, 577/640 plans skipped as pre-planned,
zero fold misses (_mix_opp == _fold_ok exactly, WALK_STATS counters); (2)
the slow mode is BISTABLE and time-clustered, not config-deterministic:
same server, 3 consecutive i2t B32 cells gave 3.748 / 3.662 / **5.567**,
and 8 later cells (fresh server, counters on) all landed 5.30-5.92. Every
slow datapoint (3.08, 3.75, 3.66) fell in one ~25-min wall-clock window on
this shared box (host load seen up to 165); nothing slow reproduced after.
The assert-heals-it observation was coincidence of timing, not masking.
Remaining truth: on-clean measures ≈5.3-5.9 vs off 6.141 — neutral to
slightly negative, and the theoretical win (skip ~1-3ms plan on the ~7% of
steps that are mixed) is <1% — so the flag STAYS default-off, but the code
is sound; no stream debugging owed. Lesson reinforced: time-separated
cells on this box can swing ±40% under foreign load; only interleaved A/B
or many repeats count. Datapoints: qb_preplan.log / qb_pponclean.log /
qb_pprecheck / qb_ppstats2 (WALK_STATS counter log in server.log).

## W5 fold-rate: MSTAR_MIXED_SINGLE_CHUNK + eager folds + occupancy floor (exp/fold-rate, 2026-07-02 late)
Counters (WALK_STATS) showed only ~35% of i2t prefill work folds into mixed
steps and 100% of standalone prefill_text steps are UNCHUNKED short spans
(<=256 tok) — excluded by the mixable gate (needs prefill_chunk_len). Fix
iterated three times, each stage caught by counters:
1. Single-chunk planner alone: catastrophic (i2t B32 6.18->3.48) — folds
   fire only at must_yield_away (~every 8th step) so chunk drain is ~8x
   slower than standalone and admission starves decode occupancy (~3900
   chain steps vs ~2300 for identical tokens).
2. + eager folds (every chain step) + n_decode>=24 floor (default only
   under the flag; P2 behavior untouched): mechanism works — folds 415 vs
   180, standalone text steps 218 vs 382, per-step _ms identical ON vs OFF
   (prefill_text 17.2 vs 17.4ms — the 1-chunk chunked path costs the same
   as single-shot; "eager fallback" theory dead). req/s ~neutral: ON 5.67/
   5.75 i2t B32, 16.6/16.7 s2t B8 vs OFF 6.00/4.05, 15.7/17.6 (OFF's own
   4th cell crashed to 4.05 — earlier "regressions" were server-age/box
   artifacts; this box lies at cell granularity).
3. THE CEILING (per-step _ms, measured): a mixed step costs 36ms but
   replaces prefill(17.2) + decode(11.8) = 29ms — folding is compute-
   NEGATIVE ~7ms/fold. The P2 net-positive came from avoided chain breaks,
   not compute. Root cause: packed capture runs decode rows through the
   PREFILL kernel path (qo_len=1 rows lose split-KV decode optimizations)
   — the POD-attention problem. NEXT LEVER: split attention inside the
   mixed capture (decode wrapper for 1-token rows + prefill wrapper for
   the chunk row, both in one graph) -> mixed step ~decode-cost -> every
   fold a real ~10ms win x ~400 folds/cell at i2t B32.
Also: MSTAR_DYNFLAGS runtime flag file + dyn_ab.sh = one-server interleaved
A/B (no restart between configs, adjacent-in-time cells cancel box noise).
Triage protocol from now on; committed numbers stay static-env.

## Split-attention microbench (mb_split_attn.py, in-graph, 2026-07-02 23:50) — LEVER CONFIRMED 4.3ms/mixed-step
The mixed step's packed attention (ONE BatchPrefillWrapper over [1]*31 +
[256]) costs 6.112 ms per 48-layer forward; decode-wrapper[31 rows,
tensor-core] + prefill-wrapper[256] back-to-back in the same graph cost
1.841 ms — 3.3x faster, 4.27 ms saved per mixed step. Note: M*'s decode
wrapper already uses use_tensor_cores=True (flashinfer routes it through
the prefill kernel internally; plain decode kernel REJECTS GQA group 7),
so the win is from per-shape PLANNING/load-balancing, not the kernel
itself — a single mixed-shape plan is what's catastrophic. This flips fold
economics (mixed 36ms - 4.3 ≈ 31.7 vs 29 replaced, + saved chain breaks)
and also speeds the already-shipped P2 mixed steps. Interleaved dyn_ab
verdict on single-chunk WITHOUT this fix: i2t B32 0.896/0.999/0.950 (−5%)
— stays off until split-attention lands. Build plan: fixed row regions
([n real decode][bs-1-n qo=1 dummies][chunk @ row bs-1][zero pads]) so the
static graph can slice at bs-1; split wrapper plans decode part + chunk
part separately. Flag MSTAR_MIXED_SPLIT_ATTN.

## MSTAR_MIXED_SPLIT_ATTN — WIN (+4.4% i2t B32 over non-split, rescues single-chunk folding)
Implemented on exp/fold-rate @ 827ab50 (FlashInferSplitMixedWrapper +
fixed-region packed layout + slot permutation through metadata/logits/
restore + bucket adjustment num_tokens += padded_bs - batch_size). A/B
(alternating servers, single-chunk ON both sides, i2t B32 ×4 cells/side):
OFF 5.678/5.448/5.819/5.790 (mean 5.684) vs ON 5.953/5.854(asserts ON!)/
5.866/6.053 (mean 5.932) = +4.4%, ON never below any adjacent OFF, zero
assert failures, tok/req sane. Fail-fast catch during bring-up: tail-merged
C=258 fold overflowed the 288 bucket's 257-token chunk window — fixed by
the bucket adjustment; env flag snapshotted process-static so dynflags
can't desync bucket math from baked captures. DISPROVEN along the way: the
eager-fold peek-cost theory (_ms_peek measured 2-4ms TOTAL per ~1500 peeks
≈ 2µs each — 100x below the estimate; backoff kept, harmless). Note
_ms_thinker_mixed did NOT drop per step (33.3 -> 34.2ms): the split plans
TWO wrappers inline on the gpu-thread (+~1-2ms CPU submit) masking the
-4.3ms GPU attention win in that counter; e2e req/s is the arbiter.
Ship-decision A/B (plain W5 vs +single-chunk+split) running as ship_ab.sh.

## MSTAR_MIXED_SINGLE_CHUNK — CLOSED, net-negative even under split+preplan
Three independent A/Bs converge: sc1 dyn_ab (no split) 0.947 geomean;
ship_ab2 (split+preplan, cross-server) ~0.97; scsplit dyn_ab (split+preplan
static, one server, adjacent cells) pairs 0.949/0.904/0.983/1.104 geomean
0.983. Mechanism understood end-to-end: eager short-span folding trades a
17-18ms standalone prefill + 12-13ms decode step for one ~30.4ms mixed step
(post-split+preplan) — roughly wall-neutral per fold — but degrades decode
occupancy ~10% (4439 chain steps vs 4012 for identical tokens; admission
rides fold slots instead of immediate standalone prefill). The occupancy
loss dominates. The lever's residual value was folding-at-yield-boundaries,
which plain W5 already does. Flag stays OFF; planner/eager/floor/backoff
code retained on exp/fold-rate for the record.

## Retained from the fold-rate campaign
- MSTAR_MIXED_SPLIT_ATTN + MSTAR_MIXED_PREPLAN cut the ORIGINAL W5
  yield-boundary folds' mixed steps 33.1 -> 30.4 ms (measured within
  ship_ab2 legs) — pure profit on the shipping fold pattern. Endgame A/B
  (ship_final.sh: W5+FAST_POSTPROC ± split+preplan) decides default-on.
- Diagnosis/velocity tooling now standard: MSTAR_WALK_STATS (+_ms, chain
  health, peek timing), MSTAR_DYNFLAGS + dyn_ab.sh (one-server interleaved
  A/B), split_ab/ship_ab alternation scripts.
- MSTAR_FAST_POSTPROC=1 was MISSING from all fold-rate-era flag sets while
  the committed v2 sweep includes it — absolute numbers tonight read ~3-7%
  low; all deltas remain valid (both sides equally affected).

## Endgame A/B verdict + protocol change (2026-07-03 ~01:45)
ship_final round 1 (FAST_POSTPROC both sides): base 6.133/5.863 vs
+split+preplan 5.758/6.229 — WASH, as arithmetic predicts: yield-boundary
folds are ~8% of steps; 2.7ms/step saving = +0.6% e2e, below this box's
noise floor. MSTAR_MIXED_SPLIT_ATTN / MSTAR_MIXED_PREPLAN stay OPT-IN
(validated, harmless, sized-correct); they become valuable only if a future
scheduler change raises fold volume without the occupancy tax. PROTOCOL:
effect-size gate — predicted-sub-2% effects get microbenches or arithmetic,
not e2e cells; QB_FAST=1 halves cell sizes for triage. GPU time moves to
the ~25% residual: fresh nsys re-decomposition (prof_winner) running.

## Sampler config-tensor cache (MSTAR_SAMPLER_CFG_CACHE) — REGRESSION; the GIL-valve insight
nsys: cg.sample_and_remap = 10.9ms/step, 85% cudaStreamSynchronize — SIX
syncs/step, repro'd exactly with set_sync_debug_mode: the six
torch.tensor(list, device=...) config uploads each do pageable-H2D + stream
sync, draining the in-flight pipeline. Cache (keyed by batch membership,
rand_offset advanced on-device) kills all 6 syncs, token-identical, 2x
faster in isolation (0.44 -> 0.20ms) — and LOSES ~5-7% e2e (4/4 adjacent
pairs, i2t B32). WHY: the blocked gpu-thread RELEASED THE GIL during those
waits; the main thread's ~10ms/step postprocess Python (route 2.5 +
check_stop 2.0 + register 1.1 + send 3.4) ran in that shade. Remove the
waits and the threads contend. **Architecture law: on the two-GIL-thread
worker, removing gpu-thread waits pays ONLY after main-thread per-step
Python shrinks or moves off-GIL.** This re-orders the roadmap: main-thread
postprocess reduction FIRST (extend FAST_POSTPROC memoization to
route_outputs; batch check_stop consumption; emit off-process), THEN
de-sync sample (flag kept for that re-test), THEN defer-sample overlap.
Default OFF (9ab0b2d, exp/overlap-sched).

## The i2t B32 wall, fully measured (2026-07-03 02:45, prof_winner trace)
Step 18.9ms, GPU 46.6% busy. Main thread 13.2ms/step Python
(postprocess_batch 9.78 = route 2.50 + check_stop 1.99 + register 1.14 +
completion-sync 0.90 + ~3.2 loop shell; send_outputs 3.41). GPU thread
11.1ms (sample_and_remap, 85% pipeline-drain syncs). Two GIL threads whose
Python sums past the step time — the sync-shade overlap is what makes it
"work" at all. Piecemeal shaving converted poorly three times tonight
(single-chunk, split e2e, sampler cache). Next swing (task tracked):
relocate per-token route/store/emit/check_stop off the hot threads —
third-process SHM-ring consumer (vLLM V1 EngineCore pattern) or full
vectorization of the per-rid loop; then re-test the sampler cache and
defer-sample, which should both flip positive once the GIL shade is gone.

## MSTAR_SLIM_EMIT (afb1099, exp/overlap-sched) — REGRESSED 5x, pending api-side debug
Steady-state token emits send SlimResultTokens (values only) after a first
full per-(rid,name) template; api server inflates from the cached template.
Outputs fully correct (tok/req ~170, zero errors/warnings) but i2t B32
collapsed 5.9-7.6 -> 1.2 req/s, jct 16.5s. Worker WALK_STATS cadence stayed
~15ms/step with intermittent stalls => the delay is DOWNSTREAM (api_server
message loop / data-worker chunk delivery throttling the closed-loop
client), not worker compute. Needs api-side timing instrumentation
(template-inflate path, data worker queue latency, chunk->client event
timing). Flag default OFF; code kept for the debug.

## Protocol caveat: QB_FAST cells
n=48 cells show ±25% same-config spread and a different token mix
(~110 tok/req vs 177 at n=96). Smoke/correctness only; ship decisions at
full n=96 with multiple adjacent pairs.

## MSTAR_SLIM_EMIT — WIN +15-25% i2t B32 (biggest e2e win since fp8)
After the template-snapshot fix (b762a5d; the data worker MUTATES
graph_edge.name on its own thread — caching the live object sent slim
items under the renamed key, loop-index accounting missed, every request
rode the 15s TTL): full-cell A/B on GPUs 0,1, adjacent pairs — r3 off
5.014/4.986 vs on 5.607/6.124 (+17%); r4 off 3.700/4.479 (degraded window)
vs on 5.311/4.962 (+26%, ON healthy through the same window). tok/req 177
both sides. Mechanism: steady-state token emits skip the per-rid GraphEdge
pickle (send SlimResultTokens; api server inflates from a per-(rid,name)
template snapshotted BEFORE routing). This is the first converting cut into
the ~13ms/step main-thread Python and validates the remove-work-not-waits
law. NEXT: stack MSTAR_SAMPLER_CFG_CACHE on top (its GIL-shade objection
weakens as the main thread lightens), then route/check_stop memoization.

## Stack test: slim + sampler-cache — cache still PARKED (wash/−1%)
Adjacent pairs on 0,1: A(slim) 5.587/5.621, 5.873/5.639 vs B(+cache)
5.364/5.636, 5.531/5.912. Cache regression shrank from −5-7% (pre-slim) to
≈−1% (with slim) — the GIL-shade account tracks quantitatively: as
main-thread Python shrinks, removing gpu-thread waits approaches breakeven.
Re-test after FAST_ROUTE (and any further main-thread trims) land.
Winning stack so far: W5 + FAST_POSTPROC + SLIM_EMIT.

## MSTAR_FAST_ROUTE — WIN +7-10% i2t B32 (267e5cf)
Memoized replicated fanout decisions per request instance (replay = clone +
_shard_dim/_total_fanin; sharded fanouts keep the full path). Adjacent
pairs on 0,1: 5.402/5.717 vs 6.164/6.114 (+10.4%); 5.637/5.309 vs
6.071/5.640 (+7.1%). Zero routing errors, tok/req 177. WINNING STACK now
W5 + FAST_POSTPROC + SLIM_EMIT + FAST_ROUTE: pair-0,1 band 5.0 -> ~6.0
(+20% tonight, compounding, all from main-thread Python removal — the
remove-work law converting cleanly twice in a row).

## Sampler cache round 3 (on slim+route) — WASH; trend −7% → −1% → ~0%
Pairs: 5.79→6.02 (+4.0%) then 5.91→5.64 (−4.6%, one 5.14 noise-dip cell).
The GIL-shade trend keeps tracking toward crossover but is not yet a
proven win. Cache stays default OFF; re-test after the next main-thread
trim (check_stop/register/loop-shell) lands. WINNING STACK (locked for
the sweep): W5 + MSTAR_FAST_POSTPROC + MSTAR_SLIM_EMIT + MSTAR_FAST_ROUTE.
NUMA note: all pair-0,1 numbers carry a cross-NUMA handicap (quick_bench
hardcodes cpunodebind=1; GPUs 0,1 are node 0) — deltas fair, absolutes
understated; the canonical sweep runs on 6,7.

## Option board 2026-07-03 (~05:00) — 12 investigation options, three research passes
NEW from M* code deep-read:
 N1. check_stop+register fast-path for decode chains (batch check_stop_batched
     over the existing pinned buffer; cache the all-inline/no-route flag per
     (rid,node,walk); stop computing _inline_emit_uuids twice) — ~1.5-2.2ms/
     step main-thread (~8-12%), 2-3 days.
 N2. Kill sleep-quantized hops: conductor unconditional time.sleep(0.001)/loop
     (conductor.py:1143) -> blocking poll; EventWakeup fd for data-worker +
     api_server 1ms polls; asyncio.Event for SSE. Latency/tails: TTFT −6-10ms
     on chunked prompts, jitter −3ms. 1 day.
 N3. set_config change-detect in prepare_batch (kv_cache_engine.py:802) +
     SCOPED cache invalidation — CRITICAL FINDING: set_config runs per rid
     per step and calls _batch_cfg_cache.clear(), so MSTAR_SAMPLER_CFG_CACHE
     NEVER HIT in any A/B (all three "trend" points measured a
     structurally-dead cache). ~0.3-0.5ms/step direct + unblocks the ~9ms
     sync lever. Half day. DO FIRST.
RE-TESTS from execution audit:
 R1. Sampler cache — REDO after N3 (previous verdicts void).
 R2. E10 two-step decode + E9 direct feed, rebased onto winning stack,
     3-arm with cache (+2-8% predicted; tested pre-every-win, self-
     handicapped by inline step-2 syncs).
 R3. GIL switch interval {unset,0.001,0.01} + NUM_SLOTS=3 piggyback —
     cheap sweep, single-cell pre-slim verdicts unreliable.
FROM vLLM-Omni source dive (their edge is 100% host-side; kernels equal;
their speech loss is structural — pickle+flock+shm stage handoffs):
 V1. Async scheduling / GPU-resident sampled ids (D2H off-thread via copy
     stream + event; placeholders repaired lazily) — largest lever, up to
     +30-60% e2e; builds on E9+N3.
 V2. Budgeted chunked-prefill interleave POLICY into our already-captured
     mixed graphs (decodes always scheduled, leftover budget = one prefill
     chunk) — kills phase drain, +15-30% under arrivals.
 V3. Persistent batch + diff application + change-triggered sampling
     metadata (steady state = O(1) checks + 32-row pinned copies).
 V4. Detok/serialization out of the conductor process (ZMQ IO threads
     release GIL; one batched EngineCoreOutputs per step).
 V5. Encoder mm_hash cache + per-step encoder budget (batched ViT call).
 V6. Gumbel/exponential sync-free sampler + resident config tensors +
     all-greedy/no-penalty gates (multinomial forces syncs).

## N3 landed + cache4 (first LIVE cache A/B) — mechanism proven, e2e verdict noise-blocked
N3 (fddc622): change-detect set_config in prepare_batch + scoped cache
invalidation. Repro-verified: cache persists across steps, syncs 6 -> 0,
invalidation still fires on real config changes. ALL prior cache A/Bs are
void (dead cache). cache4 live A/B: pair 1 +4.5% (contains the night's
best cell 6.507) but pair 2 hit a deep degraded window (4.03, jct 7.2s) —
no verdict under this noise. Cache stays flag-parked with the mechanism
proven; definitive A/B bundled with N1 in a quieter window / on 6,7.

## FINAL STACK LOCKED (2026-07-03 ~06:00) — cache+checkstop +10.5% sealed; night total +31% at i2t B32
Decisive 3-round A/B (0,1, full cells, N3 change-detect in code both sides):
locked stack 5.69/5.89/5.70 vs +MSTAR_SAMPLER_CFG_CACHE+MSTAR_FAST_CHECKSTOP
6.57/6.57/5.97 — pairs +15.4%/+11.6%/+4.7%, geomean +10.5%, tok/req 176.9
exact, zero errors. The void-verdict saga resolved: with N3 the cache is
LIVE (6 pipeline-drain syncs -> 0) and finally converts; N1-lean batched
check_stop rides along. FINAL STACK: MSTAR_MOE_FP8 + MSTAR_BATCH_EMIT +
MSTAR_FAST_POSTPROC + chunked-prefill-v2(+vision) + MSTAR_MIXED_BATCH
(+vision,+spec) + MSTAR_SLIM_EMIT + MSTAR_FAST_ROUTE +
MSTAR_SAMPLER_CFG_CACHE + MSTAR_FAST_CHECKSTOP, encoff config.
Night progression at i2t B32 on the (cross-NUMA-handicapped) 0,1 pair:
~5.0 -> ~6.4 avg, best cells 6.894/6.787 (+31%). Projection to canonical
6,7 (base band 6.0-6.2): ~7.5-7.9 req/s vs vLLM 8.21 = ~0.92-0.96x, from
0.77x at session start. Full coverage sweep of the final stack running;
canonical-pair sweep queued behind the foreign job on GPU 6.
Remaining options to close the last ~5-8%: board of 2026-07-03 (V1 async
sched / GPU-resident ids is the headliner; N1-full register/route caching;
N2 latency hops; V2 budgeted interleave policy).

## Final-stack coverage sweep (pair 0,1 — cross-NUMA caveat; canonical will read higher)
i2t: B32 6.134 / B8 3.936 / B1 0.844 — vs committed v2 (canonical pair):
6.299 / 3.361 / 0.800 and vs vLLM 8.210 / 3.455 / 0.898.
**i2t B8 now BEATS vLLM: 1.14x** (was 0.97x); B1 0.94x (was 0.89x);
B32 at committed-parity on a worse pair (canonical projection 0.9x+).
s2t: B32 31.219 / B8 18.471 (committed 31.565 / 17.197 — B8 +7%).
s2s B8 7.497 + 32.53 audio-s/s (−5%/−13% vs committed — NUMA + audio path
untouched by tonight's text opts; audio-optimal alt config unaffected).
i2s B8 1.250 / 56.15 audio-s/s (parity). No path broken; text paths up
across the board. Canonical 6,7 sweep queued behind the foreign 129GB job.

## Sidecar bundle (SLIM_EMIT2+FAST_ROUTE2+FAST_SEND+EMIT_SIDECAR, ab75034) — WIN, canonical-pair validated
Commits fd8eae5/5c87cb8/333b7dd/c8a08fc..ab75034 (route-classification memo,
int loop-keys + skip unused ResultTensors, residual emit-path Python cuts,
postprocess/emit exiled to a sidecar process — vLLM V1 EngineCore pattern —
with committed byte-identity harness). Evidence trail: pair-0,1 adjacent
quick-bench 6.825/6.842 ON vs 6.582/4.935 OFF (one clean pair +3.7%, one OFF
cell degraded); FAST_SEND standalone slightly negative (dyn_fs), kept as the
bundle's rider. CANONICAL 6,7 adjacent A/B (2026-07-03 19:39-19:59,
qb_canonoff vs qb_canonon2, same cells back-to-back): i2t B32 6.378 -> 6.962
(+9.2%), i2t B1 0.777 -> 0.841 (+8.2%), tok/req sane both sides. Verdict:
bundle POSITIVE at B1 and B32 on the canonical pair; stays in the final stack.

## Canonical-pair sweep of the final stack (qb_canon_ab75, 2026-07-03 19:27-19:39) — landed, one degraded window
First successful canonical (6,7) sweep of ab75034 + full flags. Ratios vs
committed vLLM: i2t 0.50x/0.82x/0.88x/1.16x/1.03x/0.73x (B1..B32),
s2t B8 1.14x, s2t B32 1.11x (34.233 — best s2t B32 ever recorded).
CAVEAT: the window was degraded — the same config re-measured adjacent to the
OFF leg 20 min later gave i2t B32 6.962 (0.85x) and B1 0.841 (0.94x) vs the
sweep's 5.978/0.447. Cell-granularity box noise (documented class), NOT a
config effect (adjacent A/B above proves the bundle positive). Honest
canonical statement: i2t B32 = 0.73-0.85x band, best adjacent reading 0.85x;
the +10% NUMA projection is NOT confirmed (pair-0,1 ON cells 6.82-6.84 vs
canonical ON cells 5.978/6.962 overlap in scatter). s2t B32 1.11x from the
DEGRADED window — likely understated, report as >=1.1x.

## N2 re-smoke — WEDGED again, default reverted
qb_n2smoke: 0.000 req/s (server wedged); default reverted OFF at 4a19bbb
("wakeup-source gap"). The b1c1ff1 fd guard was insufficient — root cause is
a real missed-wakeup source, not just the unguarded event fd. N2 is
RE-OPENED; the durable fix (wait_for_work inside try/except + enumerate all
wakeup sources incl. dynflags poller) is queued, do not re-enable by default
until a smoke passes with load.

## MSTAR_SCHED_PACK (opt/sched-pack da8024e) — REGRESSION, parked
Fairness-peek exponential backoff (cap 8) + shared routing flattens.
Interleaved dyn_ab on canonical 6,7 (one warm server, A=off B=on, i2t B32
x3 rounds): 7.438->6.939 (0.933), 7.353->7.000 (0.952), 4.986->5.100 (1.023,
both cells in a degraded window) = geomean 0.969, clean pairs -5..-7%.
Mechanism hypothesis: fairness yields are LOAD-BEARING — they are the fold/
admission boundaries the W5 spec-fold path fires on; backing off the peek
delays admission and taxes decode occupancy (same failure class as
MIXED_SINGLE_CHUNK). Process note: WALK_STATS was not set, so fair_peek
counters weren't captured; the consistent adjacent-pair delta is the
behavioral evidence. Retry only as (a) flatten-share alone bundled into a
bigger pack, or (b) admission-aware gating (skip peek only when the waiting
queue is provably empty — O(1) len check). Code kept on opt/sched-pack.

## Warm-server effect — dyn_ab baselines read 7.35-7.44 at i2t B32 (0.90x vs vLLM)
The same final-stack config that fresh-boot quick_bench measures at 5.98-6.96
req/s reads 7.35-7.44 on a warmed one-server dyn_ab (canonical pair, A-side
cells, adjacent rounds; jct ~4.0s, tok/s ~1294). Best B32 measurements of the
campaign = 0.90x vs vLLM 8.210. Implications: (1) committed sweep numbers
understate steady-state (fresh boot = cold JIT/caches/allocator; sweep
protocol starts a fresh server per config); (2) cross-run comparisons must
hold server age constant (dyn_ab does); (3) the honest public claim is
"0.83x fresh-boot protocol, 0.90x steady-state warm server" pending a
warm-up-normalized sweep protocol (e.g. lengthen warmup or discard the first
N minutes per server).

## Cross-NUMA penalty — MEASURED (+15.5% B32, +6.6% B1)
Direct same-pair A/B on GPUs 0,1 (fresh server per leg, same final stack,
2026-07-03 20:27-20:45): node-0 binding (correct) i2t B32 6.411 / B1 0.821
vs node-1 binding (the historical hardcoded cross-NUMA case) 5.553 / 0.770.
Penalty = +15.5% at B32, +6.6% at B1 (single cell per leg — band caveat).
The "+10%" folklore was real and at B32 understated. Consequences:
(1) all historical pair-0,1 numbers (measured cross-NUMA) understate by
~7-15%; (2) quick_bench.sh/dyn_ab.sh now DERIVE the numa node from the GPU's
PCI bus (QB_NUMA_NODE/DYN_NUMA_NODE override) — the trap is closed;
(3) correctly-bound 0,1 reads in the canonical 6,7 band (6.411 vs 6.38-6.96
fresh-boot) → both pairs are now valid bench pairs, enabling true two-track
parallelism.

## Warm-server characterization (dyn_warmchar, 6,7, 8 identical final-stack cells) — protocol finding
Same config, one server, i2t B32 cells in time order: 5.038 (first cell after
ready) then 7.268/7.223/7.049/7.095/7.043... — first cell reads ~30% low;
steady band 7.04-7.27 (~±1.6%). Server maturity (JIT/allocator/caches) spans
~100+ requests, far beyond --num-warmup. Fixes landed: quick_bench.sh now runs
a discarded warm cell per distinct path (opt-out QB_NO_WARMCELL=1);
lab_server.sh/lab_ab.sh added — persistent warm server + dynflags A/B =
~2-6 min signal per A/B pair (boot cost paid once per session). Honest B32
statement: fresh-boot protocol 0.73-0.85x, warm steady-state 0.86-0.89x
(7.04-7.27 vs 8.210). All historical fresh-boot sweep numbers (BOTH systems'
committed baselines included — vLLM's 8.210 was also a fresh-boot protocol)
carry this cold-cell bias in their FIRST cell only; our sweeps ran B32 first,
vLLM's committed run order unknown — flag when comparing.

## R2-lite: MSTAR_DIRECT_FEED on the sidecar stack (lab_main/ab_r2lite_directfeed) — WASH, confirmed again
Warm-lab interleaved A/B (one server, dynflags flips confirmed in server.log,
i2t B32 x2 rounds): 6.192->6.133 (0.990), 6.313->6.252 (0.990). E9's wash
verdict holds on the post-sidecar stack: the token-feed hop is not the
binding cost even with the lighter main thread. Direct-feed remains
correctness-validated infrastructure for a future E10-class two-step decode;
flag stays OFF. First experiment through lab_server/lab_ab: 4 full n=96
cells in ~7 min against the standing warm server.

## Cache-alone sanity on the warm lab — WASH (cache stays ON as harmless)
lab_ab cache_sanity (A=full stack, B=same minus SAMPLER_CFG_CACHE;
FAST_CHECKSTOP on BOTH sides): 6.258 vs 6.490 (-3.6%) then 6.534 vs 6.369
(+2.6%) — mixed signs, ON/OFF geomean 0.995 ~= noise. The +7-10% "cache+
checkstop" win does not decompose onto the cache alone on the current
(sidecar) stack; either checkstop carries it, the pair interacts, or the
original delta was window-inflated. Flag stays ON (never harmful). If the
decomposition ever matters: 3+ rounds each of cache-only / checkstop-only /
both on the lab.

## Checkstop-alone decomposition (lab_node0, 3 rounds) — WASH; the pair-win does not decompose
OFF/ON per round: 0.981/0.990/1.079, geomean ~1.02. With cache-alone also a
wash (see above), NEITHER half of the historical "+7-10% cache+checkstop"
reproduces in isolation on the sidecar stack. GIL-valve-consistent: the
sidecar removed the main-thread Python that those gpu-thread sync-removals
used to convert against. pair_off 3-arm (both off vs both on) queued to
close it. Flags stay ON pending that (harmless).

## Two-lab contamination caveat (protocol)
lab1 warm-coverage absolute cells (i2t B2 1.07 / B4 1.71 / B16 4.74, s2t B8
15.9 / B32 30.1 mean) ran CONCURRENT with lab2 boot+cells on the same box —
same-config A/B spreads hit ±11-15% and several cells read below both
fresh-boot and committed values. RULE: two concurrent labs are valid for
INTERLEAVED RATIOS ONLY (contention ~cancels within adjacent pairs); absolute
/scoreboard numbers require a solo-lab or quiet box. The warm-coverage
absolutes are NOT scoreboard-grade; scoreboard refresh re-queued for a solo
window.

## Decomposition FINAL (6 rounds/flag, cross-pair pooled) — the flippable-flag space is closed
FAST_ROUTE off/on pooled: 0.958/0.864/1.001 (lab2) + 0.941/1.093/1.060 (lab1)
= geomean -1.7%, spread 0.86-1.09 — NOT cleanly converting post-sidecar (the
single-lab -6.1% was one hot round). cache+checkstop pooled: -1.1% for off.
VERDICT: on the sidecar stack every individually-flippable flag measures
0-3%, below the box's practical resolution (adjacent rounds still spread
±10%). All flags stay ON (small positive direction, zero harm). The
compound stack vs pristine base remains hugely positive; the marginal
decompositions are noise. PROTOCOL: stop micro-probing; only effects
predicted >=5% get e2e rounds from here. NEXT: V1 async scheduling (est.
+5-10%, resolvable) — the last structural mechanism vLLM has that we lack.
Late-evening box drift note: absolute cells sank 7.0-7.4 -> 5.2-6.4 band
across the evening (both labs); ratios unaffected, absolutes not
scoreboard-grade.

## Warm-protocol official sweep (solo box) — ABORTED by priority change
Booted the opt/sched-pack shipping stack warm on GPUs 6,7 (solo box, numa=1,
port 8299) and started the rounds=2 official scoreboard sweep. User priority
change: no absolute-number sweeps until the experiment queue is exhausted, so
the sweep was aborted mid-run and the warm server was HANDED OVER (not killed)
for ratio A/Bs. Only i2t completed round 1: B32 7.287/7.205, B16 5.360/5.710,
B8 4.089/3.974, B4 pair (8 cell-runs, A+B x 4 cells). i2t B2/B1, s2t, s2s, i2s
and round 2 never ran. i2t B32 mean ~7.25 sits in the earlier 7.0-7.4 band
(not the evening 5.2-6.4 drift band). Partial cells in lab_sweep/ab_official/
are TRIAGE-GRADE ONLY — not scoreboard numbers.

## MSTAR_MIXED_SINGLE_CHUNK at mid-batch (scmid, sloppy POC) — WASH, hypothesis dead
Warm-lab i2t B2/B4 x2 rounds: B2 0.952/1.004, B4 1.009/1.040, geomean 1.00.
Eager folding does NOT flip positive at low batch — the B2-B4 deficit is
admission latency + fixed host floor (V1/V2 levers), not fold policy. Closed
at POC depth per sloppy-fast mode.

## #24 MoE grouped-GEMM bake-off (mb_moe_bakeoff/, GPU 2, in-graph) — CLOSED: shipped Triton kernel wins every M
In-graph per E1 law, Qwen3-Omni MoE shapes (E=128 top_k=8 H=2048 I=768),
random weights (timing only). Pure-GEMM lever (2 grouped GEMMs, ~90% of the
MoE forward): Triton fp8 w8a8 beats DeepGEMM masked at every decode M
(dg/tri 1.10-1.20x — DeepGEMM REGRESSES). Structural: DeepGEMM masked has a
64-row-per-active-expert tile floor; at decode nearly all 128 experts are
active with 2-8 real tokens each, so it pays ~E_active x 64 rows vs Triton's
token-sorted ~M x 8. It's an EP/large-batch kernel, wrong for dense low-M
decode. Tile sweep at the new 24/28 buckets: shipped BLOCK_M=16 already
optimal (best gain found 1.7% sub-gate). Inventory: sgl_kernel/flashinfer
full-fused MoE ops exist but are not drop-in GEMM swaps; the one worth a
follow-up is flashinfer trtllm_fp8_block_scale_moe (launch-fused low-latency
decode MoE) — queued. Node note: persistence_mode reads Enabled on 2/3,
PRE-EXISTING, untouched (no admin).

## Speech bundle #15 build + a premise correction (opt/speech-floor)
Item A DONE (55faf29): MSTAR_FAST_CHECKSTOP_TALKER — batched talker stop
check (one flat D2H of layer0_codes + int compares), walk-gated,
dynflags-refreshable, WALK_STATS counter, CPU parity test green; bonus: skips
the embed/codec_tokens D2H entirely (verified safe — talker routes
codec_tokens via streaming edges from GPU output, never off cpu_output).
GPU A/B queued (s2s:8/i2s:8).
Item B PREMISE CORRECTED: the default qwen3omni_2gpu.yaml ALREADY colocates
Talker+Code2Wav on rank 0 (one process per rank), and
_divide_into_worker_graphs forces every streaming consumer into its own
worker graph — no yaml can fuse the codec edge. taco.yaml decomposes
byte-identically to default (verified by building worker graphs). The
paper-§4.2 "colocate to kill per-frame IPC" is ALREADY the deployed state;
PLAN_BEAT_VLLM_V4 #15(b) and the miner-agent claim behind it were wrong.
Remaining real lever on the codec edge: intra-process per-frame overhead —
batch the 25-frame chunk handoff (one put per chunk, not per frame) and/or
fuse talker->code2wav in one walk (bigger code task).

## PREFILL_CHUNK_TOKENS 128 vs 256 (chunk128, warm lab, sloppy POC) — 128 REJECTED
A=256 vs B=128, i2t B32/B4 x2 rounds: B32 0.884/0.867 (-12%), B4 0.949/0.985.
Halving the chunk doubles per-chunk overhead (admission bookkeeping + bucket
padding) without a compensating stall win. 256 confirmed as floor; 512 arm
pending.

## V1 async scheduling (opt/async-sched 0750145..18b1244) — smoke round 1: mechanism PROVEN, one real bug caught+fixed
First live run of MSTAR_ASYNC_SCHED (deferred-postprocess-by-one-iteration on
the spec chain, requires DIRECT_FEED): counters engaged (async_sched_steps=993,
late_stop_trims=239) and the blocking sampled-token wait COLLAPSED ~1-3ms ->
~108µs/step. Smoke caught a real race: REMOVE_REQUEST tearing down a rid
whose postprocess was still deferred (KeyError in _run_deferred_postprocess;
corrupted tok/req to 200). Fix 18b1244: _remove_request guard extended to
defer removal of _deferred_pp rids + regression test. Re-smoke in flight;
then stage-2 reboot A/B. (Ops note: a server collision on 4,5:8305 — two
boots, mine and the builder's — cost one boot cycle; rule reaffirmed: one
owner per GPU pair.)

## INVALIDATION: dynflags-'{}' baseline bug in lab_ab A/Bs (caught by speech-builder agent)
dynflags.maybe_refresh applies only keys PRESENT in the JSON — so lab_ab
sides using '{}' left side-B's values ACTIVE for every round after r1. All
'{}'-baseline A/Bs tonight compared identical configs from r2 onward:
r2lite_directfeed, cache_sanity, checkstop_decomp, route_decomp(+swap),
pair_off(+swap), scmid. CONSEQUENCES: (1) the "Decomposition FINAL /
every flag 0-3%" entry is RETRACTED-AS-ARTIFACT — dilution toward 1.00 was
built in; (2) clean r1-only readings: FAST_ROUTE-off 0.958/0.941 (-4..-6%),
cache+checkstop-off 0.941/0.908 (-6..-9%) — the flag wins LIKELY STILL REAL
post-sidecar (consistent with their original deltas); the "+7-10% retracted"
note is itself withdrawn — status now "supported by clean cells, full
re-decomposition deferred" (no shipping decision hinges on it: flags stay ON
either way); (3) scmid and r2lite verdicts stand direction-wise but on r1
evidence only. FIX: lab_ab.sh now requires identical key sets in both JSONs
(refuses '{}' vs keyed), and the convention is explicit 0/1 for every
touched key. Lesson: a runtime flag-file protocol needs explicit-unset
semantics; silence is sticky.

## PREFILL_CHUNK_TOKENS 512 (chunk512, valid explicit-key A/B) — FIRST POSITIVE POC: +3-5% at B32
A=256 vs B=512, warm lab 6,7: B32 1.049/1.031 (B wins both rounds, jct also
better 4268 vs 4510), B4 1.018/0.977 (wash). Mechanism: fewer, larger chunks
amortize per-admission overhead at high batch; at B4 fewer folds are
in flight so it washes. CAVEATS before shipping: closed-loop results.json
carries no TTFT (None) — larger chunks lengthen the mixed-step tail so a
TTFT check (streaming mode or B1 jct proxy) is REQUIRED; and 768 escalation
queued to find the knee. Candidate for the integration stack as
MSTAR_PREFILL_CHUNK_TOKENS=512.

## PREFILL_CHUNK_TOKENS 768 (chunk768) — no knee past 512; STAY AT 512
A=512 vs B=768, i2t B32 x2: 0.950/1.080 — mixed signs, geomean ~1.01. The
consistent-sign win was 256->512; 768 adds nothing resolvable. Integration
candidate remains 512 (pending the B1 latency proxy, in flight).

## V1 async-sched stage 1 (fixed build 18b1244) — CORRECTNESS PASS; wait-collapse REGIME-DEPENDENT
~4800 decode steps, zero tracebacks; async_sched_steps fires every spec step;
late_stop_trims working (409). Honest mechanism reading (windowed
async_d2h_wait_us): a few windows collapse to ~50µs/step, but MOST read
0.8-1.9ms/step — the one-step deferral hides the sampled-token wait only when
the main-thread iteration outruns the GPU step. This is the predicted "V1
pays only where main-thread Python > GPU time" regime; best case = B32
steady-state. tok/req is stochastic at cell level (166-197 for identical
config) so byte-identity rides the stage-2 OFF-vs-ON compare. Stage-2 A/B
(i2t B32, warm ON cells then OFF reboot) in flight on 4,5.

## V1 stage-2 interim — ON cells show SYSTEMATIC tok/req inflation (186-199 vs 177 band)
4/4 clean async-ON i2t B32 cells: req/s 6.17-6.62 (mean 6.37), tok/req
186.7-199.2 — all above the tight 176-179 baseline band. In tok/s terms ON
~= baseline (~1229), so as-built V1 = no throughput win + NON-IDENTICAL
outputs (fails its own byte-identity bar). Hypotheses handed to the builder:
(a) late-stop trim leaking overrun tokens into the stream, or (b) deferred
postprocess shifting SAMPLING state (rep-penalty/stop bookkeeping one step
late -> legitimately longer generations). OFF cells + per-request diff will
decompose. Fix direction if (b): defer only transport/emit, keep
penalty/stop state synchronous.

## torch.compile — ALREADY ON for the whole text path (user-requested investigation)
Finding: cuda_graph_runner.py:609/2476 wraps forward_batched in
torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=False,
dynamic=False) BEFORE capture whenever the graph config sets compile=True —
and qwen3_omni's submodules set compile=True for thinker decode, prefill_text,
prefill_vision, thinker_mixed, and ALL talker walks (submodules.py:1455-2467);
only code2wav_chunk is compile=False. So M* already runs the exact vLLM
pattern (Inductor-fused kernels inside FULL CUDA graph captures) for this
model; production evidence = the E1 fp8 bug was CAUSED by dynamo re-tracing
during a Qwen3-Omni bucket capture. CONSEQUENCES: (1) the "compile gap" vs
vLLM is ONLY the scheduling glue (uncompilable control flow) — already the
campaign's target via sidecar/V1; (2) grid doc §1.1 corrected; (3) the #21
hand-fusion pack is LIKELY MOOT (Inductor already fuses norm/rope/act chains
inside the compiled region) — verify via kernel count in an existing nsys
trace before any fusion work; (4) remaining compile levers are small:
fullgraph=False graph-break audit, Inductor option tuning, code2wav compile
(speech - stopped by user directive). CUDA-graph health check same pass:
zero capture failures/eager fallbacks in all of tonight's server logs.

## tokenspeed (github lightseekorg/tokenspeed) — NOT used by vLLM-Omni
User lead checked: separate serving engine (C++ control-plane scheduler,
static compilation, Blackwell MLA kernels). Zero references in vllm-omni
source, deps, or venv. Its C++-scheduler idea = the option we scoped out.

## Decode-bucket finer grid (bgridB) — NO WIN, current grid stays
Arm A (shipped grid [1,2,4,8,16,24,28,32]): 6.603/6.456. Arm B (+20,26,30):
5.933/6.376 — finer grid reads lower (cross-boot arms, ~40min apart, so
partly drift; either way no positive signal). More buckets also cost VRAM +
boot time. CLOSED: keep the shipped grid.

## Ops: lab-kill collision 23:09 — GPU driving moved to single-owner
The sweep (8299) and speech (8311) servers died mid-cells (external kill,
zero tracebacks — agents acting on stale "kill when done" instructions while
main was firing cells). Casualties: speech ac_bundle A/B (one clean baseline
cell only — the new speech flags are NOT implicated, the A-side i2s cell
zeroed too) and the chunk-512 B1 latency gate (garbage cell). Both re-queued
on fresh labs. RULE hardened: exactly one owner for all lab boots/kills/cell
firing (main); agents are code-only.

## chunk768 mechanism note (sweeper exit report) — 768 floors to the 512 bucket
The chunk planner picks the largest CAPTURED bucket <= min(remaining, cap)
from [128,256,512,1024,2048]; cap=768 therefore runs identical 512-bucket
chunking — the wash was structural, not noise. Real next operating point:
cap=1024 (bucket already captured) — queued as a quick arm after the B1
latency gate. 512 remains the validated winner.

## chunk-512 B1 latency gate — PASSED; 512 adopted
B1 A/B (256 vs 512): req/s -2.4..-3.0%, jct +2.4..+3.5% — bounded small B1
tax, under the 5% flag bar, against +3-5% at B32. MSTAR_PREFILL_CHUNK_TOKENS=512
enters the integration stack. (768 = same as 512 structurally; 1024 arm
optional later.)

## Speech bundle A/B (FAST_CHECKSTOP_TALKER + CODEC_CHUNK_EMIT) — VALIDATED, +1-3%
Fresh 0,1 lab, default (audio-optimal) yaml, explicit-key dynflags A/B,
2 rounds x2 cells: s2s B8 1.012/1.016, i2s B8 1.006/1.031 — positive sign in
4/4 pairs. Matches the builder's scope ceiling (D2H cut + buffer-churn
reduction; the routing floor stays). Both flags validated for the
integration stack. SPEECH TRACK CLOSED per user directive.

## Compile fix first validation — graph breaks 1617 -> 771 (mid-boot)
opt/compile-fix 1733fab (pure-torch RMSNorm under compile): removes ~850
breaks (each norm break also fragmented neighboring traces). Throughput A/B
(cfix 2,3 vs gba 6,7 same-config cells) next; if positive, next break
targets: thinker.py:225 layer-loop (~48) and moe.py:478.

## Integration branch opt/integration-v4 (worktree mstar-iv4)
= opt/sched-pack (74a985c) + opt/speech-floor merged (talker fast-checkstop
55faf29 + codec chunk-emit 01856e8, both validated +1-3%; taco yaml doc).
Chunk-512 finding: the CODE DEFAULT is already 512 (qwen3_omni_model.py:282)
— the canonical flag set's explicit MSTAR_PREFILL_CHUNK_TOKENS=256 was
OVERRIDING it; integration = drop that flag from the launch set (new
canonical set: replace =256 with =512 or omit). Stacked test = final stack
flags + PREFILL_CHUNK_TOKENS=512 + FAST_CHECKSTOP_TALKER=1 +
CODEC_CHUNK_EMIT=1, pending cfix A/B (compile fix joins if it wins).

## V1 DECIDER — async OWNS the +12% output lengthening; as-built FAIL
OFF-control on the SAME worktree/stack: tok/req 173.3 (baseline band) vs ON
186-199. Not a branch artifact — MSTAR_ASYNC_SCHED changes generated content.
Ruled out: missed-EOS (unimodal lengths), deferred sampler state (RNG offset
+ penalty mask update inside sample() on the GPU thread). Live hypotheses:
(a) overrun rows (stopped-but-untrimmed, 1-2 extra steps) perturb shared
batch numerics -> sequence drift (builder's theory; but symmetric ULP drift
shouldn't SYSTEMATICALLY lengthen); (b) SAMPLER_CFG_CACHE is keyed by batch
membership and async shifts membership timing -> misaligned config rows =
wrong temp/penalty per row = systematic drift (main's theory; decomposition
test = ON + cache OFF, running). Either way V1 is PARKED as-built; if (b),
the fix is a composition-versioned cache key (small) and the design
survives; if (a), the one-step-late stop design is fundamentally at odds
with output identity for sampled workloads.

## V1 FINAL — PARKED (byte-identity fail + tok/s wash); design-doc prediction empirically confirmed
Full A/B (i2t B32, same worktree, 4 cells/side): tok/req OFF 178.3 tight vs
ON 194.3 (+9.0% longer, systematic — identity FAIL); tok/s OFF-healthy 1287
vs ON 1217-1276 (WASH — no per-token speedup; the req/s gap is purely length
inflation). Wait-collapse only partial (~1ms residual most windows): at
current regime the main thread doesn't consistently outrun the GPU step.
Mechanism by elimination: deferral shifts batch composition (2nd overrun row
per stopping rid + one-iteration admission lag) -> fp8-MoE numeric footprint
-> sampled distribution drifts longer; trim protects the stream, not the
numerics. FUNDAMENTAL TENSION: the win requires deferring check_stop; the
deferred stop causes the overrun/non-identity — for sampled workloads the
two are opposed. Flag default-OFF (off path byte-identical), substrate kept
(like W2/E9/E10) for retest after further main-thread reduction. Session
yield: clean falsification + 2 real bugs (18b1244 deferred-remove race;
the non-identity itself). Branch opt/async-sched @ 18b1244 pushed.

## ROOT CAUSE of the night's server deaths + late "degraded windows": /tmp FULL
OSError Errno 28 in the stacked server: the api server writes every request's
image/wav upload to /tmp/mstar_uploads_tim; the 70G rootfs hit 100% (20KB
free). Explains the 00:22 stacked death (NOT the integrated flags), most
likely the 23:09 server deaths, and taints the late-evening "degraded
window" absolutes (uploads failing = partial batches). Freed 3.3G; harness
hardened: lab_server.sh now exports TMPDIR=/m-coriander/coriander/tim/tmp
and cleans stale uploads at boot. LESSON: add disk-free to the pre-run gate
(CLAUDE.md already mandates /home checks — rootfs /tmp was the blind spot).

## Final-window A/Bs — INVALID (self-inflicted), re-queued as next session's first runs
Three concurrent labs = 2 on NUMA node 1 (base2+stacked, crawled ~3.1) vs 1
on node 0 (cfix2, clean 5.0-5.4): the cross-node norm-fix comparison is
invalid; the stacked-vs-base comparison (both node-1, matched contention)
read ~parity = no regression from the integrated flags, nothing more. Plus
/tmp exhaustion killed stacked r2. OPEN VERDICTS for next session, run
SEQUENTIALLY one lab at a time: (1) norm compile fix (opt/compile-fix
1733fab; breaks 1617->816 confirmed, throughput A/B pending); (2) stacked
integration test (opt/integration-v4: chunk-512 default + speech bundle);
(3) V2 budgeted admission policy = the main open build (raises fold volume,
unlocks banked split-attn, targets B2-B4 + TTFT).

## Norm compile fix — WIN +4.5% i2t B32 (clean sequential same-pair A/B); NEW CAMPAIGN BEST 7.443
Sequential solo-lab protocol on 6,7 (arm 1 base mstar-p2: 6.593/6.975/7.088/
6.976 mean 6.908; arm 2 cfix: 7.211/7.153/7.080/7.443 mean 7.222) = +4.5%,
3/4 cfix cells above every base cell, tok/req 176-178 sane (pure-torch norm
ULP drift harmless). Best cell 7.443 = 0.907x vs vLLM 8.210 — campaign
record. Mechanism: pure-torch RMSNorm under torch.compile removes ~800 graph
breaks (1617->816), letting Inductor fuse norm->residual->proj chains inside
the captured graphs. Cherry-picked to opt/integration-v4 (7aebb1d). NEXT
break targets (round 2): thinker.py:225 layer-loop (~48 breaks),
moe.py:478; attention breaks stay (natural piecewise boundary). Boot cost:
+~2min one-time Inductor compile.

## STACKED INTEGRATION TEST — +6.3%; wins COMPOSE; i2t B32 = 0.89x mean / 0.93x best
opt/integration-v4 @ 7aebb1d (sched-pack base + speech bundle + norm compile
fix) with PREFILL_CHUNK_TOKENS=512 + FAST_CHECKSTOP_TALKER=1 +
CODEC_CHUNK_EMIT=1, solo sequential protocol on 6,7: i2t B32
7.123/7.593/7.559/7.084 mean 7.340 (+6.3% vs same-protocol base 6.908),
tok/req 174-179 sane. vs vLLM 8.210: 0.894x mean, 0.925x best cell. s2t B8
18.16/20.45/19.94/19.43 mean 19.49 = 1.23x vs vLLM (best 1.29x) — also +6%
over the prior band. Composition arithmetic checks out (norm fix +4.5% +
chunk-512 ~+2%). Campaign trajectory at i2t B32: 0.53x (v0.22 release) ->
0.77x (v2) -> 0.83-0.85x (sidecar) -> 0.89x mean / 0.93x best (integrated).
NEXT: graph-break round 2 (thinker.py:225 layer-loop ~48 breaks, moe.py:478),
V2 budgeted admission policy (B2-B4 + TTFT + unlocks split-attn), then the
canonical warm+512 re-baseline sweep.

## Compile round-2 — CLOSED (both quick edits wedge the boot); custom-op route required
Removing @torch.compiler.disable from set_layer_idx OR narrowing the fp8
dispatch disable both WEDGE the boot (dynamo re-trace storm [13/23], frozen
trace, dead workers) — the disables are load-bearing: they keep dynamo out
of the cache-manager/quant object graphs entirely, and tracing in is not a
2-line change. Round-1 (+4.5%, pure-torch norm) STANDS — the difference:
norm swapped the computation for a traceable equivalent; round-2 tried to
trace INTO stateful engine objects. Correct future route: register
run_rms_norm-class FlashInfer calls and fused_experts_fp8 as torch.library
custom ops (opaque-but-traceable graph nodes, no breaks, no object tracing).
Integration branch restored to fix-1-only (7aebb1d content @ head 2e48a41+).
Remaining break census on fix-1 build: 816 total, top = attention.py:129
(48, natural piecewise boundary) + moe.py:478 (38) + long tail.

## HARDWARE: GPU 7 dropped off the bus (~02:30, 2026-07-04) — canonical pair DEAD
nvidia-smi -i 7: "No devices were found"; box enumerates 0-6 only. Every
"wedge" since 02:30 (gb2, gb3, v2pol boots on 6,7) = worker_1 "invalid
device ordinal" on the missing device, NOT code. CONSEQUENCES: (1) the gb3
set_layer_idx revert may be a FALSE NEGATIVE (never got a fair boot) —
retest cheaply when a pair frees; gb2's [13/23] recompile storm from the
fp8 narrowing was real (pre-drop) and stays reverted. (2) Canonical pair
6,7 unusable until an admin reset/reboot — userspace cannot recover a
fallen-off-bus GPU; FLAG TO USER for the box admin. (3) Remaining usable
pair: 2,3 (foreign jobs hold 0 and 5; 1,4 are singles). V2 smoke rewired
to 2,3:8313.

## V2 BUDGET POLICY round 1 — WIN AT B32 (+7.0/+10.6%) and B8 (+9.9/+0.7); floor blocks B2/B4
opt/v2-policy (0cea3e8+merge), warm 2,3 lab, dynflags A/B budget 0 vs 512,
2 rounds: B2 0.974/0.993, B4 0.965/0.946, B8 1.099/1.007, B32 1.070/1.106
(consistent sign at B32; best cell 7.218 on the node-0 pair). The
predicted-neutral B32 WON — every-step folding removes the phase-drain that
yield-boundary-only folding left. budget_skips_floor >200 = the inherited
min-decode floor (24) blocks folds at exactly B2-B8; floor2 A/B
(MIN_DECODE 24 vs 2 at B2/B4) in flight. B4 slight negative = probe
overhead without folds (floor blocks all of them) — if floor2 wins, default
becomes budget=512 + a lower floor; if it regresses, floor stays and V2
ships as a B8+/B32 lever.

## V2 FINAL — MERGED to integration; ships as a B8/B32 lever (floor stays 24)
floor2 A/B (MIN_DECODE 24 vs 2 at B2/B4): 0.980/1.090/0.989/0.903 geomean
0.988 — lowering the floor is wash-to-negative; the occupancy economics at
tiny decode batches are real and the floor guard was correct. V2 verdict:
MSTAR_MIXED_BUDGET_TOKENS=512 (floor default 24) = +7-11% at i2t B32,
+1-10% at B8, inert-by-design at B2/B4. Merged opt/v2-policy ->
opt/integration-v4. B2/B4 remain the open cells — their deficit is fixed
host cost + multi-process prompt latency, not fold policy (three levers
tried tonight all closed: single-chunk, budget, floor). Remaining V2 work:
arm-3 (budget + split-attn + preplan — fold volume is now HIGH at B32, the
banked +4.4%/mixed-step should finally convert); canonical re-baseline
blocked on GPU-7 repair.

## Money run (full integrated stack + V2, pair 2,3, ~03:45) — BAND-DEGRADED, absolutes deferred
i2t B32 5.66/4.98/5.06/5.23 (mean 5.23), s2t B8 mean 17.4 — deep 4am
degraded window on the box's weaker pair (2,3 band read 5.5-6.5 all night vs
6,7's 6.9-7.4; foreign 120GB job on GPU 0 throughout). NOT evidence against
composition: every component ratio is separately banked (norm fix +4.5%,
chunk-512 +3-5%, V2 +7-11%, all adjacent-protocol). PARITY ABSOLUTE
CONFIRMATION deferred to: GPU-7 repair + quiet box + warm solo protocol on
the canonical pair. Projected: stacked 7.34 (measured, 6,7) x V2 1.07-1.11
= 7.9-8.1 vs vLLM 8.210.

## W2-at-small-batch — NOISE-BLOCKED (box degraded), theory test deferred
Arm A (old base, TXN off) same-config B2 samples 0.988/1.005/0.854/0.708 —
±20% spread, monotonic degradation through the run (04:00, foreign 120GB job
resident). A cross-reboot arm comparison cannot resolve the predicted
5-10% effect in this. DEFERRED to a clean window; the theory (postprocess
memoization converts at B2/B4 where the host floor is unshaded — remove-work
law) remains the best-motivated B2/B4 code experiment, alongside the
prefill-merge build (in progress). GPU verdict work STOPPED for the night:
every absolute since ~03:30 is mush; ratios need adjacent cells the drift
now defeats. Code streams continue.

## Merged multimodal prefill (B1/B5, opt/prefill-merge 41200ec) — BUILT, awaiting GPU A/B
Option A: true merged prefill_multimodal walk reusing the prefill_vision
capture (union signature identical; no new capture). Enabling fact:
process_prompt strips modality placeholders (qwen3_omni_model.py:2092-2115)
so KV spans are sequential — merge = concatenation of the per-span embeds,
no splice. Invariants done: MRoPE span-threading (start_pos per span exactly
as advance_seq_lens), deepstack zero-rows alignment, byte-identical off.
14 new CPU tests + chunked/mixed suites green. CAVEATS: fires only with
CHUNKED_PREFILL_V2_VISION OFF (strategy swap vs the vision-fold path — the
A/B must compare merge vs vision-chunking, not merge vs nothing);
process-static (two-server A/B); single-image i2t scope, all else falls
back byte-identical. Targets the B2/B4 + TTFT hole (one conductor
round-trip per admission instead of two). SMOKE.md on the branch. Queued
first in the clean-window GPU queue.

## Merged multimodal prefill A/B — +5.6% at B2 AND B4 (the losing cells); MERGED as VALIDATED-OPT-IN
Two-server interleave (pmA off/0,1 vs pmB on/2,3, vision-chunking off both,
4 samples/cell): B2 1.275->1.346 (+5.6%), B4 2.014->2.127 (+5.6%), jct -5-6%
both; B1 0.819->0.783 (-4.4%, opposite sign => not a pair-band offset —
investigate at clean-window confirm); B32 excluded (same-order cell
collision between the two streams — harness lesson: stagger cell orders).
merged_prefill_walks=71+ (mechanism live). First direct hit on the B2/B4
hole all campaign. opt/prefill-merge MERGED into opt/integration-v4;
flag stays default-OFF pending the clean-window confirm (+ the B1 sign
question). Integration branch now: speech bundle + norm fix + V2 budget +
merged prefill.

## MSTAR_GC_TUNE + allocator chain — NOISE-BLOCKED (weak positive lean), chain stopped at 2 arms
gcA (off): 4.679/4.708/4.387 + one 2.17 collapse; gcB (GC_TUNE=1, carrying
concurrent crusade-boot contention): 4.911/4.972/4.650/4.225 — bands overlap,
gcB mean +2.2% vs gcA clean cells DESPITE worse contention = weak positive
lean, unresolvable at the ≥8% bar on the makeshift 0,6 pair under a 4x92GB
foreign job. Chain STOPPED before the jemalloc arm (effect-size gate).
Code kept (c879884+, default OFF, gate lines now WARNING) — cheap retest in
the clean window alongside jemalloc/mimalloc LD_PRELOAD (both on box).

## Custom-op crusade step 1 (opt/custom-ops b7fb94e) — breaks 816 -> 438, boot healthy
mstar::run_attention torch.library custom op (vLLM forward-context pattern:
plain tensors + explicit layer_idx, manager via active-manager global) kills
both the thinker attention-wrapper break (~48) AND set_layer_idx (~48) plus
cascade — census halved to 438 with zero recompile storm (contrast: the
disable-removal attempts wedged). CPU trace-proof: 1 break/2 graphs -> 0
breaks/1 graph. Remaining 438 = talker-side sites + tail; step 2 = extend
the op to the Talker's two construction sites (pre-approved). Perf cells
pending. Trajectory: 1617 -> 816 (norm fix, +4.5% e2e) -> 438 (step 1).

## Crusade steps 3-4 — fp8 class ELIMINATED (111->0); apply_rope op staged
Step 3 (be16eeb): mstar::fused_experts_fp8 op + pre-capture quant hoist —
moe.py:478 census 111 -> 0, CPU-validated constant-folding of the cache
guard. Boot healthy-but-slow: fewer breaks = bigger fused regions = longer
Inductor max-autotune (a real boot-time trade); one frame recompile-limit
hit under verification. Census now dominated by ONE class: talker apply_rope
(105) + small tail (~54). Step 4 built (mstar::apply_rope, clones q/k — no
input aliasing, mutates_args=()); commits after step-3 WARM gate; full-stack
(1-4) boot targets <200 with the final census + cell read.

## Crusade boot-time tension identified + production fix prescribed
Step-3 boot took ~40 min (NEVER_READY past the lab 20-min window; warmed
late, server healthy): fewer breaks = bigger fused regions = longer Inductor
max-autotune. This is inherent to the one-big-graph goal and must be solved
for shippability: TORCHINDUCTOR_FX_GRAPH_CACHE=1 + persistent
TORCHINDUCTOR_CACHE_DIR (pay autotune once, reuse across boots) +
TORCHDYNAMO_CACHE_SIZE_LIMIT=128 (kills the frame-[7] recompile-limit eager
fallback). Full-stack (1-4) reboot with the cache running; early census
(<200 bar) reads ~3 min into tracing; cache proof = second reboot warming
<10 min. fp8 census kill (111->0) and step-4 push confirmed.

## Crusade full-stack (steps 1-4) mid-boot census — SIX unique break sites remain
All four converted classes read ZERO in the full-stack boot. Residual:
talker.py:173 (72), thinker.py:261 (45), submodules.py:2309 (24),
submodules.py:1809 (9), talker.py:547 (9), code2wav.py:482 (6) — attributed
total ~165 (site-attributed <200 bar MET; raw grep 489 incl. dup-suppression
notices). The two new dominant sites are candidates for steps 5-6 if their
classes are convertible. Warm gate + cache-proof reboot pending (autotune).

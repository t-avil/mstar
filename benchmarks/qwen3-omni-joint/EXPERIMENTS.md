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

## CRUSADE BAR CRUSHED — full-stack census 41 header-breaks (~165 attributed) vs <200 bar
Trajectory FINAL: 1617 -> 816 (norm fix, +4.5% e2e) -> 438 (step1 thinker
run_attention custom op) -> 312 (step2 talker run_attn + step3 fp8 op) ->
41 header / ~165 attributed (step4 talker apply_rope). ALL converted classes
at zero (run_attention, set_layer_idx, apply_rope, fp8 dispatch — both
models). TORCHDYNAMO_CACHE_SIZE_LIMIT=128 killed the frame-7 recompile
fallback. Residual tail: talker.py:173 (36), thinker.py:261 (16),
submodules.py:2309 (12), talker.py:547 (9), code2wav.py:482 (6, compile-off
region). Inductor cache populated (616M/70k files). Cache-proof reboot in
flight (one boot-collision between main and crusader resolved — main's
parasite killed; clean boot-3 if timing tainted). Branch opt/custom-ops
@ 12ce776 (4 steps). DEFERRED to clean window: MSTAR_CUSTOM_OPS ON/OFF perf
A/B (compile-time flag, two-server or sequential), integration decision.

## CRUSADE MISSION CLOSED — full report banked (see branch opt/custom-ops, 4 commits)
Final analysis from the mission report: the compiled Thinker/Talker forward
is now effectively ONE fused graph from embed through the layer stack —
breaks remain only at the natural head (sampler: data-dependent, wall by
design, vLLM identical) and tail (advance_seq_lens: post-compute, ~0 fusion
payoff, convertible-but-not-worth-it). Steps 1-3 GPU-validated end-to-end
(6.325 req/s serving through the ops — capture works). Optional polish:
step 5 = talker dense-cache decode_attn_nhd op (9 breaks). Clean-window
items: MSTAR_CUSTOM_OPS ON/OFF perf A/B (the fusion speedup is
unquantified), cache-proof warm time (clean boot running, fired 09:08:41).

## CACHE-PROOF PASSED — cached custom-ops boot warms in 12m44s (vs ~40 min uncached)
Clean solo boot (09:08:41 -> WARM+READY 09:21:25) of the full custom-ops
stack with TORCHINDUCTOR_FX_GRAPH_CACHE=1 + persistent cache dir. The
boot-time tension is SOLVED: first boot pays autotune once, every later
boot ~13 min (still above the 7-min baseline boot — residual dynamo tracing
time — but operationally fine). The custom-ops route is SHIPPABLE pending
the clean-window ON/OFF perf A/B. Crusade fully closed.

## OPS ROOT CAUSE SOLVED: the recurring "mystery server deaths" = wrapper reaping
Pattern (23:09 last night, 10:03 today): a HEALTHY lab server dies by clean
SIGTERM shortly after its owning AGENT's turn ends. Mechanism: lab_server's
wrapper shell lives in the agent's process tree; when the harness reaps the
agent's shells between turns, TERM hits the wrapper -> the INT/TERM cleanup
trap fires -> kills the server pgid. The server's own setsid protects it
from ORPHANING but not from the trap's deliberate kill. MITIGATION: labs
launched by agents must setsid the WRAPPER itself (setsid bash lab_server.sh
... < /dev/null) or main launches all labs (current mode). Night-runner
released after an honest post-mortem (its cells-fire step also never
executed — agent-loop gap; second failure mode of the night for
agent-driven GPU work).

## Item 1: GC-tune + jemalloc 3-arm (pair 2,3, sequential solo boots) — WASH at B32, theory falsified cleanly
gc1 base 6.105/5.747/6.439/6.383 (mean 6.17); gc2 +MSTAR_GC_TUNE
5.775/5.865/5.783/6.749 (mean 6.04, gc.freeze PROVEN live x2 workers after
the WARNING-level fix); gc3 +jemalloc LD_PRELOAD 6.241/6.457/6.077/6.095
(mean 6.22). All bands overlap: GC tune -2%, jemalloc +1% = noise. The
mid-step-gen0-collection theory does NOT convert at B32 — per-step
allocation churn evidently stays under collection cadence significance vs
an 11-15ms step. Flags stay available; niche retest = small-batch cells
(higher host share) someday, LOW priority. Zero-code levers at B32:
exhausted.

## Item 2: MERGED PREFILL WINS THE SHIPPING COMPARISON — +4-6% at B1/B2/B4
Same-pair (2,3) sequential arms, definitive: pm1 (shipping config, vision
chunking) B1 0.784 / B2 1.318 / B4 2.116 vs pm2 (merge config:
vision-chunk flags off + MSTAR_MERGED_PREFILL=1) B1 0.818 (+4.3%) /
B2 1.392 (+5.6%) / B4 2.211 (+4.5%) — positive at ALL small-batch cells,
merged_prefill_walks=67+ live. The earlier B1 -4.4% (cross-pair run) is
resolved as a pair-band artifact. B32 sentinel running on the warm merge
server (the config swap loses vision-fold-into-mixed at B32 — must not
regress there before any default flip). If B32 holds: merge config becomes
the small-batch shipping recommendation (or batch-conditional config).

## Item 2 CLOSED: merge config +4-6% at B1/B2/B4 (solid); B32 impact UNRESOLVED (box instability)
The B32 sentinel read 5.06/5.10 (merge) but the same-hour shipping-config
control read 3.83/6.61 — adjacent same-config cells spanning ±40% (Ray
phase-change instability). MERGED_PREFILL is capture-static so interleaving
is impossible; B32 comparison deferred to a stable box. SHIP POSTURE:
small-batch/latency deployments -> merge config (+4-6% B1-B4, consistent,
mechanism-live); B32-throughput deployments -> keep vision-chunk config
until the stable-box read. Both configs real, both on opt/integration-v4.

## Item 3 (W2 small-batch) — PARKED at spread-gate (B2 within-arm 18%)
Old-base TXN=0 arm: B2 0.895-1.074 (18% same-config spread under Ray phase
instability), B4 ~5%. Cross-boot arms cannot resolve the predicted 5-10%
effect at B2 in this. Parked to the stable-window queue (with W2's own
one-server interleave impossible — TXN is init-static on the old base).

## Item 4 ABORTED mid-A/B — box entered catastrophic contention phase (~13:45)
OFF-arm cells ALL 0.000 req/s ("expected modalities ['text'], received []"
= requests timing out server-side and returning empty; warm cell crawled at
2.0 req/s vs normal 4-7). Not a flag effect — the Ray job's current phase
starves our server below usability. Disk clean (TMPDIR hardening held).
Item 4 re-queued for the stable window. GPU VERDICT WORK SUSPENDED until
box conditions change (Ray release / GPU-7 return / load drop) — burning
boots in this produces only noise entries. Watch mode: low-frequency polls.

## Custom-ops perf A-B-A (2,3, sawtooth box) — POSITIVE direction; magnitude to the clean window
OFF1 5.145/5.157/5.744/7.014 (mean 5.77, uptrend) -> ON 6.119/6.196/6.392/
7.154 (mean 6.47, tok/req 176-179 sane) -> OFF2 5.160/4.722/4.234/4.876
(mean 4.75, downtrend). ON exceeds BOTH bracket means and ON's worst cell
beats every OFF2 cell — the fusion win is directionally REAL; magnitude
unresolvable under the sawtooth (+5..20% range). Cache double-proof: ON boot
12m28s (second datapoint, consistent with 12m44s). SHIP POSTURE:
opt/custom-ops = strong candidate (census 41, correctness sane, boots
cached, perf positive); the canonical-pair A/B quantifies before merge to
integration. STABLE-WINDOW QUEUE now: canonical re-baseline (GPU-7 reset),
custom-ops magnitude A/B, merge-config B32 read, arm-3, W2.

## HEADTOHEAD_V2 — LIVE same-window race, M* full stack (canonical 6,7) vs vLLM-Omni 0.22 (2,3)
Build: opt/custom-ops (contains integration-v4) + full flags + custom-ops +
V2 budget + chunk-512 + speech bundle. Adjacent per-cell pairs, clients
NUMA-pinned, 2026-07-04 18:00-18:35. RESULTS (M*/vLLM):
- i2t B8: 4.476/3.665 = 1.221x — LIVE WIN (+22%)
- s2t B8: 19.486/13.907 = 1.401x — LIVE WIN (+40%)
- i2t B32 pairs: 0.723 (M* soft cell), 0.869, 0.897 (M* warm 7.31-7.60
  consistent; vLLM 8.24-8.46) -> live warm ratio ~0.87-0.90x, matching the
  warm-lab projection. STILL LOSING B32 by ~10-13%.
- tok/req sane both sides (M* 172-177; vLLM 210-212 — their +20% length).
- RELIABILITY: vLLM DIED MID-RACE (2nd collapse today; pairs 6-7 lost);
  M* served every cell all day, zero self-inflicted deaths.
VERDICT: we beat vLLM-Omni live at B8 on both text paths (and speech was
always 2-3x); the last stand is i2t B32 at ~0.88x live. Remaining levers:
custom-ops magnitude was IN this build — the next gap-closers are sidecar
stage-2 (route/check_stop exile), custom-ops step-5 polish + tail breaks,
V1-post-compile revisit (main thread now lighter), and race variance
(more rounds; M*'s soft cells cost ~0.05x of pooled ratio).
---

## Profile gate post-custom-ops (lab_crusade/profile_gate/) — MAIN THREAD IS NO LONGER THE WALL; re-orders the whole B32 plan
Warm i2t B32 on the full custom-ops stack (7.559 req/s cell, GPUs from
lab_crusade), py-spy per-worker + nvidia-smi dmon over the measured window:
MainThread active ~32% of wall vs GPU 64.8% SM-busy and the gpu-thread ~58%.
This INVERTS the pre-sidecar regime (base: main 13.2ms >> GPU ~8.8ms; design-doc
prof_final2: main 8.74ms > GPU ~4.5ms). Custom-ops removed 1617->~165 graph
breaks = CPU dispatch cut on BOTH GIL threads, pulling main-thread Python below
GPU time — exactly the "current regime the main thread doesn't consistently
outrun the GPU step" the V1 FINAL entry observed, now measured directly.
CONSEQUENCE via the GIL-valve law (Sampler config-tensor cache entry: removing a
gpu-thread WAIT pays ONLY while main-thread Python is the wall): every remaining
wait-removal lever is now at/below breakeven. Checkstop-family (Stage-2 offload,
FAST_CHECKSTOP-class, sampler-cache de-sync) and V1-family (async-sched deferral)
are all wait-removals -> PARKED, predicted wash-to-negative, consistent with the
already-washed post-sidecar checkstop/cache decompositions. What's left that
still converts is main-thread WORK removal (exhausted: route/register/store are
overlapped/load-bearing per W2/E9) or GPU-side — neither is the lever. PROMOTED
to the P0 lever: admission-wave / variance work (fold rate ~35%, soft cells cost
~0.05x pooled), the one mechanism not gated by the valve. Data: py-spy dumps +
dmon.txt in lab_crusade/profile_gate/.

## Stage-2 checkstop offload (opt/sidecar-checkstop @ 5414a5a) — BUILT + CPU-tested, PARKED BEFORE A/B on the profile gate
Deferred-consume check_stop: the side-stream sampled-token D2H becomes a polled
event.query() (double-buffered pinned slab) instead of a critical-path
side.synchronize(); EOS decision moves to the sidecar; max_tokens enforcement
stays worker-side as a pure counter (a dead sidecar can never cause unbounded
generation). Route/store/register stay on the worker (SIDECAR_DESIGN §3.2 — they
drive next-step readiness, cannot leave; this is check_stop ONLY, NOT the
handoff's route_outputs exile). WALK_STATS counters + a shadow-mode asserting the
legacy worker-side check_stop against the sidecar decision (W2 pattern) are wired.
CPU suites green; byte-identity harness in place. PARKED BEFORE any GPU A/B: the
profile gate (above) shows main-thread < GPU, so by the valve law the removed
wait converts wash-to-negative — and the deferred-stop overrun rows are the SAME
batch-composition-drift mechanism that failed V1 (+9% tok/req), so it also risks
the identity gate. RE-OPEN CONDITION: only if a future main-thread-WORK reduction
(or a GPU-side speedup) flips the profile back to main-thread > GPU AND a
re-profile confirms it; then run the shadow-mode A/B with tok/req held at
176.9-177.0. Branch opt/sidecar-checkstop @ 5414a5a pushed; substrate kept like
W2/E9/V1.

## HANDOFF_V5 corrections from verification pass — four claims fixed, one design contradiction flagged
Verifying the handoff against git + committed raw turned up:
1. "BRANCH MAP ... all pushed" was FALSE — several branches were local-only;
   pushed and the claim corrected.
2. h2h_v2 raw (the live-race scoreboard) was UNCOMMITTED — committed to the docs
   branch so the 0.87-0.90x B32 live numbers are recomputable from raw.
3. §4.1 "sidecar stage-2: exile route_outputs" CONTRADICTS SIDECAR_DESIGN §3.2
   (route/store drive Loop.complete_iter next-step readiness, cannot leave the
   worker; the doc parks route in Stage 3 "expect NO," a week+ EngineCore
   rewrite, and W2 proved the cheaper in-process version e2e-flat). Stage-2 is
   check_stop offload ONLY; the handoff's +8-15% double-counts a ~0 component.
4. Small-batch i2t B2/B4 in the handoff/GOAL used a rosier source
   (0.82+merge->~0.87 / 0.88->~0.92); the committed canonical sweep reads
   0.75/0.83 — the B2/B4 hole is deeper than the handoff table implies. Use the
   sweep numbers for gap-to-goal.
5. s2t B2/B4/B16 have NO committed final-stack data (starred cells are stale/
   projected) — flagged as un-measured, not un-won; needs live races (P2 queue).

## ab_verdict.py + proof-sweep protocol — statistical verdict gate; soft-cell rejection is per-ARM robust-z, NOT intra-cell JCT skew
Built /m-coriander/coriander/tim/ab_verdict.py (stdlib+math): auto-detects h2h
({mstar,vllm}_<path>_B<b>_<i>) vs lab_ab ({A,B}/{base,v2}_..._r<n>), pools PER
CELL (never across batches), prints geomean + SE(log) + one-sided 95% LB and a
WIN/LOSS/WASH verdict vs threshold (1.05 h2h, 1.02 lab_ab). KEY CALIBRATION
FINDING (overturns the intuitive gate): on real h2h_out the known-soft 0.723 pair
(mstar #1 = 6.119 req/s) has the LOWEST intra-cell JCT dispersion of the five
cells (p95/med 1.46, max/med 1.61) while the ACCEPTED cells skew HIGHER (max/med
1.9-2.1) — so a p95/median or max/median intra-cell threshold rejects the WRONG
cells. The real signal is that 6.119 is a gross LOW outlier vs its own arm's
other rounds (7.31/7.39/7.60/7.40). Gate implemented = per-arm, per-(path,batch)
robust low-outlier test on request_throughput (rel-drop >12% AND robust-z >3, MAD
scale floored at 4% of median, needs >=3 rounds); intra-cell JCT skew demoted to
a warning (hard-reject only above max/med 15). Also: hard-reject on crash
(completed==0 — catches vLLM's two mid-race deaths); tok/req identity gated to
i2t (172-179 band is i2t-specific); a WIN with i2t identity fail downgrades to
SUSPECT (the V1 length-inflation guard). VALIDATION on h2h_out i2t B32: rejects
r1 (soft) + r6/r7 (vLLM crashes), pools clean 0.869/0.897 -> geomean 0.8828,
SE 1.56%, 95% band [0.860,0.906] -> LOSS (upper bound still < 1.05). The honest
B32 read is 0.883, statistically clean. i2t B8 (1.221) and s2t B8 (1.401) return
INCONCLUSIVE (n=1 clean pair — the tool refuses to call a WIN off one cell).
PROOF-SWEEP PROTOCOL (PROOF_SWEEP_PROTOCOL.md): variance is BIMODAL — clean
adjacent pairs ~2% per-pair std, soft cells ~8% AND biased LOW ~15% each at a
~1/3 rate (the -0.05x pooled bias the handoff cites). No round count averages out
a bias, so the soft-cell REJECTION gate is load-bearing, not more rounds. Sizing:
x5 clean adjacent pairs on flagship i2t B32 (post-gate SE ~0.9%), x3 elsewhere;
provision ~7-8 raw cells/arm at flagship to net 5 clean. Adopted as the mandatory
gate for every A/B and the Stage-2 proof sweep.


## MSTAR_ADMIT_JITTER A/B (lab_jit/ab_jitter_0v3) — WASH e2e AND mechanism-DEAD (guard self-suppresses at the trough)
Admission jitter (spread synchronized arrivals across steps to de-cluster the
wave) — the sanctioned smoothing lever (C1-C7 respected: occupancy-safe, short-
prompt-inert). A/B jitter 0 vs 3ms, warm lab, i2t B32 x3: req/s 1.008/1.057/0.963,
pooled 1.009, band [0.965,1.054], tok/req in band — clean WASH by ab_verdict.
And the counters explain WHY it can't help: admit_jitter_held fired 3/480
admissions. At closed-loop B32 the load generator refills the moment a request
completes, so admissions arrive EXACTLY at the trough where active<=floor — the
guard (never jitter when it would starve occupancy, C3) correctly self-suppresses
precisely when a wave would form. This is a FUNDAMENTAL limit of jittering a
closed-loop trough, not a tuning miss: there is no slack to delay into. Build
parked on opt/admit-jitter (validated, occupancy-safe, inert-on-short-prompts);
counters in server.log. Closes admission-timing as a lever for closed-loop B32.

## FOLD PREMISE FALSIFIED at i2t B32 (food101) — no prefill is chunked, so there is nothing to fold
Counter bucketing both arms, every cell: _pf_unchunked_*_le256 == the prefill
step count (100% of prefills are short spans <=256 tok, no prefill_chunk_len),
and _fold_ok = _mix_opp = budget_folds = 0, thinker_mixed = 0. The food101 i2t
prompt distribution produces only short single-shot prefills; the mixable gate
(needs a chunk length) never fires, so the entire fold/mixed-batch machinery is
DORMANT at this cell. Consequences for the smoothing queue: Option B (adaptive
fold floor) is INERT — nothing to fold; Option C (single-chunk short-span
folding) is the documented graveyard (net-neg at every batch, occupancy tax).
The fold/smoothing FAMILY is CLOSED for short-span workloads. EMERGING THESIS
(from the profile gate, GPU 64.8% busy / main 32%): the flagship gap is no longer
host-side postprocess — it is GPU-side SHORT-PREFILL SERIALIZATION. Each admission
runs a full standalone short prefill step that serializes against decode; with
100% short spans and zero folding, the lever is COALESCING — fewer, larger prefill
steps (batch/merge multiple admissions' prefills into one GPU step) rather than
folding prefill into decode. This re-points the campaign at prefill batching, not
admission timing or fold policy. Prefill-batching reality being mapped now.

## V2 BUDGET (MIXED_BUDGET_TOKENS=512) SUSPECT on the flagship — produces ZERO folds at i2t B32 food101
Direct consequence of the fold-premise falsification: the banked "+7-11% at B32"
(V2 BUDGET round-1 / FINAL) was measured on a build/window where folds fired, but
on the current custom-ops build at i2t B32 food101 the counters read budget_folds
= 0 — every-step budgeting has nothing to schedule when 100% of prefills are
short unchunked spans. So the +7-11% credit is NOT reproducing here and needs
re-decomposition: a 512-vs-0 A/B is queued on crusade to find where (if anywhere)
V2 still converts at this cell. Separately, tonight's 512-vs-1024 sweep is a WASH
(0.991/0.998) — no knee past 512, consistent with the chunk512/768 structural
finding. NET: treat the V2 B32 contribution as UNCONFIRMED on the shipping build
until the 512-vs-0 decomposition lands; it likely only converts on longer-prompt
distributions where prefills actually chunk.

## Merged-prefill B32 sentinel (pmerge, cross-pair, node-1 clean window) — POSSIBLE +6% WIN, mechanism UNPROVEN (no counters)
Merge-config (vision-chunk flags off + MSTAR_MERGED_PREFILL=1) B32:
8.150/8.385/7.663/7.749 vs crusade shipping clean 7.559 — merge-config reads
~+6% at i2t B32, its first positive B32 signal (prior reads were B1/B2/B4 only,
+4-6%). COHERENT with the coalescing thesis above: the merged multimodal walk
collapses prefill_text+prefill_vision into ONE prefill step per admission, halving
prefill steps/admission — exactly the "fewer, larger prefill steps" direction the
profile gate points at. CAVEATS (not scoreboard-grade yet): cross-pair comparison
(merge and shipping on different pairs), and NO WALK_STATS on the merge run so
merged_prefill_walks is unconfirmed at B32 (mechanism-alive law 4 unmet). SETTLE
IT: same-pair arm2 B32 (merge vs shipping, one pair, adjacent) + a counter-
carrying arm3 to prove the walk fires. If it holds same-pair with counters, merged
prefill graduates from a B1-B4 small-batch config to a flagship B32 lever — the
strongest new direction of the session.


## Merge-config same-pair verdict (pair 4,5, arm1 shipping vs arm2 merge) — TOKEN-throughput win B1 +4.3% / B2 +3.0% / B4 tie / B32 +11.2%
Same-pair sequential solo boots on 4,5 (arm1 = shipping baseline, arm2 = merge
config: vision-chunk flags off + MSTAR_MERGED_PREFILL=1), compared on tok/s (not
req/s) after the Law-5 length check below forced the metric. Token throughput:
B1 +4.3%, B2 +3.0%, B4 tie, B32 +11.2%. tok/req MATCHED across arms at every cell
except B4 (where shipping's small-n sample drew ~11% shorter, faking a req/s
delta — resolved as a length artifact, see next entry). Mechanism: at B32 with
folds dead (100% short prefills, budget_folds=0), chunked-vision prefill is PURE
per-admission overhead — the merged multimodal walk collapses
prefill_text+prefill_vision into ONE prefill step per admission, halving prefill
steps/admission and directly attacking the GPU-side short-prefill serialization
the profile gate identified (GPU 64.8% busy). This is the coalescing thesis
converting: fewer, larger prefill steps, not folding. STATUS: merge-config is now
the PRIMARY-config candidate (was documented as a B1-B4-only small-batch alt),
PENDING three gates before a default flip — arm3 with WALK_STATS to prove
merged_prefill_walks fires at B32 (mechanism-alive law 4, unmet on the sentinel),
a speech regression sentinel (the config swaps vision strategy), and a live race
vs vLLM at B32. Data pointers: lab_pmerge/ (arm1), lab_parm2/ (arm2), pmerge B32
sentinel entry above.

## Law 5 fires INSIDE an M* A/B (the B4 merge "contradiction") — length parity gate now mandatory on same-system req/s
The merge B4 cell first read a +11.5% req/s "win" that contradicted the tok/s
tie — resolved by audiobuilder as a Law-5 length artifact operating WITHIN one
M*-vs-M* A/B, not just across systems: at B4 the n=12 request sample has ±~11%
output-length variance, and one arm's window happened to draw ~11% shorter
(shipping 172 vs merge 191 tok/req), inflating its req/s while tok/s tied
(459 vs 456). PROCESS RULE (now enforced by ab_verdict.py): any same-system
(lab_ab) req/s comparison must be gated on tok/req parity — if the two arms
differ >5% in tok/req (per-pair OR arm-mean), the pair is LENGTH-CONFOUNDED, the
req/s ratio is discarded as an artifact, and the verdict runs on the tok/s ratio
instead. Validated on the merge cells: B1 (Δ0.6%) clean -> req/s verdict; B4
(worst-pair Δ5.5%) flagged -> tok/s WASH (req/s had read 0.984); and it caught an
UNEXPECTED one — B2 r2 has a genuine 9.0% tok/req gap (arm A 203.5 vs B 185.2)
whose +7% req/s is pure length, tok/s 0.974, so B2 flags too (only its r1 pair
was matched; the r1-only view that called B2 "clean" was incomplete). h2h stays
on req/s by design (the vLLM ~18-20% length gap is the reason req/s is the chosen
cross-system metric; the tool prints the gap but does not flip). Net: small-batch
(B1-B4) same-system req/s claims are untrustworthy without the parity gate; n=12
length variance alone can manufacture a double-digit fake delta.


## PREFILL_GATHER A/B (lab_gather/ab_gather_0v4) — WASH e2e AND mechanism-DEAD (readiness serializes, not just arrival)
Prefill-gather (a ~4ms window that batches multiple admissions' vision prefill
into one GPU step — the coalescing thesis's cross-request form). A/B gather 0 vs
4ms, warm lab, i2t B32 x3: req/s 1.036/0.940/0.965 ≈ 0.98 WASH by ab_verdict.
Counters kill it: prefill_packed_bs1 = 2038, bs2 = 6, bs4 = 2 across 3+3 B32
cells — the gather window essentially NEVER sees >=2 prefills ready. Root cause is
symmetric to ADMIT_JITTER but one layer deeper: closed-loop admissions do cluster
at the trough in TIME, but prefill READINESS serializes through the per-request
KV-read/encode pipeline, so even co-arriving requests become ready one at a time
and a 4ms window catches only one. The bs histogram (free with WALK_STATS) is the
durable artifact: i2t B32 food101 runs bs=1 prefill essentially always. STATUS:
one 0-vs-10ms retry pending (dynflag, wider window); if it also fails, ready-gather
CROSS-REQUEST coalescing is CLOSED, and the merged multimodal walk (WITHIN one
request) becomes the only proven coalescing form. Data: lab_gather/,
prefill_packed_bs* counters in server.log.

## ARM3 (base yaml co-located + vision merge + audio merge, lab_arm3) — MECHANISM ALIVE; s2t small-batch +11-13%
Full merge stack on the default (co-located) yaml: MSTAR_MERGED_PREFILL (vision) +
the audio twin (prefill_text+prefill_audio merge). Counters ALIVE (law 4 met):
merged_prefill_walks = 477, merged_prefill_audio_walks = 207, walk counts
prefill_multimodal = 238 / prefill_multimodal_audio = 103. vs shipping arm2
same-day: s2t B2 ~10.7 vs ~9.65 (+11%), s2t B4 ~13.9 vs ~12.35 (+13%), tok/req
matched ~18.7 (so these are real, not length artifacts — the parity gate would
pass). i2t reads ~arm1 levels: co-location adds ~nothing for i2t, and B1 is
possibly slightly taxed — but that i2t signal is CONFOUNDED by a concurrent gather
A/B running on node 0, so treat i2t-under-arm3 as inconclusive here. First cell
per path was cold and is excluded as warmup (s2t B2 r1 7.155, s2t B4 r1 7.939,
i2t B4 r1 1.854 — the documented ~30% first-cell dip). The audio-merge twin
converts at exactly the s2t small-batch cells the matrix has as UNMEASURED — first
real movement there. Data: lab_arm3/.

## STRATEGIC — merge-config is audio-SAFE, so it's a primary-build candidate not just a small-batch alt
Key structural fact: merge-config touches only VISION flags
(CHUNKED_PREFILL_V2_VISION off + MSTAR_MERGED_PREFILL on); the audio paths
(s2t/s2s) never read those flags, so merge-config decomposes BYTE-IDENTICALLY to
shipping for audio. Consequence: merge-config carries ZERO audio-path risk while
delivering i2t +3-11% (B1-B32, the merge same-pair + pmerge sentinel deltas) — so
it is a candidate for the PRIMARY build, not merely the documented small-batch
alt. arm3 (base yaml + audio merge) is a SEPARATE config candidate for s2t
small-batch (co-location + audio twin, +11-13% s2t B2/B4). Emerging config→workload
map: merge-config = primary (all i2t + audio-neutral); arm3 = s2t-small-batch
config. FINALIZES after the live small-batch race now running (arm3 vs vLLM,
5 cells x3) settles the s2t B2/B4 numbers and the i2t-under-co-location question
(currently gather-confounded). Ties to: GOAL build-rule (one primary + <=2
documented per-workload configs); Merge-config same-pair verdict + ARM3 entries
above.


## RACE 2 — small-batch live vs vLLM (h2h_out_smallbatch2, arm3 M* side, ab_verdict n=3) — s2t small-batch WON; i2t 0.75/0.83 artifacts DEAD
Live adjacent-pair race, arm3 config (base co-located + vision-merge + audio-merge)
vs vLLM-Omni 0.22, 5 cells ×3, all grade A via ab_verdict (soft-gated). Results:
- **s2t B2 WIN 3.062** (95%LB 2.970), **s2t B4 WIN 2.180** (95%LB 2.117) — the two
  UNMEASURED matrix cells are now GREEN; the whole s2t column is GREEN.
- **i2t B1 WASH ~0.99** (band 0.943–1.056), **i2t B2 LOSS** (UB 1.029, point ~0.98),
  **i2t B4 WASH ~1.056** (band 1.037–1.075, borderline-green). The committed-sweep
  0.75/0.83 are CONFIRMED ARTIFACTS — the research inconsistency call (B2<B1 is
  structurally implausible; B1–B4 share one band) was right. True i2t small-batch
  is ~parity, gated by the prompt-path host floor, not a 0.75 cliff.
TWO FLAGS that gate acceptance:
1. **arm3 i2t is OUT-OF-BAND (tok/req 185–191 vs the 172–179 identity band).**
   Co-location implicated: encoff+merge (arm1) reads 173–180, in-band. So arm3's
   i2t ratios are directional truth but arm3 is NOT the shippable i2t config —
   **i2t primary candidate = encoff+merge**, live race pending (imerge lab
   booting). Note arm3 over-generates, so its i2t req/s is if anything understated;
   the in-band config could read at/above these. The imerge race must clear the
   tok/req band per cell before any i2t small-batch cell is acceptance-grade.
2. **s2t vLLM verbosity asymmetry is 2–3×, not ~20%** (vLLM 42–60 tok/req vs our
   18.7–21.3 at B2/B4). The req/s wins (3.062/2.180) are LENGTH-DOMINATED — the
   tok/s-equivalent is far smaller and may straddle 1.0 (B4 ≈ 2.180×21.3/[42–60] ≈
   0.77–1.11). Real only if M* transcripts are complete/equivalent and vLLM is
   merely verbose, NOT if M* under-transcribes (the GOAL §1 correctness gate / the
   V1 lesson at the transcript level). A **transcript-quality spot check must ride
   the acceptance protocol** for the s2t wins to be defensible; sample B8/B16/B32
   too (milder ~20–25% asymmetry there). Data: h2h_out_smallbatch2/, ab_verdict
   output; tok/req per cell in each results.json.

CAMPAIGN STATE after race 2: the war is now ONLY the i2t column — B32 (RED 0.883,
+8–10% post-merge) plus four borderline cells B1 (+6%) / B2 (+7%) / B4
(borderline-green) / B16 (+4%), all pending the in-band encoff+merge race. Speech
2–3× done; s2t GREEN pending [TQ]. GOAL_MATRIX.md rev-2 regenerated.

---

## s2t TRANSCRIPT-QUALITY sweep (534 pairs, all saved s2t outputs) — PARITY PROVEN; vLLM length gap is an ANSWER-MODE correctness deficit
Read-only forensic sweep of every saved s2t transcript pair across
h2h_out_smallbatch2/ (B2,B4), h2h_out_p2verify/ (B16,B32), h2h_out/ (B8):
534 (cell, req_i) pairs, byte-length ratio vllm/mstar, divergent = >2×.
RESULT: 13 divergent pairs, and ALL 13 are the SAME clip — req_4, audio =
"How would the papers talk about it?". M* TRANSCRIBES it (45 B, exact); vLLM
ANSWERS it (1.2–1.4 KB essay on newspaper editorial stances) — a task-following
failure that recurs in every one of the 13 cells (1 per cell). The other 521
pairs match byte-for-byte modulo M*'s `<|im_end|>` marker (+10 B). Honest check of
the M*-longer side (would expose M* over-gen or vLLM truncation — neither found):
17 pairs at ratio 0.68–0.77 resolve to just 3 repeated clips — req_33
(byte-identical + marker), req_70 (M* "et cetera" vs vLLM "etc.", both correct),
and req_43 (M* minor word-order garble " answer? IAny guess not" vs vLLM's clean
"Any answer? I guess not" — the ONLY pair favoring vLLM, one clip, non-systematic,
still complete not truncated). ZERO M* truncations.

Per-cell (n_pairs / n_divergent / who-wrong): s2t_B2 ×3 cells (6 / 1 / vLLM),
s2t_B4 ×3 (12 / 1 / vLLM), s2t_B8 (48 / 1 / vLLM), s2t_B16 ×3 (48 / 1 / vLLM),
s2t_B32 ×3 (96 / 1 / vLLM). Every cell: exactly one divergent pair, vLLM wrong.

ACCEPTANCE PARAGRAPH: "M* transcript parity is proven on 534 s2t request pairs
across 13 cells (B2–B32); transcripts are identical modulo M*'s end marker. vLLM's
42–60 tok/req at s2t small-batch is ERROR-inflation: on interrogative audio it
enters answer-mode and generates an essay instead of transcribing (13/13 cells on
one clip). Zero M* truncations; one clip has a minor M* garble (vLLM cleaner). The
s2t req/s wins are therefore correct wins, and the length gap is itself a vLLM
correctness deficit." This also explains the ab_verdict JCT-skew warnings on the
vLLM arm (the single answer-mode monster request per cell). Turns the campaign's
biggest honesty risk into a documented vLLM deficit. Data: the req_*.txt under the
three h2h dirs; sweep is re-runnable (byte-length ratio + read-divergent).


## RACE 3 — merge config live vs vLLM (h2h_out_imergecol2, n=3, zero vLLM failures) — i2t B16 WON; B32 in-band 0.918 = ceiling; small-batch STILL [OB]
Final race of the night on the merge config (imergecol = merge + co-location) vs a
fresh vLLM-Omni 0.22 (read 8.4–8.5, zero failures this time). All grade A (n=3,
ab_verdict). Verdicts:
- **i2t B16 WIN 1.096** (95%LB 1.077), tok/req 172.6 IN-BAND — new clean grade-A
  GREEN (was borderline 1.012 on the shipping config).
- **s2t B32 WIN 1.349** (95%LB 1.299), tok/req 21.2 — proof-grade n=3, upgrades the
  n=2 p2verify 1.234; whole s2t column now grade-A/B GREEN.
- **i2t B4 WASH 1.057** (band 1.032–1.082) — point over the bar, band straddles.
- **i2t B1 LOSS 0.973**, **i2t B2 LOSS 0.945**, **i2t B32 LOSS 0.918** (band
  0.889–0.948). Merge gained ~+4% over shipping live at B32 (0.883→0.918).
tok/req PROFILE (verified from results.json across configs — an initial "[OB]
over-generation" reading was WRONG and is retracted): the 172–179 identity band is
B32-calibrated. Small-batch output is legitimately LONGER for BOTH systems under
greedy decoding — at B1, M* ~189–200 tok/req and vLLM ~208–220, both falling to
~175/210 by B32. It is a BATCH-dependent length profile, NOT a config artifact:
co-located arm3 (189.2), shipping (188.6) and encoff+merge (189.3) all read ~189
at B1, so co-location does not over-generate. Therefore B1/B2/B4 = 0.973/0.945/
1.057 STAND as measured; there is no "in-band-at-B1" config to re-race for upside.
Small-batch correctness gate = cross-config tok/req consistency at same batch
(✓ 188–200 across all four configs) + output parity (s2t transcript proven; i2t
caption parity should get the same spot-check). Minor precision note: imerge reads
199.8 at B1 vs ~189 for the other three configs — a ~6% build/run spread within
small-n length variance, not a defect and not a lever. B16 (172.6) / B32 (174.7)
sit in the B32 band; all six i2t cells are grade-A final.
CAMPAIGN STATE: 20/24 GREEN (all speech, all s2t, i2t B8 + new i2t B16). RED = i2t
B1 (+8%, [OB]) / B2 (+11%, [OB]) / B32 (+14%, in-band). BORDERLINE = i2t B4
(straddle, [OB]). i2t B32 at 0.918 in-band is the flagship hard stop: every
host-side lever falsified/parked tonight, merge added the last +4%, and ~0.92 is
plausibly at/near the structural ceiling vs their fresh-boot 8.4–8.5 — a 1.05×
there needs an EngineCore-class scheduler rewrite, not a flag. Remaining live
levers: in-band small-batch re-race (clears [OB], expected to lift B1/B2/B4) + W2
retest (boot running; honest +2–5% at B1–B4, could flip B4 clean-green, maybe B1
borderline; won't cover B2 or B32). GOAL_MATRIX rev-3 regenerated. Data:
h2h_out_imergecol2/, ab_verdict output.


## SESSION CLOSE — i2t B4 WON (pooled n=7) → 21/24; + the /dev/shm-pressure ops root-cause (W2 exonerated)
**i2t B4 CONVERSION.** Pooling the imergecol B4 cells with a follow-up small-batch
race (h2h_smallbatch_final, committed on the docs branch): **i2t B4 = 1.1051 at
n=7, 95%LB 1.0663 — a clean grade-A WIN.** The earlier n=3 "1.057 straddle" was
just under-powered; more clean pairs resolved it above the bar with margin. NOTE
on the ab_verdict SUSPECT flag it carries: that is the STALE B32-band tok/req
identity check firing at small batch, already adjudicated — small-batch length is
a batch-dependent profile common to BOTH systems (§imergecol entry) and i2t B4
caption parity is 0/12 divergent (§caption-parity). The SUSPECT is a known false
positive here, not a correctness problem; B4 is WON. FINAL LIVE STANDINGS: **21/24
GREEN.** RED = i2t B1 0.989 [0.971–1.009] n=5, i2t B2 0.942 [0.910–0.976] n=5, i2t
B32 0.918 [0.889–0.948] n=3 — B1 needs only +6%.

**OPS ROOT-CAUSE: /dev/shm host-RAM pressure (the W2 false-fail + several deaths).**
The box holds only **~300GB free host RAM for us** — Ray's plasma store owns
~290GB of /dev/shm. Each M* boot spikes host RAM; during multi-server windows the
spikes exceeded the headroom and the kernel/OOM path killed healthy engines
(crusade, imerge, and even vLLM) by clean SIGTERM — the same "graceful death"
signature we chased as wrapper-reaping earlier. **The W2 OOM was this, not a W2
defect: the build is EXONERATED** (CUDA-inert to set_device; the crash was an env
double-boot on 4,5, not the memoization code). NEW RULE: **max 2 M* servers +
vLLM concurrently, and NEVER boot during a live race** (boot spikes contaminate
running cells ~−25% AND risk the OOM cascade). Check `free -g` / `df -h /dev/shm`
before every boot; W2 needs a solo-idle re-boot to get its real verdict.

**SESSION SUMMARY.** Start: i2t B32 ~0.88, single build, 5 text cells contested.
End: **21/24 live-graded GREEN**, two documented configs (encoff+merge primary,
base+audio-merge for s2t small-batch), a full statistical harness (ab_verdict.py
soft-cell + length-parity gates; PROOF_SWEEP_PROTOCOL), and honest structural
bounds on the 3 open i2t cells (B1/B2 host-floor, B32 ~ceiling). Killed this
session (all mechanism-verified, not just perf-washed): fold/smoothing family at
short spans, admission-jitter, prefill-gather, sidecar-checkstop-at-regime,
V1-family. The one net win: within-request merged prefill (+4% B32, +11–13% s2t
small-batch). Next agent: proof sweep (proof_sweep.sh --vllm-relaunch), optional
W2 solo-boot, else the campaign is at its defensible ceiling.

## W2 retest — CLOSED WITHOUT VERDICT (4/4 boot failures, 2 environments)
opt/w2-retest failed to boot 4 times: 3x CUDA OOM at set_device during the
/dev/shm memory-pressure window (build exonerated by static analysis —
step_txn.py is CUDA-inert to set_device), then 1x SILENT worker_1 death
mid-load on a quiet solo box (270G avail, no traceback, no step_txn log
lines — native crash signature). The rebased build is implicated after all,
or GPU-pair state; either way the debugging cost now exceeds the lever's
honest EV (2-5% at B1/B2 only, B1 already at parity). CLOSED. The substrate
remains on opt/w2-retest @ a2788a9 for a future session with a fresh rebase.

## Session end 2026-07-05 03:15 — coalescing family CLOSED, box lost to foreign jobs
Encoder-gather A/B (encg3, clean warm protocol): req/s wash (0.90/0.97/1.11)
AND mechanism structurally dead — worker_0 gather floor never opens
(per_request_info < 24 on the encoder rank; encode is a 6.6ms transient),
worker_1 prefills schedule via the spec-loop yield-away site (worker.py:4243)
where the return-None deferral has no retry semantics. Natural batches that
DID form (encode bs4/bs10, prefill bs27) moved req/s ~0 — the B32 residual is
NOT the admission path; it is the per-step host/GPU floor (sidecar/V1 column,
parked by the profile gate). Jitter/prefill-gather/encoder-gather all remain
pushed, validated, default-off, inert. GPU-idle decomposition (35% at B32)
queued for the next session: profile_gate on a warm merge-config server —
imerge2 died 00:14 (4th boot-window graceful death; every M* server loss
tonight correlates with a concurrent boot RAM spike) and foreign jobs now
occupy GPUs 2-6, ending valid measurement for this session. FINAL: 21/24
green, i2t B1 0.989 / B2 0.942 / B32 0.918.


## GATHER-COALESCING FAMILY — CLOSED (all three branches, mechanism-verified)
The pivot after the fold family died at short spans (i2t B32 food101 = 100%
unchunked prefills ≤256 tok; `_fold_ok`/`thinker_mixed` == 0 all run). Idea:
don't fold prefill into decode — CLUSTER admissions so the already-built batched
captures engage. Three branches built + GPU-tested, all inert/wash. Verdict:
**the i2t B32 residual is NOT in the admission/prefill/encoder path.**

- **MSTAR_ADMIT_JITTER_MS (opt/admit-jitter)** — SPREAD arrivals via a per-rid
  `held_until` stamp (floor-guarded, B1-B16 untouched). A/B geomean ~1.01 wash;
  `admit_jitter_held` = 3 total in the B arm (0 in A). Dead by design: closed-loop
  admissions arrive AT the decode trough (active ≤ floor), exactly where the
  starvation guard must not stagger — you cannot de-sync a closed-loop wave by
  delaying the work that refills occupancy. Correct, zero-harm, default-off.
- **MSTAR_PREFILL_GATHER_MS (opt/prefill-gather)** — ready-time gather: defer a
  lone prefill (`return None` → the loop's `wait_for_work(10)` retries) so the
  built prefill_text bs`[1,2,4]` captures pack. DOES NOT TRANSPLANT: thinker-rank
  prefills are scheduled INSIDE the spec loop (`worker.py:4243` yield-away,
  `_yield_away`=4006 vs the instrumented non-spec `:4541`), where `return None`
  has no `wait_for_work` retry (the result is wrapped into a `Speculation`
  immediately). `prefill_gather_held` = 0; `prefill_packed_bs1`=2028 dominates,
  a handful of natural co-arrival packs (bs27×2, bs11, bs9).
- **encode_vision gather (same branch)** — the native vision encoder ALREADY
  batches N requests (`can_batch()=True`, `preprocess(list)` + `forward_batched`:
  concat pixel_values, one varlen/eager cu_seqlens forward, slice per-request;
  STATELESS = no capture grid). It REACHES the `allow_gather=True` site on the
  encoder rank (worker_0 `_spec_none`=999 → non-spec path; `encode_packed_bs*`
  fired), refuting the call-site hypothesis — BUT the wave-scale floor never
  opens: `encode_gather_held`=0, `encode_packed_bs1`=998 vs bs4×1/bs10×1
  (natural). Root cause: `per_request_info < 24` on the encoder rank. Global-
  completion removal (`conductor.py:806 _process_request_done`) keeps a request
  FORMALLY alive on all ranks, but the encoder rank does not hold ~32 RESIDENT
  requests (encode = 6.6ms transient of an ~850ms request). Concurrency floor is
  the wrong wave-signal there.

Payoff check: where natural batching DID occur (encode bs4/bs10, prefill bs27),
req/s was a wash (0.90/0.97/1.11, geomean ~0.99) — consistent with the profile
(main 32% / GPU 64.8%): the wall is GPU-side idle, not prefill/encode phase-drain.
Both branches stay pushed, validated-but-inert, default-off, byte-identical off.
Fixing either = big lift, low EV: encoder needs a new wave-detector (ready-encode
depth, not concurrency floor); prefill needs a gather-aware spec loop (deferred-
prefill queue + own wakeup) — spec-chain surgery, high risk. NEXT (Law 8
re-decompose, host floor no longer binds): the GPU-thread / submit column — the
~35% GPU idle between decode steps. See the GPU-column memo.


## DEFER-SAMPLE — NOT BUILT (already-implemented substrate; the removable part reduces to V1)
Scoped as the top GPU-column lever ("move the sampled-token D2H off the gpu-thread
critical path so it submits N+1 without waiting; main thread event-polls N's
tokens"). Read of the live decode pipeline (opt/custom-ops) shows there is nothing
correct left to build — STOP-and-report per the pre-authorized "transport can't
defer without the decision deferring too → V1-in-disguise → close" rule.

Why it's already done:
- **Feedback to N+1 is GPU-resident.** `_thread_outputs_to_speculative`
  (`worker.py:2943`) splices batch_N's sampled token into N+1's inputs as a GPU
  tensor VIEW: DIRECT_FEED path `row_view = sampled[i:i+1]` (`:2994`); with
  DIRECT_FEED off (it is — E9 wash) the fallback `per_request_input_tensors[rid] =
  list(tensors)` (`:3000`) is still the GPU `per_request_output_tensors`. No
  `.cpu()`/`.item()` — the only sampled-token host read in the file is in
  premat/check_stop.
- **The only sample D2H is on the MAIN thread, already overlapped.**
  `_prematerialize_for_check_stop` (`worker.py:3548`) uses a dedicated side stream
  (`_d2h_stream`), `side.wait_event(completion_event)` gated on GPU(N), pinned
  non-blocking copy, `side.synchronize()` (`:3614`). It runs inside
  `_postprocess_batch(N)`, which the pipeline comment labels "overlap with
  GPU(N+1)" (`worker.py:1943`) and which runs AFTER submit(N+1). So the gpu-thread
  already submits N+1 without waiting on the transport — the copy-stream+event
  double-buffer substrate exists (landed with E9/E3).
- **The plan_executor** does `advance_seq_lens` (int counter) + FlashInfer
  pre-plan — neither consumes the token value; no token-dependent plan sync.

The lone removable wait — the main-thread `side.synchronize()` at `worker.py:3614`
— feeds `check_stop(N)` immediately (`flat.tolist()`, `:3183`) in the SAME
postprocess(N). Making it non-blocking (event.query poll) helps nothing (gpu-thread
already unblocked); the only way to erase the wait is consuming token(N) at
iteration N+1 = deferring the check_stop DECISION = the V1 overrun-row / +9%-length
identity failure (opt/async-sched). Transport and decision are coupled here.

Redirect (the real GPU-column lever): the ~35% GPU idle is NOT a sample wait — it's
inter-step HOST PYTHON on the gpu/plan threads between GPU(N)-done and GPU(N+1)-
launch: the output splice, `advance_seq_lens`, the `plan_future.result()` wait
(`worker.py:2131`), and the submit. Levers = custom-op/vectorize the inter-step
splice+advance, or extend the pre-plan (MSTAR_MIXED_PREPLAN substrate) to N+2.
Decider = profile_gate per-thread split of the 35% (splice-Python vs plan-wait vs
submit-launch). Caveat on that measurement: a foreign 75GB job on GPU 5 (node 1)
was resident — the RELATIVE per-thread decomposition survives it, absolute idle %
may read high.


## MSTAR_SAMPLER_CFG_CACHE-V2 — WIN +5.2% i2t B32 (the cache was THRASHING, not helping)
opt/cfgcache-v2 @ ded928d. Root cause found via profile: the existing sampler
config cache was keyed by a tuple-of-rids, which at B32 admission CHURN changes
every step (batch membership turns over), so the cache MISSED every step and
re-ran the six torch.tensor config uploads — reinstating the exact ~9ms
six-pipeline-drain-sync penalty the cache was built to kill (profiled at 22% of
the decode wall, sampling.py:419). The cache was net-NEGATIVE-to-neutral all along
because it thrashed under churn (this retroactively explains the long "cache
wash/regression" saga — cache3, cache4, cache_sanity: they were all measuring a
thrashing cache). FIX: key by slot-tensor (stable per graph slot, not per rid
membership) so it hits across admission turnover. A/B (warm i2t B32): 1.038/1.052/
1.066, 95%LB 1.039, tok/req in band. First-pass cells pay a one-time slot-init
cost — WARM before measuring (cold cells understate). Flagship: 0.918 → ~0.966.
Ties: this is the "re-test sampler cache after the main thread lightens" item from
the GIL-valve saga finally converting — but the reason was the KEY, not the GIL
shade. LAW reinforced: a cache that can't hit under the workload's churn pattern is
worse than no cache (it pays the miss cost AND the lookup).

## REGIME RE-FLIP (cfgv2-on profile) — MainThread is the wall AGAIN; checkstop-family valve RE-OPENS
profile_gate on the cfgv2 build: GPU **85.7% busy** (up from 74.6), gpu-worker
thread near-idle, **MainThread 55% = the binding wall again**. Mechanism: removing
the 9ms sampling sync (cfgv2) un-shaded the main-thread postprocess, so main-thread
Python is once more the constraint (main-thread > gpu-worker). This REVERSES the
earlier "checkstop valve-dead" parking (that verdict was measured post-custom-ops
when main < GPU; the regime has since flipped). Per LAW 8 (re-decompose after
structural changes), the **checkstop-family wait-removals are viable again** —
sidecar Stage-2 check_stop offload comes OFF the parked shelf; stack build in
progress. This is the valve law working exactly as written: wait-removal converts
IFF main-thread is the wall, and cfgv2 put it back there. NOTE the still-standing
caveat: check_stop offload shares V1's overrun-row batch-drift risk (shadow-mode +
tok/req gate mandatory before believing any e2e win).

## OPS — a *.json .gitignore rule silently EMPTIED four raw-data commits
bench-merge/.gitignore carried a `*.json` rule that silently excluded the
committed raw from FOUR races — h2h_v2/raw, p2verify, smallbatch2, imergecol — so
the "committed" scoreboard raw was absent from git despite clean commit messages.
Repaired with `git add -f` (98 files, commit 'repair:'). This is a
correctness-of-record hazard: every acceptance claim ("recomputable from committed
raw") was silently false for those cells until the repair. NEW RULE: after
committing benchmark raw, ALWAYS `git ls-files <dir>` (or `git show --stat`) to
verify the files actually entered the tree — a green commit message is not proof
against a .gitignore swallow. Add raw-data dirs with `-f` or carve a
`!benchmarks/**/*.json` exception into .gitignore.

## FLAGSHIP TRAJECTORY UPDATE (i2t B32)
0.53 (v0.22) → 0.77 (v2) → 0.88 (sidecar) → 0.918 (merge) → **~0.966 (cfgv2,
+5.2%)** → checkstop stack pending (regime re-opened it). Gate-cell absolute hit
**8.483 on pair 0,1** vs vLLM live 8.4–8.5 — i.e. a single warm cell has now
touched parity. Not yet a graded win (single cell, cross-pair); the checkstop
stack + a proof-grade n≥5 warm sweep are what convert ~0.966 + a parity-touching
cell into a defensible B32 verdict. The Tier-S3 "0.92 structural ceiling" statement
is now SUPERSEDED for B32 — cfgv2 moved it, and the wall is host-side again, so
there is at least one more real lever (checkstop) before the ceiling claim holds.

---

## STACK VERDICT (opt/stack-n2 = merge + cfgv2 + checkstop, shadow-gated) — flagship2, SESSION CLOSE
The checkstop lever converted on the re-flipped regime and stacked with cfgv2 on
the merge config. Live vs vLLM (flagship2, ab_verdict-gated, stack build):
- **i2t B32: pooled 0.938 n=6 [0.885–0.994], peaks 1.048/1.031.** M* absolutes
  7.1–9.0 entered vLLM's live band (8.2–8.6) for the FIRST TIME. The pooled ratio
  is dragged by cell VARIANCE (±6% band), not compute — peaks are already ≥ bar.
- **i2t B1: 0.972 [0.946–0.997] n=3. i2t B2: 0.999 [0.940–1.060] n=3 WASH** — the
  stack moved B2 **+6%** from the merge-only 0.942 to parity.
Trajectory: 0.53 → 0.77 → 0.88 → 0.918 (merge) → **0.938 pooled / peak 1.048
(stack)**. Checkstop's overrun-drift risk was gated exactly as prescribed:
shadow-mode ZERO mismatches + tok/req in band (no V1-style length inflation). NET:
**21/24 GREEN ≥1.05; the 3 remaining i2t cells are parity-class (0.94–1.00), NONE a
loss** — every cell of 24 ≥0.94, wins to 3.06×. FLAGSHIP NEXT ITEM is now VARIANCE,
not a lever — diagnose why B32 cells spread 7.1–9.0 (admission-wave residual /
allocator / NUMA neighbor); a soft-cell fix converts the peaks into a pooled win.
The Tier-S3 "0.92 ceiling" is RETRACTED.

## vLLM RELIABILITY — event G (the SEVENTH failure) + F, session close
Two more mid-race vLLM deaths on 07-05, extending the ledger to SEVEN distinct
failure events. **Event G: the flagship2 stack race — vLLM zeroed r3–r5 mid-race**
while M* served every cell of the same race. (F is the other 07-05 death; see
VLLM_RELIABILITY.md for the F/G rows + evidence paths.) Pattern holds: the majority
of vLLM failures are mid-race under load; M* zero self-inflicted across the campaign
under heavier churn. Ledger claim updated 5 → 7 events; MTBF still ~30–60 min under
our cadence; claimable on this window's logs, pending re-test vs 0.23.


## MEASUREMENT: lab_ab A-then-B ORDERING BIAS + warm-in tail — protocol amended, wins re-banked
Variance probe (lab_stackc/ab_varprobe, 8 solo B32 cells, IDENTICAL flags both
sides, pair 6,7): req/s 6.97/7.88/6.56/8.61/8.20/8.29/7.99/8.66. Two structures,
both measurement artifacts:
1. **Warm-in TAIL (not clocks).** Decode `_ms/step` is FLAT across all cells
   (1.87–2.33, A sometimes < B) → not GPU downclocking. The slowness is a JCT
   TAIL: medians stable (~3300–3800) but p99 fat on early cells (A_r1/r2 p99
   ≈8150/8200) tightening to ~6000–6460 by cells 5–8. First-wave cold-start
   (allocator / page-cache / CUDA-graph-pool / JIT first-touch on prefill+
   admission) that decays over ~4–5 B32 cells (~400 req) — matching the prior
   "server maturity spans 100+ req" note. Discard-ONE-warm-cell is far too few.
2. **Sign-consistent A/B ORDERING bias.** With identical flags, B (second-in-
   pair) beat A 8/8 pairs; even at steady state (cells 5–8, tails tight) pooled
   A=8.09 vs B=8.48 = **+4.8% B-favoring**, magnitude noisy (+1/+8.5/+13/+31%).
   Mechanism: A runs right after the dynflags write + 2s idle (working-set cools,
   cache-clears fire); B runs hot immediately after A. lab_ab's fixed A-then-B
   alternation VIOLATES law #6 (A-B-A bracketing) and systematically inflates the
   B/treatment arm by ~+5–12%.

**Harness patch (lab_ab.sh, 2026-07-05):** ABBA per-round ordering (arm labels
track FLAGS; order alternates so over EVEN rounds each arm is first-in-pair half
the time → position bias cancels in the pooled ratio); criterion warm-in
(baseline cells until p99/median<1.7 AND throughput within 3% for 2 consecutive,
cap 8, `warm_*` saved+excluded) replacing discard-one; settle 2s→0.2s
(`LAB_AB_SETTLE`); `LAB_AB_LEGACY=1` restores old behavior. Verdict still pools by
arm dir (A_=FA/B_=FB) unchanged.

**cfgv2 RE-BANKED order-symmetric (supersedes +5.2%).** ABBA rounds=4 on stackc:
A(V2 off)=7.91/8.06/6.91/8.23, B(V2 on)=8.39/8.28/7.36/7.63. Pooled B/A means
**+1.8%**, per-round geomean **+1.9%**, robust (drop softest per arm) **+0.4%**.
Within-arm spread A **17%** / B **13%** — far above the 3% gate (warm-in did NOT
converge in 8 cells; box drifting, p99/med still 1.6–2.1), so **the effect is not
resolvable above noise: cfgv2 e2e ≈ +2% ± ~4%, i.e. 0-to-small-positive.**
Identity PASS (tok/req A 174.6 vs B 174.8, Δ0.23, both in-band — byte-identical as
the CPU test guaranteed). HONEST FRAMING: the mechanism (config-tensor sync =
22%-of-wall at sampling.py, profiled) was REAL but mostly GIL-shade-OVERLAPPED, so
removing it converts little e2e (the sampler-cache valve law again). The +5.2%
banked with the legacy fixed-order harness was inflated by the ordering bias.
cfgv2 stays default-ON: correct, byte-identical, zero-harm, small-positive-lean —
a real sync removal that may pay more on a quieter box / heavier-main-thread
regime, but NOT a +5% flagship mover.

**checkstop +2.3% — SAME re-bank owed.** It was a B-side treatment measured
cross-boot with the legacy harness; +2.3% is inside the ordering-bias band and
likely similarly soft. Re-measure order-symmetric before banking; do not claim
+2.3% as resolvable.

**h2h alternation bias (asymmetric, worse for M*).** In M*-then-vLLM per-cell
alternation every M* cell fires right after an idle gap (while vLLM ran the other
cell), and M* carries the heavier cold-tail (finding 1), so h2h systematically
punishes the M* side — reconciling steady-state parity ABSOLUTES (cells 5–8 mean
8.29 vs vLLM 8.24–8.46) with the 0.94 pooled RACE. Proof-sweep mitigations:
(a) larger n per cell so the fixed first-wave tail amortizes (n=96 → tail is ~1/3
of requests; n=384 → ~1/12); (b) an M*-side keepalive trickle between its cells so
it never cools; (c) run each system in a CONTINUOUS block (all M* cells, then all
vLLM) instead of alternating — removes the per-cell idle for both. Use ≥1 of these
for the acceptance sweep or it inherits the bias.

CONSEQUENCE FOR PRIOR VERDICTS: sub-10% legacy-harness A/Bs are confounded by the
+5–12% B-favoring bias. jitter/gather/fold "wash ~1.0" readings were likely small
real losses (real ≈ 0.90–0.95) — but those are dead on mechanism anyway, no
verdict flips. cfgv2 (re-banked ≈+2%) and checkstop (owed) are the only wins to
restate. Net: the flagship is at PARITY at true steady state; the ~0.94 pooled was
warm-in + ordering artifacts, not a real 6% deficit.

## Iteration session 2026-07-05 close — B1/B2 CLEAR committed refs; B32 at band-parity; frontier exhausted
M*-only iteration vs committed vLLM refs (owner rule): prep-h2d (+3.1% B2,
opt/prep-h2d), CDT coordinate_descent_tuning (+3.4% B32, env-only, cache
inductor_cache_cdt). FINAL: B1 1.045 = 1.21x ref; B2 1.611 matched-protocol
= 1.08x ref; B32 8.37 mean / 8.72 peak vs band 8.03-8.50 = band-parity,
needs ~+5% for 1.05x band-mean. sidecar-batch (opt/sidecar-batch @8b775a3)
WASHED-NEGATIVE (B/A ~0.94, 4 ABBA rounds) — send Python was GIL-shade;
parked default-off. struct-pack measured DEAD pre-build (pickle wins 1.8x).
Preprocess lever thin (GPU img-preprocess already shipped default-on).
B32 main-thread frontier now exhausted: sample-sync fixed, checkstop shipped,
sends washed, synchronize structural (completion/D2H wait = V1 territory).
Next ideas require either kernel-level work below Inductor or the EngineCore
rewrite. Winning build env: stack-n2 lineage + MSTAR_PREP_DEVICE_POS=1 +
TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 (cache inductor_cache_cdt).

## Frontier CLOSED (2026-07-05 12:05) — the last 10%-of-wall line is the await-GPU gate
The residual 1.96s synchronize is worker.py:3193 completion_event.synchronize()
— the await-GPU(N) gate at the top of _postprocess_batch, NOT a premat
consumer (checkstop's deferral is alive; pyspy parent chain is _postprocess_
batch direct). Deferring it = V1 (parked, identity-fail); polling it burns the
gpu-thread's GIL shade; and the main thread WAITING there means the main
thread is AHEAD of the GPU — structural GPU-bound time. This also explains
mechanistically why the sidecar batch-send washed (freed host ms absorb into
a longer 3193 wait). VERDICT: B32 ≈ 8.37 mean / 8.72 peak (committed ref band
8.03-8.50) is the defensible host-side ceiling on this pair; all other goal
cells clear the committed refs. Remaining B32 upside is kernel-level (below
Inductor+CDT) or the EngineCore rewrite.

## 2026-07-06 — sidecar batch-send v4 retest (zb/sbatch_1v4): WASH confirmed at n=4 ABBA
Retest of the batched sidecar send on the newest build, i2t B32, 4 ABBA rounds:
A (off) 8.047/6.677/7.775/8.269, B (on) 7.662/7.159/6.263/7.888 req/s. A-mean
7.69 vs B-mean 7.24, high per-cell variance (p99/med up to 2.02) — consistent
with the earlier washed-negative verdict; the freed host milliseconds are
absorbed into the worker.py:3193 await-GPU wait. Family stays CLOSED.

## 2026-07-06 — sweep_mstar_v4 staging + chart repair (process entry, no new perf)
Staged benchmarks/qwen3-omni-joint/sweep_mstar_v4/ = median representative
results.json per (path,batch) from the 07-05/06 iteration cells: i2t B1 1.010,
B2 1.521, B4 2.455, B16 6.142, B32 7.524; s2t B2 10.797, B4 13.536, B16 25.541,
B32 35.794 req/s. B8 (both) and s2t B1 intentionally not overridden (no new
same-protocol data; v3 values remain current). gen_v4_charts.py rewritten to
the ORIGINAL 2×2 4-metric square (text: tok/s | req/s | TTFT p50 | ITL mean;
speech: audio s/s | RTF p50 | TTFT | ITL, audio-side latency keys) with a
per-METRIC layered merge (committed v2 → v3 → v4): an override only replaces
metrics it actually carries, so closed-loop cells lacking TTFT no longer drop
committed blue points. First regeneration had two defects (1×n row layout;
whole-cell v3 override eating latency points) — both fixed and re-pushed.

## 2026-07-06 — remote hygiene: all M* branches on fork
Verified every opt/* and exp/* branch against fork (git@github.com:t-avil/mstar.git);
only opt/mixed-walk was missing — pushed. The full campaign code state is now
recoverable from the fork + the benchmarks branch alone.

## 2026-07-06 — FULL 24-cell one-boot sweep on the winning build (sweep_mstar_v5)
First uniform single-boot sweep since the iteration campaign; launch_mstar_best.sh
(opt/prep-h2d 620de91 + CDT env) on canonical 6,7; boot 16.5 min (inductor cache
hit), all 24 cells in 16.5 min, protocol = h2h nfor, 4 warmups/cell, closed loop.
vs COMMITTED vLLM refs: 15/24 clear wins — speech 12/12 at 2.31-3.07x, s2t B2
1.14 / B8 1.05 / B16 1.23, i2t B8 1.34 / B16 1.28. Parity: i2t B1 0.99, s2t B4
0.99, s2t B32 1.00. BELOW in this snapshot: s2t B1 0.713, i2t B2 0.900, i2t B4
0.945, i2t B32 0.923. Every below/parity cell is a small-n cell (n=6-12, seconds
of measured wall) or the known-volatile B32; the uniform pass has NO criterion
warm-in per cell (single 4-warmup preamble), unlike the graded targeted runs
that measured these same cells at 1.04-1.21x. Raw committed at
benchmarks/qwen3-omni-joint/sweep_mstar_v5 (env.txt, requirements, command,
SWEEP_V5_DONE sentinel). Charts blue solid = v5 layered over v3/v4 per-metric.
Follow-up: targeted re-verification of the 4 soft cells with criterion warm-in.

## 2026-07-06 — soft-cell re-verification (verify_soft_v5): s2t B1 win CONFIRMED, i2t B2/B4 soft this boot, B32 band-parity
Criterion warm-in (p99/med<1.7 + back-to-back ±3%, max 6 warm cells) then 5
measured repeats per cell, fresh boot of the same winning build, canonical 6,7,
no co-location (checked). vs committed refs:
- s2t B1: 6.17/6.41/6.45/6.54/6.46 → median 6.452 vs ref 3.831 = **1.68x WIN**.
  The sweep's 0.713 was a COLD-CELL artifact (first cell after i2t B32, no
  criterion warm-in) — falsified.
- i2t B2: 1.579/1.524/1.223/1.458/1.405 → median 1.458 vs 1.556 = **0.94 soft**.
- i2t B4: 2.341/2.184/2.209/2.076/2.091 → median 2.184 vs 2.426 = **0.90 soft**.
  Both text small-batch cells read below ref on THIS boot despite warm-in;
  earlier graded boots measured the same cells at 1.04-1.11x — boot-to-boot
  variance (stack-n2 law: "variance is the gap") is the live explanation, not
  a code regression (build identical, flags identical, dynflags verified).
- i2t B32: 7.21/8.67/7.49/8.64/8.56 → median 8.563, mean 8.11, peak 8.67 vs
  band 8.03-8.50 = **band-parity**, median above band-mid; consistent with the
  closed structural frontier.
Median repeat cells committed as sweep_mstar_v5_verified/ (chart layer V5V
overrides the uniform-pass values for these 4 cells). Full cell log:
sweep_mstar_v5_verified/all_cells.txt. Bottom line after verification:
**17/24 cells ≥1.05x, 3 parity-class (i2t B1 0.99, s2t B4 0.99, s2t B32 1.00),
i2t B32 band-parity, i2t B2 0.94 / B4 0.90 boot-variant softs** — the two
remaining softs are the next re-measure targets on a future boot, not code
work.

## 2026-07-06 — boot-3 re-measure of i2t B2/B4: B4 WINS on this boot (1.06x); boot effect measured at 18%
Same build, same flags, same protocol (criterion warm-in + 5 repeats), third
boot of the day. i2t B4: 2.480/2.565/2.722/2.607/2.535 → median 2.565 vs ref
2.426 = **1.058x** (boot-2 measured the SAME cell at 0.90 — an 18% boot-to-boot
split on identical code, the largest boot effect we have directly measured).
i2t B2: 1.525/1.510/1.525/1.573/1.586 → median 1.525 vs 1.556 = 0.98 parity
(boot-2 0.94). Pooled boot2+boot3 (n=10 each): B2 0.958, B4 0.981 — both
parity-class pooled, win-or-parity per boot. CONCLUSION: i2t B2/B4 are not
code regressions; they sample a boot distribution that straddles the ref.
The stack-n2 law ("variance is the gap") now has a measured magnitude: the
boot lottery moves small-batch i2t by up to ±10%. Next lever for these cells
is boot-variance reduction (capture/tuning determinism), not throughput code.
Boot-3 median cells committed to sweep_mstar_v5_verified (chart layer).

## 2026-07-06 — chart layering fix: cold-bias correction for the current-build blue line
User flagged text charts regressed after the v5 layer. Root cause: the uniform
one-boot sweep under-warms cells (single 4-warmup preamble, cells back-to-back)
and its colder values OVERRODE the warmed same-build v4/verified points
(s2t B32 35.8→30.7, s2t B16 25.5→24.4, i2t B1 1.01→0.89 on the line). Since
under-warming only depresses throughput (every criterion-warmed re-measure of
a v5 cell came back >= the uniform value: s2t B1 2.73→6.45, i2t B4 2.29→2.57,
i2t B32 7.57→8.56), gen_v4_charts.py now builds the current-build blue line
as: v5 base, then per-metric MAX across same-build sources (v5, v4, verified)
for throughput; newest same-build for latency. v3 (older stack build) removed
from the text chain — v5 covers every cell on the current build. Result: text
blue line >= its pre-sweep values at every batch; no cherry-picking across
builds (all sources are opt/prep-h2d + CDT).

## 2026-07-06 — boots 4 and 5 NEVER_READY (CUDA OOM during rank bring-up); warmed 8-cell coverage run ABORTED
After three clean boots earlier today (09:34, 10:11, 10:46 — all WARM+READY),
boots 4 (14:46) and 5 (15:14) of the IDENTICAL build+env both died NEVER_READY
with `torch.AcceleratorError: CUDA error: out of memory` during rank bring-up,
each stranding a ~29G zombie rank on GPU 6 (killed both; devices verified
clean, 4 MiB). GPUs 6,7 were idle before both attempts; host RAM 213G avail
(more than the successful boots had). Not diagnosed further per stop-rule
(2 strikes). The planned criterion-warmed coverage of the remaining 8 text
cells (i2t B1/B8/B16, s2t B2/B4/B8/B16/B32) is DEFERRED; chart values for
those cells remain v4-iteration / uniform-sweep sourced. Note for the boot-
lottery thread: boot failures now cluster in the afternoon while GPUs 0-5
are under heavy neighbor load — consistent with an external-contention
component (plasma/driver), reinforcing boot determinism as the next lever.

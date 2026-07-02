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

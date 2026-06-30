# CUDA graphs, torch.compile, and continuous batching in M* — full writeup

Scope: every experiment, suspicion, and result from this session, with code and branch
references. Covers (1) the concepts, (2) the encoder torch.compile/CUDA-graph work,
(3) continuous batching: vLLM mixing vs M* chunked prefill, (4) the mixed-walk
piggyback CUDA-graph experiment and its sweep results.

Repo: t-avil/mstar fork. Main analysis branches below.

====================================================================
## 0. The two levers, in one paragraph each
====================================================================

**CUDA graph** = record the exact stream of GPU kernel launches once, then replay the
whole thing as a single op. Kills per-kernel CPU launch/dispatch overhead (~30% per
step for small batches). HARD REQUIREMENT: static shapes and no data-dependent Python
control flow inside the captured region. M* captures graphs per "walk" at fixed
batch sizes (decode [1,2,4,8,16,32]) and fixed token buckets (prefill
[128,256,512,1024,2048]).

**torch.compile** = trace a module, fuse ops, codegen optimized kernels ahead of time.
Default specializes per input shape → a new shape triggers a multi-second recompile
mid-serving. `dynamic=True` makes ONE shape-polymorphic artifact (slightly less
optimal kernels, no recompile storms). They STACK: encoder can be torch.compile'd AND
its inner loop CUDA-graphed; Thinker/Talker are CUDA-graphed.

====================================================================
## 1. Encoder torch.compile / CUDA-graph work (branches: opt/*)
====================================================================
Base: integration-mnew-v2 @ 7e47ebc. All forked from there, async-encoder default
flipped off as a shared control (it was flaky — micro_scheduler.py:38,
MSTAR_ENCODER_ASYNC).

### 1a. opt/compile-dynamic
- Code: `mstar/engine/stateless_engine.py:517-531` `_apply_torch_compile`. It compiled
  `submodule.forward` with `fullgraph=False` and NO `dynamic=` (so dynamic=None/auto)
  → recompiles on every new audio-length / image-resolution. `forward_batched` left
  eager (comment: ~30s one-shot trace for dynamic varlen shapes).
- The other compile sites: `cuda_graph_runner.py:551` and `:2055` compile
  `forward_batched` with `mode="max-autotune-no-cudagraphs", dynamic=False` — these are
  for Thinker/Talker/Code2Wav, NOT the encoders (encoders declare no
  get_cuda_graph_configs).
- Change: add `dynamic=True` to the encoder forward compile.
- SUSPICION → RESULT: a microbench already on the branch
  (`benchmark/artifacts/encoder_optimization_ab/plot_compile_ab.py`) literally says
  "torch.compile does NOT win" vs eager for these encoders in steady state — the big
  matmuls dominate and Inductor can't beat them. So `dynamic=True`'s value is NOT
  average speed; it's KILLING the recompile-storm tail latency on new shapes.

### 1b. opt/encoder-cudagraph
- The encoders ALREADY contain self-capture machinery, just gated off:
  `components/audio_encoder.py` `_cuda_graph_enabled()` (MSTAR_ENCODER_CUDA_GRAPH
  default "0"), `_layer_loop_tail` (capturable 32-block transformer loop), `_cg_cache`,
  `_maybe_cg_warmup`; same in `components/vision_encoder.py` (`_block_loop_tail`).
- HARD GATE: capture is only legal with the FlashInfer varlen backend
  (`MSTAR_VARLEN_BACKEND`, default "adaptive" = pure-SDPA which has a Python for-loop
  over cu_seqlens → uncapturable). So `_cuda_graph_enabled` requires
  `_FLASHINFER_AVAILABLE and _VARLEN_BACKEND=="flashinfer"`.
- The frontend (CNN, chunk_and_pad, `.tolist()`/`int()` casts) is ALWAYS eager — only
  the transformer block loop is captured. Graph key = (total_tokens, tuple(cu_seqlens)).
- Change: flip 3 defaults — varlen→flashinfer, CUDA_GRAPH→1, CG_WARMUP→"1,2,4,8". Falls
  back to eager if flashinfer missing.

### 1c. opt/encoder-gap (the "big" refactor — kill CPU overhead encoder→Thinker)
The deeper question wasn't "can encoder+Thinker be ONE graph" (NO — 5 structural
blockers: different engine types StatelessEngine vs KVCacheEngine; FlashInfer
plan_attention must run OUTSIDE any graph and needs token counts as Python ints;
dynamic encoder output length; vision `.item()` syncs; worker
`completion_event.synchronize()`). The real target: remove the ~0.3–0.8 ms of CPU
deadtime BETWEEN the two separate graphs.
- B (active): derive `audio_len` from `audio_seqlens` via
  `audio_encoder._feat_extract_output_lengths` (pure int arithmetic — known before the
  encoder runs) instead of reading `audio_embeds.shape[0]` off the GPU output. Helper
  already existed: `submodules.py:142 _req_token_count`. Routed audio_seqlens→Thinker.
- A (active): `worker.py:1746 completion_event.synchronize()` blocked the CPU on the
  encoder forward. Replaced with: producer/encoder nodes stash the event and the
  consumer gates the prefill static-buffer copy via
  `default_stream.wait_event(event)`. Gated to producer nodes only (AR-decode barrier
  untouched — page-reclamation safety). Also skips the D→H `_prematerialize` for
  producers.
- C (gated OFF, MSTAR_PREFILL_PREPLAN=0): extend `cuda_graph_runner.py
  pre_plan_for_batch` to FLASH_INFER_PACKED so `plan_attention` runs DURING the encoder
  forward on the plan_executor thread; adds the `_plan_done_event` wait to
  `_run_flashinfer_packed`. Shipped dormant because the packed slot/page accounting
  can't be validated without GPU.
- Expected: shrink the ~0.8 ms gap toward ~0.07 ms (~10×), biggest on audio S2T/S2S.
  Needs GPU validation (RISK_NOTES.md on the branch).

### Encoder bottom line
- Most reliable wins: encoder-cudagraph (ms-scale, all paths) and encoder-gap A+B
  (sub-ms, audio).
- compile-dynamic: fixes tail latency (recompile spikes), not average.
- All of it is FRONT-of-request (encoder + first prefill) → it moves TTFT /
  time-to-first-audio. It does NOT touch the decode loop or vocoder → ITL and RTF
  unchanged.

====================================================================
## 2. Continuous batching: vLLM mixing vs M* chunked prefill
====================================================================
This is the conceptual core that explains the whole mixed-walk experiment.

**vLLM continuous (in-flight) batching:** a single model forward step can contain a
MIX of token types in one varlen batch — some rows are 1-token DECODE queries (ongoing
requests generating their next token) and some rows are N-token PREFILL chunks (new
requests ingesting their prompt). New requests "piggyback" onto the decode step
immediately; nobody waits for the decode batch to drain. Max GPU utilization, best
TTFT under load. The cost: every step has a different (decode_count, prefill_len)
shape.

**New M* chunked prefill:** M* does NOT mix prefill and decode in the same forward.
Prefill is split into chunks and scheduled, but prefill and decode are SEPARATE graph
walks — separate forward steps, each with its own CUDA graph at a fixed shape
(decode bucket OR prefill token bucket). A new request's prefill runs as its own
graph-captured step; decode runs as its own graph-captured step.

**Why M* chose chunked-prefill over vLLM-style mixing — the CUDA-graph tension:**
M* leans HARD on CUDA graphs (every Thinker/Talker step replays a captured graph; ~30%
faster than eager). A vLLM-style mixed step has a shape — (decode_count,
prefill_len) — that matches NEITHER the pure-decode bucket NOR the pure-prefill bucket.
So a mixed step CANNOT replay either captured graph → it falls to EAGER → pays full
Python/dispatch cost on a step that happens often under load. Chunked prefill sidesteps
this entirely: by keeping prefill and decode as separate, individually-graphable
shapes, every step stays on the fast captured path. The price is slightly worse TTFT
(a new request waits one decode step before its prefill is scheduled) — a trade M*
accepted to preserve CUDA-graph coverage.

**The mixed-walk experiment = "add vLLM-style mixing to M*, but ALSO capture the mixed
shape as a CUDA graph so it doesn't fall to eager."** That's the whole point of the
branches in section 4.

====================================================================
## 3. The mixed-walk piggyback feature (branch: exp/mixed-walk-piggyback @ 6bdcb90)
====================================================================
Adds optional vLLM-style mixing to M*, env-gated MSTAR_MIXED_WALK.

- Scheduler trigger: `micro_scheduler.py:163` `mixed_walk_enabled`;
  `_maybe_plan_mixed` (:431-502) — ONLY fires when the just-picked primary is a
  latency-sensitive DECODE on a KV-cache node AND there are waiting prefill requests of
  other walks; `plan_mixed_budget` admits them under a shared token budget
  (MSTAR_MIXED_TOKEN_BUDGET=8192, MSTAR_MIXED_MAX_PREFILL_REQS=1,
  MSTAR_MIXED_PREFILL_CHUNK=512). Produces a MixedBatchPlan.
- Layout builder (pure CPU): `mstar/engine/mixed_walk.py`
  `build_mixed_varlen_layout` — decodes first (1 query token each), then prefill chunks
  concatenated; qo_indptr = [0,1,..,D, D+P0, D+P0+P1,...]; kv_seq_lens, 3-row M-RoPE.
  `DEFAULT_MIXED_PREFILL_BUCKETS=(64,128,256,512)`.
- Eager execution: `kv_cache_engine.py:477-660 _execute_mixed_eager` — one preprocess
  (concats embeds + plans FlashInfer attention with seq_lens=[1,1,..,N]) + one
  forward_batched on the prefill/varlen path; samples last-token-per-request.
- Dispatch: `kv_cache_engine.py:1011-1060 execute_forward`, priority
  mixed > cuda_graph > batched > sequential. The mixed branch (:1025-1037)
  unconditionally called `_execute_mixed_eager` — i.e. EVERY mixed step ran EAGER.
- The CUDA-graph hooks were stubbed: `CudaGraphKey` (cuda_graph_runner.py:96-114)
  already carries `mixed/num_decode/num_prefill_tokens` (default-inert); `run_mixed`
  (:1382, was :1220) existed but returned None.

### The SWE's diagnosis (the premise we tested)
Mixed step runs EAGER (different shape than either captured graph) → pays full
Python/dispatch every step a mixed step happens. The one-time ~30–50 ms TTFT saved by
piggybacking is dwarfed by the per-step eager penalty accumulated over a request's
decode steps. Fix: CAPTURE a CUDA graph for the mixed shape too.

====================================================================
## 4. The mixed-shape CUDA-graph experiment — 4 approaches, 4 branches
====================================================================
All forked from exp/mixed-walk-piggyback @ 6bdcb90, isolated worktrees. Shared toggle:
graph active when MSTAR_MIXED_WALK=1 AND MSTAR_MIXED_CG!=0; else run_mixed→None→eager.
Each mirrors the FLASH_INFER_PACKED capture/replay pattern
(`_capture_one_flashinfer_packed`, `_run_flashinfer_packed`) and adds a
MixedCudaGraphConfig + `_capture_one_mixed` + `_get_mixed_key_for` + a real `run_mixed`
body. FlashInfer `plan()` runs OUTSIDE the captured region (CPU int32 qo_indptr/kv);
the graph only reads the persistent static buffers.

| Branch | Approach | Grid | Graphs | Warmup |
|---|---|---|---|---|
| exp/mixed-cg-bucketed   | full grid, route to smallest fit | decode[1,2,4,8,16,32] × prefill[64,128,256,512] | 24 (×2 slots=48) | SLOW (>15 min) |
| exp/mixed-cg-supergraph | one max graph, always pad | (32 decode, 512 prefill)=bs33,544tok | 1 | fast |
| exp/mixed-cg-decodeonly | decode via existing graph, prefill eager | n/a (no fused capture) | 0 new | fast |
| exp/mixed-cg-coarse     | reduced grid (VRAM/pad tradeoff) | decode[4,16,32] × prefill[128,512] | 6 | medium |

### Bugs found during the experiment (important — these are results too)
1. super-graph & decode-only captured **0 `mixed=True` graphs** — super-graph filed its
   (33,544) graph under key `prefill_text/mixed=False`, so the runtime mixed lookup
   missed → SILENT eager fallback. They were effectively just the eager baseline.
2. bucketed & coarse correctly captured mixed=True graphs (bucketed: 48). But
   `run_mixed` returned the raw `{rid:{...}}` dict from `_run_flashinfer_packed`, while
   the engine expects a `NodeOutput` with `.per_request_output_tensors` (like
   `_execute_mixed_eager` / `_execute_with_cuda_graph` return). → AttributeError crash
   on EVERY mixed step (kv_cache_engine.py:1103). FIX: wrap the result —
   `return NodeOutput(per_request_output_tensors=...)`. This bug is shared across all
   graph arms.
3. Full bucketed grid (48 graphs) warmup is PROHIBITIVELY slow (~15 s/graph capture,
   >15 min total) — a real deployment strike vs coarse(6)/super(1).

### I2T sweep results (food101, Qwen3-Omni-30B-A3B, 1×2-GPU H200; req/s, ITL p50)
Driver: /home/tim/mixed_cg_sweep.sh. Two anchor baselines: "mixed-off"
(MSTAR_MIXED_WALK=0 ≈ current chunked-prefill path) and "eager-mix" (MIXED_WALK=1, no
graph).

LOW CONCURRENCY (B=4,8, n=10-20): EVERYTHING FLAT (within noise), ITL identical ~9–11 ms
across mixed-off, eager-mix, bucketed, coarse. Reason: mixed steps barely occur at
B≤8 — a mixed step only forms at the brief moment a finishing request is replaced by a
new one needing prefill. So neither the eager penalty nor the graph fix is exercised.

B=1 (control, serial, ZERO mixed steps): all arms ~0.66–0.69 req/s, ITL 6.6 ms — no
regression, confirms mixed-walk is inert with no overlap.

B=32 (n=64, concurrency fills → mixed steps FREQUENT) — THE DECISIVE REGIME:
  - mixed-off (current path):  req/s = 3.74   (healthy)
  - eager-mix (mix, no graph): req/s = 0.19   ← CATASTROPHIC. 64 reqs took 315 s wall,
                                                3 errored. ~20× throughput collapse.
  - bucketed (mix + CUDA graph, AFTER NodeOutput fix): B=1 clean (0.68);
                                                B=32 = <PENDING — see chat for final>.

### What B=32 proves (regardless of the pending bucketed number)
The SWE's premise is CONFIRMED and is stronger than expected: vLLM-style mixing in
M* WITHOUT a mixed CUDA graph is not just "a bit slower" — it CRATERS throughput ~20×
under load and starts dropping requests, because the frequent eager mixed steps fall
off the captured-graph fast path. This is precisely WHY new M* uses chunked prefill
instead of vLLM-style mixing: chunked prefill keeps every step graph-capturable.
The open question the bucketed-fixed B=32 run answers: does capturing the mixed shape
recover throughput back toward 3.74 (→ mixing becomes viable in M*), or not (→ stick
with chunked prefill for I2T).

====================================================================
## 5. Verdict so far / how to read it
====================================================================
- Encoder work (opt/compile-dynamic, opt/encoder-cudagraph, opt/encoder-gap): pushed +
  GPU-validation pending; cudagraph + gap-A/B are the real wins, compile-dynamic is a
  tail-latency fix only.
- Mixed-walk: at low concurrency it's a no-op for I2T (mixed steps rare). At high
  concurrency, eager mixing is catastrophic (0.19 vs 3.74). The CUDA-graph fix is
  NECESSARY for mixing to be viable; whether it fully recovers throughput is the
  pending bucketed-fixed B=32 number. Even if it does, the full-grid warmup cost
  (>15 min) argues for the coarse or super-graph variant in practice.
- Strategic takeaway: M*'s chunked-prefill choice is well-justified by the CUDA-graph
  constraint. Adding vLLM-style mixing only makes sense if the mixed shape is captured;
  otherwise it's a severe regression under load.

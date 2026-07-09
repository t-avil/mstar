# M\* Serving Engine / Runtime — Mechanisms Report (M\*-new vs M\*-old)

Read-only investigation to explain Qwen3-Omni benchmark behavior at **B=1 (launch/latency-bound)**
vs **B=32 (throughput/batching-bound)** across the 4 paths (S2T audio→text, I2T image→text,
I2S image→speech, S2S audio→speech).

Trees:
- **M\*-new** (integrated, native encoders + opts): `/home/tim/integration-wt/mstar/`
- **M\*-old** (upstream main, HF-wrapper): `/home/tim/mstar/mstar/`
- vLLM-Omni (contrast only): `/home/tim/baselines/vllm-omni/`

**Convention for diffs below:** `old→new`. A "diff hunk" like `111a112,186` means new ADDS lines
112–186 after old line 111. All file:line citations are in the **M\*-new** tree unless noted.

---

## 0. Headline finding: the engine is shared; the deltas are all encoder/preprocess/config

I diffed every engine/runtime file between the two trees. Result:

| File | old→new diff | nature of change |
|---|---|---|
| `worker/micro_scheduler.py` | **byte-identical** | — |
| `worker/node_manager_utils.py` | **byte-identical** | — |
| `graph/*` | **byte-identical** | — |
| `engine/cuda_graph_runner.py` | +0 −14 | **profiling removal only** (nvtx/perf_counter) |
| `worker/worker.py` | +1 −24 | **profiling removal only** (`enable_prof`, `WorkerProfileInfo`, `mstar.profile.*`) |
| `conductor/conductor.py` | +0 −34 | **profiling removal only** (`enable_prof`, perf_counter stamps, rx/tx_info) |
| `engine/base.py` | +3 −21 | profiling removal + minor `execute_batch(...)` inlining (no logic change) |
| `engine/stateless_engine.py` | +1 −9 | profiling removal only |
| `engine/kv_cache_engine.py` | +1 −20 | profiling removal **+ one functional delta** (see §4) |
| `worker/engine_manager.py` | +7 −15 | profiling removal; factory signatures drop `enable_prof` arg |
| `model/qwen3_omni/components/talker.py` | **byte-identical** | — |
| `model/qwen3_omni/components/code2wav.py` | **byte-identical** | — |
| `model/qwen3_omni/components/thinker.py` | **byte-identical** | — |
| `model/qwen3_omni/components/attention.py` | **byte-identical** | — |
| `model/qwen3_omni/submodules.py` | +190 −2 | **all additions in the encoder/preprocess region** (lines <700); Thinker/Talker/Code2Wav submodules (≥1000) unchanged |
| `model/qwen3_omni/qwen3_omni_model.py` | +577 −81 | native encoders, GPU mel/image preprocess, vllm-layout, sentinels |
| `model/qwen3_omni/config.py` | +29 −2 | native-encoder toggles + `codec_chunk_frames` default |
| `components/audio_encoder.py`, `vision_encoder.py` | **new files** | native varlen encoders |

**Verification commands** (run, not assumed): `diff -rq` over `worker/`, `engine/`, `conductor/`,
`graph/`, `model/qwen3_omni/`; per-file `diff | grep '^[<>]'` filtered for non-profiling lines; and
the diff hunk headers for `submodules.py` (all at lines 111/210/303/306/413/434 — the encoder
preprocess submodules, e.g. `NativeAudioEncoderSubmodule` added at `submodules.py:115`).

**Conclusion (answers task item #5):** the **continuous-batching scheduler, CUDA-graph
capture/replay machinery, plan/replay double-buffering, KV-cache engine, Talker AR loop, and
Code2Wav vocoder are the same code in both trees.** So *engine-level batch scaling is identical*
M\*-new vs M\*-old. The measured wins come from (a) **native varlen encoders** replacing HF dense
O(n²) attention (a batch story), (b) **GPU mel / image preprocess** (TTFT), and (c) one **config
default** (`codec_chunk_frames` 25→15, RTF/ITL at B=1). The native path is env-gateable back to the
HF path: `MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=0` / `…_VISION_ENCODER=0` (`config.py` reads these via
`_envflag`, default ON: `config.py:native_audio_encoder=True`, lines ~ the "Native encoder toggles"
block). With both off, M\*-new ≈ M\*-old at the encoder level — confirming the claim.

One caveat to the "engine identical" claim: a single functional engine delta exists in
`kv_cache_engine.py` — see §4. It is almost certainly a fork-point divergence, not an intentional
optimization, and I flag it honestly.

---

## 1. Continuous batching / micro-scheduler — the one-walk-per-step barrier

**Mechanism.** `worker/micro_scheduler.py` (identical in both trees). Each step:
`get_next_batch` (`micro_scheduler.py:191-307`) scans **every** worker-graph queue once, collects
all requests whose next node is ready at the engine level (`engine.check_ready`,
`micro_scheduler.py:252`), and groups them by node name into `node_name_to_requests`
(`:254-256`). Default scheduling policy is **ROUND_ROBIN** (`micro_scheduler.py:57`,
`SchedulingType.ROUND_ROBIN`).

- **ROUND_ROBIN selector** `_select_node_rr` (`micro_scheduler.py:107-123`): picks the single
  `(node_name, graph_walk)` pair with the **least-recent** `node_and_walk_to_last_batch_num` step
  (`:116-122`) — i.e. the stalest walk, for fairness across nodes/walks.
- **The barrier** (`micro_scheduler.py:272-277`): once a `(node, walk)` is chosen, entries are
  **filtered to exactly that graph_walk** (`entries = [e for e in … if e.graph_walk == graph_walk]`)
  and clipped to `max_batch_size` (`:276-277`). Everyone on a different walk **stays in the queue**
  for a later step. The PRIORITY selector enforces the same invariant explicitly
  (`_select_node_priority`, `:97-103`: *"Enforce same graph_walk for the entire batch … remaining
  requests stay in the queue"*).
- One walk is scheduled per step; `node_and_walk_to_last_batch_num` is stamped (`:298-300`) so RR
  rotates to other walks next step.

**Why this caps S2T/I2T (text-out) batching.** A request doing a **prefill** walk
(`prefill_text` / `prefill_audio` / `prefill_vision` — distinct sequential walks in
`qwen3_omni_model.py`, the Thinker has separate `PREFILL_*` graphs, §2) is on a *different*
graph_walk than a request in **decode**. The barrier forbids co-scheduling them in one step, so
**prefill cannot piggyback on a decode step.** At B=1 this is irrelevant (one request, one walk at
a time). As concurrency rises with **staggered arrivals**, decode-batch fill can fragment: a step
spent on one request's prefill walk is a step where the other requests' decode walk waits. This is
the structural reason text-out throughput can sag under non-lockstep traffic, and why
`LEVERS_REPORT.md` Lever 2 targets it.

**Why it still scales throughput.** Under the **closed-loop max-concurrency** workload actually
benchmarked (HANDOFF/FINDINGS protocol), requests advance in near-lockstep: most ready requests
share the **same decode walk**, so the single chosen walk's batch is large. RR then groups all of
them into one `forward_batched` call up to `max_batch_size` (`:276`). The decode step is a batched
CUDA-graph replay (§2), so a B=32 decode batch is one graph launch, not 32 — that is where the
~2–2.5× throughput at batch comes from. The barrier costs throughput mainly under arrival jitter or
mixed paths, not under the lockstep benchmark.

**Admission control** (`conductor/conductor.py`): `max_concurrent_requests` (`conductor.py:204-205`)
caps in-flight requests; `_try_admit_waiting` (`conductor.py:553-565`) drains the waiting queue up
to the cap; `_ingest_request` queues over-cap arrivals (`conductor.py:571-585`). If this cap is set
below 32 the B=32 decode/talker/code2wav graphs never fill — the cheapest thing to verify before
blaming the scheduler (LEVERS Lever 4).

**Effect by path / batch:**
- B=1: no effect (single walk). TTFT/ITL/RTF set by compute + launch overhead, not scheduling.
- B=32 S2T/I2T: throughput scales via batched decode replay **when lockstep**; barrier risks
  under-fill under jitter (prefill steps stealing decode steps).
- B=32 I2S/S2S: same, plus the Talker and Code2Wav are on a **separate worker** (§3), so a Thinker
  prefill walk does not stall audio generation — the barrier's downside is smaller on the audio
  paths.
- **new vs old: identical** (byte-identical file).

---

## 2. CUDA graph capture/replay + plan/replay double-buffering

**What is graph-captured, and at which batch sizes** (`model/qwen3_omni/submodules.py`,
encoder additions aside this region is unchanged old→new):

| Stage | graph_walk | capture_batch_sizes | cite |
|---|---|---|---|
| Thinker **decode** | decode | `[1,2,4,8,16,32]` | `submodules.py:1068` |
| Thinker **prefill (text/audio)** | prefill | `PREFILL_CAPTURE_BATCH_SIZES=[1,2,4]` | `:898`, used `:1078` |
| Thinker **prefill (vision)** | prefill_vision | `PREFILL_VISION_CAPTURE_BATCH_SIZES=[1]` | `:908`, used `:1112` |
| Talker **decode** | decode | `[1,2,4,8,16,32]` | `:1955` |
| Talker **prefill** | prefill | `TALKER_PREFILL_CAPTURE_BATCH_SIZES=[1,2,4]` | `:1896`, used `:1965` |
| Talker **last prefill** | (last-prefill) | `TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES=[1,2,4,8,16,32]` | `:1900`, used `:1979` |
| **Code2Wav** | `code2wav_chunk` | `[1,2,4,8,16,32]` | `:2044, :2053` |
| **Encoders (native)** | — | **not graph-captured** | `submodules.py:~118` comment: "no torch.compile / CUDA graphs … they don't uniformly help encoders" |

So **decode, Talker decode, and Code2Wav are graph-captured across the full B=1..32 bucket set**;
prefill (the TTFT path) is captured only at small batches `[1,2,4]` (text/audio) or `[1]` (vision);
encoders run eager. This matches FINDINGS §3: graphing the encoder live HURTS (graph key = clip
length → cache thrash), and prefill always replays a captured bucket (`_get_padded_num_tokens`
bisect-pads, 0 graph misses across 150 requests).

**Graph key** `CudaGraphKey(graph_walk, requires_cfg, bs, num_tokens)`
(`engine/cuda_graph_runner.py:96-101`). Lookup pads `bs`/`num_tokens` up to the nearest captured
bucket; capture iterates `for bs in reversed(sizes)` largest-first for memory reuse
(`cuda_graph_runner.py:240-248`).

**Double-buffering / speculative plan() (the overlap scheduler).** This is real and **identical in
both trees** (only nvtx/perf_counter lines were stripped):
- `NUM_SLOTS = MSTAR_NUM_SLOTS` default **2** (`cuda_graph_runner.py:152`). Each `CudaGraphKey` owns
  `NUM_SLOTS` captured graphs + persistent FlashInfer wrappers (`CudaGraphData.slots`,
  `:84-92`), so **plan(N+1) on slot[(s+1)%2] runs concurrent with replay(N) on slot[s]**
  (`:84-87, :145-151`).
- `reserve_slot` (`:901`) flips `next_slot` so the GPU thread and the plan thread agree on which
  slot an iter targets. `pre_plan_for_batch` (`:965`) runs FlashInfer `plan()` on a dedicated
  `_plan_stream` gated by the prev-batch `completion_event`/`advance_event`.
- Worker threading (`worker/worker.py:1985-2015`): a dedicated 1-worker **`gpu_executor`** runs the
  engine off the main loop; a dedicated **`plan_executor`** (default ON via `MSTAR_PRE_PLAN_SPEC=1`,
  `worker.py:2005-2010`) pre-runs `plan()` for the **speculatively-built next batch**
  (`_pre_plan_for_speculative_batch`, `worker.py:1094-1134`; speculative N+1 build at
  `worker.py:1320+`). The main thread's `await_gpu` releases the GIL so plan()'s Python work isn't
  contended.
- Spec-chain fairness: `has_ready_excluding` (`micro_scheduler.py:309-345`) lets a multi-walk worker
  break the speculative chain only when another `(node, walk)` is actually ready
  (`MSTAR_SPEC_PEEK_FOR_FAIRNESS=1`, `worker.py:2024`); single-walk workers (e.g. a dedicated
  Talker worker) speculate every iter.

**Effect by path / batch:**
- **B=1**: graphs make decode/Talker/Code2Wav **cheap and launch-overhead-robust** — the captured
  graph replaces hundreds of kernel launches with one replay. FINDINGS §3: B=1 RTF is ~98.6% Talker
  AR decode + Code2Wav, and decode is a captured graph (robust). The plan/replay overlap hides
  FlashInfer plan() Python cost behind the prior replay — important because the B=1 ITL win (0.007 s
  vs vLLM 0.012) comes from this.
- **B=32**: the bs-32 decode/Talker/Code2Wav graphs turn a 32-request step into **one batched
  replay**; ITL stays flat as batch grows (graph captured at exactly 32). This is the core
  throughput-scaling mechanism. Prefill at batch falls back to the `[1,2,4]` buckets +
  bisect-padding (no bs-32 prefill graph) — consistent with TTFT being the weaker axis.
- **new vs old: identical** — so decode-side ITL/throughput scaling is the same; any measured
  divergence at batch is from the encoder/prefill side, not the graph engine.

---

## 3. Talker AR loop + Code2Wav vocoder

Both component files (`components/talker.py`, `components/code2wav.py`) are **byte-identical** old→new.

**Talker autoregressive loop.** The Talker decodes one **audio frame** per AR step; each frame is
16 RVQ codes (`num_code_groups`/`num_quantizers=16`, `config.py:250,274`). Layer-0 code comes from
the main Talker transformer; the remaining 15 **residual** codes come from the **Code Predictor**, a
lightweight transformer run "15 autoregressive steps … in float32" (`talker.py:357-358`). Crucially
the depth loop is **unrolled into one CUDA graph**: `forward_depth_unrolled` is "the
CUDA-graph-compatible replacement … safe to call repeatedly inside a single `torch.cuda.graph`
capture" (`talker.py:446-540`), using SDPA + a static dense KV tensor and a Python-static
`lm_head_weight[i]` index (`talker.py:389-415`) instead of paged FlashInfer (the eager fallback).
So the entire 16-RVQ-depth generation of one frame for a **batch** of requests is **one graph
replay**, captured at `[1,2,4,8,16,32]` (`submodules.py:1955`); `TalkerSubmodule.MAX_BATCH_SIZE=32`
(`submodules.py:1400`) with a pre-allocated batched code-predictor KV cache
(`_get_cp_kv_cache`, `submodules.py:1434-1444`, shaped `[layers, MAX_BATCH_SIZE, 2, num_codes, …]`).

**Code2Wav vocoder.** A **non-AR, stateless** ConvNet vocoder (fp32, no autocast, no torch.compile —
`submodules.py:2033`). It runs **chunked + batched across requests**:
- `forward_batched` (`submodules.py:2126-2166`) stacks every ready request's codec tokens into one
  `[batch, num_quantizers, T]` tensor (`:2120-2124`) and calls `chunked_decode_streaming` once.
- `chunked_decode_streaming` (`code2wav.py:492-535`) runs **one** batched ConvNet forward
  (`wav = self(codes, position_ids)`, `:529`) then trims `left_context_size[i] * total_upsample`
  samples per request (`:531-534`) — per-request left-context, one shared forward.
- **Left-context chunking** (`LeftContextChunkPolicy`, described `submodules.py:2136-2151`): the
  first chunk for a request has 0 overlap frames; subsequent chunks carry
  `codec_left_context_frames` of overlap so the causal ConvNet warms up at chunk boundaries, and the
  corresponding samples are trimmed (they were already emitted). `_first_chunk_emitted`
  (`submodules.py:2015, 2147-2151`) tracks this per request.
- Code2Wav itself is **graph-captured** at `[1,2,4,8,16,32]` (`submodules.py:2044,2053`,
  graph_walk `code2wav_chunk`).

**Chunk-size config (the one talker/code2wav-adjacent delta — a config default, not engine code):**
`codec_chunk_frames=15`, `codec_left_context_frames=15` in **M\*-new** (`config.py:291-292`); M\*-old
default is 25/25. This is the `codec-chunk` win folded into the integration branch (FINDINGS §5,
keep chunk ≥ left_context).

**Why speech RTF stays low and scales:**
- **B=1**: Talker decode + Code2Wav are the whole cost (~98.6% of E2E, FINDINGS §3), but both are
  captured graphs and the depth loop is one graph → low ITL/RTF. Smaller chunk (15) lowers TTFA/ITL
  (FINDINGS §5: S2S RTF 0.167, I2S ITL −40%).
- **B=32**: the Talker decode graph and the Code2Wav graph both batch across requests into single
  replays (capture sizes include 32). **Cross-request Code2Wav batching is an M\* advantage** —
  SGLang's vocoder is batch=1, vLLM is per-call (LEVERS §"Already done"). The one tension at batch:
  the small 15-frame chunk means *more, smaller* vocoder invocations; the **open** throughput lever
  (LEVERS Lever 1) is batch-adaptive chunk size (large chunk once several requests co-vocode). Not
  yet implemented in either tree.
- **new vs old: identical Talker/Code2Wav code**; new differs only by the 15/15 chunk default
  (RTF/ITL at B=1, not a batch-scaling mechanism).

---

## 4. KV cache / memory engine — batch-scaling + the one engine delta

`engine/kv_cache_engine.py`, `engine/cache_manager.py`, `engine/kv_store.py`,
`engine/cpu_page_pool.py`. Paged KV with a `BatchedCacheManager`; CUDA-graph capture builds static
`cuda_graph_plan_states` and persistent FlashInfer wrappers re-planned per step
(`cuda_graph_runner.py:126-141`). Batch scaling: the same paged pages back a batched replay; the
admission cap (`max_concurrent_requests`, §1) bounds total KV residency. No prefix cache (KV is
request-scoped — `kv_store.py`, a no-op `flush_to_store()` hook per LEVERS Lever 6), so there is no
cross-request prefix-reuse batch effect; throughput scaling is purely from batched decode replay.

**The one functional engine delta (flag, honest):** M\*-**old** `KVCacheEngine.remove_request`
calls `submodule_mgmt.submodule.cleanup_request(request_id)` on every submodule
(`/home/tim/mstar/mstar/engine/kv_cache_engine.py:1068`, with a long comment: *"KVCacheEngine
omitted it, so any KV_CACHE submodule that tracks per-request state … leaked it … a submodule
managing a bounded resource pool (a fixed set of decode slots) drained the pool and then silently
reused one slot across concurrent requests → per-request state bleed"*). **M\*-new does NOT have
this call** — `grep cleanup_request` returns nothing in the new tree's `kv_cache_engine.py`.

- Interpretation: this is almost certainly a **fork-point divergence** — the integration worktree
  was branched from an older base (HANDOFF cites base `43ffffa`) before this cleanup fix landed on
  upstream `main`, rather than an intentional change. It is the *only* non-profiling engine diff.
- **Potential effect**: the comment ties the missing hook to a **bounded decode-slot pool** reused
  across concurrent requests — i.e. exactly a **high-batch / many-request** failure mode
  (per-request state bleed at B=32 or over many sequential requests), not a B=1 issue. The Talker's
  code-predictor KV (`_get_cp_kv_cache`, §3) and any per-request `set`/`dict` bookkeeping
  (`_eos_embed_sent`, `_first_chunk_emitted`) are candidates. **Uncertain**: I did not trace whether
  any KV_CACHE submodule in the new tree actually relies on `cleanup_request` for a bounded pool — if
  none does, the missing call is a harmless no-op (the comment notes it's a no-op for submodules that
  don't track the request). **Recommend verifying** before trusting M\*-new at sustained high
  concurrency, and before attributing any M\*-new-vs-old batch discrepancy to the encoder.

**Effect by batch:** B=1 unaffected. At B=32 / long sustained runs the missing cleanup is the one
place M\*-new could diverge from M\*-old for reasons unrelated to encoders — worth a targeted check.

---

## 5. M\*-new vs M\*-old engine deltas — verdict

**Claim ("engine is largely shared; wins are encoder/preprocess/vocoder") is VERIFIED**, with one
caveat:

1. **Scheduler / continuous batching**: byte-identical (`micro_scheduler.py`,
   `node_manager_utils.py`).
2. **CUDA-graph engine + plan/replay double-buffering**: identical except stripped profiling
   (`cuda_graph_runner.py`, `worker.py`, `base.py`, `stateless_engine.py`).
3. **Talker AR loop + Code2Wav vocoder + Thinker + attention**: byte-identical component files.
   Capture batch sizes identical. The only talker/code2wav-adjacent change is the
   `codec_chunk_frames` **config default** 25→15.
4. **KV / memory engine**: identical except profiling **and** the missing `cleanup_request` hook
   (§4) — flagged as a probable fork divergence with a possible high-concurrency effect.
5. **Real deltas are in the model layer**: native varlen audio/vision encoders
   (`components/audio_encoder.py`, `vision_encoder.py`, `NativeAudioEncoderSubmodule`
   `submodules.py:115`), GPU log-mel + GPU image preprocess, vllm-prompt-layout, audio sentinels —
   all in `qwen3_omni_model.py` (+577) / `submodules.py` (+190 additions) / `config.py`, **all
   env-gateable** (`MSTAR_QWEN3_NATIVE_AUDIO_ENCODER`, `…_VISION_ENCODER`,
   `MSTAR_GPU_IMAGE_PREPROCESS`, `MSTAR_VLLM_PROMPT_LAYOUT`).

**So engine-level batch scaling is the same in both trees.** Per FINDINGS §3, M\*-new and M\*-old are
structurally tied at B=1 (differ <1% of E2E = encoder); the native-encoder win is a **batch** story:
HF's dense O(n²) audio encoder degrades to ~2.0 RTF @ B=32 (S2S) while native varlen holds — and
that lives entirely in the encoder/preprocess layer, **not** in the shared engine.

---

## Uncertainties / things I could not fully verify (honest flags)

- **`cleanup_request` regression (§4)**: confirmed present-in-old / absent-in-new by grep, but I did
  not trace whether any active KV_CACHE submodule in M\*-new depends on it for a bounded pool. Could
  be harmless or could bite at sustained high concurrency. Verify before high-batch claims.
- **Same-walk barrier under real jitter (§1)**: the throughput cap is real in code, but its
  *magnitude* depends on arrival pattern; the benchmark protocol is closed-loop lockstep where it's
  largely hidden. Not measured here.
- I did not exhaustively diff `cache_manager.py` / `kv_store.py` line-by-line beyond confirming they
  are in the "identical / profiling-only" set via the directory `diff -rq` (they did not appear in
  the differing-files list, so they are byte-identical).
- vLLM contrast points are taken from `LEVERS_REPORT.md` citations, not re-verified against the vLLM
  tree in this pass (task scoped vLLM as "contrast only").

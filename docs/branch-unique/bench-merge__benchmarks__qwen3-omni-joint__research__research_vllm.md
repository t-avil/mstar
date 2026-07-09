# vLLM-Omni Qwen3-Omni serving — mechanisms behind the M\* vs vLLM benchmark gap

Read-only mechanism report. Goal: explain, with `file:line` citations, **where and why** vLLM-Omni
differs from M\* on the measured pattern:

- vLLM has **FLAT, low TTFT** across batch (S2T 0.14→0.28s, I2T 0.15→0.21s)
- vLLM has **LOWER throughput** than M\* at batch (S2T 305 vs 363+ tok/s; I2S 47.9 vs 94.7 audio s/s)
- vLLM has **higher ITL** than M\* at batch (≈0.05–0.06 vs M\* 0.036 at B32)
- vLLM has **higher RTF** on speech (≈2× M\* at batch)

All paths are under `/home/tim/baselines/vllm-omni/vllm_omni/` unless the path says `.venv/...`
(= upstream vLLM the fork builds on). M\* contrast paths are under `/home/tim/mstar/mstar/`.
Cross-checked against `LEVERS_REPORT.md` and `mstar/FINDINGS.md`. **Two LEVERS claims are corrected
below** (vLLM *does* batch the vocoder across requests; the bench config uses 25-frame streaming
chunks, not 300). Verified-vs-inferred is flagged per claim.

---

## 0. Architecture in one picture (this is the root of everything)

vLLM-Omni runs Qwen3-Omni as **three separate engines/workers**, each with its own scheduler and
its own GPU, wired by a `SharedMemoryConnector`:

- Pipeline topology (frozen): `model_executor/models/qwen3_omni/pipeline.py:18-66`
  - Stage 0 **Thinker** = `LLM_AR` (`pipeline.py:25`) — multimodal understanding + text decode
  - Stage 1 **Talker** = `LLM_AR` (`pipeline.py:41`) — text/hidden → RVQ layer-0 codes
  - Stage 2 **Code2Wav** = `LLM_GENERATION` (`pipeline.py:56`) — RVQ codes → waveform
- Deploy placement: Thinker on `cuda:0`; Talker+Code2Wav on `cuda:1`; connected by SHM;
  `async_chunk: true`, `codec_chunk_frames: 25`, `codec_left_context_frames: 25`
  — `deploy/qwen3_omni_moe.yaml:15-66`.

Two scheduler classes:
- `OmniARScheduler` (Thinker, Talker) — `core/sched/omni_ar_scheduler.py:47`; its `schedule()` calls
  `super().schedule()` = **upstream vLLM v1 continuous batching** (`omni_ar_scheduler.py:220`).
- `OmniGenerationScheduler` (Code2Wav) — `core/sched/omni_generation_scheduler.py:40`; a one-step
  "diffusion" fast path that schedules all available codec tokens at once
  (`omni_generation_scheduler.py:57-205`).

Contrast — M\* runs Thinker on one worker and **Talker+Code2Wav colocated on a second worker**, with
in-process non-blocking `StreamingGraphEdges`/`StreamBuffers` (no IPC between talker and vocoder)
(`mstar/FINDINGS.md:128-133`). This single architectural difference (separate engines + SHM IPC vs
colocated in-process buffers) is the spine of the RTF gap (§4).

---

## 1. TTFT — why vLLM stays FLAT across batch (vLLM's win, M\*'s weakness)

**Mechanism A — vLLM has no "prefill phase" vs "decode phase": one token budget mixes both in a
single step ("piggyback").**

- Upstream v1 scheduler, the load-bearing comment, verbatim
  (`.venv/.../vllm/v1/core/sched/scheduler.py:311-320`):
  > "There's no 'decoding phase' nor 'prefill phase' in the scheduler. … This is general enough to
  > cover chunked prefills, prefix caching, speculative decoding…"
- One pass schedules **running (decode) reqs first** (`scheduler.py:345-380`, loop at `:347`) then
  **waiting (prefill) reqs** under the *same remaining* `token_budget` (`scheduler.py:525-540`,
  loop at `:529`). A new request's prefill is admitted into whatever budget the decode batch left
  free, in the **same** step — it never has to wait for the decode batch to drain.
- Long prefills are **chunked** so they cannot monopolize a step:
  `num_new_tokens` is clipped to `long_prefill_token_threshold` (`scheduler.py:371-373`) and to the
  remaining `token_budget` (`:373`); the Thinker budget is `max_num_batched_tokens: 32768`,
  `max_num_seqs: 64` (`deploy/qwen3_omni_moe.yaml:25-26`).
- Encoder forwards are co-scheduled inside the same prefill budget with their own compute budget
  (`scheduler.py:381-397, 500-508`), so audio/image encoding is also chunk-bounded, not a
  serialized blocking call (§3).

**Effect on TTFT (flat across batch):** an arriving request's prefill is interleaved with the
running decode batch every step, so TTFT ≈ (encoder + prefill compute for *this* request) regardless
of how many requests are already decoding. It does **not** queue behind the decode batch → flat
0.14→0.28s (S2T), 0.15→0.21s (I2T).

**Contrast — M\* serializes by "graph_walk" and cannot mix prefill+decode.** M\*'s micro-scheduler
enforces one graph_walk per batch: it picks the most common walk and **leaves everyone else in the
queue** (`mstar/worker/micro_scheduler.py:97-103`; selection `:261-277`), and prefill walks
(`prefill_text/audio/vision`) are distinct sequential forwards with no chunking
(`mstar/model/qwen3_omni/qwen3_omni_model.py:206-301`). So under load a prefill walk cannot share a
step with decode-walk requests → TTFT rises with batch / staggered arrival. This is exactly LEVERS
Lever 2 + 3 (the "piggyback M\* lacks"), and it is the mechanism behind M\*'s TTFT being *behind*
vLLM (FINDINGS §1: M\* ~0.19 vs vLLM 0.118 at B=1; the gap widens at batch because M\*'s prefill
queues behind decode while vLLM's does not).

---

## 2. ITL / throughput at batch — why vLLM is SLOWER per token despite flat TTFT

This is the flip side of §1: the same mixing that flattens TTFT **taxes decode latency**.

**Mechanism B — the default CUDA-graph mode is `FULL_AND_PIECEWISE`; only *pure uniform-decode*
steps get a FULL graph. Any step that folds a prefill chunk into the decode batch is demoted to
PIECEWISE (attention runs eager).**

- Default mode is `FULL_AND_PIECEWISE` (field default `None` → v1 default):
  `.venv/.../vllm/config/compilation.py:581` (`cudagraph_mode = None`), docstring "v1 default"
  (`compilation.py:589`), set in optimization configs `.venv/.../vllm/config/vllm.py:230,252`.
  Semantics, verbatim (`compilation.py:604-606`):
  > "FULL_AND_PIECEWISE … Capture full cudagraph for decode batches and piecewise cudagraph for
  > prefill and mixed prefill-decode batches."
- A mixed step is **not** uniform-decode, so it can't match a FULL graph key:
  `_is_uniform_decode` requires `max_num_scheduled_tokens == uniform_decode_query_len` (==1) AND
  `num_tokens == max_num_scheduled_tokens * num_reqs`
  (`.venv/.../vllm/v1/worker/gpu_model_runner.py:3612-3616`). The Thinker runner computes this per
  step (`worker/gpu_ar_model_runner.py:502, 519-532`).
- Dispatch: FULL is returned only for the uniform-decode descriptor
  (`.venv/.../vllm/v1/cudagraph_dispatcher.py:301-310`); mixed batches register/resolve as PIECEWISE
  (`cudagraph_dispatcher.py:188-202, 312-317`).
- PIECEWISE keeps attention **outside** the graph, eager (`compilation.py:591-593`); and in the AR
  runner the padded/captured attention path is taken **only** when FULL:
  `pad_attn = cudagraph_mode == CUDAGraphMode.FULL` (`worker/gpu_ar_model_runner.py:544`).
- Decode attention backend = upstream FlashAttention; full-graph support is version-gated (FA3
  ALWAYS, FA2 UNIFORM_BATCH) but even FA3 stays PIECEWISE for mixed steps under the default mode
  (`.venv/.../vllm/v1/attention/backends/flash_attn.py:298-302`; `compilation.py:1319-1343`).

**Effect on ITL/throughput at B=32:** every step that piggybacks an incoming prefill chunk runs its
decode tokens with **eager attention + per-op launch overhead** instead of one captured FULL replay.
vLLM trades ITL for flat TTFT. Under sustained B=32 with continuous arrivals, mixed steps are common
→ ITL inflates to ~0.05–0.06 and Thinker token throughput sits at 305 tok/s. (Mechanism verified;
the *frequency* of mixed steps at B=32 is inferred from the large 32768 budget + closed-loop load.)

**Contrast — M\* replays one fixed-size FULL graph per decode step.** M\* captures decode graphs for
fixed buckets `[1,2,4,8,16,32]` and replays them with bisect-padding, with **0 cuda-graph misses
across 150 requests** (FINDINGS §3); its decode step is a single FULL replay (no eager attention,
no prefill folded in, because the same-walk barrier keeps decode steps pure). Result: M\* ITL 0.036
and 363+ tok/s at B=32. So M\* wins throughput/ITL precisely *because* it does **not** piggyback —
the same rigidity that hurts its TTFT (§1) protects its decode graph. The two systems sit on
opposite ends of the TTFT↔ITL trade.

(Secondary, neutral: the Thinker uses generic `FusedMoE` (`qwen3_omni_moe_thinker.py:84-85, 571-578`,
detection `qwen3_moe.py:152-161`); the python per-expert loop `Qwen3OmniMoeSparseMoeBlock`
(`qwen3_moe.py:28-127`) is **dead code**, referenced nowhere. MoE kernel choice is not the ITL gap;
but in a PIECEWISE step even the fused-MoE GEMMs lose the single-launch benefit of a FULL graph.)

---

## 3. Encoder + preprocessing — how chunked prefill hides the cost (reinforces §1)

- **Encoders run on GPU** inside the Thinker prefill: audio `audio_tower`
  (`qwen3_omni_moe_thinker.py:1051-1059`), vision `visual` (`:1170-1174`), executed via the upstream
  batched `_execute_mm_encoder` writing a shared `encoder_cache`
  (`.venv/.../vllm/v1/worker/gpu_model_runner.py:2775-2801`, cache `:511`). Encoder inputs are
  scheduled **within** the chunked-prefill step under a separate encoder compute budget
  (`.venv/.../vllm/v1/core/sched/scheduler.py:381-397, 500-508`).
- **Mel feature extraction and image resize/patchify run on CPU at the front-end** (the HF
  processor in the input-prep layer), *before* the request enters the engine core — so it is off the
  scheduler's critical step. (Verified that the stage input processors carry no mel/resize code —
  `stage_input_processors/qwen3_omni.py` handles only codec hand-off; preprocessing is upstream HF
  multimodal processing. The exact CPU cost was not measured here.)

**Effect:** because the encoder forward is chunk-bounded and co-scheduled with decode (Mechanism A),
a long audio/image prefill is split across steps and never balloons TTFT — this is why vLLM's TTFT
stays flat where **M\*-old/M\*-without-gpu-mel balloons** (M\*'s encoder runs inside a serialized
prefill walk, `qwen3_omni_model.py:1487-1532`, and HF-old's dense O(n²) audio attention degrades to
2.0 RTF @ B=32, FINDINGS §3). Note FINDINGS §3 also shows the encoder is <3% of E2E for M\*-native,
so this is primarily a TTFT-shape story, not a throughput story.

---

## 4. Speech RTF — why vLLM is ~2× M\* (the Talker + Code2Wav + IPC story)

RTF = wall / audio_seconds. Speech wall time is dominated by the Talker AR loop (one MoE forward per
audio frame) + the per-chunk vocoder + the **cross-engine hand-off**. Three mechanisms, in order of
impact:

**Mechanism C — Talker per-frame cost: 1 MoE transformer forward + 15 sequential residual
code-predictor forwards, the predictor using KV-cache-free re-prefill.**

- Layer-0 code per frame = full talker MoE forward (`qwen3_omni_moe_talker.py:259-267`,
  `compute_logits` via `codec_head` `:269-280`).
- 15 residual RVQ codes per frame via the code predictor AR loop:
  `model_executor/models/common/qwen3_code_predictor.py:806` `for step in range(1, num_groups):`
  (num_groups=16), each step a full small-transformer forward.
- The predictor **re-prefills the full short sequence each step (no KV cache)**:
  `qwen3_omni_moe_talker.py:133-139` ("re-prefills the full (short) sequence each AR step");
  `qwen3_code_predictor.py:114-118` ("No KV cache — always re-prefills").
- The predictor's **own** CUDA graphs are **disabled on GPU**:
  `qwen3_omni_moe_code_predictor_mtp.py:21-22` (`use_cuda_graphs = current_omni_platform.is_npu()`
  → False on GPU) — it relies on torch.compile only.
- Mitigation: the whole per-frame MTP (both loops) is wrapped as **one outer vLLM FULL graph**
  (`talker_mtp`) captured per batch bucket and replayed batched across requests
  (`worker/gpu_ar_model_runner.py:207-258`, capture with `CUDAGraphMode.FULL` `:252-258`; runtime
  `.venv/...`-style call in `worker/gpu_model_runner.py:1720-1726`). **Inferred caveat:** this
  helps only if FULL cudagraphs are active at runtime (stage 1 is non-eager per
  `deploy/qwen3_omni_moe.yaml:8-11`; FULL-vs-PIECEWISE selection not traced). If not FULL, the talker
  degrades to 15 launch-bound compiled forwards per frame.

**Effect on RTF:** the talker is structurally 16 forwards/frame; M\* unrolls the same 16-RVQ depth
loop into **one** CUDA graph (`mstar/model/qwen3_omni/components/talker.py:446-549`, LEVERS Lever 8).
At parity this is roughly even — so the talker is **not** by itself a 2× story (LEVERS Lever 8
confirms neither system does cross-time speculation; within-frame depth is graphed in both). The 2×
comes from C combined with D + E below.

**Mechanism D — Code2Wav batches across requests but with two efficiency leaks: zero-pad-to-max and
batch=1-only CUDA graph.**

- **vLLM DOES batch the vocoder across requests** (this *corrects* LEVERS Lever 1's "vLLM is
  per-call" / "M\* batches, vLLM doesn't"): multiple requests are packed into `[batch, 16, max_seq]`
  with **zero-padding to the longest request**, then sliced back per request via `seq_token_counts`:
  `qwen3_omni.py:413-421` (`codes = torch.zeros((batch_size,16,max_seq_len)…)`), slice at
  `qwen3_omni_code2wav.py:316-321`. The generation runner assembles the multi-request batch and
  passes per-request token counts (`worker/gpu_generation_model_runner.py:310-311`).
- **Leak 1 — padding waste:** the decoder runs over the full padded length for **every** request
  (`qwen3_omni_code2wav.py:307-310` decode whole tensor; non-async `chunked_decode` same at
  `:248-263`). Cost ∝ `batch_size × max_seq_len`, not `Σ seq_len` → one long request inflates every
  short request's vocoder cost. (Verified.)
- **Leak 2 — vocoder CUDA graph captures `capture_batch_sizes=[1]` by default**
  (`qwen3_omni_code2wav.py:154-158` → wrapper default `[1]` in
  `qwen3_tts/cuda_graph_decoder_wrapper.py`); at batch>1 the graph key misses and it falls back to
  **eager** decode. So at B=32 the vocoder is launch-bound. (Verified the default; runtime fallback
  inferred from the key.)
- **Config correction:** in the production deploy (`async_chunk: true`) the vocoder uses
  `chunked_decode_streaming` with **25-frame** streaming chunks (connector `codec_chunk_frames: 25`,
  `deploy/qwen3_omni_moe.yaml:15-21`; routing `qwen3_omni.py:567-573`), **not** `chunk_size=300`.
  The 300 in LEVERS Lever 1(b) is only the non-async `chunked_decode` path
  (`qwen3_omni.py:574-581`). So the real vLLM vocoder chunk in the bench is 25 — same as M\*'s
  default — and the LEVERS "vLLM=300" framing should be re-scoped to the non-streaming path.

**Mechanism E — cross-engine hand-off IPC: Talker→Code2Wav serialize codes through shared memory +
ZMQ per 25-frame chunk, even though both sit on `cuda:1`.**

- Codes are pulled GPU→CPU→python list in the stage processor
  (`stage_input_processors/qwen3_omni.py` `talker2code2wav_async_chunk`, `.cpu().…tolist()`),
  serialized and written to POSIX shared memory under an flock
  (`distributed/omni_connectors/connectors/shm_connector.py:53-63`, "we always use SHM" `:56-57`),
  then polled, deserialized (`shm_connector.py:88-92`;
  `distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py:183-194`) and re-uploaded
  to GPU (`qwen3_omni.py:554`).
- Granularity: **one transfer per 25 frames**, and each chunk **re-sends + re-decodes ~25
  left-context frames** that are then sliced off — `talker2code2wav_async_chunk` emit gate
  (chunk every 25 frames; left-context re-include), slice at `qwen3_omni_code2wav.py:319`. An
  N-frame utterance → ⌈N/25⌉ IPC round-trips + redundant context decode.
- Thinker→Talker is even finer: **one SHM payload per generated text token**
  (`thinker2talker_async_chunk`, one decode embedding per call).

**Effect on RTF:** M\* colocates Talker+Code2Wav in one process with in-process buffers — **zero**
IPC/serialization between codec generation and vocoding, and it batches code2wav across requests with
a working batched path (FINDINGS §5; LEVERS "already done"). vLLM pays, per utterance, ⌈N/25⌉
GPU→CPU→SHM→CPU→GPU round-trips (E) + redundant context decode (D leak) + eager batched vocoder
(D leak 2) + padding waste (D leak 1). Stacked on the per-frame talker cost (C), this is the ~2×
RTF at batch. The per-request startup (~0.2s encoder+prefill+handoff) also dominates short-audio
S2S RTF (FINDINGS §1 explains why I2S shows 1.8× but short S2S only ~1.1× at B=1).

---

## 5. Metric × path × batch summary (mechanism → effect)

| Metric | Path(s) | B=1 | B=32 | Dominant mechanism(s) |
|---|---|---|---|---|
| **TTFT** (flat, vLLM win) | S2T, I2T | vLLM 0.118 vs M\* ~0.19 (vLLM ahead) | vLLM flat 0.21–0.28; M\* balloons | **A** (mix prefill+decode, chunked prefill) + **§3** (encoder co-scheduled, CPU preprocess at front-end) vs M\* same-walk barrier |
| **ITL** (vLLM loses at batch) | S2T, I2T | M\* 0.007 vs vLLM 0.012 | M\* 0.036 vs vLLM 0.05–0.06 | **B** (mixed steps demoted FULL→PIECEWISE, eager attention) — the cost of A |
| **Throughput** (vLLM lower) | S2T 305 vs 363; I2S 47.9 vs 94.7 | M\* ahead | M\* ahead | **B** (decode not always FULL-graphed) + **D** (vocoder padding waste + batch=1 graph) + **E** (IPC) |
| **RTF** (vLLM ~2×) | S2S, I2S, T2S | I2S 1.8×, S2S ~1.1× (startup-masked) | ~2× | **C** (16 forwards/frame) + **D** (pad-to-max, eager batched vocoder) + **E** (per-chunk SHM IPC between co-located stages) |

**The single sentence:** vLLM optimizes for **flat TTFT** by mixing chunked prefill into every decode
step (A); M\* optimizes for **decode throughput/ITL/RTF** by serializing on graph_walks so its
decode/talker/vocoder always replay fixed-size FULL graphs and its talker+vocoder share one process
with no IPC. Each system's win is the direct cost of the other's design choice.

---

## 6. Corrections / cautions vs the existing reports

1. **LEVERS Lever 1(b) — "vLLM is per-call" / "vLLM default chunk 300":** vLLM **does** batch
   Code2Wav across requests (`qwen3_omni.py:413-421`). The bench (`async_chunk: true`) uses **25-frame
   streaming** chunks (`deploy/qwen3_omni_moe.yaml:15-21`, `qwen3_omni.py:567-573`), not 300. The
   real vLLM vocoder inefficiencies are **pad-to-max waste** and a **batch=1-only CUDA graph**
   (Mechanism D), not "lack of batching." M\*'s vocoder edge is the **no-IPC colocation** (E) +
   working batched-graph path, more than chunk size.
2. **LEVERS Lever 2 / 3 (piggyback, chunked prefill) — confirmed and located:** the mixing is
   upstream-inherited via `OmniARScheduler.schedule → super().schedule()`
   (`omni_ar_scheduler.py:220`) over `.venv/.../vllm/v1/core/sched/scheduler.py:310-540`. ✔
3. **Inferred, not measured here:** frequency of mixed prefill+decode steps at B=32 (Mechanism B
   magnitude); whether stage-1/2 select FULL vs PIECEWISE at runtime (Mechanism C mitigation);
   batch>1 eager fallback of the vocoder graph (Mechanism D leak 2, follows from default
   `capture_batch_sizes=[1]`); CPU preprocessing cost (§3). The *mechanisms/code paths* are all
   verified; their *quantitative weight* at B=32 is the inference.

---

### Key files (all under `/home/tim/baselines/vllm-omni/vllm_omni/`)
`core/sched/omni_ar_scheduler.py`, `core/sched/omni_generation_scheduler.py`,
`worker/gpu_ar_model_runner.py`, `worker/gpu_generation_model_runner.py`,
`model_executor/models/qwen3_omni/{pipeline.py, qwen3_omni.py, qwen3_omni_code2wav.py,
qwen3_omni_moe_talker.py, qwen3_omni_moe_thinker.py, qwen3_omni_moe_code_predictor_mtp.py}`,
`model_executor/models/common/qwen3_code_predictor.py`,
`model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py`,
`model_executor/stage_input_processors/qwen3_omni.py`,
`distributed/omni_connectors/{connectors/shm_connector.py, transfer_adapter/chunk_transfer_adapter.py}`,
`deploy/qwen3_omni_moe.yaml`; upstream `.venv/.../vllm/v1/core/sched/scheduler.py`,
`.venv/.../vllm/v1/worker/gpu_model_runner.py`, `.venv/.../vllm/v1/cudagraph_dispatcher.py`,
`.venv/.../vllm/config/compilation.py`.

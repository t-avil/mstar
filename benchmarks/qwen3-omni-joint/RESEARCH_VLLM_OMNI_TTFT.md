# How vLLM-Omni serves Qwen3-Omni i2t at flat ~170ms TTFT / ~1700-1814 tok/s (B16-32)

Research date: 2026-07-12. Author: research agent (static code reading only, no GPU runs).

## Scope & version caveat

- Base vLLM in the deploy venv is **v0.24.0** (`/m-coriander/coriander/tim/baselines/vllm-omni/.venv/lib/python3.12/site-packages/vllm`, `_version.py` -> `0.24.0+cu129`).
- vLLM-Omni repo is checked out at **tag v0.24.0** (`/m-coriander/coriander/tim/baselines/vllm-omni`). It also has tag `v0.22.0`.
- **0.22 base-scheduler line diffs are NOT reconstructable from this checkout.** The vllm-omni repo does not vendor the base `vllm/` tree; `git show v0.22.0:vllm/...` returns nothing. Only the `vllm_omni/` layer is diffable across tags. Where the omni layer is identical between 0.22 and 0.24 it is noted; the deploy yaml and stage/pipeline config are byte-identical for the parts that matter. Base-vLLM scheduler/model-runner refs below are v0.24.0-accurate; to get a true 0.22 base diff you need a base-vLLM checkout at both tags.

All file paths below are relative to either:
- `VENV = /m-coriander/coriander/tim/baselines/vllm-omni/.venv/lib/python3.12/site-packages/vllm` (base vLLM 0.24)
- `OMNI = /m-coriander/coriander/tim/baselines/vllm-omni/vllm_omni` (omni layer, tag v0.24.0)

---

## TL;DR (the mechanics to copy into M*)

1. **The Thinker is TP=1 on a single GPU (cuda:0) on H100 — there is NO all-reduce in the i2t forward.** The team-lead's TP=2 premise is wrong for CUDA; TP=2 appears only in the `platforms.npu` yaml section.
2. **i2t is stage-0-only.** Talker (stage 1) and Code2Wav (stage 2) are separate processes on the other GPU and receive zero data for a text request (`final_stage_id = 0`). No cross-stage dependency, no contention on cuda:0.
3. **One unified per-step token budget** (`max_num_batched_tokens = 32768`) mixes 31 decodes (1 token each) + 1 new prefill in the SAME forward step. A new request's first token comes out ~one scheduler step later (≈ one ITL), which is why TTFT stays flat rather than growing with queue depth.
4. **Vision tower runs eager, once per step, varlen-packing all scheduled images into one ViT forward**, on the same GPU/stream, strictly before the LLM forward. No encoder cudagraph for Qwen3-Omni.
5. **Async scheduling is on** for the single-GPU Thinker: the next step's batch is built while the current GPU forward/sample runs (no CPU-schedule bubble).
6. **Greedy (temp=0.0) batched argmax** for all 32 (skips softmax/top-k/top-p sort), **overlapped non-blocking D->H token copy** on a side stream, and **async per-step-batched detokenize** running in a *separate API process* from the GPU EngineCore.

The remaining TTFT cost is real and known: the co-scheduled image-encode+prefill lengthens the single step it lands in, briefly stalling the 31 concurrent decodes. vLLM keeps this flat (bounded to ~one step's worth) rather than eliminating it. This matches our committed M* root-cause note (one-walk-per-batch encoder+prefill freezes decodes).

---

## Q1 — Vision encoder scheduling at batch

**Batched into one varlen ViT forward, capped by a per-step encoder compute budget; overflow images defer to later steps.**

Scheduler accumulates encoder inputs across multiple requests within one step:
- `_try_schedule_encoder_inputs` called per running req (`VENV/v1/core/sched/scheduler.py:491`) and per waiting req (`:821`).
- A single `encoder_compute_budget` is threaded through the whole step and decremented across all requests: init `:415`, carry-forward `:608`/`:972`, decrement `:1429` (`encoder_compute_budget -= num_encoder_embeds`).
- Fitting images added to `scheduled_encoder_inputs[req_id]` (`:601-608`, `:965-972`).

Worker flattens all scheduled encoder items across all requests into one batched forward:
- `_batch_mm_inputs_from_scheduler` collects everything (`VENV/v1/worker/gpu_model_runner.py:2878-2888`).
- `_execute_mm_encoder` -> `group_and_batch_mm_kwargs` groups by modality and concatenates (`gpu_model_runner.py:3016`; concat in `VENV/multimodal/utils.py:188-233`).
- One `model.embed_multimodal(**mm_kwargs_batch)` per modality group runs the ViT once over all images (`gpu_model_runner.py:3086`).

**Encoder budget (max encoder batch):** two limits, both default to `max_num_batched_tokens`:
- `max_num_encoder_input_tokens = max_num_batched_tokens` (`VENV/config/scheduler.py:248`)
- `encoder_cache_size = max_num_batched_tokens` (`VENV/config/scheduler.py:249`)
- Combined in `compute_mm_encoder_budget` so at least one whole item always fits: `encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)` (`VENV/v1/core/encoder_cache_manager.py:309-316`).
- Effective per-step gate = `min(encoder_compute_budget, encoder_cache_size)` (`VENV/multimodal/encoder_budget.py` `get_encoder_budget`, ~line 183); scheduler reads it at `scheduler.py:218-219`.
- With the deploy's `max_num_batched_tokens=32768`, the encoder budget is 32768 embed-tokens/step. If images exceed it, `can_allocate` fails (`encoder_cache_manager.py:161`, `if num_embeds > encoder_compute_budget`) and the scheduler truncates that request's tokens to just before the unschedulable image (`scheduler.py:1383-1402`) — remaining images encode in later steps.

**Qwen3-Omni thinker packs all images into ONE varlen ViT forward:**
- `embed_multimodal` (`OMNI/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py:1295`) dispatches to `_process_image_input` (`:1309`).
- `_process_image_input` (inherited from `VENV/model_executor/models/qwen2_5_omni_thinker.py:970-986`) takes concatenated `pixel_values` + stacked `image_grid_thw`, calls `self.visual(pixel_values, grid_thw=grid_thw)` **once** (`:981`), then `.split(sizes)` back to per-image embeddings (`:984-986`).
- ViT builds `cu_seqlens` from stacked `grid_thw` for block-diagonal per-image attention — true varlen packing (`qwen3_omni_moe_thinker.py:171-178`, np variant `:204-207`, forward `:217-224`).

**Encode is NOT overlapped with decode.** Same GPU, same step, default stream, strictly before the LLM forward:
- `execute_model` -> `_preprocess` runs `_execute_mm_encoder(scheduler_output)` (`gpu_model_runner.py:3466`) immediately followed by `_gather_mm_embeddings` (`:3467`); the LLM forward runs after. Encoder output cached in `self.encoder_cache` (`:3094-3097`) and spliced into `inputs_embeds`.
- The only separate-stage encoder path is EPD/encoder-disaggregation via an ECConnector (early-return branch `gpu_model_runner.py:4100-4106`), which is OFF unless an ECConnector is configured. Default Qwen3-Omni i2t runs encoder+LLM in the same step.

**No encoder cudagraph for Qwen3-Omni.** Base 0.24 has an `EncoderCudaGraphManager` (`VENV/v1/worker/encoder_cudagraph.py`, wired `gpu_model_runner.py:6411-6444`), but:
- `compilation_config.cudagraph_mm_encoder = False` by default (`VENV/config/compilation.py:524`); manager not created unless set.
- Requires `SupportsEncoderCudaGraph`; only base `qwen3_vl.py:1668` implements it. The omni thinker (`qwen3_omni_moe_thinker.py:1078-1086`) does not. So the vision tower always runs eager; encoder batch size is dynamic per-step.

0.22 vs 0.24: omni thinker `embed_multimodal` + varlen `_process_image_input` unchanged (`git show v0.22.0:vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py:1288`); no encoder cudagraph in 0.22 thinker either.

---

## Q2 — Prefill admission (chunked prefill, budget, flat first-token latency)

**Which scheduler runs the Thinker:** stage 0 is `execution_type=LLM_AR`, `final_output_type="text"` (`OMNI/model_executor/models/qwen3_omni/pipeline.py:24-39`). `LLM_AR + async_scheduling` -> `OmniARAsyncScheduler` (`OMNI/config/stage_config.py:190-193`) = `OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler)` (`OMNI/core/sched/omni_ar_scheduler.py:921`). `OmniARScheduler.schedule()` does light bookkeeping then calls stock `Scheduler.schedule()` (`omni_ar_scheduler.py:235`); `_should_defer_waiting_admission()` returns False (`:137-138`), so waiting-queue admission is NOT deferred. **All budget logic is base vLLM `Scheduler.schedule()`.**

**Unified token budget mixes prefills+decodes in one step** (`VENV/v1/core/sched/scheduler.py:388`):
- No prefill/decode phase split (`:390-399`).
- One budget: `token_budget = self.max_num_scheduled_tokens` (`:408`) = `max_num_batched_tokens` = 32768 (config `:109-113`).
- Running reqs scheduled first (`:431-580`); a decode needs 1 token (`num_tokens_with_spec + num_output_placeholders - num_computed_tokens`, `:463-467`), clamped `min(num_new_tokens, token_budget)` (`:470`), budget decremented (`:578-579`).
- Waiting reqs scheduled next with remaining budget: `while (self.waiting or self.skipped_waiting) and token_budget > 0` (`:629`), decrement `:957-958`.
- Asserts: `total_num_scheduled_tokens <= max_num_scheduled_tokens` (`:990-991`), `token_budget >= 0` (`:993`).

**Chunked prefill ON by default:** `enable_chunked_prefill = True` (`VENV/config/scheduler.py:84`).

**Why first-token latency stays flat at B32:** A new request is admitted in the SAME step as the 31 decodes. In the waiting loop `num_new_tokens = request.num_tokens - num_computed_tokens` (`:796`), clamped to remaining budget (`:811`). 32 decodes use ~32 of 32768 tokens, leaving ~32736 — plenty for a full image+text prefill in the same forward. The request is appended to `self.running` (`:940`) and its tokens counted this step (`:957`). Its first output token therefore appears ~one scheduler step later (~one ITL), not after a queue drain. Admission gate is only: `len(self.running) < max_num_running_reqs` (`:630`), `token_budget > 0` (`:629`), KV blocks available (`allocate_slots`, `:874`).

**Caveat (matches our TTFT memory):** the co-scheduled prefill (image encode + prompt prefill) lengthens THAT step's forward, so the 31 concurrent decodes in that step are delayed by the prefill compute. The mechanism keeps first-token from waiting *multiple* steps; a fat prefill still stretches the single step it lands in.

**What caps a single prefill chunk per step** (`:796-811`): (1) `long_prefill_token_threshold` if set (`:797-799`, default 0 = off), (2) remaining `token_budget` (`:811`), (3) encoder budget/cache (`:815-830`), (4) KV blocks (`:874-895`, `break` if None). If chunked prefill were off, an over-budget prefill would `break`/defer the whole request instead of chunking (`:803-809`).

**Multiple simultaneous prefills:** admitted greedily FCFS, each `min(remaining_prompt, token_budget)`, until budget / `max_num_running_reqs` / KV blocks run out. Partial fits finish over later steps (chunked), tracked via `self._inflight_prefills` (`:962-963`). Note `max_num_partial_prefills` / `max_long_partial_prefills` are **V0-only, unreferenced under `vllm/v1/`** — inert here; V1 concurrency is bounded only by budget/running/KV.

**max_num_seqs=64 vs max_num_batched_tokens=32768 at B32:** neither hard-binds. At B32 only ~32 reqs run so `len(self.running) < 64` never trips; 32768 vs 32 decode-tokens leaves ~32736 so token_budget doesn't bind in steady decode. The real TTFT limiter is the co-scheduled encode+prefill stretching one step, not a scheduler cap.

**Qwen3-Omni image token count:** merged vision tokens = `(grid_h*grid_w)/(merge_size^2)`, `patch_size=14`, `spatial_merge_size=2` (`qwen3_omni_moe_thinker.py:568,688`; formula `qwen3_vl.py:941-945`) => tokens ~= pixels/784 after `smart_resize` clamps to `[min_pixels, max_pixels]`. A ~1024x1024 image ~= ~1.3k tokens; typical images low-hundreds to ~2k. Even ~2k image tokens << 32736 remaining budget, so an image prefill fits in one step.

**Async scheduling ON for the Thinker:** `async_scheduling` defaults None (`config/scheduler.py:158`), auto-resolved True for a generative, single-GPU (uniproc) executor (`VENV/config/vllm.py:964-1004`, `else: async_scheduling = True` at `:1003-1004`; `UniProcExecutor.supports_async_scheduling()` True at `VENV/v1/executor/uniproc_executor.py:145-146`). It builds the next step's batch while the current forward/sample runs, using output placeholders (`VENV/v1/core/sched/async_scheduler.py:19-49`, `num_output_placeholders += ...` `:39-41`; reconcile `:51-75`). Removes the CPU-schedule/GPU-forward bubble.

---

## Q3 — Where prefill runs vs decode; process/stage split; TP; all-reduce

**CRITICAL CORRECTION: on CUDA/H100 the Thinker is TP=1 on one GPU (cuda:0). No all-reduce in the i2t forward.**
- `StageDeployConfig.tensor_parallel_size: int | None = None` (`OMNI/config/stage_config.py:327`).
- `_build_engine_args` skips any field that is None (`OMNI/config/stage_config.py:791-794`), so an omitted `tensor_parallel_size` never enters engine_args.
- Base default then applies: `ParallelConfig.tensor_parallel_size: int = Field(default=1)` (`VENV/config/parallel.py:122`), surfaced at `VENV/engine/arg_utils.py:470`.
- The CUDA deploy yaml stage 0 sets only `devices: "0"`; `tensor_parallel_size: 2` appears ONLY in `platforms.npu` (and 4 in `platforms.xpu`). Platform overrides fire only when `current_omni_platform.device_name.lower()` matches (`stage_config.py:660-671`); H100 = `cuda`, no `platforms.cuda` block -> no override. **=> TP=1 on H100.**
- 0.22 identical: `git show v0.22.0:` shows stage 0 only `devices: "0"`, TP=2 only under npu, and `StageDeployConfig.tensor_parallel_size` default None (0.22 `stage_config.py:457`).

**Stages are separate OS processes, one GPU each, connected by POSIX shared memory:**
- Each stage/replica launched as its own subprocess via `StageEngineCoreProcManager` -> `context.Process(target=StageEngineCoreProc.run_stage_core, ...)` (`OMNI/engine/stage_engine_core_proc_manager.py:118-132`).
- GPU placement by narrowing `CUDA_VISIBLE_DEVICES` per stage from the yaml `devices:` before worker init (`OMNI/entrypoints/stage_utils.py:42-90`, `14-40`): stage 0 -> cuda:0, stages 1+2 -> cuda:1.
- Inter-stage transport = `SharedMemoryConnector`, local-only `/dev/shm` (producer `put()` writes a segment under flock or inlines small payloads; consumer `get()` by key) (`OMNI/distributed/omni_connectors/connectors/shm_connector.py:17-169`).
- Orchestrator forwards a stage's output to the next only when `stage_id < req_state.final_stage_id` (`OMNI/engine/orchestrator.py:931-936`, mirror guard `:1088`).

**i2t = stage 0 only; talker + code2wav idle:**
- Pipeline: stage 0 thinker `final_output=True, final_output_type="text"`; stage 2 code2wav `final_output_type="audio"`; stage 1 talker not a final output (`OMNI/model_executor/models/qwen3_omni/pipeline.py:25-67`).
- `final_stage_id` computed from requested output modalities (`get_final_stage_id_for_e2e`, `OMNI/entrypoints/utils.py:590-632`; called via `_compute_final_stage_id`, `OMNI/entrypoints/omni_base.py:334-339`). Text output -> `final_stage_id = 0`.
- Forward gate `0 < 0 = False` -> thinker output never forwarded; talker/code2wav get nothing. Async-chunk prewarm also skipped: `if req_state.final_stage_id <= 0: return` (`orchestrator.py:1471`).
- Thinker is fully decoupled: stage 0 `input_sources=()` (entry point), talker `input_sources=(0,)`, code2wav `input_sources=(1,)` (`pipeline.py:29-31,45,60`). Separate processes on separate GPUs => zero dependency for i2t.

**=> prefill and decode both run on the single cuda:0 Thinker process/GPU, in the same step, no TP all-reduce.** (If TP=2 on NPU: standard NCCL/HCCL all-reduce inside one engine process, no special overlap; NPU only adds `cudagraph_mode: PIECEWISE` for capture-safety, not comm overlap. Irrelevant to H100.)

---

## Q4 — tok/s mechanics (sampling, output length, detokenize, streaming)

**Greedy fast path (temp=0.0):**
- `greedy_sample` = `logits.argmax(dim=-1)`, no softmax/sort (`VENV/v1/sample/sampler.py:239-241`).
- `sample()` short-circuits when `all_greedy`: returns right after argmax (`sampler.py:261-271`), BEFORE `apply_temperature` (`:276`), logits procs (`:282`), `topk_topp_sampler` (`:286`). With all 32 reqs at temp 0.0, `all_greedy` is batch-wide => one `argmax` kernel over `[num_reqs, vocab]` for all 32.
- No logprobs requested => `compute_logprobs` (log_softmax) never runs (gather skipped, `sampler.py:96` still upcasts logits to fp32).

**Short outputs (food101 labels):** each request finishes in few decode steps, so token production is prefill/first-token weighted. Flat tok/s across B16->B32 is the signature of a saturated, well-batched decode step: per-step sampler/detok/output cost is amortized across all in-flight requests (one argmax, one batched D->H copy, one output message per step regardless of batch), so doubling concurrency ~doubles tokens/step at ~constant per-step overhead. (Reasoning, not a measured fact.)

**Detokenization: incremental, batched per step, in the API-server process (not the GPU process):**
- `IncrementalDetokenizer.from_new_request` picks Rust-backed `FastIncrementalDetokenizer` when available (`VENV/v1/engine/detokenizer.py:53-65,167-179`).
- Incremental `update()` appends only new tokens, `decode_next` per token, keeps running `output_text` (`detokenizer.py:95-142`) — no full re-decode.
- Runs in the AsyncLLM/API process via `OutputProcessor.process_outputs -> detokenizer.update(...)` (`VENV/v1/engine/output_processor.py:576,639-640`), driven by the async handler in `async_llm.py`. GPU forward+sample live in a separate `EngineCoreProc` subprocess (below). So detok overlaps the next GPU step, never blocks the forward.

**Process split + per-step batched output:**
- `AsyncLLM` holds `EngineCoreClient.make_async_mp_client(...)` (`VENV/v1/engine/async_llm.py:146`); `AsyncMPClient` = ZMQ + background EngineCore proc (`VENV/v1/engine/core_client.py:79`, imports `zmq`/`zmq.asyncio` `:20-21`). EngineCore built in a background process (`VENV/v1/engine/core.py:896`, `run_engine_core` `:1154`, output pump `process_output_sockets` `:1589`). => GPU forward/sample in `EngineCoreProc`; detok + streaming in `AsyncLLM`.
- Async loop pulls one `EngineCoreOutputs` batch per step, chunked by `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` (`async_llm.py:660-677`), default **128** (`VENV/envs.py:149,1299-1300`) >= B32, so all 32 reqs' new tokens process in a single `process_outputs` call per step (`output_processor.py:576-660`).

**What keeps tok/s high at B32:**
- One batched `argmax` for all 32 (skips softmax/top-k/top-p sort).
- Overlapped non-blocking D->H copy of sampled tokens on a dedicated `async_output_copy_stream` with an event; `.tolist()` deferred to `get_output` (`VENV/v1/worker/gpu_model_runner.py:242-277`, stream `:692-699`, tolist `:297-312`) — GPU proceeds to next step while previous tokens drain to host.
- Async, batched output processing off the GPU process; `await asyncio.sleep(0)` yields between chunks (`async_llm.py:681-683`).

**Qwen3-Omni-specific output handling (thinker text path):** omni subclasses base processor (`OMNI/engine/output_processor.py:10`, `MultimodalOutputProcessor`, built via `build_llm_stage_output_processor`, `stage_init_utils.py:814-834`). For stage-0 text it uses the base `IncrementalDetokenizer` (omni overrides only fire for generation stages where `detokenizer is None`, i.e. audio/codec, `output_processor.py:347-372`). One knob: `stream_interval` (`output_processor.py:296-312,468`) batches client emission when `>1`, but default is 1 and the stage-0 yaml doesn't set it, so text streams every step.

---

## Q5 — Qwen3-Omni-specific things that make batch TTFT flat

- **async_chunk: true** in the yaml, but its prewarm path is skipped for text (`orchestrator.py:1471`, `final_stage_id <= 0` return). async_chunk matters for audio/codec streaming (talker/code2wav), NOT for i2t TTFT.
- **enable_prefix_caching: false** on all stages (yaml). No prefix-cache lookup cost; also no cross-request prompt reuse — irrelevant to steady-state decode, and prevents prefix-cache variance in TTFT.
- **No encoder cudagraph, no encoder fixed batch sizes** for Qwen3-Omni (Q1): vision tower always eager, dynamic batch per step. So there is no "capture-size cliff" that would spike TTFT for uncaptured encoder batch sizes.
- **Thinker LLM cudagraph on by default** (`enforce_eager` unset => false; yaml header comment). Decode steps replay captured graphs; the encode+prefill step is the eager/uncaptured cost.
- **The flat-TTFT recipe = (a) TP=1 single-GPU thinker (no all-reduce), (b) unified-budget one-step co-scheduling of new prefill with 31 decodes so first-token = ~1 step, (c) async scheduling to remove the CPU bubble, (d) big 32768 token budget so a full image prefill always fits in one step.** The residual TTFT is one step's encode+prefill compute; vLLM bounds it to ~one step rather than eliminating it.

---

## Direct implications for M* (i2t)

1. **Do not chase TP=2 for parity — vLLM's thinker is TP=1.** Our own note already found TP=2 net-negative for M* (multi-process arch); this confirms vLLM isn't beating us via TP.
2. **The win is unified single-step admission of the new prefill alongside decodes, with async scheduling hiding the CPU schedule.** M*'s "one-walk-per-batch encoder+prefill freezes decodes" + 2s GPU-idle host-sync stall is exactly what vLLM avoids: vLLM's encode+prefill is one bounded step, and async scheduling means no host-sync bubble between steps.
3. **Encoder is eager and varlen-packed into one ViT forward** — no magic capture; the encoder cost itself is not hidden, just bounded to one step and not repeated per-request. M*'s ENCODER_ASYNC lever (off-stack, -30% TTFT) targets the right thing.
4. **Greedy batched argmax + overlapped D->H copy + async detok in a separate process** keep tok/s flat. Check M* isn't doing a blocking D->H sync per step for check_stop/postprocess (our decode profile: check_stop D->H 18%, _postprocess_batch 23%) — vLLM moves all of that off the GPU process.

# opt/encoder-gap — implementation & risk notes

Branch `opt/encoder-gap`, worktree `/home/tim/opt-preplan-wt`. NOT tested on GPU
(no device here). Everything below is correctness-by-construction with explicit
fallbacks. Read the "First GPU run watch-list" before enabling anything.

Files changed (see `git diff --stat`):
- `mstar/model/qwen3_omni/submodules.py` — Change B consumer + helper.
- `mstar/model/qwen3_omni/qwen3_omni_model.py` — Change B graph wiring.
- `mstar/worker/worker.py` — Change A producer gating + Change C trigger/handoff.
- `mstar/engine/cuda_graph_runner.py` — Change C packed pre-plan + plan-wait;
  Change A consumer wait.
- `mstar/engine/kv_cache_engine.py` — Change A event threading; Change C engine surface.
- `mstar/engine/base.py` — Change C base no-op surfaces.

---

## CHANGE B — audio_len precompute

**What I implemented**
- `ThinkerSubmodule._audio_len_from_inputs(inputs, audio_embeds)` (new staticmethod):
  computes the audio token count from `audio_seqlens` via
  `_feat_extract_output_lengths(...).sum()` (pure CPU integer arithmetic — no GPU
  output needed), else falls back to `audio_embeds.shape[0]`.
- `prepare_inputs` prefill_audio branch now calls that helper instead of
  `audio_len = audio_embeds.shape[0]`.
- Graph wiring: added `"audio_seqlens"` to the Thinker prefill_audio node's
  `input_names`, and emit an `audio_seqlens -> Thinker` edge in
  `_get_thinker_prefill_inputs` (unconditional, empty tensor_info when absent).
- The worker's Change-C trigger derives the same count directly from the encoder
  batch's `audio_seqlens` input (`_encoder_audio_len`) — that is the real
  consumer of the precompute (it runs before the encoder forward finishes).

**Change-B decision: NEW EDGE (route existing `audio_seqlens` to the Thinker node).**
- Why: traced the graph — `prefill_audio` is `Sequential([audio_encoder(inputs
  audio_features, audio_seqlens) -> Thinker(inputs audio_embeds)])`. `audio_seqlens`
  reaches the **encoder** node only; it does **not** reach the Thinker node. So the
  spec's "preferred: no new edge" branch does not apply.
- I routed the **existing** `audio_seqlens` CPU tensor to the Thinker node and reuse
  `_feat_extract_output_lengths`, rather than threading a brand-new `audio_out_lens`
  field. Reason: it mirrors the **already-working** vision pattern
  (`image_grid_thw -> Thinker`, qwen3_omni_model.py ~1050), reuses the canonical
  converter (guaranteed equal to `audio_embeds.shape[0]`), and touches far fewer
  sites than inventing/threading a new precomputed field through process_prompt +
  schedule + edges.

**Fallbacks**
- `_audio_len_from_inputs` returns `audio_embeds.shape[0]` whenever `audio_seqlens`
  is missing, an empty list, `[None]`, or the arithmetic raises. So a wiring gap or
  the test path (no AutoProcessor) degrades to exactly today's value.
- The `audio_seqlens -> Thinker` edge is emitted unconditionally with empty
  tensor_info when the source is absent (mirrors vision), so the node's ready
  condition is always satisfiable — never a hang.

**Unverified risk**
- That adding an input_name to the Thinker prefill_audio node does not change
  ready-signal timing in some multi-segment / vLLM-layout edge case. I confirmed
  the empty-tensor_info edge path (`tensors.py:get_ready_tensors` line ~624 passes
  empty edges through as ready) and that the consumer is empty-safe, so the failure
  mode is "falls back to shape[0]", not "hangs". Still worth eyeballing the first
  multi-audio-segment request on GPU.

---

## CHANGE A — encoder host-sync -> producer-gated GPU ordering

**What I implemented (worker.py `_postprocess_batch`)**
- New helper `Worker._is_producer_node(node_name)` = `node_name in
  {audio_encoder, vision_encoder}` (mirrors the existing encoder gate at
  worker.py ~1268).
- For producer nodes: do NOT call `output.completion_event.synchronize()`. Instead
  stash the event per-rid in `self._pending_producer_events`.
- For producer nodes: SKIP `_prematerialize_for_check_stop` (pass the raw GPU
  `output` to `check_stop_for_batch`).
- Non-producer / AR-decode nodes: **unchanged** (keep host sync + prematerialize).
- Consumer side: the worker attaches the stashed event to the next consumer batch's
  `metadata["producer_completion_event"]` (`_attach_producer_event`), the kv-cache
  engine forwards it to `runner.run(...)`, and `_run_flashinfer_packed` does
  `torch.cuda.default_stream(device).wait_event(producer_event)` immediately before
  the static-buffer copy + `graph.replay()`.

**Why this is GPU-correct without the host sync**
- `_execute_on_gpu_thread` already records `default_stream.wait_stream(side_stream)`
  at encoder dispatch (worker.py ~1302). That orders ALL later default-stream work
  (incl. the Thinker `graph.replay()`, which runs on the default stream) after the
  encoder, and also makes any later default-stream **buffer reuse** of the encoder's
  inputs safe. The host `.synchronize()` was redundant for ordering.
- The encoder output is routed via `store_and_populate_graph_edges(skip_cuda_sync=
  True)` (refs only), and encoder `check_stop` is the no-op default
  (`submodule_base.py:351` returns `set()` without reading tensors). So no CPU path
  reads encoder DATA in postprocess.

**RISK I was asked to verify — `_prematerialize` masking the win: CONFIRMED REAL.**
- `_prematerialize_for_check_stop` (worker.py ~1972) does `side.wait_event(event)`
  then `side.synchronize()` over `output.per_request_output_tensors`. Encoder nodes
  DO populate `per_request_output_tensors` with the (large) `audio_embeds`/vision
  embeds, so the unmodified path would D->H-copy them and host-sync — re-introducing
  exactly the encoder-forward block Change A removes. Hence I skip `_prematerialize`
  for producer nodes (safe because their `check_stop` is a no-op). **If this skip is
  ever removed, the Change A win is fully masked.**

**Fallbacks**
- `producer_completion_event` absent -> no `wait_event` -> ordering still provided by
  the existing `wait_stream`. Eager (non-graph) consumers run on the default stream,
  already ordered. AR-decode path never receives a producer event and its barrier is
  untouched (requirement 2).
- The consumer wait is in `_run_flashinfer_packed` ONLY, never `_run_basic_batched`.

**Unverified risk**
- The producer event wait is supplementary; if `wait_stream` ordering were somehow
  NOT in effect (e.g. a future refactor moves graph.replay off the default stream),
  the event wait would be the only guard and the cross-batch carry would matter more.
- Multi-rid encoder batches whose consumers split across batches: I keep one event
  (ordering is still covered by per-encoder `wait_stream`), so this is best-effort,
  not a correctness dependency.

---

## CHANGE C — pre_plan for FLASH_INFER_PACKED + encoder->prefill trigger

**Status: PLUMBING ALWAYS PRESENT; the live trigger is GATED OFF by default
(`MSTAR_PREFILL_PREPLAN`, default `0`).** With the flag off, behavior is byte-for-byte
today's (inline plan runs). This is deliberate: the packed pre-plan's page/slot
aliasing cannot be validated without a GPU, so it ships dormant.

**What I implemented**
1. `runner.pre_plan_for_batch` gains `num_tokens: int|None=None`. When supplied
   (packed), the key is resolved via `_get_key_for` (which pads num_tokens to the
   bucket) — the SAME lookup `run()` uses at replay (spec task 5). BASIC path
   unchanged (still `_get_basic_batched_key_for`).
2. Packed seq_lens: `[num_tokens]*real_bs + [0]*pad` (zero-length padded slots,
   mirroring `_run_flashinfer_packed`'s `zero_padding_input`). `dtype=None` (->
   kv_cache dtype, matching inline `preprocess`) and `is_causal=config.causal_attention`.
   So `qo_indptr` sums to the REAL token count, matching the inline plan that the
   replay-time `plan_attention` SKIP then records.
3. `_run_flashinfer_packed` now has the `_plan_done_event` wait block before replay
   (copied from `_run_basic_batched`); no-op when no pre-plan ran.
4. Engine surface: `reserve_packed_slot` + `pre_plan_packed_for` on `kv_cache_engine`
   (base no-ops in `base.py`). Worker trigger `_maybe_trigger_prefill_preplan`
   (bs==1 audio only) reserves a slot, submits the plan on `plan_executor`, and
   carries `(slot, future)` per rid. At prefill_audio dispatch
   `_consume_prefill_preplan` reuses the slot (skips `reserve_replay_slot`) and the
   GPU thread awaits the future before replay.
5. `_get_key_for` used for both pre-plan and replay (task 5).

**plan_rope: LEFT INLINE.** I did NOT add a `_pre_planned_labels` skip to `plan_rope`
(cache_manager has no such skip and adding one risks stale static pos-ids). So at
replay, `plan_attention` SKIPS (overlapped) but `plan_rope` still runs inline — this
**partially negates** the overlap win (RoPE pos-id copy is small vs the FlashInfer
plan, but non-zero). Noted per spec.

**Why the packed pre-plan should be correct when enabled (mirrors a proven path)**
- The existing BASIC pre-plan aliases real rids onto `static_cm.request_ids` (front)
  + dummy tail, plans, then restores. Both replay paths (`_run_basic_batched` line
  1299 and `_run_flashinfer_packed` line 1518) use the IDENTICAL dummy->real
  `request_states[dummy_rid][label] = real_state` swap. Since BASIC pre-plan +
  BASIC replay work in production with this real-rid-vs-dummy-rid split, packed
  pre-plan + packed replay should behave the same — the FlashInfer wrapper buffers
  depend on the request STATES (same object via aliasing), not the rid strings.

**Best-effort fallbacks**
- `reserve_packed_slot`/`pre_plan_packed_for` return `None`/`False` on any miss or
  exception (wrapped in try/except) -> caller skips, inline plan runs.
- `_maybe_trigger_prefill_preplan` is wrapped in try/except, handles bs==1 only,
  and only fires when `MSTAR_PREFILL_PREPLAN=1` AND `plan_executor` exists.
- If the trigger doesn't fire (or the consumer batch never carries a handoff),
  dispatch reserves its slot and plans inline exactly as today.

**Unverified risks (validate on GPU before trusting the flag)**
- **Packed slot/dummy-rid aliasing & page accounting** (#1): pre-plan allocs pages on
  the REAL rid; replay's swap aliases dummy->real and SKIPS the inline plan (so skips
  re-alloc). `_restore_dummy_states` frees DUMMY-rid pages. The real-rid pages must
  persist correctly and not double-alloc/leak. This is the same shape as the proven
  BASIC path but packed adds zero-length padding slots — unverified here.
- **Dropped-batch stale slot**: if a request is removed after its pre-plan ran but
  before the prefill_audio replay consumed it, the slot's `_pre_planned_labels` stays
  set until some later replay on that slot clears it (then that replay would wrongly
  SKIP its own plan). The packed key can't be re-derived by the BASIC-only
  `reset_pre_plan_state_for_slot`, so I only drain the future on remove (worker.py
  `_remove_request`). Low exposure at concurrency 1; risky if many A2T requests churn.
- **Single-worker `plan_executor` serialization**: the prefill pre-plan shares the
  one plan thread with spec-decode pre-plan (max_workers=1). Under mixed load they
  serialize; a slow prefill plan could delay a decode plan and vice-versa.
- **Bucket padding consistency**: pre-plan and replay both call `_get_key_for`, so
  the padded bucket matches; but the plan seq_lens use the REAL num_tokens. If a
  future change makes replay pad seq_lens (not just the graph), they'd diverge.
- **plan_rope inline** (above): masks part of the overlap.

---

## First GPU run watch-list

Enablement / bisection switches (each piece can be disabled independently):
- Change C trigger: `MSTAR_PREFILL_PREPLAN=0` (default) fully disables it.
- Change A: there is no env flag; to bisect, temporarily revert the producer
  branches in `_postprocess_batch` (restore `output.completion_event.synchronize()`
  and unconditional `_prematerialize`). The consumer `wait_event` self-disables when
  no event is stashed.
- Change B: set nothing — to neutralize, the consumer already falls back to
  `shape[0]`; to force the fallback, revert the `audio_seqlens` input_name + edge.

Symptoms to watch:
1. **Hang / deadlock right after an audio request starts prefill_audio** — suspect
   Change B wiring (Thinker node waiting on an `audio_seqlens` edge that was never
   emitted). Check: every prefill_audio step must emit the `audio_seqlens->Thinker`
   edge. Disable by reverting the input_name addition.
2. **Hang on the GPU thread `await_plan`** — suspect Change C: a pre-plan future that
   never completes (plan_executor wedged) or a slot mismatch. `MSTAR_PREFILL_PREPLAN=0`.
3. **Wrong / garbage audio transcription with pre-plan ON only** — suspect Change C
   plan/key or page mismatch (pre-plan planned a different bucket/seq_len than replay
   used, or page aliasing corruption). Turn the flag OFF; if it goes away, it's the
   packed pre-plan. Compare `_get_key_for(bs, audio_len+2, "prefill_audio")` at
   trigger vs `run()`'s `_get_key_for(bs, real_num_tokens, "prefill_audio")`.
4. **No latency win between encoder-end and prefill_audio replay** — suspect the
   `_prematerialize` skip was lost, or `wait_stream` not in effect: profile for a
   host sync / large D->H copy in postprocess on the encoder node. Confirm
   `_is_producer_node` returns True for `audio_encoder`.
5. **Win smaller than expected with pre-plan ON** — expected: `plan_rope` is still
   inline (documented). The `plan_attention` should show as
   `cache.plan_attention.skipped_pre_planned` in NVTX at replay.
6. **Page-pool exhaustion / leak over a long A2T run with the flag ON** — the dropped
   pre-plan slot / real-rid page accounting (risk #1/#2). Flag OFF.

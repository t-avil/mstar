# Resumable Chunked Thinker Prefill (MSTAR_CHUNKED_PREFILL)

## Goal

Reduce Speech-to-Text (S2T) and Image-to-Text (I2T) **TTFT at batch** by splitting
a long audio / vision / text Thinker prefill into token-budgeted chunks spread
across scheduler steps, so one long prefill does not monopolize a scheduler step
and stall the first tokens of other in-flight requests.

Today a Thinker prefill walk is **one forward over ALL tokens of that walk**
(`ThinkerSubmodule.prepare_inputs` / `forward` in
`mstar/model/qwen3_omni/submodules.py`). At batch, a 30s audio clip (~750 audio
tokens) or a large image (thousands of vision tokens) runs as a single step that
blocks every other request's decode/prefill for the duration of that forward.

## Flags (default OFF -> byte-identical to today)

- `MSTAR_CHUNKED_PREFILL` (bool, default OFF): master switch.
  `qwen3_omni_model.chunked_prefill_enabled()`.
- `MSTAR_LONG_PREFILL_TOKEN_THRESHOLD` (int, default `512`): chunk size = max new
  prefill tokens admitted per request per scheduler step.
  `qwen3_omni_model.long_prefill_token_threshold()`. Keep aligned with one of
  `ThinkerSubmodule.PREFILL_TOKEN_BUCKETS` so full chunks replay an existing CUDA
  graph capture.

Parity contract: with the flag OFF the single-shot path is untouched. With the
flag ON, a prefill whose span is `<= threshold` is processed as a single chunk
and is **byte-identical** to single-shot.

## The M-RoPE problem and the vLLM recipe (implemented)

Qwen3-Omni uses interleaved 3D M-RoPE. The per-token position is a `(3, seq_len)`
grid: text is linear in all 3 dims; audio ramps the temporal dim per frame and
pins height/width; vision uses a 2D spatial grid plus temporal. A naive
"recompute positions for this chunk" would be fragile at a chunk boundary that
lands mid audio-span or mid vision-grid.

We copy vLLM's recipe (`gpu_model_runner.py` `_calc_mrope_positions` /
`_init_mrope_positions`): **precompute the full 3D position tensor once**, then
each chunk just indexes `positions[:, computed : computed + chunk]`. No per-chunk
grid math; mid-span boundaries are exact by construction.

Implemented in:

- `mstar/model/qwen3_omni/components/rope.py`
  - `slice_mrope_positions(full_pos_ids, computed, chunk_len)` -- the per-chunk
    index (a view; bit-exact).
  - `prefill_mrope_pos_advance(full_pos_ids, start_pos)` -- the unified
    per-request 1-D advance `max(pos_ids) + 1 - start_pos`. This is vLLM's
    `position_delta` analog and **equals the existing single-shot advances**:
    `seq_len` for text/audio and the vision `mrope_pos_advance`
    (`end_pos_base + 1 - start_pos`). Applied ONCE after the final chunk so
    decode continues linearly regardless of how the span was chunked.
- `mstar/model/qwen3_omni/submodules.py`
  - `ThinkerSubmodule._maybe_chunk_prefill(graph_walk, fwd_info, node_inputs)`:
    when the flag is ON, slices the precomputed full-span `input_embeds`,
    `custom_pos_ids` (3D positions), and `masks_for_talker` to the window
    `[prefill_chunk_offset : prefill_chunk_offset + prefill_chunk_len]` read from
    `fwd_info.step_metadata`. The three prefill branches of `prepare_inputs`
    (text / audio / vision) build the FULL tensors exactly as today and return
    through this helper.

### Correctness argument (why chunked == single-shot)

1. **Positions**: `cat([slice(0,k), slice(k,S)], dim=1) == full` (exact). Each
   token gets the identical 3D position whether or not the span is chunked.
2. **RoPE**: cos/sin are computed position-wise (`compute_3d_cos_sin`), so the
   cos/sin for a chunk's tokens equal the single-shot cos/sin for those tokens
   exactly. Identical Q/K rotation.
3. **KV append**: `BatchedCacheManager._plan_attention_impl` already writes `sl`
   new tokens at offset `state.seq_len` and grows pages to `state.seq_len + sl`.
   Feeding chunks across steps appends KV at the right offsets; FlashInfer
   causal prefill makes chunk N's queries attend over all KV chunks `< N` wrote
   plus its own causal prefix -- standard chunked-prefill attention.
4. **Position advance**: summing per-chunk seq_lens reproduces the full advance;
   `prefill_mrope_pos_advance` gives the same final `position_id_start` the
   single-shot path lands on, so decode is unaffected.

(1)-(2) are pinned by `test/modular/test_qwen3_omni_chunked_prefill_parity.py`
on CPU; (3)-(4) are GPU-only.

## PROGRESS (2026-06 — chunked prefill IMPLEMENTED for text AND audio)

Status update: the conductor-driven re-enqueue is now implemented for BOTH
the **text** and **audio** prefill paths (text-output requests only;
`audio_output=False`).

Why the conductor and not the micro-scheduler: in this engine a prefill walk's
input edges are owned by the conductor. The scheduler only schedules nodes the
conductor has already emitted inputs for; it cannot independently "re-run" a
prefill node. So "run the same prefill node again next step" is realized by the
conductor re-emitting the SAME walk with an advanced offset until the span is
consumed. `MicroScheduler._maybe_reenqueue_prefill_remainder` is now a tolerant
no-op/sanity guard (no longer raises) and documents this.

### Text chunking (unchanged from prior commit)

- `qwen3_omni_model.plan_text_prefill_chunk(total, offset, threshold)` — pure,
  unit-tested chunk planner returning `(chunk_len, walk_done)` or `None`.
- `Qwen3OmniModel._text_chunk_bounds(...)` — gates chunking to:
  `prefill_text` walks, flag ON, span > threshold, AND `audio_output is False`.

### Audio chunking (NEW — encoder-split)

The blocker for audio chunking was that the conductor did not know the audio
token count until AFTER the `prefill_audio` Sequential (encoder + Thinker)
ran as one unit. The fix: **split the audio encoder into its own conductor
step**.

Graph walk change:

- **Before**: `prefill_audio = Sequential([audio_encoder, Thinker])` — one walk,
  encoder and Thinker run together, conductor has no control between them.
- **After**: Two separate walks:
  - `encode_audio = GraphNode(name="audio_encoder", ...)` — runs the encoder,
    persists `audio_embeds` with `dims[0]` = audio token count.
  - `prefill_audio = GraphNode(name="Thinker", ...)` — consumes
    `audio_embeds` from persist_signals; now chunkable.

The prefill schedule changed from `[("prefill_audio", audio_entry)]` to
`[("encode_audio", audio_entry), ("prefill_audio", {})]`. The `encode_audio`
step carries the encoder inputs (audio_features, audio_seqlens); the
`prefill_audio` step gets `audio_embeds` from persist_signals.

What was implemented for audio:

- `plan_audio_prefill_chunk(audio_token_count, offset, threshold)` — pure chunk
  planner accounting for the +2 sentinel tokens (audio_start + audio_end).
- `Qwen3OmniModel._audio_chunk_bounds(...)` — reads `audio_embeds.dims[0]` from
  persist_signals; same gating as text (flag ON, audio_output False).
- `Qwen3OmniModel._chunk_bounds(...)` — unified resolver, tries text then audio.
- `get_graph_walk_graphs()` — `encode_audio` as a standalone `GraphNode` with
  `persist=True` on the output; `prefill_audio` as a standalone `GraphNode`
  (no longer a Sequential).
- `get_partitions()` — `"encode_audio"` added to Thinker partition's `graph_walks`.
- `_build_thinker_prefill_schedule()` — `("prefill_audio", entry)` split into
  `("encode_audio", entry)` + `("prefill_audio", {})`.
- `_get_thinker_prefill_inputs()` — `encode_audio` targets `audio_encoder`;
  `prefill_audio` targets `Thinker` and reads `audio_embeds` from persist.
- `_get_thinker_forward()` — uses `_chunk_bounds` (text + audio) instead of
  `_text_chunk_bounds` only.
- `num_thinker_prefill_steps` — excludes `encode_audio` (no `thinker_states`).
- `_THINKER_PREFILL_WALKS` in micro_scheduler — includes `encode_audio`.

### Common conductor mechanics (text + audio)

- `Qwen3OmniModel._get_thinker_initial_args` / `_get_thinker_forward` —
  per-walk committed-token cursor `metadata.kwargs['prefill_chunk_offset']`; on a
  non-final chunk the SAME walk is re-emitted (schedule `prefill_step` NOT
  advanced) with `step_metadata['prefill_chunk_offset' / 'prefill_chunk_len']`;
  `is_last_prefill` is set only on the final chunk of the final walk; the input
  tensor is held alive across chunks (unpersist deferred to the final chunk).
- `ThinkerSubmodule._maybe_chunk_prefill` (already present) slices the precomputed
  full-span embeds + 3D M-RoPE + talker masks to the window. Per-chunk
  `advance_seq_lens(seq_len)` makes the summed position advance equal single-shot
  (text/audio). KV append is already resumable in the cache manager.

### What is deliberately NAIVE / still single-shot (documented limitations)

- **audio_output=True (Talker active) is NOT chunked.** The Talker consumes one
  `thinker_states` chunk per WALK; the submodule emits `thinker_states` on every
  prefill step, so chunking a walk would emit N per walk and drift the Talker's
  `num_thinker_prefill_steps`. Correct fix needs accumulating thinker_states
  across chunks and emitting once on the final chunk (encoder/state persistence).
- **prefill_vision is NOT chunked.** Vision additionally needs per-chunk
  `deepstack_<i>` slicing. `_maybe_chunk_prefill` still raises for a vision chunk
  window, so vision can never be silently chunked. The encoder-split pattern from
  audio could be replicated for vision once the deepstack slicing is resolved.
- **No cross-chunk encoder reuse needed for text** (text has no encoder). For
  audio, the encoder runs once in `encode_audio` and its output persists across
  all Thinker chunks — no encoder re-execution.

Net effect: long **text-output** prefills — both text spans (T2T, and the text
portion of S2T/I2T) AND audio spans (S2T where the audio Thinker prefill is the
stall source) — are streamed in <= `threshold`-token chunks; everything else
runs exactly as before. Flag OFF => byte-identical.

CPU tests:
- `test/modular/test_qwen3_omni_chunked_prefill_parity.py` (model-side M-RoPE).
- `test/modular/test_qwen3_omni_chunked_prefill_scheduler.py` (conductor
  re-enqueue loop — text + audio chunk boundaries, is_last_prefill only on
  final chunk, unpersist deferral, encoder-split walk sequence, audio_output
  bypass, flag-off bypass).

GPU A/B (flag ON vs OFF), S2T / long text-output prompt at batch:

```
/home/tim/launch_mstar_wt.sh /home/tim/exp/chunk-wt <gpus> <numa> <port> \
  chunk_sock <log> MSTAR_CHUNKED_PREFILL=1 MSTAR_LONG_PREFILL_TOKEN_THRESHOLD=512
```

(threshold 512 == a `ThinkerSubmodule.PREFILL_TOKEN_BUCKETS` entry, so full
chunks replay an existing CUDA-graph capture; the ragged final chunk uses the
smallest bucket >= its size, like a short single-shot prefill — no new captures.)

## What WAS STUBBED (historical — most now DONE)

### DONE (items 1-4 are now implemented)

1. **Per-request computed-tokens counter.** DONE — `prefill_chunk_offset` in
   `metadata.kwargs`, advanced per chunk by `_get_thinker_forward`. Reset to 0
   when the walk completes and the schedule advances.
2. **Cap + re-enqueue.** DONE — conductor-driven in `_get_thinker_forward`:
   `_chunk_bounds` computes the window, non-final chunks re-emit the same walk,
   only the final chunk sets `is_last_prefill=True`.
3. **Conductor coordination.** DONE — `_get_thinker_forward` does not advance
   `prefill_step` until the walk's span is fully consumed.
   `num_thinker_prefill_steps` excludes `encode_audio` walks.
4. **Encoder-output persistence (audio).** DONE — the audio encoder is split
   into `encode_audio` (standalone walk, `persist=True` on `audio_embeds`
   output). The conductor reads `dims[0]` from the persist signal to know the
   audio token count, then chunks `prefill_audio`. `audio_embeds` persists
   across chunk steps (unpersist deferred to final chunk).

### Remaining TODO

5. **Vision chunking.** `prefill_vision` needs the same encoder-split as audio
   PLUS per-chunk `deepstack_<i>` slicing. `_maybe_chunk_prefill` still raises
   for vision. The encoder-split pattern from audio can be replicated.
6. **`seen_token_mask`.** `add_tokens` must run once per token; today it runs on
   the full span in the text branch. Move it to the chunk window.
7. **audio_output=True (Talker active) chunking.** Needs accumulating
   `thinker_states` across chunks and emitting once on the final chunk.
8. **CUDA graphs.** A fixed chunk == one `PREFILL_TOKEN_BUCKETS` entry, so full
   chunks replay existing captures; the ragged final chunk uses the smallest
   bucket `>=` its size, exactly like a short single-shot prefill today. No new
   captures required.

- `MicroScheduler._maybe_reenqueue_prefill_remainder(...)` — still a tolerant
  no-op/sanity guard. Actual re-enqueue is conductor-driven.
- `ThinkerSubmodule._maybe_chunk_prefill` raises for `prefill_vision` with a
  real chunk window (vision needs deepstack slicing).

## Validation (GPU)

Parity test (chunked vs single-shot):

```bash
MSTAR_QWEN3_OMNI_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct \
  pytest -q test/modular/test_qwen3_omni_chunked_prefill_parity.py
```

The CPU property tests run anywhere; the full engine-level KV/first-token-logit
parity test is skipped until the re-enqueue stub above is implemented.

TTFT-at-batch benchmark (S2T), flag ON vs OFF, p50 at B = 8, 16, 32 -- see the
returned GPU command in the task summary.

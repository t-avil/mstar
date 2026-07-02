# W5 Phase 1 — Chunked Thinker Prefill (V2)

Flag: `MSTAR_CHUNKED_PREFILL_V2=1` (default **OFF**). Flag-off is byte-identical.

## Problem

Today one Thinker admission runs its **entire** prefill as a single forward step
(~27.5 ms graph for a large i2t vision block) between decode steps. That single
step stalls every in-flight decoder (~8.7 ms/step amortized at B32) for the full
prefill duration. Target: i2t B32 0.77x -> ~0.85-0.90x vs vLLM by splitting the
prefill into chunks of <=C tokens, run as separate normal prefill steps that the
worker round-robin interleaves with decode steps.

This is P1: still uniform-walk batches (a chunk step is a pure prefill step, a
decode step is a pure decode step; they alternate). **No mixed forward** — that is
P2.

## Mechanism (conductor-driven cursor)

Chunking is driven entirely at the **conductor** (`qwen3_omni_model.py`), reusing
the existing per-walk conductor<->worker round-trip. Verified control flow:

- `conductor.py:1088` main loop drains `WORKER_GRAPHS_DONE` messages per request,
  each independently, then calls `_process_done_forward` (`:1127`).
- `_process_done_forward` (`:906`) calls `get_partition_forward_pass_args` ->
  `_get_thinker_forward` (`:1066`), which advances the prefill schedule and emits
  the next walk's inputs. Each walk is one conductor round-trip.
- The worker `MicroScheduler._select_node_rr` (`micro_scheduler.py:107`) groups
  ready nodes by (node, walk) and schedules the least-recently-run group. A
  request's pending prefill **chunk** (a `prefill_text`/vision Thinker walk) and
  another request's `thinker_decode` are different walk groups, so RR naturally
  alternates them. This is the interleaving: no scheduler change needed for P1.

To chunk a walk, the conductor **re-emits the SAME walk** with an advanced
`prefill_chunk_offset` in `metadata.kwargs` instead of advancing the schedule.
Only once the walk span is fully consumed (`walk_done`) does it advance to the
next schedule entry. The submodule slices the staged embeds / pos_ids /
deepstack by `[offset:offset+chunk_len]`.

`mark_node_complete` -> `Loop.complete_iter` is NOT involved for prefill walks
(prefill walks are plain `GraphNode`s, not the decode `Loop`); the re-emit is a
fresh conductor-issued forward, so the E9 concern does not apply to P1's
mechanism. (The brief's worker-side `PrefillProgress`/`complete_iter` re-queue is
an alternative mechanism; the conductor cursor is simpler, matches the verified
control flow, and reuses the proven June prior art
`exp/chunked-prefill` af836ee.)

## Chunk size C

`ThinkerSubmodule.PREFILL_TOKEN_BUCKETS = [128,256,512,1024,2048]`. Pick the
largest bucket <= remaining span, min 128. `MSTAR_PREFILL_CHUNK_TOKENS` (default
512) caps C. A walk whose span <= C is not chunked (single full-span step,
byte-identical to flag-off).

## is_last_prefill / sampling

`is_last_prefill` gates logits emission (`submodules.py:912`), new_token return
(`:1382`), and thinker_states emission. With chunking it must be true ONLY on the
**last chunk of the last prefill walk**. Non-last chunks run forward + append KV
and emit nothing. `hidden[-1:]` on the final chunk is the last real token of the
full prefix (KV holds the whole prefix), so the sampled first token is identical
to the unchunked run.

## KV append / causality (verified)

- `advance_seq_lens` (`cache_manager.py:820`) bumps `state.seq_len += C` after
  each chunk step. The next chunk's `plan_attention` computes
  `total_len = state.seq_len + C` (`:290`), so the KV append offset is the
  already-appended prefix. FlashInfer prefill with `qo_indptr`=C,
  `kv_indptr`=prefix+C and `is_causal=True` gives bottom-right causal alignment:
  a chunk (q=C, kv=prefix+C) attends causally over the whole prefix. Correct.

## MRoPE position advance (verified)

`advance_seq_lens` advances `position_id_start` by `seq_len` by default, or by the
per-request `custom_pos_advance` side-channel (`set_custom_pos_advance`,
`cache_manager.py:786`) when set. For **text/audio** chunks the MRoPE span == token
count, so the default `+= C` per chunk is correct and sums to the full span across
chunks. The submodule computes each chunk's `pos_ids` from `start_pos`
(= `position_id_start`) which has been advanced by the previous chunks, so
positions are contiguous and the post-prefill decode `position_id_start` equals
the unchunked value bit-exactly. No custom advance needed for text/audio.

Audio has the same property (its wrapped span's positions are contiguous from
`start_pos`), so audio chunks also use the default `+= C`.

For **vision** the MRoPE 3D-grid span exceeds the token count, so the last vision
chunk must apply the full remaining `mrope_pos_advance` jump via
`set_custom_pos_advance`; non-last chunks advance by their `chunk_len`. (See the
vision implementation spec below.)

## Scope: what landed vs. what is specced for GPU implementation

**Landed (commit `feat(qwen3_omni): chunked Thinker TEXT prefill`)**: text-prefill
chunking, fully wired, tested (CPU scheduler test), flag-off byte-identical.

**Specced but NOT implemented — vision chunking (the i2t B32 win)**: For i2t the
text prompt is tiny (~7 tokens, "Please describe this image in detail") and the
prefill is dominated by the image's vision tokens (258+; up to thousands for
video). So the ~27.5ms mega-step that stalls decoders is the **vision** Thinker
prefill, and hitting the 0.77x->0.85-0.90x target requires chunking it — text
chunking alone does not move i2t. This piece needs the encoder-split + staged
window slicing + per-chunk MRoPE advance, all of which are silent-wrong-on-error
and require GPU validation to land safely; it is specced below, not written, and
gated `MSTAR_CHUNKED_PREFILL_V2_VISION` (default OFF within V2) when built.

### Vision implementation spec (GPU-required)

1. **Graph split** (`_get_thinker_graph_walks`, gate on `chunked_prefill_v2_vision
   _enabled()`): replace `prefill_vision = Sequential([vision_encoder, Thinker])`
   with two walks — `encode_vision` (the `vision_encoder` GraphNode, persisting
   `vision_embeds` + `deepstack` to `EMPTY_DESTINATION`) and a Thinker-only
   `prefill_vision`. Register `encode_vision` in the Thinker partition
   `graph_walks`. Mirror audio's June split exactly (af836ee).

2. **Schedule** (`_build_thinker_prefill_schedule`): for image/video modalities
   emit `("encode_vision", {pixel_values, image_grid_thw, ...})` then
   `("prefill_vision", {})`. `num_thinker_prefill_steps` (Talker accounting) must
   exclude `encode_vision` (encoder-only, no thinker_states) — but note chunking is
   gated `audio_output=False` so the Talker is absent anyway.

3. **Chunk bounds** (`_vision_chunk_bounds`, add to `_chunk_bounds`): read the
   vision token count from the persisted `vision_embeds.dims[0]`; span = count + 2
   (sentinels). Same gating as text (`chunked_prefill_v2_vision_enabled()` +
   `audio_output=False` + walk == `prefill_vision`).

4. **Submodule staging** (`ThinkerSubmodule`): on chunk 0 (offset == 0) of a
   `prefill_vision` walk, compute the FULL wrap once — exactly today's
   submodules.py:674-754 block: `wrapped_embeds` (total_len, hidden) via
   `_wrap_vision_input`, `pos_ids` (3, total_len) via `get_rope_index_vision` from
   the ORIGINAL `start_pos`, `mm_mask` (sentinels at [0, -1] = 0), and per-layer
   `full_deepstack[mm_mask] = deepstack_i` — and stash them in
   `self._vision_stage[rid] = PrefillProgress(wrapped_embeds, pos_ids,
   deepstack_list, mm_mask, total_len, mrope_pos_advance, start_pos)`. Add
   `cleanup_request(rid)` (pop the stage) — wire it the same way Code2Wav's
   `cleanup_request` is (submodules.py:2111) so aborted/finished rids don't leak
   the large staged GPU tensors.

5. **Per-chunk slice**: each chunk (including chunk 0) returns
   `ARNodeInputs(input_seq_len=C, input_embeds=stage.wrapped_embeds[off:off+C],
   custom_pos_ids=stage.pos_ids[:, off:off+C], tensor_inputs={deepstack_i:
   stage.deepstack[i][off:off+C], masks_for_talker: ...})`. The forward uses the
   STAGED pos_ids (never recomputed from the advanced start_pos), so intra-block
   3D positions stay exact.

6. **Per-chunk MRoPE advance** (the subtle part). Unchunked vision advances
   `position_id_start` by `mrope_pos_advance` (the full 3D span, > total_len) while
   `seq_len += total_len`. Chunked: `seq_len += C` per chunk (KV append, automatic
   via plan_attention). For `position_id_start` to end at
   `original_start_pos + mrope_pos_advance` exactly:
     - non-last chunks: `set_custom_pos_advance([C])` (advance by the chunk len).
       Value is irrelevant to correctness of THIS or later chunks' forwards
       because those use staged pos_ids, but keeping it = C keeps seq_len and
       position loosely in step and simplifies the assert.
     - last chunk: `set_custom_pos_advance([mrope_pos_advance - sum(prior C)])` so
       the running `position_id_start` lands on exactly the unchunked post-vision
       value. The submodule already emits `mrope_pos_advance` via
       `preprocess`/`set_custom_pos_advance`; per-chunk it emits the adjusted
       value.
   `MSTAR_CHUNKED_PREFILL_V2_ASSERT`: after the last vision chunk, assert
   `state.position_id_start == original_start_pos + mrope_pos_advance` and
   `state.seq_len == pre_walk_seq_len + total_len`.

7. **Batch-vision assert** (submodules.py:799): a chunk is still one request's
   vision slice, so the single-request-per-step assert holds per chunk; no change
   needed unless combining with `MSTAR_BATCH_VISION_PREFILL` (out of P1 scope).

- **Audio prefill**: same encoder-split shape as vision but simpler (no
  deepstack, MRoPE span == token count so uniform `+= C` advance like text).
  Deferred behind the same secondary gate; the June af836ee audio split is the
  reference.
- **Talker coupling**: chunking is gated to `audio_output=False` (the i2t/i2t-B32
  target). When the Talker is conditioned (`audio_output=True`, i2s/a2s), each
  Thinker prefill walk emits exactly one `thinker_states` chunk that the Talker
  counts (`num_thinker_prefill_steps`); chunking a walk would emit N of them and
  break that accounting. P1 leaves audio-output paths unchunked. Documented.

## Validation hooks

- `MSTAR_CHUNKED_PREFILL_ASSERT=1`: after the last chunk of a walk, assert the
  conductor's running token/position total equals the unchunked span for that
  walk (cheap, conductor-side, no GPU state read).
- DEBUG log on the first chunk per rid per walk: `chunks planned=N, C=<c>,
  span=<s>`.

## Flag-off invariant

Every code path checks `chunked_prefill_v2_enabled()` (and, for vision/audio, the
secondary gate) before diverging. Flag-off: no schedule split, no chunk metadata,
submodule sees no `prefill_chunk_*` keys and runs the full span exactly as today.

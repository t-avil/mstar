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

For **vision** the MRoPE 3D-grid span exceeds the token count, so the last vision
chunk must apply the full remaining `mrope_pos_advance` jump via
`set_custom_pos_advance`; non-last chunks advance by exactly their `chunk_len`.
(See vision section.)

## Scope decision: text (+ audio) core; vision as separate staged commit

- **Text prefill**: chunked directly. The `text_inputs` tensor is staged (kept
  persisted across chunks); the submodule slices `text_ids[offset:offset+C]`,
  builds `pos_ids` from the advanced `start_pos`. Lowest risk; primary correctness
  vehicle.
- **Vision prefill (i2t's dominant cost)**: requires the June encoder-split —
  `vision_encoder` becomes its own conductor walk that persists `vision_embeds` +
  `deepstack`; the Thinker walk that consumes them is then chunkable. The Thinker
  wrap (`_wrap_vision_input`, +2 sentinels), `get_rope_index_vision` pos_ids, and
  per-layer `full_deepstack[mm_mask]` are computed ONCE on chunk 0 and staged;
  each chunk slices the staged `embeds`, `pos_ids`, and `deepstack` by the same
  window. The deepstack/mm_mask window alignment is the fiddliest part; it is in
  its own commit and gated by `MSTAR_CHUNKED_PREFILL_V2_VISION` (default OFF within
  V2) so text chunking can ship even if vision needs GPU validation.
- **Audio prefill**: same encoder-split shape as vision but simpler (no
  deepstack). Deferred behind the same secondary gate; documented.
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

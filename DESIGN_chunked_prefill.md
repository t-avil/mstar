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

## What is STUBBED: scheduler / conductor re-enqueue

The piece that cannot be validated without a GPU is the cross-step re-enqueue of
a prefill remainder. It is guarded behind the flag and raises a clear
`NotImplementedError` if ever actually triggered:

- `mstar/worker/micro_scheduler.py`
  `MicroScheduler._maybe_reenqueue_prefill_remainder(...)` -- detailed TODO; the
  guard fires only when a request has been conductor-marked
  (`requires_prefill_chunking`), which nothing sets yet, so it is dormant.
- `ThinkerSubmodule._maybe_chunk_prefill` raises for `prefill_vision` with a
  real chunk window (vision needs deepstack slicing + encoder persistence).

### TODO to finish (in priority order)

1. **Per-request computed-tokens counter.** Track new prefill tokens admitted so
   far per `(request_id, prefill walk)`. The total span length is known to the
   conductor (text token count; audio/vision token count after the encoder node),
   not the scheduler -- publish it on
   `CurrentForwardPassInfo.step_metadata['prefill_total_len']`.
2. **Cap + re-enqueue** (`MicroScheduler.get_next_batch`). When
   `computed + threshold < total`: set `step_metadata['prefill_chunk_offset']`
   and `['prefill_chunk_len'] = min(threshold, total - computed)` for this step,
   advance the counter, and re-push the SAME prefill node onto the request's
   per-request queue so it is ready next cycle. Only the FINAL chunk sets
   `is_last_prefill=True` (logits sampled once -- `ThinkerSubmodule.forward`
   ~L868; conductor `_get_thinker_forward` ~L1098).
3. **Conductor coordination** (`qwen3_omni_model.py`). `_get_thinker_forward`
   must not advance `prefill_step` until the current walk's span is fully
   consumed. Keep the Talker's `num_thinker_prefill_steps` counting one streamed
   `thinker_states` chunk per WALK, not per token-chunk (simplest: stream
   `thinker_states` only on each walk's final token-chunk), or the Talker
   last-prefill detection (`_get_talker_forward`) drifts.
4. **Encoder-output persistence (audio/vision).** The encoder node runs once; its
   output (`audio_embeds` / `vision_embeds` + `deepstack`) must persist across
   the walk's chunk steps. `prefill_vision` additionally needs per-chunk
   `deepstack_<i>` slicing -- until then `_maybe_chunk_prefill` raises for vision.
5. **`seen_token_mask`.** `add_tokens` must run once per token; today it runs on
   the full span in the text branch. Move it to the chunk window.
6. **CUDA graphs.** A fixed chunk == one `PREFILL_TOKEN_BUCKETS` entry, so full
   chunks replay existing captures; the ragged final chunk uses the smallest
   bucket `>=` its size, exactly like a short single-shot prefill today. No new
   captures required.

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

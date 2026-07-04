# Merged multimodal prefill walk (`MSTAR_MERGED_PREFILL`)

Old plan rows B1/B5: "merge `prefill_text` + `prefill_vision` into one walk —
drop the conductor round-trip." This documents the design chosen, why, and the
invariants that were the hard part. CODE ONLY; GPU A/B is owned by main.

## The problem (measured)

An i2t admission runs `prefill_text` AND `prefill_vision` as SEPARATE graph
walks. Each walk boundary is a conductor round-trip: the worker emits
`WORKER_GRAPHS_DONE` (ZMQ) → the conductor's `_process_done_forward` calls
`get_partition_forward_pass_args` to pick the next walk → the conductor sends the
next walk's `InputSignals` back (ZMQ) → the worker runs it
(`conductor.py:918-990`). For i2t that is one full round-trip between the text
span and the vision span, on the critical TTFT path (i2t TTFT 0.43s vs vLLM
0.16s at B32; i2t B2/B4 0.82x/0.88x).

## Why the two walks are separate today

1. **Node structure.** `prefill_text` is a single `Thinker` `GraphNode`;
   `prefill_vision` is a `Sequential[vision_encoder → Thinker]` — the encoder
   must run first to produce `vision_embeds` + `deepstack`
   (`qwen3_omni_model.py:756`, `:815`).
2. **Post-preprocess tensor signature.** `prefill_text`'s packed signature is
   `input_embeds + cos_3d + sin_3d`. `prefill_vision` adds per-layer
   `deepstack_<i>` statics and flows a per-request `mrope_pos_advance` through
   the `_PlanState` side-channel (`submodules.py:1079-1110`,
   `get_cuda_graph_configs:1493-1521`). Different signatures ⇒ different captures.
3. **Position advance.** Text advances `position_id_start` by token count; vision
   advances by the 3D-grid span (`_build_vision_full.mrope_pos_advance`).

## Key enabling fact

M* deliberately STRIPS modality placeholders from `text_inputs`
(`process_prompt`, `qwen3_omni_model.py:2092-2115`): the KV cache is built as
SEQUENTIAL, cleanly-separated spans (`[text][<vstart>vision<vend>]` or
`[<vstart>vision<vend>][text]`), NOT an interleaved placeholder splice like HF.
So a merged prefill needs no placeholder-position scatter — it is exactly the
CONCATENATION of the per-span embeddings the separate walks already compute.

For image input, `text_inputs` is a SINGLE full span and the image is ONE
separate block (`process_prompt` sets `text_inputs=[input_ids]`; the vLLM
prefix/suffix split fires only for AUDIO). So i2t = exactly one `prefill_text` +
one `prefill_vision`, in modality order.

## Design chosen: Option A — a merged walk that reuses the vision capture

A `prefill_multimodal` walk (`Sequential[vision_encoder → Thinker]`), registered
only under `MSTAR_MERGED_PREFILL`. The Thinker node additionally declares
`text_inputs`. Its `prepare_inputs` builds ONE `ARNodeInputs` spanning the whole
prompt by REUSING the existing per-span computations and concatenating them,
threading the MRoPE start position across spans exactly as the separate walks
thread it via `advance_seq_lens`:

- vision span: `_build_vision_full(inputs, cur, device)` (verbatim) → wrapped
  embeds, 3D pos_ids, per-layer deepstack, mm_mask, `mrope_pos_advance`.
- text span: `embed_tokens` + `get_rope_index_text(seq_len, cur)` + talker masks
  (verbatim from the `prefill_text` branch), advancing `cur` by `seq_len`.
- concat in modality order: `input_embeds` (dim 0), `custom_pos_ids` (dim 1),
  per-layer `deepstack_<i>` (dim 0, text rows zero-filled), `masks_for_talker`
  (dim 1). One row, `input_seq_len = total`, single `mrope_pos_advance =
  final_pos − start_pos`.

The post-preprocess signature is then IDENTICAL to `prefill_vision`
(`input_embeds + cos_3d + sin_3d + deepstack_<i>`, `mrope_pos_advance`
side-channel), so `prefill_multimodal` REUSES the `prefill_vision` capture
(`replay_graph_walks=["prefill_vision","prefill_multimodal"]`). No new capture,
no extra warmup.

### Why this is numerically equivalent (modulo ULP)

Causal attention over the concatenated `[A][B]` span gives each span attention to
itself + everything before it — exactly what the sequential walks produce (span B
attends to A's already-resident KV). The per-span embeds/pos/deepstack are
bit-identical (same helpers, same threaded `start_pos`). The KV cache ends
identical. The only difference is ONE forward vs N, so kernel tiling / FlashInfer
split-KV reduction order differs → last-bit drift, the same property accepted for
chunked prefill (`EXPERIMENTS.md` W5-P1) and vLLM chunked prefill.

### Why NOT Option B (walk chaining)

The conductor↔worker protocol returns control after every walk; the state
machine that picks the next walk (`get_partition_forward_pass_args`) lives in the
conductor, and KV/position state lives on the worker. "Continue to the next walk
without a round-trip" would require either moving the model state machine onto the
worker or a new multi-walk dispatch/continuation protocol — a far larger and
riskier change to the engine than Option A, which lands entirely inside the
existing single-walk execution model (a walk is already allowed to be a
`Sequential[encoder → Thinker]`, and the Thinker forward already supports the
union signature from W5-P3). Option A also removes the round-trip outright rather
than hiding it, and reuses validated machinery. Rejected B.

## Scope / eligibility (byte-identical when any fails)

Merge fires only when ALL hold, else the unmerged multi-walk schedule is used
unchanged:
- `MSTAR_MERGED_PREFILL` on (default OFF).
- Output is text-only (no audio output) — keeps the Talker's
  `num_thinker_prefill_steps` accounting and `thinker_states` streaming on the
  untouched path; i2t (the measured regression) is text-out.
- `MSTAR_CHUNKED_PREFILL_V2_VISION` OFF (that flag swaps `prefill_vision` for the
  `encode_vision`+chunk pair; incompatible span model).
- The schedule is EXACTLY one `prefill_text` + one `prefill_vision` (either
  order). Multiple images, audio-interleaved, or text-only prompts fall back.

## The hard invariants

1. **MRoPE position continuity across spans.** The merged `prepare_inputs`
   threads `cur` through the spans in modality order and reproduces each span's
   `start_pos` exactly as the separate walks would see it after `advance_seq_lens`
   (text advance = `seq_len`; vision advance = `mrope_pos_advance`). The single
   returned `mrope_pos_advance` must land `position_id_start` on the same final
   value the two walks reach: `text_len + vision_mrope_advance`.
2. **Deepstack alignment.** Deepstack is additive-spliced at the post-attention
   layers indexed by `deepstack_visual_indexes`. The merged per-layer tensor is
   `(total, hidden)`, nonzero only on the vision sub-span (text rows zero-filled),
   so the splice hits the same absolute rows as the standalone vision forward.
   Reuses the W5-P3 zero-rows pattern.
3. **Talker masks.** `masks_for_talker` is concatenated per span in order so the
   multimodal/text partition streamed to the Talker is identical (belt-and-braces:
   merge is text-out, so no Talker, but kept correct).
4. **seen_token_mask.** The text sub-span still calls
   `seen_token_mask.add_tokens(text_ids)` exactly as `prefill_text` does.

## Telemetry

`MSTAR_WALK_STATS` already counts executed `(node, graph_walk)`; a
`prefill_multimodal` execution shows up automatically. A dedicated
`merged_prefill_walks` counter is bumped in the worker for the explicit fold
count.

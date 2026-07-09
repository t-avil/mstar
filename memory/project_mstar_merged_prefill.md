---
name: project_mstar_merged_prefill
description: "M* merged multimodal prefill walk (MSTAR_MERGED_PREFILL, plan B1/B5) — branch, design, key gotcha"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

Merged multimodal prefill for M* Qwen3-Omni (old plan rows B1/B5): collapse an
i2t admission's separate `prefill_text` + `prefill_vision` walks into ONE
`prefill_multimodal` walk (Sequential[vision_encoder→Thinker]) that runs both
spans in one Thinker forward, dropping the conductor round-trip. Targets i2t
TTFT/JCT at B1-B4. Branch `opt/prefill-merge` (commit 41200ec, on the fork),
worktree `/m-coriander/coriander/tim/mstar-pmerge`. Flag `MSTAR_MERGED_PREFILL`
(default OFF, byte-identical off). Built CODE-ONLY 2026-07-04; GPU A/B pending.

Design: Option A — the merged walk REUSES the `prefill_vision` CUDA-graph capture
(identical post-preprocess signature: input_embeds+cos+sin+deepstack_<i>,
mrope_pos_advance side-channel). Enabled by the fact that M* strips modality
placeholders (process_prompt), so KV spans are cleanly separated and the merge is
just concatenation of the per-span embeds, no HF-style placeholder scatter. Hard
invariant = MRoPE position threading across spans (text advance=seq_len, vision
advance=3D-grid mrope_pos_advance). Rejected Option B (walk chaining): the
conductor owns the walk-picking state machine, worker owns KV/position state — a
much bigger protocol change.

B4 CONTRADICTION RESOLVED (2026-07-04, custom-ops base, 4,5 same-pair): merge
"lost" i2t B4 in req/s (2.39 vs shipping 2.66) but that was a LENGTH ARTIFACT
(Law 5 within M*, not cross-system). Token-throughput was a TIE (456 vs 459);
shipping's B4 cells just drew ~11% shorter food101 requests (172 vs 191 tok/req,
n=12 sample variance across the two separate servers). On the fair metric
(text_token_throughput) the merge config is >= shipping at EVERY batch size:
B1 +4.3%, B2 +3.0%, B4 tie(+0.7%), B32 +11.2% — no crossover, no awkward B2/B4
boundary. Recommend the small-batch merge config across B1-B4 (+B32). Ruled out
chunked-vision pipelining (would show in tok/s; didn't). PROCESS: two-server
config A/Bs must gate on tok/req parity or use token-throughput/JCT, never raw
req/s — independent request samples fake ~11% req/s deltas at n=12.

KEY GOTCHA: it is a vision-STRATEGY SWAP, not additive. The merge only fires when
`MSTAR_CHUNKED_PREFILL_V2_VISION` is OFF (that flag rewrites vision into
encode_vision+chunk, a 3-entry schedule the merge won't match). So it is an
ALTERNATIVE to the MIXED_BATCH_VISION fold path, not stacked on it — A/B both arms
run vision as separate walks, toggle the flag; process-static (two servers).
Scope: single-image i2t, text output; everything else falls back byte-identical.
See DESIGN_merged_prefill.md + SMOKE.md on the branch. Related:
[[project_mstar_decode_bottleneck]] [[project_mstar_mixed_batch_p2]].

AUDIO TWIN (built 2026-07-04, branch `opt/prefill-merge-audio` commit b7a4333,
worktree `/m-coriander/coriander/tim/mstar-audiomerge`, pushed to fork). Flag
`MSTAR_MERGED_PREFILL_AUDIO` (default OFF, INDEPENDENT of the vision flag).
Collapses one-text + one-audio into `prefill_multimodal_audio`
(Sequential[audio_encoder→Thinker]). Targets s2t B2/B4. KEY DIFFERENCE from
vision: audio has NO deepstack and NO 3D-grid MRoPE jump — its positions
increment +1/token so the walk's MRoPE advance == its seq_len (verified via
get_rope_index_audio). So the merged signature == prefill_text
(input_embeds+cos+sin+masks_for_talker, no side-channel) and it replays on the
`prefill_text` capture, NOT the vision capture; default advance_seq_lens is
exact. Also, unlike vision (bs=1 assert), merged-audio rides the packed
prefill_text capture so it CAN batch >1 request — a plus at B2/B4. NO
chunked-audio walk exists, so — unlike vision — there is NO chunked-prefill flag
to disable; the ONLY interaction constraint is MSTAR_VLLM_PROMPT_LAYOUT, which
produces a 3-entry interleaved schedule (text/audio/text) the 2-entry merge
won't match (so leave it OFF for the A/B, matching the winning build). Gated to
text output (s2t eligible; s2s/i2s keep unmerged so Talker per-walk
thinker_states accounting stays aligned). A/B: A=off, B=+MSTAR_MERGED_PREFILL_AUDIO=1
(process-static, two servers). Counter `merged_prefill_audio_walks` (WARNING).

CRITICAL CORRECTION (2026-07-04, commit a2d2f47): MSTAR_VLLM_PROMPT_LAYOUT
DEFAULTS TRUE (qwen3_omni_model.py:103) and IS the benchmarked s2t config — my
earlier "keep it OFF / it's off in the winning build" was BACKWARDS. Under it,
process_prompt splits s2t text into prefix+suffix -> the schedule is 3-entry
[prefill_text(prefix), prefill_audio, prefill_text(suffix)], NOT 2-entry. The
original 2-entry-only matcher was therefore STRUCTURALLY DEAD on the real s2t
workload (merged_prefill_audio_walks would be 0). Do NOT set VLLM_PROMPT_LAYOUT=0
to force 2-entry — it changes task semantics (layout=0 => model TRANSCRIBES;
default layout=1 => model ANSWERS) + loses vLLM MRoPE parity. FIX (a2d2f47):
matcher now ALSO folds the 3-entry interleaved layout into one
prefill_multimodal_audio walk (concat prefix+audio+suffix, linear MRoPE, prefix
under text_inputs / suffix under text_inputs_suffix, Thinker node declares both;
2-entry emits suffix edge empty for readiness). Matcher returns merged_audio_order
= "audio_first"/"text_first"/"interleaved". 34 CPU tests green. i2t vision merge
is UNAFFECTED (prompt_layout split is audio-gated). GPU A/B pending. Also feeds
ARM-3 (base yaml + both merges, small-batch config).

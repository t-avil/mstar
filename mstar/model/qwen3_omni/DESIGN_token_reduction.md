# Multimodal token reduction for Qwen3-Omni prefill (TTFT)

Goal: cut **Thinker prefill cost** by reducing the number of multimodal tokens
the Thinker must prefill. This directly lowers **S2T/S2S TTFT** (audio path) and
**I2T/I2S TTFT** (image path) — the failing metrics. It is **orthogonal to the
vision-prefill batching** already landed, so the wins stack: batching makes each
encoder forward cheaper; this makes the *Thinker* sequence shorter.

## What this is NOT

We do **not** do FastV-style attention pruning. That keeps the sequence length
unchanged (so KV cache and the O(n²) prefill attention do not shrink), breaks
FlashInfer/FlashAttention's packed varlen layout, and degrades OCR/grounding.
Instead we **shrink the actual token sequence** before the Thinker, then
recompute M-RoPE positions for the reduced count so the rest of the sequence
stays contiguous.

Two env-gated, default-OFF knobs:

| Env var | Default | Path | Status |
|---|---|---|---|
| `MSTAR_AUDIO_TOKEN_STRIDE` | `1` (off) | S2T/S2S | **Implemented** (fully wired) |
| `MSTAR_VISION_TOKEN_MERGE` | `1` (off) | I2T/I2S | **Scaffold** (plumbing wired, merge op is a placeholder) |

At the default value both are exact no-ops → byte-identical to baseline.

---

## 1. Token counts per modality (derived from this codebase)

### Audio (`components/audio_encoder.py`)

Mel frontend is a Whisper-style `WhisperFeatureExtractor` (16 kHz, n_fft 400,
hop 160) → **100 mel frames/sec**. The CNN output-length formula
`_feat_extract_output_lengths` (audio_encoder.py:257) is:

```
out = ((in%100 -1)//2+1 ... )//2 + 1  +  (in//100)*13
```

i.e. **13 audio tokens per 100 mel frames = ~13 tokens per second of audio**
(≈ one token / 77 ms). Each token is `output_dim = 3584`-wide and is handed to
the Thinker as `audio_embeds`.

Examples (audio tokens entering the Thinker, before the 2 BOS/EOS sentinels):

| Clip length | stride 1 | stride 2 | stride 4 |
|---|---|---|---|
| 5 s | ~65 | ~33 | ~17 |
| 30 s | ~390 | ~195 | ~98 |

### Vision (`components/vision_encoder.py`, `config.py`)

`patch_size = 14`, `temporal_patch_size = 2`, `spatial_merge_size = 2`. After the
encoder's native 2×2 spatial merge, tokens per image =
`T * (H/14/2) * (W/14/2)`:

| Image (after smart-resize) | merged tokens (factor 1) | factor 4 | factor 9 |
|---|---|---|---|
| 448×448 | 256 | 64 | ~28 |
| 896×896 | 1024 | 256 | ~113 |

`MSTAR_VISION_TOKEN_MERGE` applies an **extra** per-axis merge of `sqrt(F)` on
top of the native merge (so F=4 ⇒ 2× per axis ⇒ ¼ the tokens). DeepStack
features (3 levels) are merged identically.

---

## 2. Where the reduction inserts

Architecture: each prefill is a **single-modality graph walk** whose token count
is **data-driven** — derived from the encoder output tensor's row count, not
from a pre-tokenized placeholder count in the prompt. This is the key enabler:
shrinking the encoder output automatically propagates to sequence length, KV
allocation (`plan_attention`/`plan_rope` read `seq_lens` from the inputs),
talker masks, and position advance — **with no placeholder-count mismatch**, the
usual failure mode for in-model token reduction.

### Audio — `ThinkerSubmodule.prepare_inputs`, `prefill_audio` branch (submodules.py)

```
audio_embeds = inputs["audio_embeds"][0]          # (N, 3584) from encoder
_dump_obj(..., audio_embeds)                       # RAW dump (pre-reduction) for parity
audio_embeds = _pool_audio_tokens(audio_embeds, audio_token_stride())   # (ceil(N/S), 3584)
audio_len = audio_embeds.shape[0]                  # everything below uses reduced count
```

`_pool_audio_tokens` is `avg_pool1d(kernel=stride, stride=stride,
ceil_mode=True, count_include_pad=False)` over the time axis — segmentwise
average pooling (Qwen2.5-Omni-style). The ragged tail collapses into one
correctly-averaged token. Single request per walk ⇒ no cross-request bleed.

### Vision — `ThinkerSubmodule.prepare_inputs`, `prefill_vision` branch

```
vision_embeds, deepstack, eff_sm = _merge_vision_tokens(
    vision_embeds, deepstack, grid_thw, vision_token_merge_factor(), base_sm)
```

`_merge_vision_tokens` reshapes each image's row-major `(T, H/sm, W/sm, C)`
token block and `avg_pool2d(kernel=axis, stride=axis)` over (H, W), then returns
an **effective spatial merge size** `eff_sm = base_sm * axis`. It validates that
`axis` divides the post-merge grid for *every* image and **bails safely**
(returns inputs unchanged) on any non-divisible grid or missing
`image_grid_thw`, so it can never desync token count vs positions.

---

## 3. M-RoPE position handling (the tricky part)

Token-count changes must propagate to the 3D position IDs, or attention planning
and parity break.

### Audio
`get_rope_index_audio(audio_len, start_pos+1, ...)` builds a temporal ramp
`arange(audio_len)+start_pos+1` (h/w pinned, or = temporal under
`MSTAR_VLLM_PROMPT_LAYOUT`). Because `audio_len` is read **after** pooling, the
ramp is shorter and the trailing EOS sentinel sits at `start_pos+1+audio_len`.
The downstream `advance_seq_lens` advances `position_id_start` by
`seq_len = audio_len+2`, so the next turn's positions remain contiguous and
monotonic. No other call site needs to change.

### Vision
`get_rope_index_vision` derives positions from `grid_thw` divided by the spatial
merge size. We feed it `eff_sm` (not the config's `base_sm`), so the height/width
ramps have exactly `(H/eff_sm)·(W/eff_sm)` entries = the merged token count.
`mrope_pos_advance = end_pos_base + 1 - start_pos` is recomputed from the new
positions, so the post-vision position advance is correct. The DeepStack scatter
(`full_deepstack[mm_mask] = deepstack_inp`) stays consistent because the
DeepStack tensors were merged to the same length.

---

## 4. Expected TTFT reduction

Thinker prefill cost on the multimodal span is roughly
`a·n + b·n²` (linear MLP/projection + quadratic attention) where `n` = tokens.

- **Audio, stride S**: `n → n/S`. Linear term ↓ S×, quadratic ↓ S²×. For audio
  spans that dominate the prompt (long clips), measured TTFT reduction is
  **near-proportional to the token cut** (≈ 2× at S=2, ≈ 4× at S=4) minus the
  fixed text-prompt overhead. The pooling op itself is negligible.
- **Vision, factor F**: `n → n/F`. Same structure; biggest wins on
  high-resolution images where vision tokens dominate.

Both stack on top of vision-prefill batching (which amortizes the *encoder*
cost, a different bottleneck).

---

## 5. Quality risk + gate

**Audio (low-moderate risk).** Literature (Qwen2.5-Omni-style downsampling)
reports ~3× near-lossless on ASR/S2TT, but it MUST be validated here. Risk rises
with stride and with content needing fine temporal resolution (fast speech,
phonetic detail, overlapping speakers). Average pooling across the encoder's
internal window boundaries is acceptable (Qwen pools globally) but is a quality
knob.

**Vision (higher risk).** Uniform spatial averaging is a deliberately simple
**scaffold** — it is exactly the regime the literature warns hurts
OCR/grounding/small-text. The plumbing (count + position + DeepStack
recomputation) is the reusable, correct-by-construction part; the
averaging step should be replaced by a content-aware, quality-gated selection
(ToMe bipartite matching / PruMerge+ keep-then-merge) before any production use.

### Gate (required before enabling either knob > 1)

1. **Output parity / quality, fixed decode (greedy)** at stride/factor
   `1` vs `2` vs `4` (vision: `1` vs `4` vs `9`):
   - Audio: WER / CER on a held-out ASR set; BLEU/COMET on S2TT; plus exact-token
     output parity vs stride-1 on a smoke set (expect divergence — measure it).
   - Vision: task accuracy on VQA + an OCR/grounding-heavy set (the sensitive
     case), plus output parity vs factor-1.
2. **Accept** a setting only if the quality delta vs baseline is within the
   project's tolerance AND the TTFT win is realized. Record the chosen value per
   modality; ship default OFF.

---

## 6. Implemented vs stubbed

- **Implemented & fully wired (audio):** `audio_token_stride()` reader,
  `_pool_audio_tokens`, insertion in `prefill_audio`, M-RoPE recompute via the
  reduced `audio_len`, talker-mask + seq-len + position-advance propagation.
  Default 1 = exact no-op.
- **Scaffold (vision):** `vision_token_merge_factor()` reader,
  `_merge_vision_tokens` with full count/position/DeepStack plumbing + safe
  bail-out, effective-merge-size into `get_rope_index_vision`. The **merge
  operator itself is a placeholder average-merge** — swap for a quality-gated
  ToMe/PruMerge+ selection. Default 1 = exact no-op.
- **Not done (needs GPU):** the quality/WER validation runs themselves (this was
  a no-GPU design+scaffold pass; py_compile only).

---

## 7. GPU validation commands

Use the project's fixed GPU set; confirm idle first (`nvidia-smi`); wrap in
`timeout`; one run per setting. Pseudocommands — adapt to the serving/eval entry
point in `benchmark/` and `configs/qwen3omni*.yaml`.

```bash
# ---- S2T: TTFT + WER/parity at stride 1 vs 2 vs 4 ----
for S in 1 2 4; do
  MSTAR_AUDIO_TOKEN_STRIDE=$S timeout 1800 \
    python -m mstar.serve --config configs/qwen3omni_colocated.yaml \
      --bench s2t --report ttft,itl --out runs/s2t_stride_$S.json
  # WER/parity: greedy decode over the ASR/S2TT eval set, compare to stride 1
  MSTAR_AUDIO_TOKEN_STRIDE=$S timeout 3600 \
    python benchmark/qwen3_omni_eval.py --task asr --greedy \
      --baseline runs/s2t_stride_1.json --out runs/s2t_wer_$S.json
done

# ---- I2T: TTFT + quality at merge factor 1 vs 4 vs 9 ----
for F in 1 4 9; do
  MSTAR_VISION_TOKEN_MERGE=$F timeout 1800 \
    python -m mstar.serve --config configs/qwen3omni_colocated.yaml \
      --bench i2t --report ttft,itl --out runs/i2t_merge_$F.json
  MSTAR_VISION_TOKEN_MERGE=$F timeout 3600 \
    python benchmark/qwen3_omni_eval.py --task vqa,ocr --greedy \
      --baseline runs/i2t_merge_1.json --out runs/i2t_quality_$F.json
done
```

Expected: TTFT for S2T at stride 2 ≈ ½ the stride-1 audio-prefill component;
WER delta small at stride 2, larger at 4 (the gate decides the ship value). For
vision, expect a strong TTFT drop with the average-merge but watch OCR — that is
the signal to replace the placeholder with a content-aware merge.

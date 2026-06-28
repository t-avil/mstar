# CUDA-graph coverage for Qwen3-Omni (Thinker / Talker / vocoder)

Branch: `exp/cuda-graph-bucketing` (based on `integration-mnew` = M\*-new).
Goal: cut TTFT (multimodal prefill) and decode overhead by maximizing CUDA-graph
coverage at **zero quality cost** — round prefills up to a captured token-budget
graph instead of running them eager, and make sure every uniform AR-decode region
is full-graphed.

This document records (1) what is **already** graph-captured, (2) the **eager
gaps** found, (3) the proposed bucketing, (4) expected TTFT/overhead reduction
and memory cost, (5) what is **implemented vs stubbed**, (6) GPU validation
commands.

---

## 1. What is already graph-captured (do NOT redo)

The CUDA-graph runner key is already token-aware:

```python
# mstar/engine/cuda_graph_runner.py:96
@dataclass(frozen=True)
class CudaGraphKey:
    graph_walk: str
    requires_cfg: bool
    bs: int
    num_tokens: int        # <-- num_tokens IS part of the key
```

Capture iterates every `(bs, num_tokens)` bucket a config declares
(`warmup_and_capture`, lines ~240-265), largest-first so the shared static-buffer
pool is sized by the max bucket.

| Submodule / walk | Config type | bs captured | token buckets | Status |
|---|---|---|---|---|
| Thinker `thinker_decode` | BasicBatched (compile) | 1,2,4,8,16,32 | n/a (1 tok/req) | **full graph** |
| Thinker `prefill_text` (+`prefill_audio` replay) | FlashInferPacked (compile) | 1,2,4 | **128,256,512,1024,2048** | graphed + pad-up |
| Thinker `prefill_vision` | FlashInferPacked (compile) | 1 | **128…16384** | graphed + pad-up |
| Talker `talker_decode` | BasicBatched (compile) | 1,2,4,8,16,32 | n/a (1 tok/req) | **full graph incl. RVQ depth** |
| Talker `talker_prefill` | FlashInferPacked (compile) | 1,2,4 | **128,256,512,1024** | graphed + pad-up |
| Talker `talker_last_prefill` | BasicBatched (compile) | 1,2,4,8,16,32 | 6 tok/req fixed | full graph |
| Code2Wav `code2wav_chunk` | BasicBatched (fp32, **no compile**) | 1,2,4,8,16,32 | fixed `full_seqlen` | **graphed** |
| Vision ViT encoder | private `torch.cuda.CUDAGraph` (vision_encoder.py:225) | — | per cu_seqlens key | graphed |
| Audio AuT encoder | private `torch.cuda.CUDAGraph` (audio_encoder.py:458) | — | per key | graphed |

### 1a. Prefill pad-up ALREADY exists (the key finding)

An arbitrary prefill length is **already rounded UP** to the smallest captured
bucket that fits — it is *not* recaptured per length and *not* run eager when it
falls between buckets:

```python
# mstar/engine/cuda_graph_runner.py:823
def _get_padded_num_tokens(self, num_tokens, padded_bs, config):
    sizes = sorted(config.get_total_tokens(padded_bs))
    idx = bisect.bisect_left(sizes, num_tokens)
    if idx >= len(sizes):
        return None            # <-- ONLY here do we miss -> eager
    return sizes[idx]          # round UP to next captured bucket
```

`get_total_tokens` returns the bucket keys for a FlashInferPacked config
(`cuda_graph_config.py:102`). Dispatch chain:
`KVCacheEngine._can_use_cuda_graph` (kv_cache_engine.py:536) →
`runner.can_run` → `_get_key_for` → `_get_padded_num_tokens`. On a miss it logs
`[cuda-graph miss] … falling back to eager` (kv_cache_engine.py:611) and runs the
eager forward. FlashInfer attention uses `qo_indptr` over the *real* per-request
token boundaries, so the trailing pad tokens of a rounded-up bucket are ignored —
**pad-up is zero quality cost.**

### 1b. Talker decode is already full-graphed including RVQ depth

`talker_decode` (BasicBatched, `compile=True`) captures the entire
`forward_batched → _forward_decode_like` (submodules.py:1670-1754): Talker LLM
backbone, `codec_head`, suppress-mask, in-graph repetition-penalty sampling of
layer-0, **and the full residual-VQ depth loop** — `for group_idx in
range(1, num_codes)` unrolls `code_predictor.forward_depth_unrolled` (talker.py:446)
+ per-codebook sampling into the single captured graph. `forward_depth_unrolled`
is graph-safe by construction: dense preallocated KV (no `plan()`), Python-static
`cache_pos`/`seq_len`, position ids passed in directly. **No eager region in the
Talker decode step.**

### 1c. Vocoder runs graphed

Code2Wav is captured (`code2wav_chunk`, BasicBatched, fp32, `compile=False`).
`prepare_inputs` pads codec tokens to the fixed `full_seqlen`
(`codec_left_context_frames + codec_chunk_frames`); `can_use_cuda_graphs`
(submodules.py:2195) requires `shape[1] == full_seqlen`, so standard streaming
chunks replay the fixed graph in fp32.

---

## 2. Eager gaps found

1. **Prefill eager cliff (primary TTFT gap).** Any prefill *longer than the top
   bucket* misses (`_get_padded_num_tokens` → `None`) and runs **eager**:
   - Thinker `prefill_text`/`prefill_audio`: **> 2048 tokens** → eager. A long
     S2T audio prompt or a long text instruction loses the whole prefill graph
     win — exactly the longest, most TTFT-expensive prompts.
   - Talker `talker_prefill`: **> 1024 tokens** → eager.
   - Thinker `prefill_vision`: **> 16384 tokens** → eager (only very long video).

2. **Round-up waste (secondary overhead).** Buckets are coarse (powers of two).
   A 130-token text prefill replays the **256** bucket (~2× the needed attention/
   MLP compute on pad tokens); a 600-token prefill replays **1024**. Pure
   overhead, not a correctness/quality issue.

3. **Vocoder oversized chunks.** If a chunk arrives with `shape[1] > full_seqlen`
   (e.g. an adaptive/large-chunk policy), `can_use_cuda_graphs` returns False →
   eager vocoder. Not hit by the default `LeftContextChunkPolicy` (chunks are
   bounded at `full_seqlen`), but real under an adaptive-chunk branch.

Gaps that **do not** exist (verified, so not touched): per-length recapture
(none — pad-up reuses one graph), eager Talker decode (none — fully graphed),
eager decode-step RVQ depth (none — unrolled in-graph).

---

## 3. Proposed bucketing

Lever A is largely *already in place* (round-up dispatch). The remaining win is
**a better bucket SET**: finer near common short prompts (less round-up waste) and
a wider top bucket (defer the eager cliff). Gated by
`MSTAR_PREFILL_BUCKET_PADUP` (default OFF), so the default capture set and
dispatch stay byte-identical.

Resolution per walk (`_resolve_prefill_buckets`, submodules.py):

- **OFF (default):** returns the existing hardcoded list unchanged (same object,
  same order). Byte-identical.
- **ON:** union the default with a finer low/mid grid
  `{64,96,192,384,768,1536,3072,6144}` (clamped `≤ max(default)`) and add one
  wider top bucket at `2 × max(default)`.
- **ON + explicit override** (`MSTAR_PREFILL_BUCKETS`,
  `MSTAR_PREFILL_VISION_BUCKETS`, `MSTAR_TALKER_PREFILL_BUCKETS`): use that
  comma-separated list verbatim — the deployment escape hatch to fit buckets to a
  measured prompt-length histogram.

Resulting ON sets:

| Walk | OFF (default) | ON (widened) |
|---|---|---|
| Thinker text/audio | 128,256,512,1024,2048 | 64,96,128,192,256,384,512,768,1024,1536,2048,**4096** |
| Talker prefill | 128,256,512,1024 | 64,96,128,192,256,384,512,768,1024,**2048** |
| Thinker vision | 128…16384 | 64…16384 (finer) + **32768** |

---

## 4. Expected TTFT / overhead reduction and memory cost

**TTFT / overhead (no quality cost — pad-up is exact):**
- Eliminating the eager cliff for 2049–4096-token text/audio prefills: an eager
  multimodal prefill on the 30B MoE Thinker is the dominant TTFT term; replacing
  it with a captured-graph replay removes per-op CPU launch overhead on the long
  prompts that need it most. vLLM's analogous ViT token-budget bucketing recipe
  reports **+11–20% single-GPU latency**; the I2T/S2T prefill here is the direct
  analogue.
- Round-up waste: halving the gap below each bucket roughly **halves the pad-token
  compute** for prompts that land just above a power-of-two (e.g. 130→192 instead
  of 130→256; 600→768 instead of 600→1024).

**Memory cost (the trade):** each extra bucket is one extra capture = persistent
FlashInfer prefill wrapper(s) + static input buffers, sized at that bucket, times
`NUM_SLOTS` (default 2) times `capture_batch_sizes`. Static buffers are
leading-dim slices of one shared max-bucket allocation per `(config, key)`
(`_intern_static_buffer`, cuda_graph_runner.py:380) so they do **not** scale with
bucket count — only the per-bucket FlashInfer workspaces + the wider top bucket's
max allocation grow. Thinker text/audio goes 5→12 buckets, Talker 4→10, vision
8→17. The new top bucket (4096 text / 32768 vision) enlarges the shared max
buffer (`input_embeds`,`cos_3d`,`sin_3d`,`deepstack_i`) proportionally to its
size. Budget capture VRAM headroom before enabling; tune with the explicit
override to capture only the buckets a real length histogram needs.

---

## 5. Implemented vs stubbed

**Implemented (behind `MSTAR_PREFILL_BUCKET_PADUP`, default OFF):**
- `_prefill_padup_enabled()` / `_resolve_prefill_buckets()` (submodules.py).
- Wired into `ThinkerSubmodule.get_cuda_graph_configs` (text/audio + vision) and
  `TalkerSubmodule.get_cuda_graph_configs` (talker_prefill).
- Per-walk explicit overrides: `MSTAR_PREFILL_BUCKETS`,
  `MSTAR_PREFILL_VISION_BUCKETS`, `MSTAR_TALKER_PREFILL_BUCKETS`.
- Default OFF verified byte-identical (returns the same list object); `py_compile`
  clean.

**Already present (confirmed, not re-implemented):** round-up prefill dispatch
(`_get_padded_num_tokens`), full Talker decode graph incl. RVQ depth unroll,
graphed fp32 vocoder, graphed vision/audio encoders.

**Stubbed / not done (need GPU to validate; out of scope here):**
- *Fully* eliminating the eager cliff for arbitrarily long prompts. The widened
  top bucket only defers it. True elimination needs either a max-context top
  bucket (large VRAM) or **chunked prefill** (split a long prefill into
  bucket-sized captured segments) — not implemented.
- Vocoder oversized-chunk bucketing (multiple `code2wav` chunk-size captures) —
  not implemented; default policy stays within `full_seqlen`.
- Auto-deriving buckets from an observed prompt-length histogram — manual via the
  override env vars for now.

---

## 6. GPU validation commands (no GPU was used here)

Parity / default-OFF byte-identity (must pass unchanged):

```bash
# Captured bucket set is unchanged with the flag OFF.
python - <<'PY'
import torch
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule, TalkerSubmodule
# build/load model per the existing test harness, then:
# assert sorted(cfg.get_total_tokens(1)) == [128,256,512,1024,2048]  # text/audio OFF
PY

# Existing Qwen3-Omni parity suite — output must be identical OFF.
pytest tests/ -k qwen3_omni -q
```

A/B latency (one fixed GPU set, per CLAUDE.md: confirm idle via nvidia-smi, pin
`CUDA_VISIBLE_DEVICES`, wrap in `timeout`, monitor, clean up):

```bash
export CUDA_VISIBLE_DEVICES=0
# Baseline (eager cliff present):
timeout 1800 python bench_ttft.py --modality s2t --prompt-lens 1500,2500,3500
# Widened buckets (cliff deferred, finer grid):
MSTAR_PREFILL_BUCKET_PADUP=1 timeout 1800 python bench_ttft.py --modality s2t \
    --prompt-lens 1500,2500,3500
# Tuned to a measured histogram:
MSTAR_PREFILL_BUCKET_PADUP=1 MSTAR_PREFILL_BUCKETS=256,512,768,1280,2048,3072 \
    timeout 1800 python bench_ttft.py --modality s2t --prompt-lens 1500,2500,3500
```

Check the engine log for `[cuda-graph miss] … falling back to eager` on the
2049–4096 token prefills: present at baseline, **absent** with the flag ON.
Capture-time VRAM delta is reported by `CudaGraphRunner[...]: warmup_and_capture
done … cuda alloc delta` — compare OFF vs ON to size the memory cost.

# Qwen3-Omni on M* — joint benchmark results (#131)

Interim joint result (audio + image paths). Authoritative numbers recompute from the per-request
`raw_<path>.json` via `aggregate.py`. Device-isolated per-system 2×H200 pairs. `mstar_new` = native
encoder build (vllm-layout, prompt-parity); `mstar_new_gpumel` = + GPU log-mel; `mstar_new_gpuimgON`
= + GPU image preprocess. A win "counts" only at ≥10% over **both** M*-old and vLLM, parity green.

## Headline (honest)

**vs vLLM-Omni:**
- **I2S ≈2×**: throughput 1.85× (B1) / 1.98× and RTF 2.03× (B4). ✅ (image speech)
- **I2T 1.3–1.7×** (B1–B16), ~tie at B32. ✅
- **S2S +GPU-mel wins every batch**: RTF 1.25–1.41×, TTFT 1.39–1.78× (stays real-time through B=32). ✅
- **S2T +GPU-mel**: throughput win (1.2–1.4×); **TTFT loses at high batch** (vLLM tighter prefill — a
  prefill lever, not mel). ❌ on S2T-TTFT-at-batch.

**vs M*-old (the ≥10% bar):**
- **Audio (+GPU-mel): decisive** — S2T TTFT 5.3s→0.43s @B32 (≈12×), throughput ≈4.6×; S2S RTF 1.8→0.61.
  Root cause: HF CPU WhisperFeatureExtractor mel is an intrinsic ~240ms cost that serializes across the
  batch; GPU mel (torch.stft) removes it. ✅
- **Image: ~tie** — native ≈ M*-old on this flash_attn box (both use flash_attn varlen for vision; the
  native>HF gap is patch-embed Conv3d only, modest). So image clears the bar vs vLLM, **not** vs M*-old.

## Optimization ledger
| Lever | Flag | Verdict |
|---|---|---|
| GPU log-mel | `MSTAR_GPU_MEL` | ✅ decisive audio win; recommend default-ON (bf16-equiv, cos≥0.9999 — same class as accepted native encoder) |
| Native encoders | (default) | ✅ parity green; I2S≈2× vs vLLM; ~tie vs old |
| codec_chunk | config | S2S win (Agent D finalizing fixed-larger-chunk on the graph path) |
| GPU image preprocess | `MSTAR_GPU_IMAGE_PREPROCESS` | ✅ 12–440× per-image on large images; neutral on 512px food101 (benchmark) |
| Code2Wav SP | `MSTAR_CODE2WAV_SP` | ❌ negative (slower than compiled single-GPU; vocoder launch-bound on H200); not landed |
| varlen backend recalibration | `MSTAR_VARLEN_BACKEND` | ⚪ inert in production (flash_attn always used) |

## Parity
18-case backend-equivalence green; native==HF (cos≈1.0); GPU-mel cos≥0.9999 (99.97% bf16-identical).

## Caveats
- Clients co-located on node-0 during some runs (A/B deltas clean; absolute latency may be slightly
  inflated — re-confirm with a remote client).
- I2S currently B1/B4; more batches + the integrated (all-flags-on) headline land in a follow-up commit.
- Audio numbers from Agent B's sequential single-pair isolated 4-way (most internally consistent).

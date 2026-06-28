# Qwen3-Omni on M* — joint benchmark results (#131)

Final joint result. `mstar_new` = **integrated optimized build** (native encoders + `MSTAR_VLLM_PROMPT_LAYOUT=1`
[matched audio length / same-audio] + `MSTAR_GPU_MEL=1` + `MSTAR_GPU_IMAGE_PREPROCESS=1` + codec-chunk).
Baselines: `mstar_old` (upstream HF-wrapper) and `vllm` (vLLM-Omni). Authoritative numbers recompute from
per-request `raw_<path>.json` via `aggregate.py`. Device-isolated 2×H200 pairs. A win counts only at ≥10%
over **both** baselines with parity green. Audio baselines = Agent B's sequential isolated 4-way; image
throughput = Agent D's vLLM proof; image variants = Agent C.

## Headline — integrated M*-new vs vLLM (and vs M*-old), B=1→32

| Path | vs vLLM (throughput) | vs vLLM (RTF) | vs M*-old | ≥10% over BOTH |
|---|---|---|---|---|
| **S2S** audio→speech | **1.6–3.6×** (2.1–3.6× at B8–32) | **1.53–1.78×** | 1.75–4.1× | ✅ every batch |
| **S2T** audio→text | **1.40–1.62×** | (text) | 2.4–5.5× | ✅ every batch |
| **I2S** image→speech | **1.85–2.21×** (~2×) | **1.86–2.21×** | ~tie (1.0–1.12×) | ✅ vs vLLM every batch |
| **I2T** image→text | **1.42–1.98×** | (text) | 1.03–1.22× | ✅ vs vLLM; vs old at B2/8/16/32 |

**The targets:** S2S and I2S hit ~2× vs vLLM at batch (S2S throughput 2–3.6×, RTF 1.5–1.8×; I2S ~2×). S2T and
I2T win vLLM solidly (1.4–2×). vs M*-old the audio paths are decisive (2.4–5.5×, driven by GPU-mel); image is
~tie (native ≈ old on this flash_attn box — the native>HF gap is patch-embed only).

## Why it works (the composition)
- **GPU log-mel** (`MSTAR_GPU_MEL`): the HF CPU WhisperFeatureExtractor mel is an intrinsic ~240 ms cost that
  **serializes across the batch** (un-optimized S2T plateaus at B8, S2S vocoder+mel saturate to RTF 2.0–2.3 @B32
  and *lose* to vLLM). Moving it to `torch.stft` on GPU (~0.4 ms) is what flips S2T and S2S from losses into
  wins at batch. This is the single decisive lever.
- **codec-chunk** + **prompt-layout** (matched audio length → fair RTF) + **native encoders** compose on top.
- Length fairness: integrated S2S median audio ≈4.3 s vs vLLM ≈5.0 s (ratio ~0.85 — comparable, RTF is fair).

## Optimization ledger
| Lever | Flag | Verdict |
|---|---|---|
| GPU log-mel | `MSTAR_GPU_MEL` | ✅ decisive (flips S2T+S2S to batch wins); recommend default-ON (bf16-equiv cos≥0.9999) |
| Native encoders | default | ✅ parity green; I2S ~2× vs vLLM; ~tie vs old |
| codec-chunk | config | ✅ part of the integrated win; D's fixed-larger-chunk lever in progress |
| GPU image preprocess | `MSTAR_GPU_IMAGE_PREPROCESS` | ✅ 12–440× per-image on large images; neutral on 512px food101 |
| Code2Wav SP | `MSTAR_CODE2WAV_SP` | ❌ negative (vocoder launch-bound on H200); not landed |
| varlen recalibration | `MSTAR_VARLEN_BACKEND` | ⚪ inert in prod (flash_attn always used) |

## Honest negatives / caveats
- **S2T TTFT loses to vLLM at high batch** (vLLM tighter prefill; gpu-mel fixes throughput but not the prefill
  gap) — a separate future lever, not mel.
- Clients co-located on node-0 during runs; A/B deltas clean, absolute latency may be slightly inflated
  (re-confirm with a remote client).
- I2S baseline coverage is B1/4/16/32 (B2/8 lack comparators); audio/image baselines from different agents'
  runs (consistent methodology, isolated where it matters).

## Parity
18-case backend-equivalence green; native==HF (cos≈1.0); GPU-mel cos≥0.9999 (99.97% bf16-identical, same class
as the accepted native encoder). Charts: `charts/{audio,image}_to_{text,speech}_throughput_rtf.png`
(regenerable from `raw_<path>.json` via `aggregate.py`).

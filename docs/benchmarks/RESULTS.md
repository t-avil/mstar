# Qwen3-Omni on M* — joint benchmark results (#131)

Final joint result, all 4 paths × B=1→32. `mstar_new` = **integrated optimized build** (native encoders +
`MSTAR_VLLM_PROMPT_LAYOUT=1` [matched audio length / same-audio] + `MSTAR_GPU_MEL=1` + `MSTAR_GPU_IMAGE_PREPROCESS=1`,
codec-chunk@15). Baselines: `mstar_old` (upstream HF-wrapper) and `vllm` (vLLM-Omni). Numbers recompute from
per-request `raw_<path>.json` via `aggregate.py`. Device-isolated 2×H200 pairs; all runs fair (sequential /
same-load paired — never cross-time-window on the shared node). All numbers are plain ratios from `NUMBERS.md`
(M\*-new/vLLM and M\*-new/M\*-old per metric; for lower-is-better RTF/TTFT/ITL, **>1.00× = M\*-new faster**).
Sources: audio = Agent B's isolated 4-way; image = Agent C's sequential 3-way; integrated S2T/I2T =
Agent D's post-fix same-session re-run vs vLLM; integrated S2S/I2S = coordinator integrated sweep.

## Headline — integrated M*-new vs vLLM (and vs M*-old), B=1→32

| Path | vs vLLM (>1.00× = M*-new faster) | vs M*-old |
|---|---|---|
| **S2T** audio→text | tok/s **1.39–1.65×**, req/s **1.58–1.89×** (B32 20.1 vs 12.7 req/s) | tok/s **2.15–5.51×** |
| **S2S** audio→speech | RTF **1.55–1.77×**, tput **1.64–2.02×** | tput **1.78–4.09×** |
| **I2S** image→speech | tput **1.80–2.18×**, RTF **1.82–2.21×** (~2×) | ~tie (0.99–1.14×) |
| **I2T** image→text | tput **1.35–1.94×** (TTFT still >vLLM at high B) | ~tie (0.98–1.17×) |

**Headline read (length-fair):** speech is reported on **RTF** (+ audio s/s); text on **tok/s** (each system
self-counts with its own Qwen vocab). req/s is secondary (length-confounded). M\*-new emits ~0.78 (I2S) / ~0.82
(S2S) of vLLM's audio length; M\*-old emits only ~0.57 (S2S) / ~0.76 (I2S) of vLLM's — so M\*-old's RTF/throughput
look better-than-fair and M\*-new vs M\*-old on speech must be read on length-normalized RTF. **Targets:** S2S & I2S
~2× vs vLLM at batch; S2T req/s 1.58–1.89× (tok/s 1.39–1.65×); I2T tput 1.35–1.94×. vs M\*-old the audio paths are
decisive (S2S tput 1.78–4.09×, GPU-mel); image is ~tie (native ≈ old on this flash_attn box; GPU-img adds the
margin). **The paper's "~2–2.5× throughput at batch over vLLM" holds once preprocessing is on-GPU.** Outputs are
**not** identical (stochastic talker@0.9 + thinker@0.7, unseeded — M\*'s native defaults, same as main); parity per
#131 is encoder + performance, not audio-output identity.

## Why it works — and the corrected text-path story
- **GPU log-mel** (`MSTAR_GPU_MEL`) is the single decisive audio lever. The HF CPU WhisperFeatureExtractor mel is
  an intrinsic ~240 ms cost that **serializes across the batch**; on the CPU-mel path (M\*-old) S2T req/s plateaus
  (~5.4–5.9 for B≥8) and S2T TTFT(text) p50 climbs to 5.35 s @B32. On-GPU (`torch.stft`, ~0.4 ms) it's gone:
  **S2T req/s scales to 1.58–1.89× vLLM and TTFT(text) p50 stays flat 0.10→0.37 s.** Same mechanism keeps S2S
  real-time (RTF<1) through B=32.
- **GPU image preprocess** (`MSTAR_GPU_IMAGE_PREPROCESS`) is the analogous image lever: the HF CPU resize/patchify
  serializes on the host (up to ~175 ms/image on ~3000 px inputs; ~7–100× faster on-GPU, e.g. 1900×1300 64 ms→0.6 ms).
  On food101 (512 px) I2T TTFT(text) p50 runs 0.31→0.76 s and **I2T scales 1.35–1.94× vLLM throughput** — though
  I2T TTFT (0.76 s @B32) is still ~3.7× vLLM's 0.21 s (residual image-prefill gap).
- **Root-cause clarity:** the high-batch text-path plateaus were **CPU-preprocessing serialization, NOT a
  scheduler/architecture deficit.** (The scheduler uses ROUND_ROBIN, already fair; a reorder lever was evaluated
  by code analysis and rejected as order-invariant — zero GPU spent.) Moving preprocessing on-device is the fix.
  A separate piggyback/chunked-prefill change could add more headroom but was not needed and is deferred
  (see honest negatives for the considered-and-deferred rationale).
- **prompt-layout** (matched audio length → fair RTF; S2S median audio ≈4.3 s vs vLLM ≈5.2 s, ratio ~0.82; I2S
  ratio ~0.78) + **native encoders** + codec-chunk@15 compose on top. Caveat: M\*-old emits much shorter audio
  (~0.57 of vLLM on S2S), so M\*-old's RTF/throughput look better-than-fair; read M\*-new vs M\*-old on speech via
  length-normalized RTF (and even then M\*-old's short outputs flatter it).

## Optimization ledger
| Lever | Flag | Verdict |
|---|---|---|
| GPU log-mel | `MSTAR_GPU_MEL` | ✅ decisive (S2T scales ~2×, S2S real-time); recommend default-ON (bf16-equiv cos≥0.9999) |
| GPU image preprocess | `MSTAR_GPU_IMAGE_PREPROCESS` | ✅ I2T scales 1.35–1.94×; also 7–100× per-image on large images (64 ms→0.6 ms @1900×1300) |
| Native encoders | default | ✅ parity green; I2S ~2× vs vLLM; ~tie vs old (native>HF = patch-embed only here) |
| codec-chunk @15 | config | ✅ default 15 kept (correct). ❌ larger-chunk-for-batch DISPROVEN (on-graph A/B: I2S +5–7% <bar, S2S −18%); not landed |
| Code2Wav SP | `MSTAR_CODE2WAV_SP` | ❌ negative (vocoder launch-bound on H200); not landed |
| varlen recalibration | `MSTAR_VARLEN_BACKEND` | ⚪ inert in prod (flash_attn always used) |
| scheduler reorder (Lever 2) | — | ❌ order-invariant, can't help (RR already fair); rejected by analysis, 0 GPU |

## Honest negatives / caveats
- vs **M*-old, image is ~tie** (native ≈ old on this flash_attn box; the native>HF gap is patch-embed Conv3d only).
  GPU-img is what gives image its margin, and it's neutral on the 512 px food101 benchmark (big only on large images).
- S2T **req/s 1.58–1.89×** and **tok/s 1.39–1.65×** (both wins; metric-dependent magnitude).
- **I2T TTFT** stays above vLLM at high batch (0.76 s vs 0.21 s @B32) — throughput scales/wins but TTFT is the one
  honest remaining image gap (a future image-prefill lever, same family as S2T-TTFT).
- Clients co-located on node-0 during some runs; A/B deltas clean, absolute latency may be slightly inflated
  (re-confirm with a remote client). Contention-depressed first numbers were caught and re-run fair.
- Further S2T/I2T TTFT-at-batch headroom (piggyback/chunked-prefill) was **considered and deferred**: the in-scope
  scheduler reorder (Lever 2, option-i) was ruled out by code analysis as order-invariant (RR already fair); the
  only remaining fix is piggyback/chunked-prefill (a new mixed prefill+decode walk + combined CUDA-graph key +
  same-walk-invariant change + full re-parity) — high-risk and **out of #131 scope** (encoder + perf per the issue).
  GPU-mel already won the throughput headline, so piggyback would only shave residual text-path TTFT-at-batch
  (S2T B32 0.37 vs 0.28 s; I2T B32 0.76 vs 0.21 s), ships default-OFF, and is deferred to a separate supervised effort.

## Parity
18-case backend-equivalence green; native==HF (cos≈1.0); GPU-mel cos≥0.9999 (99.97% bf16-identical, same class as
the accepted native encoder). Charts: `charts/{audio,image}_to_{text,speech}_throughput_rtf.png` (regenerable
from `raw_<path>.json` via `aggregate.py`). Productized build: branch `integration-mnew`.

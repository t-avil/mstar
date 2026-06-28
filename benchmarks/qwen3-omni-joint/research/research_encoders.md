# Qwen3-Omni Encoders + Input Preprocessing — Mechanisms Report

Read-only, file:line-cited comparison of **M\*-new** (`/home/tim/integration-wt/mstar`,
native encoders, default ON), **M\*-old** (`/home/tim/mstar`, HF-wrapper encoders), and
**vLLM-Omni** (`/home/tim/baselines/vllm-omni`). Goal: explain Qwen3-Omni benchmark
behavior (TTFT / throughput / RTF) at B=1 vs B=32 per path (S2T, I2T, I2S, S2S).

Cross-checked against `/home/tim/exp_rebench/LEVERS_REPORT.md`, `/home/tim/mstar/FINDINGS.md`,
and memory notes `[[mstar-numa-contention-ttft]]`, `[[mstar-gpu-env-quirks]]`.

Effect numbers come from those two sources (server A/B + microbench), not from a run I did
here — flagged as such. Code mechanism citations are first-hand from the files below.

---

## 0. Default wiring (what actually runs in prod)

- **M\*-new defaults to native encoders, both ON**: `config.py:380-381`
  (`native_audio_encoder: bool = True`, `native_vision_encoder: bool = True`), env-overridable
  `MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER` at `config.py:397-400`.
- **M\*-old has no native option** — it always builds the HF `Qwen3OmniMoe{Audio,Vision}Encoder`
  with `attn_implementation="flash_attention_2"` (`/home/tim/mstar/.../qwen3_omni_model.py:1488-1490`,
  `:1531-1533`) and runs each encoder **once per request, not batched**
  (`/home/tim/mstar/.../submodules.py:49-57`, "Runs once per request").
- **GPU mel and GPU image preprocess exist ONLY in M\*-new** and are **default OFF**
  (`MSTAR_GPU_MEL` `qwen3_omni_model.py:61`; `MSTAR_GPU_IMAGE_PREPROCESS` `:205-208`).
  Confirmed M\*-old has zero references to either flag.
- Preprocessing site for all M\* paths: `data_worker.py:_process_input` → `process_prompt`
  (`data_worker.py:241-248`), i.e. on the **host**, per request, *before* the request enters
  the batched Thinker prefill.

---

## 1. GPU log-mel (`MSTAR_GPU_MEL`) — the decisive audio lever (S2T, S2S)

### Mechanism (HF CPU mel vs torch.stft GPU mel)

**Default / M\*-old / M\*-new-OFF path = HF `WhisperFeatureExtractor` on the CPU.**
In `process_prompt`, each raw waveform is moved to host numpy and handed to the HF feature
extractor:
- `qwen3_omni_model.py:1378-1381` — `np_audios.append(waveform.cpu().numpy())` (only when GPU mel OFF).
- `qwen3_omni_model.py:1556-1577` — `feat_extractor(audio, ...)` runs HF's single-threaded
  numpy STFT + mel filterbank + log on the host CPU, then permutes/masks to `(n_mel, T)`.
- M\*-old is identical (`/home/tim/mstar/.../qwen3_omni_model.py:1103` `feat_extractor = ...`).

**GPU path (`MSTAR_GPU_MEL=1`, M\*-new only).**
- Dispatch gate: `qwen3_omni_model.py:1375-1377` (`_use_gpu_mel = _GPU_MEL and processor and cuda`).
- Per-clip call: `qwen3_omni_model.py:1551-1555` → `_audio_mel_gpu` (`:1276-1308`) → module-level
  `gpu_log_mel` (`:64-81`): `torch.stft(center=True, pad_mode="reflect", return_complex=True)`,
  drop-last-frame, power spectrogram, `mel_filters.T @ mag`, `log10`, per-clip max-8 clamp,
  `(x+4)/4`. Filterbank + hann window are cached per device (`:1293-1303`).
- Output contract is byte-compatible with the HF path (`(n_mel, T)` float32 on CPU, `:1306-1307`),
  so everything downstream (encoder, Thinker splice) is unchanged. Parity: cos≥0.9999, ~1e-5
  max-abs (`test_qwen3_omni_gpu_mel_parity.py`, cited in docstring `:1286-1289`).

### Why it's an intrinsic ~240 ms CPU cost that serializes across the batch

- The HF mel is a **single-threaded numpy STFT over a 480k-sample / 3000-frame 30 s clip**.
  Per `[[mstar-numa-contention-ttft]]` this is **~240 ms/clip and intrinsic, not a contention
  artifact**: measured S2T B=1 TTFT ~0.34 s with node-0 quiet vs ~0.10 s with GPU mel — the
  ~240 ms delta *is* the CPU mel. The encoder *forward* is a separate, tiny cost
  (FINDINGS §3: audio encoder ~16–25 ms, <3 % of E2E, ~99.7 % launch overhead).
- **Serialization**: mel runs in `data_worker._process_input` (`data_worker.py:241`) on the
  host, per request, *before* the request joins the batched prefill. It never touches the GPU,
  so B requests' mel work cannot be amortized by GPU batching and instead stacks up on limited
  host/data-worker CPU capacity. Empirically near-linear in B (baseline S2T TTFT: B=8 **1.69 s**,
  B=32 **5.24 s**, FINDINGS §1 / `[[mstar-numa-contention-ttft]]`). The GPU path collapses each
  clip to ~0.3–0.4 ms incl. H2D+D2H, so it neither serializes nor balloons.
  *Uncertainty:* I did not confirm the exact number of data-worker processes; the
  "serializes" claim rests on the measured near-linear TTFT growth, not on reading a worker-count.

### Effect (from `[[mstar-numa-contention-ttft]]`, same-binary A/B, GPU-mel OFF→ON)

| Path | Metric | B=1 | B=32 |
|---|---|---|---|
| **S2T** | TTFT (first text token) | 3.5× faster | **12.2× faster** |
| **S2T** | text throughput | — | 2.3–4.4× |
| **S2S** | RTF | 1.5× | **3.3×** (stays RTF<1 where baseline blew to 2.0) |

Decisive lever for audio at batch. Without it M\*-new is 12–18× worse than vLLM on batch S2T
TTFT and goes non-real-time on S2S; with it M\* beats vLLM on S2S RTF (1.25–1.41× @B16/32).
Residual S2T-TTFT gap to vLLM at high batch is a *separate prefill lever*, not mel.

---

## 2. Native audio encoder vs HF wrapper (S2T, S2S)

### Mechanism

- **M\*-new native AuT** (`components/audio_encoder.py`, whole file). Attention goes through
  `varlen_attention` (`:242-251`): if `flash_attn` imports, it calls
  **`flash_attn_varlen_func`** directly with `cu_seqlens` (the per-window packing), bidirectional,
  no padding. The deterministic frontend (chunking, valid-index, `cu_seqlens`, CNN output-length)
  replicates HF bit-for-bit (`:257-296`).
- **Cross-request batching** is the structural win: `NativeAudioEncoderSubmodule`
  (`submodules.py:112-184`) concatenates N requests' mel along time (`preprocess`, `:145-153`),
  runs **one** varlen forward, and slices outputs back per request (`forward_batched`, `:155-169`).
  `can_batch` only engages when each request has exactly one `feature_lens` segment (`:177-184`).
- **M\*-old HF wrapper** (`AudioEncoderSubmodule`, `/home/tim/mstar/.../submodules.py:49-109`)
  runs the HF encoder **once per request** (docstring `:56` "not batched across requests"), built
  with `flash_attention_2` (`qwen3_omni_model.py:1488-1490`).

### Is it the same flash_attn in prod? (MSTAR_VARLEN_BACKEND inert)

**Yes — same flash_attn primitive on both sides.** Per `[[mstar-gpu-env-quirks]]` fact #1,
`flash_attn` (2.8.3) is installed in the shared venv, so `varlen_attention` *always* takes the
`flash_attn_varlen_func` path (`audio_encoder.py:244-250`). Consequently
**`MSTAR_VARLEN_BACKEND`** (`audio_encoder.py:230`) and the SDPA backend matrix
(`_sdpa_varlen_*`, `:49-139`) and the adaptive-τ heuristic (`:110-139`) are **inert in the
served path** — they matter only as a no-flash fallback or via the flashinfer+CUDA-graph path
(`:202-225`, `_cuda_graph_enabled` `:441-444`, opt-in `MSTAR_ENCODER_CUDA_GRAPH`). The
in-code comment in `audio_encoder.py:72` ("this H200" lacks flash-attn) is **stale/wrong**.

So native-vs-HF is **not** a faster attention kernel; both call the same FA2 varlen. The
difference is (a) cross-request batching and (b) decoupling from `transformers`.

### Effect (FINDINGS §3, §2; native encoder docstring `qwen3_omni_model.py:1922-1927`)

- **B=1**: native ≈ HF. Encoder forward is <3 % of E2E and launch-bound; B=1 RTF is ~98.6 %
  Talker AR decode + Code2Wav. Microbench shows only ~1.2–1.7× (peak B=4–8) and it's the
  batching, not the kernel (`qwen3_omni_model.py:1925-1927`).
- **B=32**: this is where native matters. M\*-old's HF path degrades — at S2S it hits
  **2.0 RTF @ B=32** (FINDINGS §3, dense O(n²) cross-segment attention when not varlen-packed
  per request) while native varlen holds up. Most valuable as #131 acceptance #2 evidence; as a
  raw speedup it is secondary to GPU mel (Lever 1). LEVERS_REPORT Lever 5 frames this as the
  native-vs-HF batch story.
- **Note**: at B=32 the *encoder forward* improvement is partly masked by the CPU-mel balloon
  (§1) on the default path — i.e. native encoder without GPU mel still inherits the ~5 s mel
  serialization. The two levers are complementary: GPU mel removes the host-CPU wall, native
  varlen keeps the on-GPU encoder cheap at batch.

---

## 3. Native vision encoder + GPU image preprocess (I2T, I2S)

Two independent mechanisms here. Both bite at TTFT, at different places.

### 3a. Patch-embed: native `F.linear` vs HF bf16 `Conv3d` cuDNN cliff

- **The dominant per-image cost in the HF vision encoder is the patch embed**, not attention.
  `VisionPatchEmbed` (`components/vision_encoder.py:80-108`) stores the weight as a `Conv3d`
  (kernel==stride==patch, so checkpoints load unchanged) but computes it as an **`F.linear`
  matmul** (`:106-108`). Docstring `:81-92`: HF's **bf16 `Conv3d` for this kernel==stride shape
  hits a cuDNN low-precision cliff — ~3.2 s/image bf16 on H100** vs ~0.2 ms fp32 conv and ~40 µs
  for the matmul. Exact in fp32; bf16 differs only by accumulation rounding (≤~1.6e-2/elt),
  bounded by parity (`qwen3_omni_encoder_parity.py`).
- Same FA2 varlen attention as audio (`vision_encoder.py:35` imports `varlen_attention`;
  `VisionAttention.forward` `:153-159`). So again the attention kernel is identical to HF;
  the patch-embed swap is the speedup (`qwen3_omni_model.py:1983-1988`: "large per-image speedup
  comes almost entirely from computing the patch embed as F.linear ... same swap could in
  principle be applied to the HF path").
- **Cross-request batching**: `NativeVisionEncoderSubmodule` (`submodules.py:286-364`)
  concatenates patch rows + `grid_thw` rows, one forward, slices merged tokens **and each
  DeepStack level** back per request (`preprocess` `:319-327`, `forward_batched` `:339-354`).
  M\*-old `VisionEncoderSubmodule` runs once per request (`/home/tim/mstar/.../submodules.py:192-200`).

### 3b. GPU image preprocess (`MSTAR_GPU_IMAGE_PREPROCESS`)

- **Default / HF path**: each GPU image is moved to host numpy
  (`qwen3_omni_model.py:1357-1370`, `img_u8.cpu().contiguous().numpy()`) and run through HF's
  `Qwen2VLImageProcessor` on CPU — smart_resize + rescale + normalize + patchify
  (`:1544-1549`). Comment `:190-193`: "that CPU round-trip + numpy processing is the single
  biggest I2T TTFT cost (~175 ms)".
- **GPU path** (`:205-208`, `_gpu_image_preprocess` `:239-317`): identical algorithm fully
  on-device — `_smart_resize` port (`:211-236`), torchvision `resize` (bicubic+antialias) on the
  CUDA uint8 tensor (the same kernel HF's fast backend calls, `:276-283`), fused rescale+normalize
  (`:285-292`), patchify reshape/permute (`:295-316`). Image never leaves the GPU. Dispatch:
  `:1528-1543`. Parity: grid_thw bit-exact, pixel_values cos>0.9999 (`:198-202`).

### Why I2T TTFT balloons at batch on CPU-preprocess and is fixed on-GPU; why native≈HF on food101

- The 175 ms CPU resize/patchify is, like mel, a **host-CPU cost in `process_prompt`** that does
  not run on GPU → it serializes across the batch the same way (LEVERS / FINDINGS §3:
  "image `process_prompt` CPU resize/patch up to 175 ms ... biggest, most variable I2T cost").
  GPU preprocess removes the round-trip and the host serialization.
- **But the 175 ms is image-size-driven** (FINDINGS lines 170-174): it only materializes at
  ~3000 px. On **food101 (512×512)** CPU preprocess is only **~2 ms**, so at B=1 I2T TTFT barely
  moves (0.238→0.232 s) and **gpu-img is effectively neutral there**. The dominant food101 I2T
  TTFT cost is prefill walks + ~60 ms/walk conductor round-trips, not preprocess.
- Likewise the **native vision encoder ≈ HF at B=1 on small images**: the patch-embed cliff is
  per-image and the encoder forward is 0.55 % of I2S E2E (FINDINGS §3). The native win shows up
  (i) on **large images** (the Conv3d cliff scales with patches) and (ii) at **batch**
  (cross-request varlen vs per-request HF). So for the food101 benchmark specifically, expect
  native-vision and gpu-img to look ~flat at B=1 and to separate from HF only at batch / on
  large-image datasets.

### Effect summary (I2T, I2S)

- **B=1, food101 512px**: native vision ≈ HF; gpu-img ≈ neutral. TTFT dominated by prefill
  walks + conductor round-trips, not encoder/preprocess.
- **B=32 and/or large images**: gpu-img removes the host-CPU resize serialization (up to ~175 ms
  ×B on ~3000 px); native vision keeps the on-GPU encoder cheap (no Conv3d cliff, varlen batch).
  Both lift I2T TTFT / I2S RTF at batch; on food101 the headline win is modest because the dataset
  doesn't exercise the large-image cost.

---

## 4. How vLLM-Omni does the SAME preprocessing (and why its TTFT is flatter)

### vLLM runs mel + image preprocess on CPU too (confirmed from code)

- **Audio mel = HF `WhisperFeatureExtractor` on CPU**, same as M\*:
  `qwen3_omni_moe_thinker.py:45` imports it; `:604-608` `get_feature_extractor` asserts it's a
  `WhisperFeatureExtractor`; `:669-742` `_call_hf_processor` pads to hop_length and calls the HF
  processor (`super()._call_hf_processor`, `:719-724`) which runs the same numpy STFT.
- **Image = HF `Qwen2VLImageProcessor` on CPU**: `qwen3_omni_moe_thinker.py:861`
  `image_processor = self.info.get_image_processor(...)`, merge math `:907`.

So **vLLM pays the same intrinsic CPU mel + CPU image-preprocess cost** — it is *not* doing
GPU mel or GPU image preprocess. (The `torch.stft` GPU mels elsewhere in vllm-omni — e.g.
`qwen3_tts/.../whisper_encoder.py:75`, `glm_tts/voice_clone.py:110` — are the *TTS speech
tokenizer*, a different pipeline, not the Qwen3-Omni thinker's input mel.)

### Why vLLM's TTFT is nonetheless flatter (different prefill scheduling, not cheaper mel)

This is an architectural difference, not a preprocessing one:

1. **MM processing is decoupled from the GPU loop and cached.** vLLM v1 runs the HF processor in
   the front-end input-processing stage, with a multimodal processor cache
   (`config/stage_config.py:463` `mm_processor_cache_gb`) and cached mm outputs
   (`core/prefix_cache.py:48-157`, `mm_outputs_cache` / `mm_cache_keys`), plus async-chunk
   next-stage processing (`stage_config.py:218`, `:799-801`, `:914-927`). So repeated/identical
   mm inputs skip reprocessing and the cost overlaps the engine rather than sitting inline on the
   single GPU worker's admission path the way M\*'s `data_worker.process_prompt` does.
2. **Continuous batching + chunked prefill** keep TTFT flat as B grows: vLLM mixes prefill+decode
   in one step and budgets/chunks prefill (LEVERS_REPORT Lever 2/3:
   `omni_ar_scheduler.py:220`, `omni_generation_scheduler.py:105`, `enable_chunked_prefill`
   `stage_config.py:516`). M\* has the *same-graph_walk-per-batch barrier*
   (`micro_scheduler.py:97-103`) and **no chunked prefill**, so prefills don't interleave and a
   batch's per-request host preprocessing stacks up.

Net: vLLM and M\* share the *same* intrinsic CPU mel/image cost; vLLM hides it behind
front-end decoupling + mm caching + continuous/chunked batching, whereas M\*'s default path
exposes it inline and serially → the balloon. M\*'s **GPU mel / GPU image preprocess remove the
cost entirely** (cheaper than hiding it), which is why M\*-new+GPU-mel can *beat* vLLM on S2S
RTF/TTFT at batch even though vLLM never had a 240 ms wall to begin with — but M\* still trails
vLLM on S2T TTFT at high batch because of the separate prefill-scheduling gap (§1 residual).

*Uncertainty:* I confirmed vLLM's mm-cache and chunked-prefill knobs exist in code, and that the
HF processor is the mel/image path; I did not trace the exact process/thread the front-end runs
on in the deployed vLLM-Omni config, so the "separate process, overlapped" claim is from the v1
architecture + these knobs, not a launch-config read.

---

## Per-path bottom line (TTFT / throughput / RTF, B=1 vs B=32)

| Path | B=1 | B=32 | Dominant lever |
|---|---|---|---|
| **S2T** | TTFT ~0.34 s default, ~0.10 s w/ GPU mel (mel = the gap); native enc ≈ HF | TTFT default ~5.24 s; GPU mel → 12.2× faster TTFT, 2.3–4.4× tput; native varlen holds vs HF dense | **GPU mel** (§1); native enc (§2) secondary |
| **S2S** | RTF mostly Talker+vocoder; GPU mel 1.5× | RTF: GPU mel 3.3×, stays <1 vs baseline 2.0; native varlen vs HF 2.0 RTF | **GPU mel** (§1) + native audio enc (§2) |
| **I2T** | food101 512px: native vision ≈ HF, gpu-img neutral (~2 ms); TTFT = prefill walks+round-trips | gpu-img removes host resize serialization; native vision avoids Conv3d cliff + varlen-batches | gpu-img + native vision (§3), size/batch-gated |
| **I2S** | encoder 0.55 % of E2E; RTF = Talker+vocoder | native vision + gpu-img lift encoder side at batch; RTF still vocoder-bound (LEVERS Lever 1) | §3 for prefill; vocoder chunk for RTF |

**The one decisive audio lever is GPU mel.** The native encoders are a *batch/large-input*
correctness-and-throughput story (same FA2 kernel, cross-request varlen + no Conv3d cliff), not a
B=1 speedup. `MSTAR_VARLEN_BACKEND`/adaptive-τ are inert in prod (flash_attn present). vLLM pays
the same CPU mel/image cost but hides it via front-end decoupling + mm caching + chunked
continuous batching; M\* removes it outright with GPU mel/image preprocess.

Citations are first-hand for code mechanisms; effect magnitudes are from FINDINGS.md /
LEVERS_REPORT.md / the two memory notes (server A/B + microbench), not re-measured here.

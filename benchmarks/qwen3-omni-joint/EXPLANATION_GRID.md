# Why M\*-new wins (and where it doesn't): the code-rooted grid

For each of the 4 paths × {throughput, TTFT, ITL, RTF} × {B=1, B=32}: a **technical** explanation
(rooted in the M\*-new / M\*-old / vLLM-Omni code) and a **one-sentence** plain explanation.
Numbers are p50/mean from the committed `raw_<path>.json` (integrated M\*-new vs vLLM, B=1→32).

**The two facts everything reduces to:**
1. **M\*'s serving engine is byte-identical between M\*-new and M\*-old** (scheduler, CUDA-graph
   engine, plan/replay, Talker, Code2Wav are the same files — `research_engine.md` §0). So every
   M\*-new *vs M\*-old* win is the **encoder/preprocess/vocoder-config** layer, not the engine.
2. **M\* and vLLM sit on opposite ends of one trade-off** (`research_vllm.md` §1–2): vLLM **mixes
   chunked-prefill into every decode step** (`vllm/v1/core/sched/scheduler.py:311-320,345-540`) →
   *flat TTFT* but decode steps get demoted `FULL→PIECEWISE` (eager attention,
   `compilation.py:604-606`, `gpu_model_runner.py:3612-3616`) → *worse ITL/throughput*. M\* enforces
   **one graph-walk per step** (`micro_scheduler.py:107-123,272-277`) → decode/Talker/Code2Wav always
   replay **fixed-size FULL CUDA graphs** captured at `[1,2,4,8,16,32]`
   (`submodules.py:1068,1955,2044`) → *better ITL/throughput/RTF*, but prefill can't piggyback →
   *worse TTFT on the text/prefill axis at batch*.

Legend: ✅ M\* wins · ⚠️ M\* loses · ≈ tie.

---

## S2T — audio → text  (metrics: throughput, TTFT, ITL; RTF n/a)

| metric | B=1 (M\*-new / vLLM) | B=32 (M\*-new / vLLM) |
|---|---|---|
| throughput tok/s | 85 / 57 ✅ | 423 / 305 ✅ |
| TTFT s (p50) | 0.097 / 0.140 ✅ | 0.373 / 0.283 ⚠️ |
| ITL s (mean) | 0.007 / 0.012 ✅ | 0.058 / 0.077 ✅ |

**Throughput · B=1** — *Technical:* the audio encoder forward is <3% of E2E and launch-bound; M\* B=1
decode is a captured FULL graph with plan/replay double-buffering (`NUM_SLOTS=2`,
`cuda_graph_runner.py:152`) hiding FlashInfer `plan()` behind the prior replay, so M\* emits tokens
with less per-token launch overhead than vLLM. *Plain:* M\* spends less time launching GPU work per
token, so even a single request finishes faster.

**Throughput · B=32** — *Technical:* M\* groups the 32 concurrent decodes into **one** batched FULL-graph
replay (`submodules.py:1068`, capture size 32); vLLM, under sustained arrivals, folds incoming prefill
chunks into decode steps which demote to PIECEWISE/eager attention (`compilation.py:604-606`), taxing
per-token cost — so M\* 423 vs 305 tok/s. The CPU→GPU mel move (`MSTAR_GPU_MEL`, `gpu_log_mel`
`qwen3_omni_model.py:64-81`) is what lets B=32 prefill keep up at all (else the ~240 ms CPU mel
serializes, `research_encoders.md` §1). *Plain:* M\* decodes the whole batch in one pre-baked GPU
program while vLLM keeps falling back to slower step-by-step execution.

**TTFT · B=1** — *Technical:* M\* wins because `MSTAR_GPU_MEL` replaces the ~240 ms single-threaded
HF `WhisperFeatureExtractor` CPU mel (`qwen3_omni_model.py:1556-1577`) with a ~0.4 ms `torch.stft`
GPU mel (`:1276-1308`); first-token latency drops to the prefill compute. *Plain:* M\* computes the
audio's spectrogram on the GPU in under a millisecond instead of a quarter-second on the CPU.

**TTFT · B=32** — *Technical:* the one place M\* loses. vLLM's chunked-prefill mixes a new request's
prefill into the running decode step (`scheduler.py:345-540`) so TTFT stays ~flat (0.283 s); M\*'s
same-graph-walk barrier (`micro_scheduler.py:272-277`) makes a prefill walk wait for a decode step,
so first-token latency rises to 0.373 s. gpu-mel removed the *balloon* (was ~5 s), leaving only this
scheduling gap. *Plain:* with many requests in flight, vLLM slips a newcomer's first step in alongside
everyone else's, while M\* makes it wait its turn.

**ITL · B=1 and B=32** — *Technical:* M\* wins both because every decode step is a fixed-size FULL graph
replay (no eager attention), whereas vLLM's default `FULL_AND_PIECEWISE` runs decode attention eagerly
on any mixed step (`gpu_ar_model_runner.py:544`, `cudagraph_dispatcher.py:301-317`). *Plain:* once
M\* is generating, each next token comes out on a pre-recorded fast path; vLLM keeps re-deciding how
to run the step.

---

## I2T — image → text  (metrics: throughput, TTFT, ITL; RTF n/a)

| metric | B=1 (M\*-new / vLLM) | B=32 (M\*-new / vLLM) |
|---|---|---|
| throughput tok/s | 116 / 77 ✅ | 719 / 531 ✅ |
| TTFT s (p50) | 0.309 / 0.151 ⚠️ | 0.760 / 0.205 ⚠️ |
| ITL s (mean) | 0.007 / 0.012 ✅ | 0.036 / 0.057 ✅ |

**Throughput · B=1** — *Technical:* same engine advantage as S2T (FULL-graph decode + plan/replay);
the native vision encoder computes the patch embed as `F.linear` (`vision_encoder.py:106-108`) instead
of HF's bf16 `Conv3d`, but on 512 px food101 that's a small share at B=1 — the win is mostly the decode
engine. *Plain:* M\* launches less GPU overhead per token.

**Throughput · B=32** — *Technical:* M\* batches the 32 decodes into one FULL-graph replay; additionally
the native vision encoder concatenates all requests into one varlen forward
(`NativeVisionEncoderSubmodule`, `submodules.py:286-364`) vs HF's per-request encode
(`/home/tim/mstar/.../submodules.py:192-200`), and `F.linear` avoids the bf16-`Conv3d` cuDNN cliff
(`vision_encoder.py:81-92`). 719 vs 531 tok/s. *Plain:* M\* encodes all the images together and decodes
the batch in one pre-baked program.

**TTFT · B=1** — *Technical:* M\* **loses** (0.309 vs 0.151). On small food101 images both encoders are
cheap and GPU-img-preprocess is ~neutral (`research_encoders.md` §3b); M\*'s first-token path crosses
4 processes with ~60 ms/walk conductor round-trips + sequential prefill walks
(`qwen3_omni_model.py:206-301`), while vLLM's single-engine prefill is tighter. *Plain:* for one small
image, vLLM's leaner prefill path reaches the first word sooner.

**TTFT · B=32** — *Technical:* M\* loses more (0.760 vs 0.205) — the residual honest gap. Image
resize/patchify + prefill are host/serialized in M\*'s prefill walk; vLLM chunk-prefills and
piggybacks so per-request first-token stays flat (`scheduler.py:311-320`). gpu-img-preprocess
(`_gpu_image_preprocess` `:239-317`) removes the host resize but the prefill-scheduling gap remains.
*Plain:* M\* still makes a new image request wait behind the decode batch for its first word; vLLM
doesn't.

**ITL · B=1 and B=32** — *Technical:* M\* wins (B32 0.036 vs 0.057) for the same reason as S2T: pure
FULL-graph decode steps vs vLLM's PIECEWISE-demoted mixed steps. *Plain:* M\* streams each subsequent
token faster because it never leaves its fast pre-recorded path.

---

## I2S — image → speech  (throughput, TTFT, ITL, RTF)

| metric | B=1 (M\*-new / vLLM) | B=32 (M\*-new / vLLM) |
|---|---|---|
| throughput audio s/s | 11.57 / 6.39 ✅ | 94.73 / 47.85 ✅ (~2×) |
| RTF p50 (lower=better) | 0.086 / 0.157 ✅ | 0.322 / 0.655 ✅ (~2×) |
| TTFT s (p50, audio) | 0.447 / 0.558 ✅ | 1.071 / 2.494 ✅ |
| ITL s (mean, audio) | 0.092 / 0.297 ✅ | 0.346 / 1.227 ✅ |

**Throughput / RTF · B=1** — *Technical:* speech wall-time is Talker + Code2Wav; M\* unrolls the 16-RVQ
depth loop into one CUDA graph (`talker.py:446-540`) and runs a graph-captured, fp32 Code2Wav
(`submodules.py:2033,2044`) at `codec_chunk_frames=15` (`config.py:291-292`). vLLM pays ⌈N/25⌉
GPU→CPU→SHM→GPU round-trips between its *separate* Talker and Code2Wav engines
(`shm_connector.py:53-63`) + a code-predictor that re-prefills with no KV cache
(`qwen3_code_predictor.py:114-118`). ~1.8× already at B=1. *Plain:* M\* makes the audio in one place
on a pre-baked path; vLLM keeps shipping data between two engines.

**Throughput / RTF · B=32** — *Technical:* M\* batches Code2Wav **across requests** into one
`[batch,16,T]` FULL-graph replay captured at 32 (`code2wav.py:492-535`, `submodules.py:2044`). vLLM
*also* batches but (a) **zero-pads to the longest request** so cost ∝ batch×max_len
(`qwen3_omni_code2wav.py:307-321`) and (b) its vocoder CUDA graph captures **batch=[1] only**
(`qwen3_omni_code2wav.py:154-158`) → eager at B=32, plus the per-chunk IPC. So M\* ~2× (94.7 vs 47.9
audio s/s; RTF 0.322 vs 0.655) and stays real-time. *Plain:* at batch M\* vocodes everyone together in
one fast program while vLLM falls back to a slow per-step path and wastes work on padding.

**TTFT · B=1 and B=32** — *Technical:* M\* wins both (1.07 vs 2.49 at B=32). First-audio-token needs the
Thinker→Talker→Code2Wav warmup; M\* colocates Talker+Code2Wav in one process with in-process buffers
(no IPC, `FINDINGS.md:128-133`), and gpu-mel/native-vision keep the prefix cheap; vLLM eats encoder +
cross-engine handoff before first sound. *Plain:* M\* produces the first audio sooner because the
speech stages live together instead of messaging across GPUs.

**ITL · B=1 and B=32** — *Technical:* M\* wins large (0.35 vs 1.23 at B=32): per-audio-frame the Talker
depth loop is one graph replay vs vLLM's 16 forwards/frame with a non-graphed (NPU-only graphs,
`qwen3_omni_moe_code_predictor_mtp.py:21-22`) re-prefilling predictor, made worse at batch by the
pad-to-max vocoder. *Plain:* each successive chunk of audio comes out far faster on M\*.

---

## S2S — audio → speech  (throughput, TTFT, ITL, RTF)

| metric | B=1 (M\*-new / vLLM) | B=32 (M\*-new / vLLM) |
|---|---|---|
| throughput audio s/s | 9.58 / 5.59 ✅ | 62.24 / 33.52 ✅ |
| RTF p50 (lower=better) | 0.107 / 0.189 ✅ | 0.501 / 0.778 ✅ (real-time vs degrading) |
| TTFT s (p50, audio) | 0.229 / 0.534 ✅ | 1.299 / 2.316 ✅ |
| ITL s (mean, audio) | 0.072 / 0.239 ✅ | 0.283 / 0.852 ✅ |

**Throughput / RTF · B=1** — *Technical:* same speech engine as I2S, plus the input audio uses gpu-mel
so the prefix is cheap; the smaller `codec_chunk_frames=15` (vs M\*-old 25) lowers TTFA/ITL
(`research_engine.md` §3). RTF 0.107 vs 0.189. The S2S B=1 margin (~1.7×) is smaller than I2S's
because short ~3–4 s answers are startup-dominated. *Plain:* M\* starts and runs the voice pipeline
faster end-to-end on a single clip.

**Throughput / RTF · B=32** — *Technical:* M\* keeps S2S **real-time (RTF 0.501)** at B=32 where vLLM
hits 0.778, via batched FULL-graph Talker decode + cross-request batched Code2Wav and zero IPC; vLLM
compounds pad-to-max + batch=1 vocoder graph + per-25-frame Talker→Code2Wav SHM round-trips
(`research_vllm.md` §4). Throughput 62.2 vs 33.5 audio s/s (~1.85×). (M\*-old, by contrast, blows past
RTF 1.0 at B=32 because its HF dense-O(n²) audio encoder + CPU mel serialize — the native+gpu-mel
combo is what holds it.) *Plain:* under heavy load M\* still generates speech faster than real-time
while vLLM slows down.

**TTFT · B=1 and B=32** — *Technical:* M\* wins both (1.30 vs 2.32 at B=32) — gpu-mel removes the input
mel wall and the colocated talker/vocoder removes the handoff before first sound. *Plain:* M\* speaks
the first syllable sooner.

**ITL · B=1 and B=32** — *Technical:* M\* wins (0.283 vs 0.852 at B=32): one-graph depth loop + batched
vocoder vs vLLM's per-frame multi-forward + eager batched vocoder. *Plain:* each next bit of audio
arrives much faster on M\*.

---

## The honest exceptions (where M\* does not win)
- **I2T TTFT at B=1 and B=32** (0.31/0.76 vs vLLM 0.15/0.21): M\*'s prefill can't piggyback on decode
  steps (same-walk barrier) and image-prefill is the costliest prefix; vLLM's chunked-prefill keeps
  first-token flat. Throughput/ITL still win.
- **S2T TTFT at B=32** (0.373 vs 0.283): same scheduling gap (gpu-mel removed the balloon; the
  residual is the missing piggyback). M\* wins S2T TTFT at B=1.
- These are the documented **future lever**: piggyback / chunked-prefill (a mixed prefill+decode walk +
  a combined CUDA-graph key) — high-risk, deferred to a supervised effort. M\*-old vs M\*-new on image
  is ~tie (the engine is shared; native-vision/gpu-img help mainly at batch / on large images).

*Sources: `research_engine.md`, `research_encoders.md`, `research_vllm.md` (file:line-cited), the
committed `raw_<path>.json`, `FINDINGS.md`, `LEVERS_REPORT.md`.*

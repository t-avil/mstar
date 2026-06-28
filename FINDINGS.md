# Qwen3-Omni on M\* — Findings & Optimization Plan

Investigation of M\* issue **#131 (port Qwen3-Omni encoders to native M\*)** plus a
head-to-head against vLLM-Omni and SGLang-Omni on `Qwen/Qwen3-Omni-30B-A3B-Instruct`
(8×H200). Systems: **M\*-new** (native encoders), **M\*-old** (upstream main, HF-wrapper
encoders), **vLLM-Omni** (v0.21.0). Protocol everywhere: closed-loop max-concurrency
continuous batching, seed=42.

Paths: T2S text→speech, I2S image→speech, S2S audio→speech, S2T/A2T audio→text, I2T image→text.

---

## 1. Headline benchmark numbers

**Figure 5 (Seed-TTS T2S, 2-GPU sweep, B∈{1..32}):** M\* beats vLLM-Omni on **both** RTF
and throughput at every batch, **~2–2.5×** throughput — reproduces the paper's claim.

**B=1 (×50, seed=42) across paths — RTF↓ (speech) / TTFT↓·ITL↓ (text):**

| Path | M\*-new | M\*-old (HF) | vLLM | M\* vs vLLM |
|---|---|---|---|---|
| S2S RTF / tput | 0.172 / 6.03 | 0.206 / 4.81 | 0.193 / 5.62 | **1.1×** |
| I2S RTF / tput | 0.087 / 11.8 | 0.088 / 11.6 | 0.158 / 6.33 | **1.8×** |
| S2T TTFT / ITL | ~0.19 / 0.007 | — | 0.118 / 0.012 | TTFT behind, ITL ahead |
| I2T TTFT / ITL | ~0.20 / 0.007 | — | 0.118 / 0.012 | TTFT behind, ITL ahead |

**Where M\* already wins:** throughput at batch (~2–2.5×), and **ITL** (0.007 vs 0.012).
**Gaps:** **TTFT** (~0.19 vs 0.118) and **B=1/short-audio throughput**.

### Why I2S is ~1.8× but S2S only ~1.1× vs vLLM
RTF = wall / audio_seconds, but a **fixed ~0.2 s per-request startup** (encoder + Thinker
prefill + handoff/preprocess) can't be normalized away. It's ~5% of long I2S audio (~42 s)
→ the true ~1.8× gen-speed advantage shows; it's ~40% of short S2S audio (~3 s) → the
advantage is buried. **Plus** vLLM emits ~2× longer S2S audio (it answers/expands — §4),
amortizing its own startup better and flattering its S2S RTF. Fix = cut the startup cost (§5).

---

## 2. Encoder parity — #131 acceptance #1: ✅ MET

Native encoders are **numerically identical to HF** (= vLLM, which subclasses HF), random-weight
implementation-equivalence test:

| dtype | vision | audio |
|---|---|---|
| fp32 | cos=1.000000, relL2=6.9e-04 | cos=1.000000, relL2=5.4e-04 |
| bf16 | cos=1.000000, relL2=0.0 | cos=0.999925, relL2=1.2e-02 |

0 missing / 0 unexpected weights; DeepStack levels + token counts match across resolutions.
Added **`test/modular/test_qwen3_omni_varlen_backend_parity.py`** (18 cases) asserting every
varlen backend (flash_attn/flashinfer/dense/per_segment/padded/adaptive) matches the dense
reference + FlashInfer head-dim padding exactness — the regression guard for backend changes.

---

## 3. The encoder is NOT the B=1 bottleneck (measured)

- Audio encoder forward = **~16–25 ms = <3% of E2E** (launch-bound: ~99.7% kernel-launch
  overhead, real GPU math ~0.04 ms). Vision encoder = **0.55% of I2S E2E**.
- **B=1 RTF is ~98.6% Talker AR decode + Code2Wav vocoder.**
- **CUDA-graphing the encoder live HURTS** (graph key = exact clip length → cache thrash +
  capture cost lands in measured requests). Confirmed on both audio and vision.
- **"Thinker-prefill graph thrash" hypothesis = DISPROVEN.** `grep -c "cuda-graph miss"` = **0
  across 150 requests**; prefill always replays a captured bucket (`_get_padded_num_tokens`
  bisect-pads; `prefill_audio` aliases `prefill_text`'s graph). The real TTFT cost is genuine
  encoder+prefill compute + **~60 ms per-walk conductor round-trips** + **image `process_prompt`
  CPU resize/patch up to 175 ms** (biggest, most variable I2T cost).

**M\*-new vs M\*-old are structurally tied at B=1** — same Thinker/Talker/Code2Wav, differing only
in the encoder (<1% of E2E). The native encoder's win is a **batch** story: M\*-old's HF encoder
(dense O(n²) attention) degrades to **2.0 RTF @ B=32 (S2S)** while native varlen holds up.

---

## 4. vLLM "answers" vs M\* "transcribes" — root cause = prompt positioning

Not encoder, sampling, repetition_penalty, or EOS. It's **where the audio sits relative to the
instruction**:
- **vLLM** (stock HF chat template): audio **inside the user turn, before** the instruction →
  trained "spoken-query→reply" layout → it **answers** (and answers spoken questions at length).
- **M\*** (`process_prompt`): text-only chat + audio as a **bare block outside** any turn →
  instruction governs → it **transcribes**.

**Proof** (clip4 "How would the papers talk about it?", temp 0, seed 42): audio-then-instruction
→ 84 prompt tokens, 256-tok essay (finish=length); instruction-then-audio → **84 (identical)**
tokens, 9-tok verbatim transcription (finish=stop). Same tokens reordered → opposite behavior.

Only **1/50 librispeech clips** (the lone spoken *question*) runs long; the other 49 (statements)
transcribe identically even in vLLM's layout. The "100 s degeneration" = the Talker **faithfully
vocalizing the Thinker's essay** (~0.63 s/word, linear) — no babble (hence rep_penalty had no
effect). Alignable with **no code change** (reorder instruction-first, or isolate audio in its
own turn). M\* is the faithful transcriber. An env-gated `MSTAR_VLLM_PROMPT_LAYOUT` to make M\*
replicate vLLM's behavior + a full diff (text/audio/tensors) is in progress.

### Latent M\* bugs found (harmless for transcription, worth fixing)
- Audio sentinel token IDs **151647/151648** are labeled `<|audio_bos|>/<|audio_eos|>` in
  `config.py` but are actually `<|object_ref_end|>/<|box_start|>` in the tokenizer; the real
  audio markers are **151669/151670** (what vLLM uses).
- `flatten_messages` ("v1 simplification", `adapters.py:57`) drops multi-turn role structure —
  fine for single-turn, latent for multi-turn.

---

## 5. Optimization plan → 2× vLLM (throughput + TTFT), beat ITL

**Key realization:** M\* already has the fast architecture — separate-process partitions
(cross-request pipeline overlap), non-blocking StreamingGraphEdges + StreamBuffers
(Thinker→Talker→Code2Wav), continuous batching, async plan/replay double-buffering, the Talker's
16-RVQ depth loop unrolled **inside one CUDA graph**, and **Talker+Code2Wav colocated on one
worker** (paper-confirmed, eliminates their inter-process codec IPC). So the wins are mostly
**config + a few targeted changes**.

| # | Change | Target | Effort | M\*-only? |
|---|---|---|---|---|
| 1 | **PD-disaggregation** (`qwen3omni_pd_disaggregated.yaml` exists) — prefill/decode split | TTFT + tput | config | ✓ |
| 2 | `max_concurrent_requests ≥ 32` — saturate the bs-32 Talker decode graph | throughput | config | — |
| 3 | **Code2Wav SEQUENCE PARALLELISM** — shard the vocoder frame-dim across both GPUs (long-audio I2S) | tput/RTF | medium | **✓✓ vLLM can't** |
| 4 | Image `process_prompt` → GPU / overlap (the 175 ms CPU cost) | TTFT (I2T) | medium | — |
| 5 | Merge `prefill_text`+`prefill_audio/vision` into one Thinker walk (drop ~60 ms round-trip) | TTFT | medium | — |
| 6 | Sweep `codec_chunk_frames` 25→15 + Talker TP2 (`full_tp2.yaml`) | ITL | config/low | — |

**M\* can, vLLM can't:** Code2Wav **sequence parallelism** (per-signal frame-dim resharding via
`ShardingConfig.compute_fanout`; Code2Wav is stateless + non-AR + already has a left-context halo)
and **per-walk re-placement** (PD-disagg / colocated / TP2 from the *same* Walk-Graph by config).
vLLM-Omni bakes each stage onto a device — no per-stage sequence-split, no per-phase replacement.

**Adopt from vLLM/SGLang:** prewarm-downstream + background SHM chunk streaming (TTFP), Talker
structural MTP (full RVQ frame/step), stateless re-prefill + `torch.compile(epilogue_fusion=False)`
for the code predictor, batch+graph the vocoder (SGLang's is batch=1 — M\* can beat it).

**First, run the config-only sweep** (PD-disagg vs colocated vs TP2 vs max_concurrent=32) — high
information, zero code — before committing to the Code2Wav-SP build.

---

## 6. Deliverables & status

- **Pushed to fork** (CLAUDE.md git workflow): `bench/qwen3-omni-{i2s,s2s}-vllm` and
  `…-mstar-old`, each merged to `benchmarks`. Each carries `raw.json` + RTF/throughput charts +
  10 audio samples/batch.
- **Committed:** backend-parity unit tests (`test_qwen3_omni_varlen_backend_parity.py`).
- **Pending:** M\*-new branches; 4 comparison charts (I2S/S2S × tput/RTF); the config-sweep;
  Code2Wav-SP; the `MSTAR_VLLM_PROMPT_LAYOUT` equivalence proof.

**Fairness note:** vLLM's longer S2S/I2S audio (it answers vs transcribes) confounds raw RTF —
use **median RTF + length-independent TTFT/ITL** for the headline comparison.

**Infra:** the harness reaps long-lived background-bash servers — launch servers detached with
`setsid` (reparented to init) and unique `--socket-path-prefix` per server to survive + avoid
the shared-`/tmp/mstar` ZMQ collision.

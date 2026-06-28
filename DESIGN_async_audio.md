# Async audio pipeline / encode-prefill overlap — design + measurement

Branch: `exp/async-audio-pipeline` (based on `integration-mnew`, Qwen3-Omni, M*-new).

Goal: cut audio-output latency (RTF, TTFA) and lift throughput on I2S / S2S / T2S by
overlapping pipeline stages and smarter streaming scheduling.

Status of this commit: **measurement implemented, overlap levers designed/scaffolded.**
Default path is byte-identical (all new behavior is env-gated, default OFF).

---

## 0. TL;DR for reviewers

The single most important finding: **M* already pipelines the audio path across
partitions.** Qwen3-Omni is a 3-partition streaming graph (Thinker → Talker →
Code2Wav) and the default config (`configs/qwen3omni.yaml`) places each partition
on a **separate worker/GPU**, connected by `StreamingGraphEdge` + `StreamBuffer`.
So Code2Wav (vocoder) for chunk N **already runs concurrently with Talker decode of
chunk N+1** — the headline VocalNet-style "overlap the vocoder" win is already
captured architecturally. On top of that, M* already has:

- **cross-request vocoder batching** (`Code2WavSubmodule.preprocess` /
  `forward_batched`, CUDA-graph capture for bs ∈ {1,2,4,8,16,32}),
- **streaming/stateful detok** (per-request left-context frame carry-over,
  `chunked_decode_streaming` + `_first_chunk_emitted` / `_latest_seq_len`),
- **within-partition CPU-plan/GPU-execute double-buffering** (`MSTAR_PRE_PLAN_SPEC`,
  `MSTAR_NUM_SLOTS=2`, the speculative pre-plan loop in `worker.py`).

Therefore the remaining **gaps** (highest value first) are:

1. **Sub-lever A — ENCODE→PREFILL overlap (RServe-style).** `prefill_audio` /
   `prefill_vision` are `Sequential([encoder, Thinker])`: the encoder runs to
   completion before Thinker prefill begins. This is the largest *unrealized*
   latency lever for TTFT/TTFA on I2S/S2S/I2T/S2T. **Designed, not implemented**
   (needs chunked encoder output + incremental Thinker prefill; GPU validation
   required).
2. **Sub-lever C — VoxServe-style streaming scheduler.** The conductor is a
   model-driven state machine and the per-worker `MicroScheduler` is
   priority/round-robin by **engine type**, not by **request deadline**. There is
   no "prioritize a request until its first audio chunk, then chunk-duration
   soft-deadline" logic. **Designed, not implemented.**
3. **Sub-lever B — async vocoder/detok overlap.** Largely **already present** for
   the default (disaggregated) topology. The only residual gap is the **colocated**
   topology (`configs/qwen3omni_colocated.yaml` and friends), where a shared worker
   serializes Talker before Code2Wav via `MicroScheduler` engine priority
   (`KV_CACHE=0` < `STATELESS=2`). Closing that means interleaving the vocoder
   between talker decode steps on one GPU — lower value than A/C and only matters
   when colocated.

**Implemented now:** the latency-split measurement (`MSTAR_AUDIO_LATENCY_SPLIT`)
that is the "critical first step" — it tells us how much of RTF is vocoder vs
detok, and (with `MSTAR_PHASE_TIMING` per worker) talker-decode vs vocoder.

---

## 1. Current pipeline map (file:line)

Graph walks & partitions — `mstar/model/qwen3_omni/qwen3_omni_model.py`:
- `get_graph_walk_graphs()` L441–667. Walks:
  - `prefill_audio` = `Sequential([audio_encoder, Thinker])` L485–516 — **encoder
    then Thinker, sequential**.
  - `prefill_vision` = `Sequential([vision_encoder, Thinker])` L518–552 — same.
  - `thinker_decode` Loop L556–586 → streams `thinker_states` to Talker.
  - `talker_prefill` / `talker_last_prefill` / `talker_decode` L592–643 → stream
    `codec_tokens` to Code2Wav.
  - `code2wav_chunk` L646–656 → emits `audio_chunk` to client.
- `get_partitions()` L673–696: Thinker / Talker / Code2Wav, each a partition,
  `producer_partitions` chain Thinker→Talker→Code2Wav.
- `get_partition_topology()` L698+: streaming `Connection`s,
  `FixedChunkPolicy(chunk_size=1)` for thinker_states, left-context chunk policy for
  codec_tokens.

Device placement:
- `configs/qwen3omni.yaml` — default: `Code2Wav` (+ encoders) on rank 0, `Thinker`
  on rank 1, `Talker` on rank 2 → **3 separate workers/GPUs** = real
  cross-partition pipelining.
- `configs/qwen3omni_colocated.yaml`, `qwen3omni_pd_disaggregated.yaml` — alternate
  placements (Thinker+Talker colocated, P/D split).

Streaming machinery:
- `mstar/streaming/stream_buffer.py` `has_chunk_ready` L76–89 — consumer partition
  self-triggers when its buffer holds a full chunk.
- `mstar/streaming/chunk_policy.py` L79–122 — LeftContext policy: first chunk fires
  at `codec_chunk_frames`, later chunks at `+ codec_left_context_frames`.
- `mstar/worker/worker.py` `_poll_stream_buffers` L710+ drives consumer triggers.

Vocoder + detok:
- `Code2WavSubmodule` `mstar/model/qwen3_omni/submodules.py`:
  - `prepare_inputs` L~2058 — eos filter, pad to `full_seqlen`, record
    `_latest_seq_len[rid]`.
  - `preprocess` L~2105 — stack requests → `(bs, Q, T)` (cross-request batching).
  - `forward_batched` L~2126 — `self.code2wav(...)` vocoder + int16 conversion.
  - `postprocess` L~2201 — per-request left-context trim using
    `_first_chunk_emitted` / `_latest_seq_len`.
- `mstar/model/qwen3_omni/components/code2wav.py`:
  - `Qwen3OmniMoeCode2Wav.forward` L474–490 — embedding → pre_transformer →
    upsample stack → decoder ConvNet (torch.compiled in `consolidate`, L460–472).
  - `chunked_decode_streaming` L492–535 — batched decode + per-request trim.

Within-partition overlap (already present):
- `worker.py` speculative double-buffer loop L~2066–2341: CPU preamble + plan(N+1)
  overlap GPU(N); `_pre_plan_for_speculative_batch` L1094, `_execute_on_gpu_thread`
  L1158. Flags: `MSTAR_PRE_PLAN_SPEC` (default ON), `MSTAR_NUM_SLOTS=2`,
  `MSTAR_PHASE_TIMING` (per-iter phase histogram, L2040). **Scope: one partition's
  AR loop**, not cross-partition.

Per-worker scheduler:
- `mstar/worker/micro_scheduler.py` — `PRIORITY = {KV_CACHE:0, STATELESS:2}` L36–39,
  `SchedulingType` PRIORITY / ROUND_ROBIN. Picks ready node groups by engine type;
  **no request-level deadline/priority.**

---

## 2. Expected RTF latency split (from code structure)

We cannot run GPU here, so this is a structural estimate to be confirmed by the
measurement in §3. Reasoning:

- The vocoder (`Qwen3OmniMoeCode2Wav.forward`) is a transformer pre-net + a deep
  transposed-ConvNet upsample/decoder stack with total upsample 1920× — it expands a
  short codec-frame chunk into ~1920 samples/frame of fp32 audio. This is the
  classic dominant cost in VocalNet/Qwen-Omni-class stacks (~70% of audio RTF in
  the VocalNet literature).
- Detok (`(wav.clamp*32767).to(int16)` + per-request slice trim) is an elementwise
  cast + a view-slice: **cheap, single-digit % of the chunk**.
- Talker decode is one autoregressive transformer step per codec frame; per *frame*
  it is cheaper than the vocoder *chunk* it feeds, but it runs `codec_chunk_frames`
  times per vocoder chunk, so per *chunk-of-audio* talker-decode and vocoder are the
  same order of magnitude and the ratio depends on `codec_chunk_frames`.

**Predicted split per emitted audio chunk (to verify):**
`vocoder_fwd` ≫ `detok_int16` (expect detok < 5%); talker-decode-per-chunk
comparable to vocoder, with vocoder the larger single op. Because the two run on
**separate GPUs concurrently** in the default config, the **steady-state RTF is
bounded by max(talker-chunk, vocoder-chunk)**, not the sum — which is exactly why
the already-present cross-partition pipeline is the dominant win and why sub-lever B
has little headroom left in the disaggregated topology.

Implication for prioritization: with B already realized, **TTFA/TTFT** is gated by
the **serial prologue** — encode → Thinker prefill → first thinker token → talker
prefill → first codec chunk → first vocoder chunk. That prologue is where
sub-lever A (encode→prefill overlap) and sub-lever C (prioritize-to-first-chunk)
pay off. Hence A and C are higher-value than further B work.

---

## 3. Measurement plan (the critical first step) — IMPLEMENTED

Two complementary, env-gated instruments, both default OFF and byte-identical when
off:

### 3a. Vocoder vs detok split — `MSTAR_AUDIO_LATENCY_SPLIT` (NEW, this commit)
`Code2WavSubmodule.forward_batched` (`submodules.py`). Set
`MSTAR_AUDIO_LATENCY_SPLIT=<period_in_chunks>` (e.g. 100). Every `period` chunks it
logs:
```
Code2Wav audio-latency-split iter=N: detok_int16: p50=.. p95=.. mean=.. n=.. |
    vocoder_fwd: p50=.. p95=.. mean=.. n=.. | batch_size: mean=.. n=..
```
When the flag is > 0 it inserts `torch.cuda.synchronize` around the vocoder forward
and the int16 conversion so wall-clock attributes correctly to each region (the
syncs perturb absolute throughput slightly — it is a diagnostic mode, documented as
such). When the flag is 0 (default) none of that code runs.

### 3b. Per-partition step time — `MSTAR_PHASE_TIMING` (EXISTING, reuse)
`worker.py` L2040. Because each partition is its own worker in the default config,
running with `MSTAR_PHASE_TIMING=200` gives, **per worker**, the `iter_total` /
`await_gpu` histogram = that partition's per-step GPU time:
- Talker worker → talker-decode step time,
- Code2Wav worker → vocoder chunk step time,
- Thinker worker → prefill/decode step time.

Combined, 3a + 3b give the full **talker-decode vs vocoder vs detok** split the task
asked for, plus the encode/prefill prologue cost.

---

## 4. Sub-lever designs (scaffolded, NOT implemented)

### A. ENCODE → PREFILL overlap (RServe-style) — highest unrealized value
Today `prefill_audio = Sequential([audio_encoder, Thinker])`: full encode, then
prefill. Proposed: chunk the encoder input along time, emit `audio_embeds` per
segment over a `StreamingGraphEdge`, and let the Thinker prefill consume segments
incrementally (it is already the streaming-producer pattern used Thinker→Talker).
- Graph change: replace the `Sequential` with an encoder `Loop`/segmented node that
  streams `audio_embeds` chunks to a Thinker node that prefills incrementally.
- Lossless: identical math, only the scheduling boundary moves.
- Proposed flag: `MSTAR_ENCODE_PREFILL_OVERLAP=1` (default OFF) selecting the
  streamed graph in `get_graph_walk_graphs()`.
- Expected gain: removes encode time from the TTFT critical path (hides it under
  prefill). On long audio prompts (S2S/S2T) and large images (I2S/I2T) this is the
  bulk of the prologue. Estimate: TTFT/TTFA down by ~encode_time − overlap_residual.
- Risk / why not done here: needs chunk-boundary correctness for encoder attention
  masks + position IDs and GPU validation against the sequential baseline (parity).
  Not safe to land blind.

### C. VoxServe-style streaming scheduler — high value under load
Add request-level scheduling on top of the engine-type priority:
1. **Prioritize to first audio chunk:** any request that has not yet emitted its
   first `audio_chunk` gets boosted priority across Thinker/Talker/Code2Wav so TTFA
   is minimized; after the first chunk it drops to normal.
2. **Chunk-duration soft deadline:** once streaming, each request must produce the
   next audio chunk before the previously emitted chunk finishes playing
   (`chunk_frames / sample_rate` of wallclock). Schedule by earliest deadline; this
   pairs with the existing cross-request vocoder batching (batch requests whose
   deadlines are close).
- Hook points: `mstar/worker/micro_scheduler.py` (per-worker ready-node ordering)
  and `mstar/conductor/conductor.py` `_process_done_forward` (global state machine)
  — add a per-request priority/deadline field carried in `request_info`.
- Proposed flag: `MSTAR_STREAM_SCHED=voxserve` (default = current behavior).
- Expected gain: lower TTFA tail and fewer audio underruns at high concurrency; net
  throughput from tighter vocoder batch formation.
- Risk / why not done here: touches the shared scheduler hot path across all models;
  needs load testing on GPU to avoid starvation/regressions.

### B. Colocated async vocoder/detok overlap — residual, low priority
Only relevant for colocated placements. Today `MicroScheduler` runs Talker
(KV_CACHE, prio 0) before Code2Wav (STATELESS, prio 2) on a shared worker, so the
vocoder waits. Option: detok overlap (run int16 cast / trim of chunk N on a side
stream while the vocoder/talker does N+1) and/or a colocated interleave policy. Low
value because the default topology already overlaps via separate workers.

---

## 5. GPU validation commands (run on the GPU box)

Latency split (critical first step), I2S / S2S / T2S:
```bash
# Vocoder vs detok split (Code2Wav worker logs the histogram)
MSTAR_AUDIO_LATENCY_SPLIT=100 <serve-cmd> ...      # then run an I2S/S2S/T2S workload
# Per-partition step time: talker-decode vs vocoder vs prefill (each worker logs)
MSTAR_PHASE_TIMING=200 <serve-cmd> ...
# Both together
MSTAR_AUDIO_LATENCY_SPLIT=100 MSTAR_PHASE_TIMING=200 <serve-cmd> ...
```
Grep results: `Code2Wav audio-latency-split` (vocoder_fwd vs detok_int16) and
`Worker N phase-timing` (per-partition iter_total/await_gpu).

A/B/RTF + TTFA comparison once a lever is implemented (flag ON vs OFF):
```bash
# Baseline (all flags OFF — byte-identical default path)
<bench-harness> --task {i2s,s2s,t2s} --metrics ttfa,ttft,rtf,throughput   # OFF
# Encode->prefill overlap ON
MSTAR_ENCODE_PREFILL_OVERLAP=1 <bench-harness> --task {i2s,s2s,t2s} --metrics ttfa,ttft,rtf,throughput
# VoxServe scheduler ON (under concurrency)
MSTAR_STREAM_SCHED=voxserve <bench-harness> --task {i2s,s2s,t2s} --concurrency 32 --metrics ttfa,rtf,throughput
```
Acceptance: flag-ON must match flag-OFF on output tokens/audio (parity / losslessness)
and improve TTFA/TTFT (A) or TTFA-tail + throughput under load (C); RTF unchanged or
better. Use the existing parity tests (`test/`) to confirm byte-identical audio.

---

## 6. Implemented vs stubbed

| Item | State |
|---|---|
| Vocoder-vs-detok latency split (`MSTAR_AUDIO_LATENCY_SPLIT`) | **Implemented** (`submodules.py`, py_compile OK) |
| Per-partition step timing (`MSTAR_PHASE_TIMING`, reuse) | Already present |
| Cross-partition pipeline (Talker∥Code2Wav) | Already present (separate-worker topology) |
| Cross-request vocoder batching | Already present |
| Stateful streaming detok (left-context carry) | Already present |
| A: encode→prefill overlap | **Designed / scaffolded** (this doc) |
| C: VoxServe streaming scheduler | **Designed / scaffolded** (this doc) |
| B: colocated vocoder interleave | Designed (low priority) |

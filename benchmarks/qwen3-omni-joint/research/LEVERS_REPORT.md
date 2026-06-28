# Qwen3-Omni on M\* — Throughput / Talker / Vocoder Optimization Levers

Read-only comparison of M\* (`/home/tim/mstar`) vs vLLM-Omni (`/home/tim/baselines/vllm-omni`)
and SGLang-Omni (`/home/tim/baselines/sglang-omni`). Goal: concrete, actionable levers to make
M\* faster across batch sizes (B=4..32), focused on throughput, the Talker, and the Code2Wav vocoder.

Cross-checked against `/home/tim/mstar/HANDOFF.md` and `FINDINGS.md` — each lever is tagged with
whether it is already done / ruled out / still open.

**Scope note / honesty:** M\* is already architecturally strong. It already has: plan/replay
double-buffering + CPU/GPU overlap (the equivalent of SGLang's overlap scheduler), cross-request
batching with CUDA-graph buckets up to B=32, the 16-RVQ Talker depth loop unrolled into ONE graph,
and **cross-request Code2Wav batching that both baselines lack**. So the biggest wins are (a) one
config knob (vocoder chunk size at batch), (b) removing the same-walk batching barrier, and (c)
chunked prefill. Several "obvious" levers are already exhausted — flagged below so nobody re-burns GPU on them.

---

## TL;DR — ranked by expected throughput impact at B=4..32

| # | Lever | Path(s) | Impact | Risk | Status |
|---|---|---|---|---|---|
| 1 | **Vocoder chunk size adaptive to batch** (M\* 15/25 vs vLLM 300) | I2S, S2S, T2S | **High** | Low–Med | Open (only B=1 tuned) |
| 2 | **Drop the same-graph_walk-per-batch barrier** → allow prefill+decode mixing / piggyback | all | **High** | High | Open, not tried |
| 3 | **Chunked prefill in the Thinker** (long audio/vision) | S2S, S2T, I2S | Med–High | High | Open, not tried |
| 4 | **Saturate the Talker decode graph** (`max_concurrent_requests ≥ 32`) + verify batch fill | all S2S/I2S/T2S | Med–High | Low | Partially noted; verify it actually batches |
| 5 | **Encoder cross-request micro-batch coalescing** (SGLang-style wait window) | batch S2T/S2S/I2T | Med | Med | Open; matters most native-vs-HF |
| 6 | **Prefix caching for the Thinker** (shared system+instruction prefix) | I2T, S2T at batch | Low–Med | Med | Open; M\* has no prefix cache |
| 7 | **Recalibrate `adaptive` varlen backend τ** for audio's many ~50-tok windows | encoder at batch | Low–Med | Low | Open (HANDOFF Phase 4) |
| 8 | Talker speculative multi-FRAME decode (MTP across time) | S2S/I2S | Low | High | **Speculative** — likely not worth it |
| — | Overlap scheduler / double-buffering | all | — | — | **Already done** (don't redo) |
| — | B=1 placement (colocated / PD-disagg / TP2) | all | — | — | **Ruled out at B=1** (FINDINGS §5) |

---

## Lever 1 — Vocoder chunk size should scale with batch (biggest easy win)

**(a) What it is.** M\* runs Code2Wav with very small streaming chunks tuned for B=1 latency. vLLM
runs it with `chunk_size=300`. Small chunks at batch means many tiny, launch-bound vocoder
invocations and poor GPU utilization. Since M\* *already* batches the vocoder across requests, a
larger chunk in throughput-mode compounds with cross-request batching for a big win on the
audio-output paths.

**(b) Evidence.**
- vLLM default chunk is **300 frames**: `/home/tim/baselines/vllm-omni/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py:216-263` (`chunked_decode(codes, chunk_size=300, left_context_size=...)`). Qwen3-TTS uses `_decode_chunk_frames = 300`, `_decode_left_context_frames = 25` (`.../qwen3_tts/qwen3_tts_code2wav.py:67-68`) and even does **variable-length bucket batching** of chunks (`qwen3_tts_code2wav.py:69-71`).
- SGLang uses `stream_chunk_size = 10`, `left_context_size = 25`, and is **batch=1** (`/home/tim/baselines/sglang-omni/sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py:60-61, 246`) — so M\* can beat SGLang here regardless.
- M\* default is **25** with 25-frame left context (`/home/tim/mstar/mstar/model/qwen3_omni/config.py:280-292`), and the `codec-chunk` branch lowered it to **15** for B=1 TTFA. That is a *latency* optimization; it is the wrong direction for batch throughput.

**(c) Where it goes in M\*.**
- Default: `/home/tim/mstar/mstar/model/qwen3_omni/config.py:291-292` (`codec_chunk_frames`, `codec_left_context_frames`).
- Policy wiring: `/home/tim/mstar/mstar/model/qwen3_omni/qwen3_omni_model.py:467-470` (`LeftContextChunkPolicy(chunk=..., left_context=...)`).
- Batched vocoder forward already exists: `/home/tim/mstar/mstar/model/qwen3_omni/components/code2wav.py:492-535` (`chunked_decode_streaming`, takes `left_context_size` per request, trims per request). It already accepts `[batch, num_quantizers, T]`.

**(d) Impact + risk.** High for I2S/S2S/T2S throughput at B≥8 — fewer, larger, fully-batched
vocoder launches. Risk: low–medium. **Constraint from FINDINGS:** keep `chunk ≥ left_context`
(15/15 is safe; naive 25→15 with lc=25 caused a −10 pop stride and ~38% audio corruption). The
clean design is **batch-adaptive chunking**: small chunk (15) when the vocoder batch is small
(B=1 latency), large chunk (e.g. 150–300) once several requests are co-vocoding. Parity gate:
boundary-focused waveform A/B (a seam is audible), same as the Code2Wav-SP plan.

**(e) Status.** Only the B=1 direction (25→15) has been explored (`codec-chunk` branch ✅).
The large-chunk-for-throughput direction is **open and untried** and is the cleanest high-impact knob.

---

## Lever 2 — Remove the "same graph_walk per batch" barrier (enables piggyback)

**(a) What it is.** M\*'s micro-scheduler can only batch requests that are on the *identical*
graph_walk. It picks the single most common walk and leaves everyone else in the queue. So a
request doing a prefill walk cannot share a step with requests doing a decode walk, and the
audio/vision/text prefill walks cannot co-batch. This is the structural reason throughput can sag
when requests are not in lockstep (staggered arrivals, mixed paths). vLLM and SGLang both **mix
prefill + decode in one step** (chunked prefill + continuous batching), which keeps the decode
batch full while prefills stream in.

**(b) Evidence.**
- M\* barrier: `/home/tim/mstar/mstar/worker/micro_scheduler.py:97-103` — *"Enforce same graph_walk
  for the entire batch. Pick the most common graph_walk… remaining requests stay in the queue."*
  Selection + clipping at `micro_scheduler.py:261-277`.
- M\* separate prefill walks (no mixing, no chunking): `/home/tim/mstar/mstar/model/qwen3_omni/qwen3_omni_model.py:206-301` — `prefill_text`, `prefill_audio`, `prefill_vision` are distinct sequential walks.
- vLLM mixes: `/home/tim/baselines/vllm-omni/vllm_omni/core/sched/omni_ar_scheduler.py:220` (`super().schedule()` inherits vLLM v1 continuous batching that mixes prefill+decode) and chunked-prefill gate at `omni_generation_scheduler.py:105`.
- SGLang mixes via `enable_mixed_chunk`: `/home/tim/baselines/sglang-omni/sglang_omni/scheduling/omni_scheduler.py:201` (`self.is_mixed_chunk = ... and server_args.enable_mixed_chunk`).

**(c) Where it goes in M\*.** `micro_scheduler.py:_select_node_priority` (lines 78-105) and the
entry-collection loop (`get_next_batch`, lines 227-307). Two sub-options, increasing difficulty:
(i) batch *decode-walk* requests together more aggressively (already same walk — verify it isn't
fragmenting by `requires_cfg` or per-request graph id); (ii) allow a captured "mixed" graph that
admits both a prefill chunk and decode tokens (true piggyback) — large change, needs a new graph
key in `cuda_graph_runner.py:99-103` (`CudaGraphKey(graph_walk, requires_cfg, bs, num_tokens)`).

**(d) Impact + risk.** High at B≥8 with realistic (non-lockstep) traffic — this is the classic
continuous-batching win the paper claims. Risk: **high** — touches the scheduler invariant and
CUDA-graph keying; parity and the 18-case backend test must stay green. Recommend prototyping
option (i) first (cheap) and measuring batch-fill before attempting true piggyback.

**(e) Status.** Open, not tried. HANDOFF Phase 3 assumes the closed-loop lockstep workload where
this barrier is mostly hidden; under staggered arrival it will bite. Worth an A/B with arrival jitter.

---

## Lever 3 — Chunked prefill in the Thinker

**(a) What it is.** Split a long audio/vision/text prefill into token-budgeted chunks across
scheduler steps, so a 2k-token S2S prefill does not monopolize the Thinker partition and stall
other requests' work. Both baselines do this; M\* does not.

**(b) Evidence.**
- vLLM: `enable_chunked_prefill` (`/home/tim/baselines/vllm-omni/vllm_omni/config/stage_config.py:516`), budget gate at `omni_generation_scheduler.py:105`, token budget at `:65`.
- SGLang: `chunked_prefill_size` + `PrefillAdder` (`/home/tim/baselines/sglang-omni/sglang_omni/scheduling/sglang_backend/prefill.py:84-92`), tracked at `omni_scheduler.py:196-201`. (Note: SGLang *disables* it for the talker — `talker_scheduler.py:35` — chunked prefill is a Thinker-side lever, not a Talker one.)
- M\*: prefill is one forward per walk over all tokens; buckets exist but no splitting — `submodules.py:709` (`PREFILL_TOKEN_BUCKETS = [128,256,512,1024,2048]`), walks at `qwen3_omni_model.py:206-301`.

**(c) Where it goes in M\*.** Thinker prefill submodule `submodules.py` (token-bucket region
~`:709-720`) plus the walk dispatch in `qwen3_omni_model.py:206-301`; scheduler would need to
re-enqueue the remaining prefill chunk (`micro_scheduler.get_next_batch`).

**(d) Impact + risk.** Medium–high for long-input paths (S2S/S2T/I2S) at batch; low for short
inputs. Risk: **high** (interacts with M-RoPE position bookkeeping and KV append across chunks).
Note the mitigating factor: M\*'s Thinker is on a **separate worker** from Talker+Code2Wav, so a
long prefill stalls only Thinker-side work, not the audio generation pipeline — which *reduces* the
upside vs a monolithic engine. Measure the Thinker-partition stall before committing.

**(e) Status.** Open, not tried. Lower priority than 1 & 2 given the separate-worker mitigation.

---

## Lever 4 — Saturate the Talker decode graph; confirm cross-request batching actually fills

**(a) What it is.** M\* *can* batch the Talker AR decode across requests (graphs captured for
B=1,2,4,8,16,32). The lever is operational: make sure `max_concurrent_requests ≥ 32` so the B=32
decode graph is actually exercised, and verify the scheduler is grouping multiple requests'
Talker-decode steps into one batched replay rather than running B=1 graphs back-to-back.

**(b) Evidence.**
- M\* Talker batches: capture sizes `[1,2,4,8,16,32]` (`/home/tim/mstar/mstar/model/qwen3_omni/submodules.py:880`), `TalkerSubmodule.MAX_BATCH_SIZE = 32` (`submodules.py:1212`), depth loop unrolled in one graph (`components/talker.py:446-549`).
- Scheduler groups across requests: `micro_scheduler.py:227-307` (collects all ready requests for a node, batches up to `max_batch_size`).
- Admission cap: `/home/tim/mstar/mstar/conductor/conductor.py:223-225` (`max_concurrent_requests` from YAML), drain at `:573-602`.
- vLLM/SGLang both batch the talker the same way (vLLM `talker_mtp` graphs `gpu_ar_model_runner.py:212-267`; SGLang batched predictor `talker.py:1089-1098`). So this is parity, not a deficit — but only if M\*'s batch actually fills.

**(c) Where it goes.** YAML `max_concurrent_requests` (read at `conductor.py:223`); verify with a
batch-fill counter around the Talker replay in `worker.py` decode path / `kv_cache_engine.py:829-856`.

**(d) Impact + risk.** Medium–high if the batch is currently under-filling (e.g., requests
desync'd by the same-walk barrier — ties into Lever 2). Low risk (config + instrumentation).

**(e) Status.** HANDOFF lists `max_concurrent_requests ≥ 32` as a known config lever (FINDINGS §5,
row 2) but **the actual Talker batch-fill at B=32 has not been verified** — do this first in Phase 3;
it's the cheapest way to find out if Lever 2 is needed.

---

## Lever 5 — Encoder cross-request micro-batch coalescing

**(a) What it is.** Coalesce several requests' audio/vision encoder forwards into one batched call
with a short wait window, instead of relying on incidental same-walk grouping. SGLang does this
explicitly with a dedicated encoder micro-batcher.

**(b) Evidence.**
- SGLang: `SimpleScheduler(..., max_batch_size=32, max_batch_wait_ms=50, max_batch_cost=10GB)` for image and audio encoders (`/home/tim/baselines/sglang-omni/sglang_omni/models/qwen3_omni/stages.py:844-850, 854-917`).
- vLLM: batched `_execute_mm_encoder` + `encoder_cache` shared across the batch (`/home/tim/baselines/vllm-omni/vllm_omni/worker/gpu_model_runner.py:1396-1402`).
- M\*: encoders run inside the prefill_audio/prefill_vision walks (`qwen3_omni_model.py:1487-1532`, hardcoded `flash_attention_2`); batching is whatever the general scheduler happens to group, no dedicated coalescing window.

**(c) Where it goes in M\*.** A coalescing window in `micro_scheduler.get_next_batch` for encoder
nodes, or a prefill-walk batcher around `qwen3_omni_model.py:206-301`.

**(d) Impact + risk.** Medium for the **native-vs-HF batch story** (acceptance #2): the native
varlen encoder holds at B=32 while HF's dense O(n²) degrades to 2.0 RTF (FINDINGS §3). For
native-vs-native throughput the impact is smaller because FINDINGS measured the encoder at
<3% of E2E and launch-bound at B=1. Risk: medium (adds latency via the wait window). Use a small
window (e.g. ≤ one decode step) so it doesn't hurt TTFT.

**(e) Status.** Open. Most valuable as evidence for acceptance #2, less as a raw speedup.

---

## Lever 6 — Prefix caching for the Thinker (shared system + instruction prefix)

**(a) What it is.** Cache and reuse the KV of the shared prefix (system prompt + identical
instruction) across a batch of requests, so it is computed once. Audio/image content differs per
request, so only the leading text prefix is reusable.

**(b) Evidence.**
- M\* has **no** prefix cache: KV store is request-scoped (`/home/tim/mstar/mstar/engine/kv_store.py:394-482`); the only hook is a no-op `flush_to_store()` at `kv_store.py:451`.
- vLLM has it but **off by default** (`/home/tim/baselines/vllm-omni/vllm_omni/engine/stage_init_utils.py:759`, `enable_prefix_caching=False`).
- SGLang RadixCache (`/home/tim/baselines/sglang-omni/sglang_omni/scheduling/sglang_backend/cache.py:21-33`), but the **talker disables radix** (`talker_scheduler.py:34`) — so prefix caching is a Thinker-only lever there too.

**(c) Where it goes in M\*.** New cache layer behind `kv_store.py` (the `flush_to_store` hook at
`:451`) + a prefix-match check in `cache_manager.py` before Thinker prefill.

**(d) Impact + risk.** Low–medium: Qwen3-Omni prompts are short (84–145 tokens per FINDINGS §4),
so the reusable text prefix is small relative to audio/vision tokens — limited savings. Higher for
I2T/S2T at batch with a long shared instruction. Risk: medium (correctness of partial-prefix reuse
with M-RoPE positions). **Both baselines keep it off by default for omni**, which is a signal the
payoff is marginal here.

**(e) Status.** Open but **deprioritized** — short prompts cap the upside; the same-walk barrier
(Lever 2) and vocoder chunk (Lever 1) dominate.

---

## Lever 7 — Recalibrate the `adaptive` varlen-attention backend threshold

**(a) What it is.** M\* selects a varlen attention backend; the `adaptive` heuristic switches on a
work threshold (τ≈5e5 per HANDOFF). Audio produces many small ~50-token windows, for which the
threshold is likely miscalibrated, picking a sub-optimal backend at batch.

**(b) Evidence.**
- M\* backend set is real and parity-tested: flash_attn / flashinfer / dense / per_segment / padded /
  adaptive (HANDOFF, `test/modular/test_qwen3_omni_varlen_backend_parity.py`, 18 cases; FINDINGS §2).
  Encoders currently pin `flash_attention_2` (`qwen3_omni_model.py:1487-1532`).
- Baselines select varlen flash per backend selector (vLLM `diffusion/attention/selector.py:95`;
  SGLang varlen `cu_seqlens` path `ming_omni/components/vision_encoder.py:366-387`).

**(c) Where it goes.** The `MSTAR_VARLEN_BACKEND` selection / adaptive-τ logic (component attention
path, `model/qwen3_omni/components/attention.py` + the varlen dispatch). Keep the 18-case parity green.

**(d) Impact + risk.** Low–medium, encoder-only; encoder is <3% of E2E so the absolute win is
small except where HF-old degrades. Low risk (env-gated, parity-tested).

**(e) Status.** HANDOFF Phase 4. Open; do it for the backend-curve story, not for a headline speedup.

---

## Lever 8 — Talker speculative multi-FRAME decode (SPECULATIVE — likely not worth it)

**(a) What it is.** Predict multiple *time-step* audio frames per Talker forward (true MTP across
time), not just the within-frame RVQ depth.

**(b) Evidence / why it's weak.** What vLLM calls "talker MTP" and what SGLang's predictor does is
the **within-frame residual-code** prediction (layer-0 → 15 residual codes), which M\* **already**
does and already unrolls into one CUDA graph (`talker.py:446-549`; vLLM `gpu_ar_model_runner.py:212-267`;
SGLang `talker.py:1223-1309`). Neither baseline does cross-time speculative frame decode for the
Qwen3-Omni talker. So M\* is **not** leaving within-frame talker throughput on the table.
Cross-time speculation would need a draft model + verification and risks audio quality.

**(c) Where it would go.** `components/talker.py` AR loop — but would require a verifier and new graphs.

**(d) Impact + risk.** Low expected throughput / high risk + audio-quality jeopardy.

**(e) Status.** Speculative; recommend **not** pursuing unless 1–4 are exhausted.

---

## Already done / ruled out (do NOT re-spend GPU here)

- **Overlap scheduler / CPU-GPU double-buffering** — M\* already has it: 2 slots
  (`cuda_graph_runner.py:154`, `MSTAR_NUM_SLOTS=2`), `plan_executor` + `gpu_executor` with
  plan(N+1)∥replay(N) (`worker.py:2008-2038, 2206, 2417`; `kv_cache_engine.py:953-1035`). This is
  the equivalent of SGLang's overlap scheduler (`omni_scheduler.py:925-968`). **Parity, not a gap.**
- **Cross-request Code2Wav batching** — M\* already batches the vocoder across requests
  (`code2wav.py:492-535`); SGLang is batch=1 (`code2wav_scheduler.py:246`), vLLM is per-call. M\* is
  ahead — the lever is chunk *size* (Lever 1), not whether to batch.
- **B=1 placement** (colocated / PD-disaggregated / TP2) — FINDINGS §5 ruled these out at B=1
  (default `qwen3omni_2gpu` already optimal; colocated −12..−36%; PD-disagg regresses + needs 3 GPUs).
  PD-disagg may still help *throughput* at batch, but it is a Phase-3 throughput experiment, not a new idea.
- **CUDA-graphing the encoder live** — DISPROVEN, it HURTS (graph key = clip length → cache thrash),
  FINDINGS §3. Don't.
- **merge-prefill-walks / prefill bucket padding** — no TTFT win (HANDOFF: `merge-prefill-walks` ✅ negative).

---

## Recommended order for the integration phase

1. **Verify Talker batch-fill at B=32** (Lever 4, cheap) — tells you whether you even need Lever 2.
2. **Batch-adaptive vocoder chunk** (Lever 1, config + small code) — highest impact-per-effort on I2S/S2S/T2S.
3. **Prototype scheduler decode-batch consolidation** (Lever 2 option i) under staggered arrival; measure batch-fill.
4. Only if 1–3 leave throughput on the table: true piggyback (Lever 2 option ii) and/or Thinker chunked prefill (Lever 3).
5. Side quests for the #131 story (not headline speedups): encoder coalescing (Lever 5, acceptance #2) and varlen-τ (Lever 7).

All changes env-gated, default OFF, byte-identical baseline; 18-case varlen parity + encoder-vs-HF
parity green; boundary-focused waveform A/B for any vocoder change. A win only counts if ≥10% over
BOTH M\*-old and vLLM with parity green (HANDOFF methodology §5).

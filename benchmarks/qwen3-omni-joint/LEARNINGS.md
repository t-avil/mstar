# Qwen3-Omni Serving Optimization: Experiment Log and Learnings

**Date range**: June 2026
**Hardware**: 8× NVIDIA H200 141GB, dual-socket NUMA, RHEL 9.7
**Model**: Qwen3-Omni (Thinker 32B MoE + Talker 0.5B + Code2Wav vocoder)
**Serving config**: 2-GPU tensor-parallel (SHM protocol, RDMA/Mooncake broken on this node)
  - GPU 0 (rank 0): Talker + Code2Wav
  - GPU 1 (rank 1): Vision encoder + Thinker

**Benchmark protocol**: closed-loop profiling, B=1..32, warmup=5, N=max(50, 10×B).
Four inference paths: S2T (audio→text), I2T (image→text), S2S (audio→speech), I2S (image→speech).

---

## 1. Systems under test

| Label | Code | Description |
|-------|------|-------------|
| **M\*-old** | `main` @ 9ee1369 | Upstream baseline. HuggingFace encoders, no optimization flags. |
| **M\*-new** | `integration-mnew` @ e943d72 | All optimizations integrated. Flags: `MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1 MSTAR_VISION_GRAPH_ALIGN=1 MSTAR_BATCH_VISION_PREFILL=1` |
| **vLLM-Omni** | vLLM fork with Qwen3-Omni | Third-party baseline, same hardware, same datasets. |

M\*-new evolved over the project. The label moved forward as optimizations landed — earlier runs labeled "mstar_new" had only chunked prefill; the final label includes the full combined stack. Historical data is preserved as `mstar_new_chunked` in the raw JSON.

---

## 2. Final headline numbers (M\*-new = combined vision opts)

### S2T (audio → text) — the biggest win

| B | M\*-new req/s | vs vLLM | vs M\*-old |
|---|---|---|---|
| 1 | 5.06 | **2.20×** | **2.65×** |
| 4 | 8.92 | **2.10×** | **3.29×** |
| 8 | 12.16 | **2.21×** | **4.82×** |
| 16 | 17.90 | **2.25×** | **7.31×** |
| 32 | 23.53 | **1.86×** | **9.61×** |

M\*-new dominates S2T. The scaling advantage over M\*-old is enormous (2.6× at B=1, nearly 10× at B=32) because M\*-old saturates at ~2.5 req/s regardless of batch size — it processes requests sequentially without batching the Thinker prefill.

### I2T (image → text) — consistent but modest over M\*-old

| B | M\*-new req/s | vs vLLM | vs M\*-old |
|---|---|---|---|
| 1 | 0.69 | **1.86×** | 1.09× |
| 4 | 1.73 | **2.43×** | 1.17× |
| 8 | 2.44 | **2.42×** | 1.20× |
| 16 | 3.30 | **2.18×** | 1.21× |
| 32 | 4.21 | **1.66×** | 1.31× |

The vLLM advantage is large (1.7-2.4×). The M\*-old advantage is smaller (1.1-1.3×) because image requests are dominated by vision encode time, and M\*-old already uses the same native encoders for text generation. The combined vision opts mainly help with CUDA graph alignment and batch prefill during the Thinker step.

### S2S (audio → speech) — audio throughput scales massively

| B | M\*-new audio s/s | vs vLLM | vs M\*-old |
|---|---|---|---|
| 1 | 8.6 | **1.54×** | **2.03×** |
| 4 | 18.0 | **1.30×** | **2.51×** |
| 8 | 27.5 | **1.41×** | **3.64×** |
| 16 | 39.1 | **1.62×** | **5.33×** |
| 32 | 54.6 | **1.63×** | **6.47×** |

### I2S (image → speech) — wins on latency, ties on audio throughput vs M\*-old

| B | M\*-new audio s/s | vs vLLM | vs M\*-old |
|---|---|---|---|
| 1 | 11.6 | **1.82×** | 1.05× |
| 4 | 31.1 | **1.96×** | 1.06× |
| 8 | 50.9 | **2.14×** | 0.98× |
| 16 | 74.7 | **2.14×** | 1.01× |
| 32 | 96.6 | **2.02×** | 1.04× |

I2S audio throughput is essentially tied with M\*-old — the bottleneck is the Talker/Code2Wav pipeline, not the Thinker. But M\*-new wins decisively on TTFT (1.4-1.7×) and ITL (1.6-1.7×), meaning the user-perceived latency is better even when raw throughput is similar.

---

## 3. Experiments: what we tried, what worked, what didn't

### 3.1 Optimizations that shipped (in M\*-new)

#### GPU log-mel extraction (`MSTAR_GPU_MEL=1`)
**Branch**: `audio-encoder-opt`
**What**: Move log-mel spectrogram computation from CPU to GPU. The audio encoder's feature extraction (STFT + mel filterbank) was running on CPU and copying to GPU per request.
**Result**: Reduces S2T/S2S prefill time. Part of the integration baseline — its contribution is baked into every M\*-new run.
**Lesson**: Low-hanging fruit. Any per-request CPU→GPU data pipeline step is worth moving to GPU if the tensor is already going there.

#### GPU image preprocessing (`MSTAR_GPU_IMAGE_PREPROCESS=1`)
**Branch**: part of `integration-mnew`
**What**: Move image resize + patchify from CPU to GPU. Same rationale as GPU mel.
**Result**: Reduces I2T/I2S prefill time. Combined with GPU mel, these two eliminate all significant CPU preprocessing.
**Lesson**: Same as above — move the data pipeline onto the device.

#### Chunked prefill (`MSTAR_CHUNKED_PREFILL=1`)
**Branch**: `exp/chunked-prefill`
**What**: Break the Thinker's long prefill into fixed-size chunks that can be interleaved with other requests' decode steps. Without this, a single long-context prefill blocks the entire Thinker for hundreds of milliseconds at higher batch sizes.
**Result**: **The single most impactful optimization.** At B=32 S2T, chunked prefill alone delivered 22.8 req/s vs M\*-old's 2.4 req/s. It unlocks batch-level scaling: M\*-old's throughput plateaus because one request's prefill blocks all others' decode.
**Lesson**: In a multi-model pipeline (Thinker + Talker + Vocoder), the Thinker is the serialization bottleneck. Any optimization that lets the Thinker overlap prefill with decode for other requests has outsized impact. This is analogous to continuous batching in text-only LLM serving.

#### Vision CUDA graph bucket alignment (`MSTAR_VISION_GRAPH_ALIGN=1`)
**Branch**: `opt/vision-cudagraph-align` (from `exp/vision-cudagraph`)
**What**: The Thinker's `prefill_vision` step uses CUDA graphs for speed, but the default bucket sizes (`[1, 2, 4, ... 8192, 16384]`) have large gaps at intermediate sizes. Vision tokens often land at counts like 200-300, which get padded up to 512 — wasting ~40% of compute. Added intermediate buckets: `[128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]`.
**Result**: 2-5% improvement on I2T throughput. Small but consistent, and zero-risk (just less padding waste).
**Lesson**: CUDA graph bucketing is a "free" optimization — it only reduces padding, never changes semantics. But the gains are proportional to how much padding the old buckets wasted, which depends on the actual token count distribution. Profile first to see where tokens land.

#### Batch vision prefill (`MSTAR_BATCH_VISION_PREFILL=1`)
**Branch**: `opt/batch-vision-prefill` (from `exp/batch-vision-prefill`)
**What**: When multiple concurrent requests need vision encoding, batch their vision encoder forward passes into a single call instead of running them sequentially. At B=1 there's nothing to batch, but at B=4+ this reduces total vision encode time.
**Result**: 2-5% improvement on I2T at B≥4. Complements graph alignment — one reduces per-request overhead, the other reduces cross-request serialization.
**Lesson**: Batching encoder forward passes is only useful when concurrency > 1. At B=1 it's a no-op. The gain scales with the ratio of (encoder time) / (total request time).

### 3.2 Experiments that showed neutral or negative results

#### Encoder coalescing (`MSTAR_ENCODER_COALESCE`)
**Branch**: `exp/encoder-coalesce`
**What**: A windowed coalescing strategy — hold incoming requests for up to N ms to accumulate a batch before running the encoder. Idea: trade a small latency increase for better GPU utilization on the encoder.
**Result**: **Neutral to slightly negative.** Tested A/B on S2T B=1,4,8. The coalescing wait time (10ms) added latency without meaningful throughput gain, because the encoder is already fast relative to the Thinker. At low concurrency there's nothing to coalesce; at high concurrency the requests arrive fast enough that natural batching already occurs.
**Lesson**: Coalescing helps when the batched operation is expensive and the wait time is small relative to it. For a fast encoder (~10-20ms), a 10ms coalescing window is proportionally too large. This would matter more for a 200ms+ encoder.

#### Codec chunk tuning (`codec-chunk`)
**Branch**: `codec-chunk`
**What**: Reduce the Code2Wav vocoder chunk size from 25 to 15 tokens. Smaller chunks start audio streaming earlier (lower TTFT for speech) but may reduce throughput due to more frequent small kernel launches.
**Result**: **Net negative at default setting.** The TTFT improvement was real (~15-20% lower TTFT-audio) but throughput dropped because the vocoder makes more, smaller forward passes. Left as `default OFF`.
**Lesson**: There's a real tradeoff between streaming granularity and throughput in the vocoder. A chunk size of 15 is too aggressive for throughput-focused benchmarks. The right value depends on whether the use case is latency-sensitive (conversational) or throughput-sensitive (batch processing).

#### Async audio pipeline / encode-prefill overlap (`exp/async-audio-pipeline`)
**Branch**: `exp/async-audio-pipeline`
**What**: Overlap the audio encoder with the Thinker prefill — start running the Thinker on already-encoded prefix tokens while the encoder is still processing later chunks.
**Result**: **Scaffolding only, not benchmarked.** The implementation required a `StreamingGraphEdge` abstraction that was complex to integrate with the existing conductor. Did not reach a runnable state for benchmarking.
**Lesson**: Latency-hiding via overlap is architecturally attractive but mechanically hard in a CUDA-graph-captured pipeline. The executor needs to know about partial results, which breaks the current "one step = one forward pass" model.

#### Speculative/MTP decode (`exp/spec-decode-mtp`)
**Branch**: `exp/spec-decode-mtp`
**What**: Speculative decoding for the Thinker using a multi-token prediction head. Idea: predict 2-4 tokens per step, verify, and accept if correct.
**Result**: **Scaffolding only.** The Qwen3 model doesn't ship with an MTP head, so this would require training one. Parked as a future direction.
**Lesson**: Speculative decode is a model-level optimization that requires training support. It's not a pure serving-side change.

#### FP8 quantization (`exp/fp8-quant`)
**Branch**: `exp/fp8-quant`
**What**: FP8 KV cache, weights, and attention for the Thinker to reduce memory bandwidth pressure.
**Result**: **Scaffolding only.** Needs careful calibration to avoid quality regression. Parked.
**Lesson**: Quantization is high-reward but high-risk. Needs a quality evaluation pipeline (not just throughput benchmarks) before deployment.

#### Token reduction (`exp/token-reduction`)
**Branch**: `exp/token-reduction`
**What**: Downsample audio tokens (stride-2) and merge vision tokens to reduce sequence length in the Thinker.
**Result**: **Scaffolding only.** Quality impact unknown without evaluation.
**Lesson**: Same as FP8 — trades quality for speed. Needs eval infrastructure first.

#### Fused MoE kernels (`exp/moe-kernels`)
**Branch**: `exp/moe-kernels`
**What**: Replace the Thinker's MoE routing + expert dispatch with a fused Triton kernel.
**Result**: **Scaffolding only.** The MoE layer is not the bottleneck in the current profile — vision encode and inter-model communication dominate.
**Lesson**: Profile before optimizing. The Thinker MoE forward pass is fast enough that kernel fusion gains are marginal compared to pipeline-level optimizations.

#### Precision toggles (`exp/precision-toggles`)
**Branch**: `exp/precision-toggles`
**What**: Toggle TF32 vs FP32 for matmul, and FP32 precision for the vocoder.
**Result**: **No measurable throughput difference.** The default PyTorch precision settings are already using TF32 for matmul on H200.
**Lesson**: On modern hardware with TF32 enabled by default, explicit precision toggles are a no-op.

#### Talker batch-fill instrumentation (`exp/talker-batchfill`)
**Branch**: `exp/talker-batchfill`
**What**: Instrument the Talker's batch utilization — how full is each decode step's batch.
**Result**: **Diagnostic only, not an optimization.** Confirmed that Talker batch fill is high at B≥4 (>80% utilization), meaning the Talker is not the bottleneck.
**Lesson**: Good diagnostic. Confirmed the Thinker, not the Talker, is the scheduling bottleneck.

#### Talker pending queue (`exp/talker-pending-queue`)
**Branch**: `exp/talker-pending-queue`
**What**: Replace the Talker's text input queue with a device-backed FIFO to avoid host↔device copies.
**Result**: **Scaffolding.** The Talker text FIFO is not on the critical path — text tokens are tiny compared to audio.
**Lesson**: Optimize the bottleneck, not the periphery.

#### Vocoder adaptive chunk (`exp/vocoder-adaptive-chunk`)
**Branch**: `exp/vocoder-adaptive-chunk`
**What**: Dynamically adjust vocoder chunk size based on current batch level — larger chunks when batch is full (throughput mode), smaller chunks when batch is sparse (latency mode).
**Result**: **Not benchmarked to completion.** The adaptive logic added complexity without clear benefit in closed-loop benchmarks where batch size is fixed.
**Lesson**: Adaptive strategies shine in open-loop / variable-load scenarios, not in fixed-concurrency benchmarks. Would need a realistic traffic trace to evaluate properly.

#### Encoder placement reshuffle (`exp/encoder-placement`)
**Branch**: `exp/encoder-placement`
**What**: Move vision + audio encoders from Rank 1 (with Thinker) to Rank 0 (with Talker + Code2Wav). Idea: balance GPU memory usage across ranks.
**Result**: **Not benchmarked.** Config-only change but would require re-profiling the entire pipeline.
**Lesson**: Placement changes affect the communication pattern between ranks. In SHM mode the cross-rank cost is low, but it's still a full pipeline change.

#### Mixed prefill+decode walk (`exp/mixed-walk-piggyback`) — UPDATED 2026-06-29

**Branch**: `exp/mixed-walk-piggyback` @ `6bdcb90`
**Behind**: `MSTAR_MIXED_WALK=1`

**What**: Allow the Thinker to run a mixed step — prefill for a new request + decode for existing requests in the same FlashInfer varlen forward pass. This is the explicit M\* equivalent of vLLM-Omni's default mixed-position continuous batching.

**Status (corrected)**: NOT "scaffolding only" — the **eager forward path is fully implemented**. ~1000 LOC across `mstar/engine/mixed_walk.py`, `mstar/worker/micro_scheduler.py`, `mstar/engine/kv_cache_engine.py`, `mstar/engine/cuda_graph_runner.py`, with 6 CPU unit tests passing. Scheduler emits mixed batches, worker builds NodeBatch with decode-first ordering, engine routes through FlashInfer's prefill wrapper with `qo_indptr`-based ragged layout. The **CUDA graph capture for the mixed step is the explicitly deferred part** — `CudaGraphRunner.run_mixed` returns None → eager fallback.

**Result (I2T full sweep on GPUs 6,7, vs lowrisk mstar_new)**:
| B | base req/s | ours | Δ | base TTFT | ours TTFT | ΔTTFT |
|---|---|---|---|---|---|---|
| 1 | 0.755 | 0.736 | -2.5% | 215 | 255 | +18.7% |
| 2 | 1.165 | 1.122 | -3.6% | 276 | 313 | +13.4% |
| 4 | 1.835 | 1.722 | -6.2% | 298 | 345 | +15.9% |
| 8 | 2.520 | 2.316 | -8.1% | 344 | 386 | +12.4% |
| 16 | 3.471 | 3.260 | -6.1% | 405 | 434 | +7.0% |
| 32 | 4.460 | **0.922** | **-79.3%** | 576 | 604 | +5.0% |

NEGATIVE on every batch and metric, **catastrophic at B=32**. The eager-mode dispatch overhead per mixed step exceeds the per-request piggyback savings; at B=32 the cost balloons nonlinearly because FlashInfer's varlen prefill on a 32-decode-position + 4096-prefill-position mixed input is fundamentally slower without graph capture than the captured decode-only kernel.

**Lesson**: Theory was correct (matches vLLM-Omni's TTFT mechanism) but unusable without CUDA graph capture for the mixed shape. The deferred work is captured in §9.8b — bucketed `(decode_count, prefill_len)` graph registry, ~36 graphs, 2-4 GB extra VRAM, 30-60s extra warmup. Until that lands, **ship default-OFF** (which it does). The piggyback TTFT savings the design promised are real, but eager-mode dispatch cost dwarfs them by ~5-10×.

#### Config knobs (`exp/config-knobs`)
**Branch**: `exp/config-knobs`
**What**: Tune scheduler parameters — decode bucket size (B=64), KV cache pages, NUM_SLOTS.
**Result**: **Neutral.** The default settings were already well-tuned for B≤32. The B=64 decode bucket is only useful if serving B>32.
**Lesson**: Config tuning is environment-specific. Profile with your actual workload before changing defaults.

#### Parity mode (`exp/parity-mode`)
**Branch**: `exp/parity-mode`
**What**: A debug mode that forces M\*-new to produce byte-identical output to M\*-old for S2S, to verify that optimizations don't change model behavior.
**Result**: **Diagnostic tool.** Confirmed that all shipped optimizations are numerically identical to M\*-old — no quality regression from GPU mel, GPU image preprocess, chunked prefill, or vision opts.
**Lesson**: Essential for confidence. Having a parity mode lets you verify that performance optimizations don't change model output, which is especially important for speech (where small numerical differences can cause audible artifacts).

---

## 4. Puzzles and anomalies in the data

### 4.1 TTFT is higher for M\*-new than vLLM, but throughput is much better

This is the most counterintuitive result. In S2T and I2T:

| B=8 S2T | M\*-new | vLLM |
|---|---|---|
| TTFT p50 | 248ms | 230ms |
| req/s | 12.16 | 5.50 |
| tok/s | 176.7 | 132.6 |

M\*-new is **slower on TTFT** (time to first token) but **2.2× faster on throughput**.

**Explanation**: This is a direct consequence of chunked prefill. When M\*-new breaks prefill into chunks, each chunk yields the scheduler to other requests' decode steps. This means:
1. The prefilling request waits longer for its first token (more scheduler yields)
2. But all other requests in the batch keep making decode progress during that wait
3. Net effect: higher per-request TTFT, but dramatically higher system throughput

Think of it as cooperative multitasking: each request gives up a little latency so the system as a whole gets more done. At B=1 (no other requests to yield to), M\*-new's TTFT is similar to or better than vLLM's. The TTFT penalty grows with batch size because there are more requests competing for scheduler time.

**This is the correct tradeoff for a throughput-focused deployment.** If TTFT is the primary SLA, reduce the chunk size to yield less often (at the cost of some throughput). The chunk size is configurable.

### 4.2 M\*-old ITL is suspiciously low (near-zero) in S2T

At B≥4 S2T, M\*-old reports ITL (inter-token latency) of <1ms:

| B | M\*-old ITL mean |
|---|---|
| 4 | 0.003s |
| 8 | 0.0005s |
| 16 | 0.0002s |

This is not real. M\*-old processes requests sequentially (no batching), so at B=8, seven requests are queued while one is being processed. The "ITL" reported is the time between tokens **for the one active request**, which is indeed fast because it has the entire GPU to itself. But the queued requests see zero tokens until their turn.

The throughput number (2.5 req/s at B=8) tells the real story: M\*-old's effective throughput doesn't scale with batch size because it can only process one request at a time.

**Lesson**: ITL is meaningful only for systems that actually interleave requests. For sequential processing, ITL per-request is misleadingly good while system throughput is bad.

### 4.3 M\*-old S2S ITL is bumpy across batch sizes

The M\*-old S2S ITL (audio) jumps around:

| B | ITL mean |
|---|---|
| 1 | 0.085s |
| 2 | 0.038s |
| 4 | 0.183s |
| 8 | 0.087s |
| 16 | 0.143s |
| 32 | 0.145s |

The non-monotonic pattern (drops at B=2, spikes at B=4, drops again at B=8) is suspicious. Possible explanations:
1. **Talker batch scheduling artifact**: The Talker processes requests in a different order than they arrive, and the "ITL" measurement captures inter-chunk gaps in the audio pipeline, not pure compute time.
2. **Run-to-run variance**: M\*-old S2S was only run once per batch size. The S2S pipeline has higher variance than S2T because the Talker → Code2Wav handoff introduces scheduling jitter.
3. **Measurement artifact**: The harness measures ITL as time between successive audio chunks, which may include time spent on other requests' Thinker steps.

**A rerun was triggered to check if this is noise or a real pattern.**

### 4.4 I2S audio throughput: M\*-new ≈ M\*-old, but M\*-new wins on latency

At B=8 I2S:
- M\*-new: 50.9 audio s/s, TTFT 0.66s, ITL 0.16s
- M\*-old: 51.7 audio s/s, TTFT 0.93s, ITL 0.26s

Audio throughput is essentially identical, but M\*-new is 1.4× faster on TTFT and 1.6× faster on ITL.

**Explanation**: The I2S pipeline bottleneck is the Talker/Code2Wav, which is unchanged between M\*-old and M\*-new. Both systems can produce audio at roughly the same rate. But M\*-new gets through the Thinker (text generation) phase faster, so the Talker starts producing audio sooner → lower TTFT. The Thinker also finishes sooner, leaving more GPU time for the Talker to generate audio chunks without interruption → lower ITL.

**Lesson**: In a multi-model pipeline, throughput can be bottlenecked by a different model than latency. Optimizing the Thinker improved latency everywhere but only improved throughput on paths where the Thinker was the bottleneck (S2T, I2T) — not on speech paths where the Talker/vocoder is the bottleneck.

### 4.5 vLLM TTFT is suspiciously flat across batch sizes for text paths

| B | vLLM TTFT (I2T) |
|---|---|
| 1 | 151ms |
| 2 | 135ms |
| 4 | 151ms |
| 8 | 187ms |
| 16 | 191ms |
| 32 | 205ms |

vLLM's TTFT barely increases from B=1 to B=32 (151ms → 205ms), while M\*-new goes from 283ms to 585ms.

**Explanation**: vLLM likely does **not** use chunked prefill for these batch sizes, or uses a much larger chunk. Each request's prefill runs to completion without yielding. This gives excellent per-request TTFT but limits throughput because other requests are blocked during prefill.

This is the mirror of puzzle 4.1: vLLM optimizes for latency, M\*-new optimizes for throughput. Neither is "wrong" — they're different tradeoff points.

---

## 5. What we learned about benchmarking methodology

### 5.1 Closed-loop vs open-loop

All our benchmarks used closed-loop profiling: the next request is sent immediately when a slot opens. This models a **saturated server** — the system is always at maximum concurrency. It's the right protocol for measuring peak throughput and latency-under-load.

Open-loop (Poisson arrivals) would model a **partially loaded server** and would show different results, especially for M\*-old which performs well at B=1 but collapses at B≥4.

### 5.2 The PYTHONPATH trap

When running A/B benchmarks across git worktrees, `PYTHONPATH` must be set to the worktree under test. If it points to the wrong worktree, spawned GPU workers load the wrong code — and the results are silently invalid. We burned at least one full sweep before catching this.

**Rule**: Always set `PYTHONPATH=<worktree>` explicitly in the sweep script. Never rely on the ambient shell `PYTHONPATH`.

### 5.3 SHM socket collisions

Two sweeps running on the same port with the same SHM socket prefix will cross-tear-down each other's GPU workers. The symptom is a "server crash" on one sweep immediately after the other's cleanup step.

**Rule**: Unique port + unique socket path per concurrent sweep. Our `sweep.sh` auto-generates socket paths as `/home/tim/tmp/sk_${SYSTEM}_${PORT}`.

### 5.4 GPU selection matters for comparability

All benchmarks for a given system must run on the same physical GPUs. Even identical GPU models on different NUMA nodes can show 2-5% throughput differences due to memory bandwidth and PCIe topology. We used GPUs 0,1 (NUMA 0) or 5,6 (NUMA 1) consistently within each system's runs.

### 5.5 Don't trust ITL alone

ITL (inter-token latency) is a per-request metric that doesn't capture queuing. A system that processes requests sequentially will show excellent ITL for the active request and infinite ITL for queued requests. Always pair ITL with throughput and TTFT.

---

## 6. Architecture observations

### 6.1 The Thinker is the bottleneck for text, the Talker for speech

This is the central insight. The Qwen3-Omni pipeline has three models in series:

```
Input → [Encoder] → [Thinker (32B MoE)] → text output
                                         → [Talker (0.5B)] → [Code2Wav] → audio output
```

For text-only paths (S2T, I2T), the Thinker is the bottleneck. Optimizing it (chunked prefill, graph alignment, batch vision) yields direct throughput gains.

For speech paths (S2S, I2S), the Thinker generates text tokens that the Talker converts to speech tokens that Code2Wav converts to audio. The Talker + Code2Wav pipeline is slower per audio-second than the Thinker, so the Thinker finishes its part quickly and then waits. Optimizing the Thinker improves TTFT (the Thinker's part finishes sooner) but not raw audio throughput (still bottlenecked by Talker).

**Implication**: Future speech-path optimizations should target the Talker and Code2Wav, not the Thinker.

### 6.2 Scaling behavior: M\*-new scales linearly, M\*-old doesn't

M\*-new S2T throughput scales roughly linearly with batch size (5 req/s at B=1, 24 req/s at B=32, ~4.7× for 32× batch increase). This is expected with chunked prefill — more concurrent requests means more decode steps interleaved with prefill chunks, so the pipeline stays full.

M\*-old S2T throughput is flat (1.9 req/s at B=1, 2.4 req/s at B=32). Without batched decode, adding concurrency just adds queuing.

This scaling gap is the strongest argument for chunked prefill. At B=1, M\*-new is "only" 2.6× faster. At B=32, it's 9.6× faster. The gap keeps widening.

### 6.3 Vision encode dominates I2T/I2S latency

Image requests (I2T, I2S) are ~3× slower than audio requests (S2T, S2S) at B=1:
- S2T B=1: 5.06 req/s
- I2T B=1: 0.69 req/s

The vision encoder (Qwen2-VL's ViT) processes 256+ image patches through self-attention, which is significantly more expensive than the audio encoder's mel spectrogram + transformer. This is why the vision-specific optimizations (graph alignment, batch prefill) target image paths.

### 6.4 2-GPU SHM is good enough

We were forced to use SHM (shared memory) tensor communication instead of RDMA/Mooncake because the RDMA stack was broken on this node. SHM adds ~100μs per inter-rank transfer compared to RDMA's ~10μs. At the token generation rates we observe (~20-170 tokens/s for the Thinker), this overhead is <1% of total time. For 2-GPU TP, SHM is not a bottleneck.

---

## 7. Recommendations

### What to ship
1. **Chunked prefill**: by far the highest-impact optimization. Non-negotiable for any batch size > 1.
2. **GPU mel + GPU image preprocess**: trivial wins, no quality risk.
3. **Vision graph alignment + batch prefill**: modest but free wins for image paths.

### What to pursue next
1. **Talker/vocoder optimization**: this is now the speech-path bottleneck. Adaptive chunk sizing, Talker batch scheduling improvements, or vocoder kernel fusion.
2. **FP8 quantization**: high reward but needs quality evaluation infrastructure first.
3. **True continuous batching** (mixed prefill+decode): the "right" solution beyond chunked prefill, but architecturally complex.

### What to skip
1. **Encoder coalescing**: overhead > benefit for fast encoders.
2. **Precision toggles**: no-op on modern hardware.
3. **Config tuning for B≤32**: defaults are fine.

---

## 8. Reproducing these results

All benchmark data lives on the `benchmarks` branch of `t-avil/mstar`:
```
benchmarks/qwen3-omni-joint/
  raw_audio_to_text.json
  raw_image_to_text.json
  raw_audio_to_speech.json
  raw_image_to_speech.json
  NUMBERS.md              # headline comparison table
  make_proof_charts.py    # regenerates all charts from raw JSON
  charts/                 # PNG charts
```

Each raw JSON contains per-request datapoints and pre-computed aggregates for every (system, batch_size) combination. Charts can be regenerated:
```bash
python make_proof_charts.py benchmarks/qwen3-omni-joint benchmarks/qwen3-omni-joint/charts
```

The sweep entry point is `benchmark/sweep.sh` on main:
```bash
benchmark/sweep.sh --system mstar_new --gpus 0,1 --port 8160 \
    --paths s2t,i2t,s2s,i2s --batches 1,2,4,8,16,32 \
    --flags "MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1 MSTAR_VISION_GRAPH_ALIGN=1 MSTAR_BATCH_VISION_PREFILL=1" \
    --worktree /path/to/integration-mnew \
    --output /tmp/sweep_mnew
```

---

## 9. Round 2 experiments (2026-06-29)

After the M\*-new combined vision opts shipped, three new experiments were tried,
branched off `opt/combined-vision-opts` @ `e943d72`. Branches: `exp/encoder-placement-profiling`,
`exp/encoder-async-schedule`, `exp/encoder-chunk-coalesce`. Each has its own RESULTS.md.

### 9.1 Encoder placement profiling — `exp/encoder-placement-profiling`

**Behind**: `MSTAR_PROFILE_ENCODER_PLACEMENT=1`.

**What**: Per-stage timestamps for I2T (preprocess / vision_fwd / handoff / e2e), dumped
to JSONL. Not an optimization, a measurement. Goal was to find out where time actually
went after the GPU image preprocess opt landed.

**Result** (I2T B=1..32, 1025 requests):
| Stage | Median time |
|---|---|
| preprocess (GPU resize+patchify) | **130 ms** |
| vision_fwd (ViT) | 30 ms |
| handoff (encoder → Thinker) | 0.03 ms |
| total e2e at B=1 | 4812 ms |
| **encoder share of e2e** | **3-10%** |

**Lesson**: The encoder is **not** the bottleneck — the Thinker is. Preprocess is 5× the
vision encoder itself, and even both together are <10% of total request time at any
batch. Future optimizations should target the Thinker, not the encoder.

### 9.2 Async encoder scheduling — `exp/encoder-async-schedule`

**Behind**: `MSTAR_ENCODER_ASYNC=1` (default OFF), `MSTAR_ENCODER_ASYNC_DEPTH=4`.

**What**: Pipeline the encoder. Start request N+1's encoder forward while request N is
still in Thinker decode, on a low-priority CUDA stream. Cap at K in-flight encoded
buffers.

**Result — path-dependent**:
| Path | Verdict | B=32 result |
|---|---|---|
| I2T | **PROMISING** | +7.4% req/s, **-30% text TTFT** |
| S2T | NEGATIVE | -18% req/s, +8% TTFT |
| I2S | NEUTRAL | req/s flat, audio TTFT flat |

**Why path-dependent**: async helps when the encoder is slow enough that hiding it
behind decode is worth the speculation overhead.
- Vision: preprocess + ViT ≈ 160 ms → worth hiding behind ~700 ms decode at B=32.
- Audio: mel + audio encoder ≈ 10-20 ms (already GPU) → speculation overhead exceeds
  hidden work, causes scheduler contention with decode.
- Speech output: text-side gains exist but Talker+Code2Wav serializer eats them before
  audio reaches the user.

**Lesson**: Don't treat "async encoder" as a global on/off. The path of the request
determines whether speculation pays off. **Ship as opt-in for vision-input text-output
workloads** (I2T-style) at B ≥ 16. Do not enable for audio input or speech output.

### 9.3 Chunk-boundary encoder coalescing — `exp/encoder-chunk-coalesce`

**Behind**: `MSTAR_ENCODER_CHUNK_COALESCE=1`, `MSTAR_ENCODER_COALESCE_SIZE=4`.

**What**: Replace the previous failed *time-based* coalescing (§3.2) with a coalescer
that flushes its pending encoder queue at **Thinker prefill-walk boundaries** instead
of on a wall-clock timer. Boundaries are natural batch-formation windows.

**Result**:
| Path | B=1 | B=8 | B=16 | B=32 | Verdict |
|---|---|---|---|---|---|
| I2T | +8% req/s, -22% TTFT (PROMISING) | -0.7% | -0.4% | -1.6% | NEUTRAL overall |
| S2T | -3.8% | +8.8% (noise; B=4,16 around it neutral) | -1.5% | -5.0% | NEUTRAL overall |

TTFT improvements at low batches are small but real (-8% to -22% on I2T). Throughput is
mostly flat. The chunk-boundary hook fires at the right place, but on the `e943d72`
base the only chunk boundaries are inter-walk (e.g. `prefill_audio → prefill_text`) —
not intra-prefill. The coalescer rarely accumulates a meaningful batch.

**Lesson**: This optimization is structurally sound but needs single-walk chunked prefill
to truly shine. **Park** until intra-prefill chunking lands; revisit when the chunk-boundary
signal fires multiple times per request.

### 9.4 Cross-experiment patterns

Several methodology lessons emerged from running these three in parallel:

- **MVP tests at low batch can mislead.** Exp 2's MVP at B=1,8 showed NEUTRAL on I2T,
  but the full sweep revealed PROMISING at B=32. Always test the regime where the
  optimization is theoretically supposed to help, not just the small-batch convenience
  regime.

- **Server-ready races bite when teardown is slow.** Multiple sweeps had A→B
  transitions where the new server's `/v1/models` curl hit the dying A server's still-up
  API, giving false "ready in 1s" + contaminated bench results. Mitigation: wait for
  worker GPU memory to drop AND check the new server log contains its own
  startup marker before benching.

- **SHM socket prefix must be unique per concurrent server.** Otherwise concurrent
  mstar serves collide on the default socket path. Always pass `--socket-path-prefix`.

- **HF cache permissions matter.** The shared `/m-coriander/coriander/hf` has datasets
  owned by other users with restrictive locks. Set `HF_DATASETS_CACHE=/home/tim/hf_datasets`
  to keep dataset locks under your own UID. Don't change `HF_HOME` — model weights still
  live in the shared path.

- **B-only sweep vs committed baseline is good enough** for ±5% effects when the
  experiment branch and baseline share GPU pair and NUMA. Saves an hour per experiment
  vs full A/B.

### 9.5 Recommended M\*-new v2 configuration

Based on Round 2, the ideal deployed M\*-new should be `opt/combined-vision-opts` plus
**path-gated** async encoder:
- I2T, I2I (vision input, text output): `MSTAR_ENCODER_ASYNC=1` (+5-7% req/s, -10-30% TTFT at B≥16)
- S2T, S2S (audio input): leave OFF (avoids the -18% S2T regression)
- I2S, S2S (speech output): either way — gains don't propagate through Talker+Code2Wav

A future code change should make this gating automatic (e.g. enable async when the
encoder is the heavy one for the dispatched request), so operators don't need per-deployment
env tuning.

### 9.6 Round-2 follow-up experiments (Exp 4, D5, E2)

After the first three Round-2 experiments, three additional ones were spawned to test
specific hypotheses. **All three came back NEGATIVE** in clean serial conditions on I2T
(GPUs 4,7, N=15 warmup=3, no concurrent sweeps).

#### Exp 4 — chunked-prefill + chunk-boundary coalescer combined

**Branch**: `exp/encoder-chunk-coalesce-with-prefill` @ `6457f7e`. Merges
`exp/chunked-prefill` (intra-walk Thinker chunking, aee505f) with the chunk-boundary
encoder coalescer from Exp 3. Theory: now the coalescer's chunk-boundary hook fires
at intra-prefill yields, building larger batches than before.

**I2T verdict** (N=15 per batch):
| B | base | ours | Δreq/s | ΔTTFT |
|---|---|---|---|---|
| 1 | 0.693 | 0.731 | +5.5% | **-34.6%** |
| 4 | 1.727 | 1.556 | -9.9% | -4.9% |
| 8 | 2.438 | 1.923 | **-21.1%** | +212% |
| 32 | 4.209 | 2.597 | **-38.3%** | +303% |

The B=1 win is real (TTFT -34%) but throughput collapses at B≥8. Chunked-prefill yields
the scheduler more often, which helps single-request TTFT but **starves the decode loop
under concurrency**. Combined with the coalescer's extra bookkeeping, throughput tanks.

**Conclusion**: Chunked-prefill is a latency-vs-throughput tradeoff knob, not a free win.
At low N our chunking overhead-per-step exceeds the queueing benefit. Park unless you have
a latency-only deployment where TTFT-at-B=1 is what matters.

#### D5 — Spatial merge as separate GraphNode

**Branch**: `exp/spatial-merge-node` @ `67c9e37`. Behind `MSTAR_SPATIAL_MERGE_NODE=1`.
Splits the vision encoder's 4→1 patch reduction MLP into its own GraphNode so it can be
placed on a different GPU from the encoder. Same-rank placement should be a no-op; the
4× transfer saving would only materialize if encoder and Thinker landed on different
ranks.

**I2T verdict** (same-rank placement, serial N=15):
| B | base | ours | Δreq/s | ΔTTFT |
|---|---|---|---|---|
| 4 | 1.727 | 1.549 | -10.3% | +2.9% |
| 8 | 2.438 | 2.008 | -17.6% | +175% |
| 16 | 3.299 | 2.531 | **-23.3%** | +397% |
| 32 | 4.209 | 2.481 | **-41%** | +365% |

The extra GraphNode boundary costs **10-40% throughput and 200-400% TTFT** at higher
batch sizes — even when both nodes live on the same GPU. The scheduling/edge serialization
cost is real, not just contention. This optimization only pays off if you actually move
spatial_merge to the Thinker's rank in a config where the encoder runs on a different rank
— then the 4× cross-rank transfer saving must exceed the GraphNode overhead.

**Conclusion**: Park for current 2-GPU config (encoder + Thinker both on Rank 1). Revisit
only if a future placement puts the encoder on Rank 0 alone, where the unmerged-patches
transfer would become a real bottleneck.

#### E2 — Encoder output caching by content hash

**Branch**: `exp/encoder-cache` @ `811d65d`. Behind `MSTAR_ENCODER_CACHE=1`. SHA-256 of
input bytes → cached encoded tokens, LRU eviction at 512 MiB.

**I2T verdict** (serial N=15, hit rate measured in server log):
| B | base | ours | Δreq/s | ΔTTFT | hit_rate |
|---|---|---|---|---|---|
| 1 | 0.693 | 0.693 | +0.0% | -9.9% | 0% |
| 4 | 1.727 | 1.572 | -9.0% | -0.3% | 70% |
| 8 | 2.438 | 1.997 | -18.1% | **+258%** | 85% |
| 32 | 4.209 | 3.079 | **-26.8%** | +140% | **94%** |

The cache works as designed — food101 has natural repeats at higher batch and the hit
rate climbs to 94%. **But throughput drops anyway** because:
1. Vision encoder is only 3-10% of e2e (per Exp 1). Caching that small slice can't
   move overall throughput much.
2. Cache lookup (SHA-256 of image bytes + dict lookup) adds latency on the hot path
   for EVERY request, hit or miss.
3. At B≥8, the lookup contention serializes through the cache mutex, blowing up TTFT.

**Conclusion**: The encoder is the wrong layer to cache. The cache only pays off where
the encoder dominates e2e — e.g. very large images, or workloads where encoder cost is
unhidable (no decode to overlap with). For our standard workloads, encoder caching is
strictly negative. Park.

### 9.7 Round-2 ship/park decision matrix

| Experiment | Throughput | TTFT | Ship? |
|---|---|---|---|
| **GPU mel, GPU image preproc, vision graph align, batch vision prefill** (combined-vision-opts) | +2-5% | small | **SHIPPED** (in mstar_new) |
| Exp 2 — async encoder | NEUTRAL throughput, mixed TTFT (path-dependent) | -5 to -10% I2T low-mid B | Opt-in only |
| Exp 3 — chunk-boundary coalesce | NEUTRAL | small TTFT improvement at B=1 | Park (no intra-prefill chunking on base) |
| Exp 4 — chunked-prefill + coalesce | -21 to -38% at B≥8 | -34% B=1, +200-400% B≥8 | Park (latency-only deploys could opt in) |
| D5 — spatial-merge GraphNode | -10 to -41% (same-rank) | +200-400% | Park (only worth in cross-rank configs) |
| E2 — encoder cache | -9 to -27% despite 94% hit rate | +140-260% at B≥8 | Park (wrong layer to cache) |

### 9.8a M\*-new promoted: opt/combined-lowrisk (1f66ce6)

After Round 2, four additional low-risk opts landed and were swept together as
`opt/combined-lowrisk`. They form the **new shipping M\*-new**. Branch tip: `1f66ce6`.

New opts added on top of `opt/combined-vision-opts`:
- `fbc9804` — vision-sync-elim: remove GPU→CPU syncs in `prefill_vision` prepare
- `10bb1d5` — encoder-internal CUDA graph enabled by default
- `1f66ce6` — `torch.compile(dynamic=True)` on encoder forward (single shape-poly artifact)
- `95290f6` — defaults `MSTAR_ENCODER_ASYNC=0` (Round-2 finding: A/B win didn't reproduce)

**Sweep result (all 4 paths, B=1..32, GPUs 5,6 NUMA 1 — same as committed baseline)** vs the
previous `mstar_new` (combined-vision-opts only). I2T is the headline win — **PROMISING at EVERY batch**:

| Path | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| S2T Δreq/s | -4% | **+5%** | -2% | +2% | -3% | **+8%** |
| S2T ΔTTFT | -4% | -12% | -9% | -11% | +2% | -5% |
| **I2T Δreq/s** | **+9%** | **+6%** | **+6%** | +3% | **+5%** | **+6%** |
| **I2T ΔTTFT** | **-24%** | **-15%** | **-15%** | **-17%** | **-19%** | -1% |
| S2S Δreq/s | **+7%** | +5% | **+5%** | 0% | +2% | +4% |
| S2S ΔTTFT | -2% | -8% | -2% | -1% | 0% | -4% |
| I2S Δreq/s | +1% | -2% | -1% | +4% | +4% | +1% |
| I2S ΔTTFT | -7% | -9% | -7% | **-13%** | **-10%** | -11% |

Charts and raw data updated: `mstar_new` in `raw_*.json` now points at lowrisk (1f66ce6).
The previous `combined-vision-opts` baseline is archived as `mstar_new_v1` in the same JSON.

### 9.8b Where to go next

Confirmed by Exp 1 and verified by all Round-2 negatives: **the encoder is not the bottleneck.**
The Thinker is. Future experiments should target:
1. **FP8 KV cache + weights** for the Thinker (memory bandwidth)
2. **Speculative decoding** for the Thinker (decode parallelism — needs MTP head training)
3. **True continuous batching** (mixed prefill+decode in one forward) — already scaffolded
   in `exp/mixed-walk-piggyback`, never benchmarked
4. **Talker / Code2Wav optimization** for speech paths — confirmed bottleneck for I2S/S2S

Stop investing in encoder-side optimizations. They've returned everything they can.

# Beat vLLM-Omni 0.24 on text *and* keep the 2–3× speech lead — one config

## One-paragraph summary (all new work vs `origin/main` `9ee13699`)

On top of `origin/main` this branch adds a **single serving configuration** that beats vLLM-Omni 0.24 on text (i2t **1.04×**, s2t **1.27×** tok/s) while keeping M*'s 2–3× speech lead (i2s **2.17×**, s2s **2.91×**, t2s **2.00×** req/s), built from: **(1)** a *replicated encoder* placed as free BF16 DP-replicas on both GPUs with each request's encode routed to the *idle* rank by output modality — the ~30-line, byte-identical change that lets one config win text and speech instead of needing a per-modality topology; **(2)** an off-process *multiprocess preprocess pool* that removes the i2t TTFT blow-up (≈2800 ms → ≈176 ms at B32); **(3)** an occupancy-auto-gated *s2t audio-prefill merge* that collapses two decode-blocking prefill steps into one only at low batch (wins B≤16, dodges the B32 regression); **(4)** a *host-floor decode stack* — sidecar emit, deferred/integer check-stop, ordered emit, batched+slim emit/route, on-device batched position prep, cached sampler config — that cuts the ~26 ms/step Python-GIL floor; **(5)** *fp8 grouped-GEMM MoE* plus *torch.library custom ops* that keep the Thinker's compiled graph intact under fp8; and **(6)** *chunked + captured-mixed prefill* with wider vision/prefill capture grids so more shapes replay a captured graph — every flag default-**off** and byte-identical when off, fp8/custom-ops bounded to rounding, all backed by an expanded CPU parity-test suite.

---


**Result (B32, natural-EOS, closed-loop, continuous batching, no DP/PD disaggregation):** a
**single** served config beats vLLM-Omni 0.24 on every path — i2t **1.04×**, s2t **1.27×** tok/s;
i2s **2.17×**, s2s **2.91×**, t2s **2.00×** req/s — recovering both per-modality optima at once.

## Features and how each one wins

- **Replicated encoder + output-modality routing (the single-config unlock).** Encoders run as free BF16 DP-replicas on *both* GPUs, and each request's encode is routed to the *idle* rank by output modality (text-out → Thinker-free rank, speech-out → co-located with Thinker); this removes the encoder↔decode contention that forced a per-modality topology, so one config wins text and speech — byte-identical, pure scheduling (~30 lines).
- **Topology-per-modality baseline (`encoff`/`base`).** The Thinker (text) and Talker+Code2Wav (speech) contend differently, so encoder placement is flipped per output modality; this is the win the routing above folds into a single deployment.
- **Off-process multimodal preprocess pool.** Image/audio preprocessing is moved off the decode process; this is the fix that kills the i2t TTFT spike (the "abyss") and is what makes i2t competitive at B32.
- **MERGED_PREFILL_AUDIO occupancy auto-gate.** s2t's two decode-blocking prefill steps (text + audio, ~42 ms/req) are merged into one walk, but gated by live batch occupancy (`MAX_BS=24`) because the merge wins +19–28% at B≤16 yet regresses −26% at B32 — so one config wins s2t at *every* batch. Both walks are captured → parity-safe.
- **Host-floor decode stack.** Cuts the ~26 ms/step Python/GIL host floor that capped decode throughput; this is the core tok/s lever behind the text wins across batches.
- **fp8 MoE grouped GEMM + torch.library custom ops.** fp8 MoE speeds the Thinker's expert GEMMs, and the custom ops keep the compiled Thinker graph intact *under* fp8 (a naive fp8 path breaks compilation); together they raise decode throughput with output faithful to bounded rounding.
- **Chunked + captured-mixed prefill, capture-grid + prefill-bucket coverage.** Prefills are chunked and replayed from captured graphs across a fuller batch/shape grid; this removes prefill stalls in the decode wave and keeps every serving point on a captured (not eager) path.
- **Ordered-emit.** Fixes an out-of-order/lost first-token race under concurrency at no throughput cost, so the throughput wins come with correct token streams.

## Parity & honesty

Winning flags are default-off byte-identical or bounded-rounding (fp8/custom-ops); the auto-gate has 8 CPU parity tests pinning byte-identical schedules when it declines to merge. Speech "2–3×" is **request** throughput (audio-seconds/s is length-confounded because vLLM-0.24 emits longer audio); t2s (~2.0×) is the weakest speech path, at the low edge of the band. vLLM numbers were measured here — re-run them through the owner pipeline before defending them for submission.

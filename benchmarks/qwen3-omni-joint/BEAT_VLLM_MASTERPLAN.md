# BEAT vLLM — Master Plan: +50% on TTFT, ITL, req/s, tok/s (Qwen3-Omni 30B-A3B, H200)

**Date:** 2026-07-07. **Method:** 8 parallel research agents (decode-loop/MRV2, attention,
KV, speculative decoding, MoE kernels, batching/TTFT, competitive+fairness, fusion/sampler/
encoder), each instructed to distrust marketing and cite primary papers/PRs/commits. This is a
multi-quarter feature plan: **30 options** across 6 workstreams, each with mechanism, per-metric
impact, M* implementation, effort/quarter, risk, and a citation. Read §0 and §1 first — they
change what "winning" even means.

---

## §0. PREREQUISITE — fix the benchmark before optimizing (do this WEEK 1)

We found M* emits **~176 tokens/request vs vLLM's ~212** on the same prompts (a 17% gap). That
confound sits inside BOTH throughput metrics: it deflates M*'s tok/s **and inflates M*'s req/s
"wins"** (fewer tokens/req → each request finishes sooner). **No optimization number is trustworthy
until this is closed.** Smoking gun from the local vLLM-Omni checkout: commit **`519cf3ec` (#4137,
2026-06-04)** set the Thinker reference to **`temperature=0.0` greedy, no top_p/top_k, no
`repetition_penalty`**. If M* runs anything else (temp>0, `repetition_penalty=1.05`, old `top_k=1`),
token streams diverge and EOS fires at a different step.

**Fair protocol (industry standard — MLPerf/NVIDIA/vLLM-bench):**
- **`ignore_eos=true` + fixed `max_tokens=N`** (N=256 or 512) on BOTH engines → every request emits
  exactly N decode tokens. (vLLM `bench serve --ignore-eos --random-output-len N`; SGLang has it on
  by default; M*: add the flag or pin `min_tokens=max_tokens=N`.)
- Pin sampling identically: temp=0 greedy, top_p=1, top_k off, all penalties=0, same seed, same
  tokenizer for serving AND counting.
- Verify: `finish_reason=="length"` for ~100% of requests on both.
- Headline metric = **output-token throughput** (decode tokens ÷ wall time); never total (input+output)
  token throughput. Store per-request `completion_tokens`, `finish_reason`, TTFT, ITL in raw.json.

**Expected effect:** the current 0.89× B32 req/s and the "1.1–1.15× B1–B16 wins" will both move —
possibly a lot. Re-run the whole h2h matrix under this protocol on a **quiet box** (B32/B16 are so
CPU-floor-bound that any co-tenant depresses them 15–25%) before committing engineering to §2.
*Source: vLLM bench serve docs; NVIDIA "LLM Inference Benchmarking Fundamentals"; MLPerf inference_rules; local commit 519cf3ec.*

---

## §1. Per-metric diagnosis (where the gap actually is)

From our warmed h2h (all batches), the honest picture per metric:

| metric | M* vs vLLM 0.22 | root cause |
|---|---|---|
| **ITL** | **M\* already WINS** (9.3 vs 15.4 ms @B32) | M*'s decode forward is efficient; not the problem |
| **TTFT** | **M\* loses badly, explodes with batch** (181→2532 ms vs vLLM flat 87→228) | **prefill serialization** — encoder+Thinker prefill runs as an exclusive step that stalls decode |
| **req/s** | M* wins B1–B16, loses B32 (0.89×) | the TTFT explosion drags JCT at high batch; + output-length confound (§0) |
| **tok/s** | M* lower everywhere | ~2/3 output-length confound (§0), ~1/3 the real B32 gap |

**The "stupid simple" things you suspected, confirmed, in priority order:**
1. **The benchmark isn't apples-to-apples** (§0) — likely the single biggest correction.
2. **A per-step CPU/GIL floor (~26 ms @B32)** the whole industry already removed (MRV2, SGLang overlap
   scheduler). vLLM v0.6 profiling: scheduling+API were **62% of wall-clock** before they killed it.
3. **The MoE grouped-GEMM ships a HARDCODED tile config with no autotune** (`utils/fused_moe/kernels.py:401-420`,
   `BLOCK_M=16,N=32,K=64`) — a near-free +15–35%.
4. **Prefill is never mixed with decode** — the one-`graph_walk`-per-batch rule is the TTFT killer.

**Critical nuance on MRV2:** the vLLM-Omni **MoE Thinker is NOT on Model Runner V2 yet** (RFC #1770
is a proposed ~11-week plan, not landed; the checkout's Thinker uses "standard CUDA-graph capture").
Upstream MRV2 became default for Qwen3 *dense* in v0.22 and for *quantized/MoE* in v0.24 — so when
vLLM-Omni rebases onto ≥0.24, the Thinker gets it. **Implication: build M*'s MRV2-equivalent
proactively — it's both the current CPU-floor fix AND the leapfrog before vLLM-Omni catches the
MoE-MRV2 wave.**

---

## §2. The 30 options

Legend: **Metrics** = which of TTFT/ITL/req·s/tok·s it moves. **Q** = target quarter (Q3'26 = now).
Effort S/M/L/XL. Every % is from cited adjacent-system measurements, not yet M*-measured.

### Workstream A — Decode-loop CPU/host-overhead elimination (owns ITL, tok/s, req/s)
*This is M*'s MRV2. Stacking A1+A2+A3+A5 plausibly clears +50% on ITL/tok/s/req·s in the B32 CPU-bound
regime — MRV2 alone hit +56% on the analogous small-model stress test.*

1. **GPU-native input prep + persistent batch state table** (MRV2 core). Build input_ids/positions/
   query_start_loc/seq_lens on-GPU (Triton), gathered each step from a fixed max_reqs-row table;
   requests hold a permanent slot. Kills the per-rid Python prepare loop. *Metrics: tok/s +40–56%,
   req/s +30–50%, ITL −25–35%. M*: add device `DecodeState` + Triton gather; mirror omni RFC #1770's
   index-based buffer (drop rid-keyed dicts). L, Q3–Q4. Risk: TM-RoPE must be gatherable (RFC's flagged
   gotcha). Src: MRV2 design doc; RFC #1770.*
2. **Persistent/incremental scheduler ready-set.** Replace "rescan all queues per-rid per-node every
   step" with an event-driven slot+free-list; O(Δ) updates. *Metrics: tok/s +20–40%, ITL −20–30%. This
   IS the 26 ms floor. M*: micro_scheduler rewrite; substrate for A1. M, Q3. Risk: membership parity
   under preemption. Src: MRV2 persistent batch; vLLM v0.6 (sched=29%).*
3. **Zero-sync decode loop.** Purge per-step `.item()/.cpu()/.tolist()`; keep sampled token on-GPU;
   pinned staged H2D (MRV2 `StagedWriteTensor`). *Metrics: ITL −5–15%; unblocks A7 graph capture. M,
   Q3. Risk: one residual sync erases it — Nsight-verify. Src: MRV2; vLLM #7172.*
4. **Async output processing.** Overlap postprocess(N) (detok/stop/logprobs) with GPU(N+1) via an
   output queue + consumer thread. *Metrics: ITL −8–20%, tok/s +8–20%. M*: split check_stop/detok off
   the walk (builds on opt/sidecar-checkstop). S–M, Q3 — cheapest, do FIRST. Src: vLLM v0.6 async output.*
5. **Multi-step scheduling.** Schedule once, replay the decode graph n steps before returning to Python
   — divides the 26 ms by n. *Metrics: tok/s/req·s +25–28%. Caveat TTFT+ at low load; disable until warm.
   M*: n-step plan in decode walk. M, Q3. Risk: batch can't change mid-window; keep output bit-identical
   (async-sched failed on non-identical output). Src: vLLM v0.6 multi-step.*
6. **Process separation / EngineCore (GIL relief).** API+tokenize+detok in a separate process (ZMQ/
   SHM); engine core = scheduler+executor only. *Metrics: tok/s +10–25%, TTFT −10–20% under load. M*:
   reuse SHM protocol; pass multimodal tensors by handle. S–M, Q3, parallelizable. Src: vLLM V1 EngineCore.*
7. **Expand CUDA-graph coverage (FULL_AND_PIECEWISE).** Full graph over more uniform-decode buckets;
   piecewise graph (attention eager) for prefill/mixed → fewer eager fallbacks. *Metrics: ITL −10–20%,
   tok/s +10–20%. M*: add decode buckets + piecewise mixed capture (needs A3). M, Q4. Src: vLLM CUDA
   graphs design; "most performant for MoEs" (#25444).*
8. **Fused sampler + object pooling.** FlashInfer sorting-free top-k/top-p from pre-softmax logits (no
   vocab sort, no full-softmax); pool per-step metadata objects. *Metrics: ITL −5–10%, tok/s +10–24%
   (vLLM object-cache alone +24%). M*: swap custom sampler → FlashInfer fused; keeps token on-GPU (feeds
   A3). S, Q3. Src: MRV2 Triton sampler; vLLM #7162/#7117; FlashInfer sampling.*

### Workstream B — TTFT / prefill-decode interleaving (owns TTFT — the measured weakness)
*This is the ONLY path to +50% TTFT. B1+B7(graph)+B2 is the core; B3 attacks the multimodal half.*

9. **Stall-free chunked-prefill token-budget scheduler (Sarathi-Serve).** One mixed step: admit all
   decodes first, fill slack with prefill chunks (chunk = budget − Σdecode); long prefills split across
   steps; decode cadence constant. *Metrics: TTFT flattens (target 2532→~250 ms @B32), req/s +>50% @B≥16.
   Sarathi: 2.6–5.6× capacity at fixed SLO. M*: productionize opt/v2-policy + exp/mixed-batch-p2 (already
   in flight). M, Q3 — THE fix. Risk: MoE reload tax on tiny chunks (→ B10). Src: Sarathi-Serve arXiv:2403.02310.*
10. **MoE-aware LAYERED prefill.** Chunked prefill on MoE reloads all experts per chunk (+39% traffic);
    instead interleave prefill/decode across **layer-groups** so each expert set loads once/step.
    *Metrics: TTFT −up to 70%, e2e latency −41%. M*: layer over B9; per-request layer cursor. M–L, Q4.
    Risk: deepest scheduler surgery; graph boundaries. Src: "From Tokens to Layers" arXiv:2510.08055.*
11. **Encoder side-stream overlap (fix MSTAR_SIDE_PREFILL).** Run ViT/AuT encoder on a dedicated stream
    concurrent with decode; fix the paged-prefill `qo_indptr` packing race (side stream and decode
    co-write the indptr buffer — give the side path its own static buffers). *Metrics: TTFT −up to 71%
    (EPD), P99 TTFT −30–50%. M*: fix the race on opt/side-prefill; bound encoder SMs. S–M, Q3 — branch
    already wired. Src: EPD arXiv:2501.05460; vLLM EPD blog.*
12. **Encoder CUDA graphs (ViT + AuT).** Capture full encoder forward at token-budget buckets
    (power-of-2, greedy bin-pack, eager fallback above max). *Metrics: TTFT (i2t/s2t) +11.8–19.6%
    single-GPU. M*: capture SigLIP2 ViT (543M) + AuT (650M); AuT dynamic 1–8 s window → bucket by
    (tokens×window) or fix window/graph. M, Q3. Src: vLLM ViT CUDA graphs.*
13. **FP8 encoders + varlen/packed encoder attention.** fp8 ViT/AuT linear+attn projections (Hopper
    native); pack sequences with cu_seqlens (1 kernel/layer, −63% padding). *Metrics: TTFT ~1.5× ViT/
    ~1.6× prefill. M*: reuse fp8 infra on encoder modules; keep norm/softmax bf16, maybe AuT first/last
    blocks bf16. M, Q3–Q4. Risk: ASR-WER/vision-grounding — quality-gate. Src: arXiv:2502.01070; LiteVLM 2506.07416.*
14. **Prefill/Encode-Decode disaggregation (DistServe/EPD/Mooncake).** Separate prefill+encode and decode
    worker pools; KV over NVLink. *Metrics: 7.4× more req or 12.6× tighter SLO (DistServe). M*: prefill/
    decode roles + KV connector. L–XL, Q4–Q1 — only if B9+B10 leave a residual SLO conflict. Src: DistServe
    arXiv:2401.09670; Mooncake arXiv:2407.00079.*

### Workstream C — Speculative / parallel decoding (owns ITL + tok/s, esp. low/mid batch)
*Unusually strong for M*: the CPU floor means every accepted extra token rides an already-paid 26 ms
step (ITL ≈ 26/τ); and MoE stays bandwidth-bound so verification tokens stay cheap to higher batch than
dense models — BUT the verify-batch expert-union taxes high batch, so cap spec length (n≈2) at B32.*

15. **Qwen3 native MTP head, self-speculation (SHIP FIRST).** Run the MTP head k=1–2 ahead, verify in
    one packed forward, accept-longest-prefix, lossless (rejection-sampled). *Metrics: ITL −40–45%,
    tok/s ×1.8 @B1–8 (DeepSeek-V3 MTP 1.8×; Qwen3 community 1.4–2.2×); ~1.4–1.6× to B16–32. M*: Loop
    primitive already runs multi-token bodies; reuse packed-prefill verify tile; MSTAR_MTP_SPEC. Low, Q3
    — IF Qwen3-Omni ships/we train the head. Risk: multimodal τ unknown; MoE expert-union caps n at high
    batch. Src: DeepSeek-V3 arXiv:2412.19437; MoESD arXiv:2505.19645.*
16. **Dynamic draft-tree verification in the Loop primitive (SUBSTRATE for 15/17).** EAGLE-2 context-
    aware tree (expand-by-confidence + rerank + one-forward tree-mask verify). *Metrics: +20–40% over
    linear draft. M*: tree-attention-mask verify tile; bucket tree shapes for CUDA-graph capture. M, Q4.
    Src: EAGLE-2 arXiv:2406.16858.*
17. **EAGLE-3 draft head on Qwen3-Omni features (HIGHEST CEILING).** One layer autoregresses at feature
    level (fuses low/mid/high features; training-time test). *Metrics: τ=5–7.5 → up to ~4× tok/s low
    batch, and the ONLY spec method net-positive at B56 (1.01×). M*: train head on Thinker hiddens
    (multimodal feature drafting = novel); tree-verify via C16. High, Q4. Risk: training cost; audio/vision
    τ unproven. Src: EAGLE-3 arXiv:2503.01840.*
18. **Prompt-lookup / n-gram drafting (training-free).** Match last-n tokens against prompt/context,
    propose the following span. *Metrics: ITL −30–50% on echo-heavy outputs — **ideal for S2T transcription**
    and structured/JSON; ≈0 on open generation. M*: trivial Loop body; always-on cheap layer, per-modality
    gate. Trivial, Q3. Src: PLD; REST.*
19. **Parallel single-pass draft + batch-adaptive spec length.** Generate K draft tokens in one forward
    (fewer Python iters → hits the 26 ms floor); adapt K to batch load to dodge the straggler cliff.
    *Metrics: P-EAGLE up to 1.69× over EAGLE-3; adaptation converts a −30% heavy-load loss into ~0. M*:
    layer on C15–16 + scheduler policy. M, Q4. Src: P-EAGLE (vLLM); arXiv:2510.22876.*

### Workstream D — MoE & quantization kernels (owns ITL/tok·s decode + prefill TTFT GEMM)
*M*'s MoE kernel is arguably ahead of public SOTA at tiny decode-M (BLOCK_M=16 beats DeepGEMM masked) —
a perishable moat. D20+D21 are near-free; D22 is the biggest decode lever.*

20. **Per-shape autotuned Triton MoE config (NEAR-FREE, DO FIRST).** Replace the hardcoded tile
    (`kernels.py:401-420`) with an offline-tuned per-(E,N,K,dtype,M-bucket) lookup + split-K in the search
    space. *Metrics: MoE GEMM +15–35% → decode ITL/tok·s + prefill TTFT (vLLM tuned Qwen3-30B-A3B: +34%
    req/s). M*: port vLLM benchmark_moe.py + config-JSON. Low, Q3. Risk: ~zero (numerically identical);
    a missing config silently tanks decode. Src: vLLM PR#19455/#31442.*
21. **Fused moe_align + fused-topk kernel.** Routing/align/sort cost is fixed in E=128, doesn't shrink
    with M → 20–40% of the MoE block at decode + many tiny launches. Fuse SGLang's single-kernel
    moe_align (padding folded) + fused topk; int32 not int64. *Metrics: align ~3×; single-digit-% ITL but
    hits the CPU/launch floor. M*: swap utils/fused_moe/align.py. Low, Q3 — pairs with D20. Src: SGLang MoE
    align design.*
22. **W4A16 int4 experts (Marlin/Machete) — biggest decode win.** int4 weights, bf16 acts, dequant in
    mainloop; halves weight bytes vs fp8 → attacks the bandwidth-bound decode. Keep router+shared+attn
    high-precision. *Metrics: Marlin ~3.87× GEMM to B16–32 (2.8× e2e vLLM); Machete +29–32% tok/s @70B,
    no prefill regression → ITL/tok·s ~2–3× vs bf16. M*: add int4 dequant branch (stripped path exists in
    the SGLang source we ported) + packing/group-scales. M, Q3–Q4. Risk: 1–3% MMLU; validate SPEECH
    quality (moat perishable); weight-only only (Qwen3 act-quant sensitive). Src: Marlin arXiv:2408.11743;
    Machete/Red Hat.*
23. **swap-AB tiling + routing-aware tile dispatch (RaMP).** Remap token-M onto WGMMA's flexible N axis
    (BLOCK_M=32, less boundary waste); dispatch tile counts from the actual topk histogram (routing is
    skewed). *Metrics: swap-AB +8% @B2/4 → ~2% @B≥20; RaMP 1.30× vs Triton-fp8, up to 1.70× @M=32 skewed.
    M*: rewrite fused_moe_kernel (transpose+re-tile) / add RaMP dispatch. M–H, Q4. (Ignore "+180–290%" —
    pathological microbench.) Src: LMSYS swap-AB; RaMP arXiv:2604.26039.*
24. **Guard the fp8 BLOCK_M=16 moat; W4A8 for prefill only (DEFENSIVE).** Keep the winning Triton fp8
    grouped-GEMM as decode default; do NOT port DeepGEMM-masked (block_m=64 floor regresses at our M),
    W4A8-MoE (large-batch play), or FP4 (Hopper-emulated, Blackwell-only). Optional W4A8 for the prefill
    GEMM. *Metrics: protects the 1.13× fp8 win. Ongoing. Src: DeepGEMM #85/#98; SGLang #23896.*

### Workstream E — Attention & KV cache (req/s via batch headroom, TTFT via reuse, B1 latency)
25. **FP8 (e4m3) KV cache.** Store K/V fp8 (FlashInfer native); ~50% KV → up to 2× batch. *Metrics:
    req/s/tok·s + (bigger batch), ITL −15% at long ctx (break-even ~7k tokens — multimodal is past it).
    M*: allocate paged pool e4m3 + dequant scales into FlashInfer wrappers; per-tensor→per-head. L–M, Q3.
    Risk: fp8 KV is 12–35% SLOWER for short/compute-bound prefill → gate by seq-len; verify FA3 two-level-
    accumulation fix for long ctx; validate speech. Src: vLLM FP8 KV blog; FlashInfer #1753.*
26. **FP8 prefill via FA3 / trtllm-gen kernel.** FP8-qkv Hopper prefill kernel for long multimodal
    context. *Metrics: TTFT — attention 1.55× @8K/1.80× @32K, +15–25% fp8 → e2e TTFT ~15–20% @8K. M*:
    swap prefill wrapper to trtllm-gen backend; fp8 off ≥128k. M, Q3 (gated on FlashInfer landing). Note:
    "FA3 rejected" was for DECODE; this is prefill-only, where the H200 curve pays. Src: FA3 arXiv:2407.08608.*
27. **Tensor-core GQA decode + seq-aware split-KV.** Pack the 8 Q-heads/KV-head as the MMA row for a
    tensor-core decode; force KV-splits at B≤4 (4 KV heads leave >90% SMs idle). *Metrics: ITL/tok·s at
    B1 latency (FlashInfer 2–3× vs vLLM decode); split helps mainly H_KV≤2 so ~B1 only for us. M*: confirm
    decode wrapper uses tensor-core GQA path; add plan()-time split heuristic. S–M, Q3. Risk: washed by the
    CPU floor at B32 — B1-latency lever. Src: FlashInfer arXiv:2501.01005; seq-split arXiv:2604.00028.*
28. **RadixAttention/cascade prefix + multimodal encoder-embedding cache (TTFT).** Radix tree over paged
    KV keyed by token-IDs **+ media hash**; cache ViT/AuT outputs by media hash → repeated image/clip skips
    encode. *Metrics: TTFT — encoder-cache 7.8–28× on repeats; prefix reuse large on shared system prompts/
    multi-turn (SGLang up to 5–6.4×). M*: scheduler radix match + encoder-output LRU. **Correctness: prefix
    key MUST include media hash** (vLLM shipped a wrong-output bug omitting it, #20261); TM-RoPE offsets must
    match. M–L, Q3 (encoder cache first, cheap) → Q4 (radix KV). Src: SGLang arXiv:2312.07104; vLLM #20261.*
29. **KV compression for media tokens (SnapKV / PyramidKV).** Score prompt tokens by recent-window
    attention; keep top-k media KV + recent window; pyramidal per-layer budget. *Metrics: big KV shrink →
    bigger batch → req/s/tok·s + shorter cache → lower ITL. M*: decode-time budget (compress once after
    prefill). H, Q4 — quality-gated. Risk: HIGHEST quality risk (OCR/dense-caption/speech detail); per-
    modality budgets + eval harness. Src: PyramidKV arXiv:2406.02069; FastV arXiv:2503.18278.*

### Workstream F — Compile / fusion (owns the residual ITL/tok·s via launch-overhead)
30. **Kill custom-op/FlashInfer graph breaks + fx fusion passes + fused TM-RoPE.** (a) Give custom
    torch.library ops proper `register_fake`/meta so Dynamo traces THROUGH them (M* has ~311 FlashInfer
    breaks/boot); inline thin wrappers; keep only attention+fp8-GEMM custom, wrapped in fusion passes.
    (b) fx passes fusing (RMSNorm+residual+fp8-quant), (SiLU-Mul+fp8-quant), (attn+fp8-quant epilogue) —
    vLLM: attn+quant ~7%, basket ~5–10%. (c) Fused FlashInfer RoPE+Q+KV-update kernel, extended to 3-axis
    TM-RoPE, captured in the decode graph. *Metrics: ITL/tok·s — torch.compile+graphs cut per-step CPU
    30–50%; directly attacks the 26 ms floor. M*: audit MSTAR_CUSTOM_OPS ops; fx passes modeled on vLLM
    FusionPass. M, Q3 (partly in flight on opt/custom-ops). Risk: inlining can lose a hand-kernel — A/B each
    op. Src: vLLM torch.compile blog; #24629; vLLM fusion docs.*

---

## §3. Multi-quarter roadmap (dependency-ordered)

**Q3 2026 — "measure right + harvest the CPU floor + flatten TTFT" (the +50% base case):**
- Week 1: **§0 fairness fix** + quiet-box re-measure (gates everything).
- Near-free kernel wins: **D20** (autotune MoE), **D21** (fused align), **A8** (fused sampler), **A4**
  (async output). Each is S/Low and independently shippable.
- CPU-floor foundation: **A2** (persistent ready-set) → **A3** (zero-sync) → **A6** (process split).
- TTFT: **B9** (stall-free mixed step) + **B11** (fix side-prefill race) + **B12** (encoder graphs) + **B28**
  (encoder-embedding cache).
- Spec: **C15** (MTP self-spec) + **C18** (n-gram for S2T).
- Compile: **F30** (graph-break/fusion) — partly in flight.

**Q4 2026 — "the structural leapfrog + the big decode lever":**
- **A1** (GPU-native input prep / MRV2 core) on the A2 substrate — the marquee item.
- **A5** (multi-step) + **A7** (FULL_AND_PIECEWISE graphs).
- **B10** (MoE layered prefill) — the MoE-correct TTFT multiplier.
- **D22** (W4A16 int4 experts) — biggest decode ITL/tok·s lever; **C16→C17** (tree verify → EAGLE-3).
- **E25** (fp8 KV, 2× batch) + **E26** (fp8 prefill) + **E28** (radix KV).

**Q1 2027 — "diminishing-returns + high-ceiling bets":**
- **B14** (PD/EPD disaggregation) if a residual SLO conflict remains.
- **C19** (parallel/adaptive spec), **D23** (swap-AB/RaMP), **E29** (KV compression), **E27** (B1 tensor-core).

---

## §4. Does this reach +50% on ALL FOUR? (honest math)

- **ITL:** already winning; A1+A3+A7+F30 keep the lead and add margin. **Clears +50% easily.** ✅
- **tok/s:** §0 removes ~2/3 of the apparent gap; then A1+A5 (MRV2, +40–56%) + D22 (int4, ~2×) + C15 (MTP,
  1.8×) — these stack multiplicatively on decode. **+50% very likely once §0 is fixed.** ✅
- **req/s:** A-stack (+30–50%) + B9 (TTFT flattening lifts B32 JCT) + D20/D22. **+50% likely at B≥8.** ✅
- **TTFT:** the hard one — CPU-floor work barely helps first-token, and A5 can hurt it. **Needs B9+B10+
  B11+B12+B13+B28 (the entire WS-B).** Target: 2532→~250 ms @B32 (a 10× cut, well past +50%). Achievable
  but it's the workstream that must not slip. ⚠️→✅

**Bottom line:** +50% on ITL/tok·s/req·s is a stacking exercise on the CPU-floor + MoE-kernel + spec
levers (A + D + C). +50% on **TTFT is a single-workstream bet on WS-B** (mixed prefill+decode). And §0
may reveal the current gap is smaller than measured — do it first.

## §5. The 5 "if you only do five" picks
1. **§0 fairness fix** (week 1 — may erase half the apparent gap).
2. **D20 autotune MoE config** (near-free +15–35%, the hardcoded-tile bug).
3. **B9 stall-free mixed prefill+decode** (the TTFT fix; branches exist).
4. **A1+A2 MRV2-equivalent** (kills the 26 ms floor; leapfrogs vLLM-Omni before it rebases to 0.24).
5. **C15 MTP self-spec** (~1.8× tok/s where we're weakest; nearly free given the CPU floor + MoE).

*All 30 options carry primary-source citations inline. Numbers are adjacent-system measurements, not yet
M*-measured — validate each in-CUDA-graph on a quiet box (per our contention finding) before trusting.*

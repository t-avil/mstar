# M* branch index (encoders-implemented-md)

Every branch on the fork, newest first. ★ = headline results. `-delta` branches (on the
fork) hold local tips that had diverged from the remote — pushed under a suffix so nothing
was clobbered; diff `<branch>` vs `<branch>-delta` to see the local delta.

## Headline branches (best results)

- **`opt/moe-autotune`** — ★ CURRENT GOD-BRANCH. Full winning stack (custom-ops, moe-fp8, mixed-batch+merged-prefill, sidecar-checkstop, fast-route/send, prep-device-pos-batched) + this session's adds: MoE-parity fix (BLOCK_SIZE_K=64 byte-identical), MSTAR_SYMM_ALLREDUCE (PyTorch symm-mem all-reduce for TP=2), DP-replica configs + MSTAR_DP_ROUND_ROBIN, TP=2 boot fixes. Wins i2t B1-16, s2t ALL batches, ITL everywhere; loses only i2t B32.
- **`opt/prep-pos-batched-v9`** — E1 — the canonical documented BEST build (launch_mstar_best.sh). vs vLLM: B1 1.21x, B8 1.22x, B16 1.10x; s2t/s2s/i2s all won. + PREP_DEVICE_POS_BATCHED (+5.9% B32).
- **`opt/stack-n2`** — flagship2 — best-measured i2t B32 (0.938x +12%, peaks 1.048 touching vLLM band); B1 0.972 +8%; all 5 modality cells 8.10-9.01 req/s. = decode-merge + cfgcache-v2 + sidecar-checkstop.
- **`opt/integration-v4`** — Six composed wins (sidecar +9% B32, ...).
- **`opt/prefill-merge-audio`** — s2t merge win: B2 +11%, B4 +13% (a2d2f47).

## All branches

 | date | branch | tip | description / commit subject |
 |---|---|---|---|
 | 2026-07-08 | `benchmarks` | e406dedf | Aggregation branch — all benchmark artifacts (raw.json, charts, results). |
 | 2026-07-08 | `encoders-implemeneted-benchmarked-mstar-v2` | e406dedf | Bench working branch — charts + docs (win-map, i2t/s2t 2x2 scoreboards, reviews, EXPERIMENTS). |
 | 2026-07-08 | `opt/moe-autotune` | 6df89397 | ★ CURRENT GOD-BRANCH. Full winning stack (custom-ops, moe-fp8, mixed-batch+merged-prefill, sidecar-checkstop, fast-route/send, prep-device-pos-batched) + this s |
 | 2026-07-08 | `test/fullgraph` | 32d89781 | feat(side): MSTAR_SIDE_ENCODER_ONLY — route only stateless encoder to side stream |
 | 2026-07-08 | `test/moeparity` | 32d89781 | feat(side): MSTAR_SIDE_ENCODER_ONLY — route only stateless encoder to side stream |
 | 2026-07-07 | `opt/sideprefill-fix` | 586fcfa1 | thread-local _ACTIVE_MANAGER + isolated side FlashInfer workspace (fixes SIDE_PREFILL shape-race). |
 | 2026-07-07 | `opt/decode-multistep` | b787e0bf | MSTAR_DECODE_MULTISTEP: greedy-identical but B32 regresses 3.5x (inline re-plan). Parked. |
 | 2026-07-06 | `opt/prep-pos-batched-v9` | 915ab8f3 | E1 — the canonical documented BEST build (launch_mstar_best.sh). vs vLLM: B1 1.21x, B8 1.22x, B16 1.10x; s2t/s2s/i2s all won. + PREP_DEVICE_POS_BATCHED (+5.9% B |
 | 2026-07-05 | `opt/sidecar-batch` | 8b775a31 | perf(sidecar): MSTAR_SIDECAR_BATCH — coalesce worker→sidecar sends (B32 11.5% line) |
 | 2026-07-05 | `opt/prep-h2d` | 620de912 | Winning-build base @620de91 (E1's predecessor); MSTAR_PREP_DEVICE_POS dynflags-refreshable. |
 | 2026-07-05 | `opt/stack-n2` | 7ef5150b | flagship2 — best-measured i2t B32 (0.938x +12%, peaks 1.048 touching vLLM band); B1 0.972 +8%; all 5 modality cells 8.10-9.01 req/s. = decode-merge + cfgcache-v |
 | 2026-07-05 | `opt/cfgcache-v2` | ded928d7 | fix(sampler): re-sync V2 per-slot rand on OFF→ON flip; document ring depth |
 | 2026-07-05 | `opt/prefill-gather` | 092d18f2 | feat(sched): extend gather to encode_vision (batched encoder admission) |
 | 2026-07-04 | `opt/prefill-merge-audio` | a2d2f476 | s2t merge win: B2 +11%, B4 +13% (a2d2f47). |
 | 2026-07-04 | `opt/w2-retest` | a2788a97 | feat(worker): MSTAR_STEP_TXN — memoized decode-step transaction, rebased onto opt/custom-ops (W2 retest) |
 | 2026-07-04 | `opt/admit-jitter` | 0c682680 | feat(sched): MSTAR_ADMIT_JITTER_MS — admission-wave smoothing (lever a) |
 | 2026-07-04 | `opt/sidecar-checkstop` | 5414a5ae | MSTAR_SIDECAR_CHECKSTOP Stage-2 deferred-consume check_stop offload. |
 | 2026-07-04 | `opt/custom-ops` | 12ce776d | MSTAR_CUSTOM_OPS torch.library ops (run_attention/fused_experts_fp8/apply_rope) — removes graph breaks for fuller compile. |
 | 2026-07-04 | `opt/integration-v4` | 52d91165 | Six composed wins (sidecar +9% B32, ...). |
 | 2026-07-04 | `opt/prefill-merge` | 41200ecd | feat(prefill): merged multimodal prefill walk (MSTAR_MERGED_PREFILL, B1/B5) |
 | 2026-07-04 | `opt/v2-policy` | 56e65f8d | MSTAR_MIXED_BUDGET_TOKENS every-step chunk fold (default off). |
 | 2026-07-03 | `opt/compile-fix` | 1733fabf | perf(compile): pure-torch RMSNorm under torch.compile — the FlashInfer call graph-broke at every norm (~311 breaks/boot), blocking Inductor fusion of norm->resi |
 | 2026-07-03 | `opt/speech-floor` | 01856e80 | speech-floor Item C: MSTAR_CODEC_CHUNK_EMIT chunk-batched codec edge |
 | 2026-07-03 | `opt/async-sched` | 18b12446 | fix(async-sched): guard deferred-postprocess rids in the direct remove path |
 | 2026-07-03 | `opt/sched-pack` | 74a985cd | triage: MSTAR_DIRECT_FEED dynflags-refreshable (per-step branch, E9-validated value-identical paths) |
 | 2026-07-03 | `many-charts` | 894b719d | many-charts: turnkey pipeline (make_charts.sh), 2x2 grids, tok/s panels, d2d1983 s2t-old, vLLM version labels |
 | 2026-07-03 | `exp/overlap-sched` | ab750346 | test(sidecar): byte-identity harness — flag-on vs legacy message streams |
 | 2026-07-03 | `exp/fold-rate` | a65e0ed5 | feat(split-attn): re-enable packed pre-plan under split — engine pads the fixed-region shape |
 | 2026-07-02 | `encoders-implemeneted-benchmarked-v3-w5-datapoints` | 9b916ea9 | docs(w5-datapoints): addendum — pre-plan slow datapoints were host contention; theory retracted |
 | 2026-07-02 | `exp/mixed-batch-p2` | 4d9d6e37 | diag: classify standalone prefill steps by chunked/unchunked and size bucket |
 | 2026-07-02 | `encoders-implemeneted-benchmarked-v3` | caf9a3ed | docs: packed pre-plan rejected pending stream-race debug (clean run -50%; asserts masked the race) |
 | 2026-07-02 | `exp/chunked-prefill-v2` | 23d07ee5 | fix(chunked-vision): always emit declared input edges — readiness requires every input_names entry |
 | 2026-07-02 | `exp/prefill-buckets` | 08e71238 | exp(graphs): denser prefill token buckets (384, 768, 1536) |
 | 2026-07-02 | `exp/step-txn` | b98de665 | feat(worker): MSTAR_STEP_TXN — memoized continuing-decode step transaction (default OFF) |
 | 2026-07-02 | `encoders-implemeneted-v2` | 2239005b | opt(moe): tuned fp8 prefill tiles — BLOCK_M=64/GROUP=8, 1.63x on prefill MoE |
 | 2026-07-02 | `opt/decode-v2` | 2239005b | opt(moe): tuned fp8 prefill tiles — BLOCK_M=64/GROUP=8, 1.63x on prefill MoE |
 | 2026-07-02 | `exp/batched-postprocess-v2` | 1e171e1e | Merge branch 'exp/batched-postprocess' into exp/batched-postprocess-v2 |
 | 2026-07-02 | `exp/combo-w1w7` | c8cebc76 | Merge branch 'exp/bucket24' into exp/combo-w1w7 |
 | 2026-07-02 | `exp/bucket24` | 845faff7 | exp(graphs): denser thinker decode buckets (24, 28) to cut B32 padding waste |
 | 2026-07-02 | `exp/batched-postprocess` | 4ce01253 | opt(worker): memoized decode postprocess store/populate (MSTAR_FAST_POSTPROC) |
 | 2026-07-02 | `exp/two-step-decode` | 6bf61365 | opt(worker+engine): two-step decode in one submission (MSTAR_MULTISTEP_DECODE) |
 | 2026-07-02 | `exp/direct-feed` | 6644913a | opt(worker): direct token feed for AR decode speculation (MSTAR_DIRECT_FEED) |
 | 2026-07-02 | `encoders-implemeneted-benchmarked-mstar-new-v2` | 7d427d76 | fix(numbers): mark stale vLLM columns invalid — contended 1f66ce6 run superseded by raw_*.json |
 | 2026-07-01 | `opt/mixed-walk` | 4334810e | design + eager mixed prefill+decode (DESIGN_mixed_walk_graph.md). |
 | 2026-07-01 | `opt/vllm-parity` | 360bdf3b | opt(moe): H200-tuned decode tiles + block-fp8 w8a8 MoE (gated) |
 | 2026-07-01 | `encoders-implemeneted-benchmarked-mstar-old-v2` | c0d90856 | bench(qwen3-omni): refresh M*-old (ae7d173) on clean GPUs 6,7 (v2) |
 | 2026-07-01 | `encoders-implemeneted-benchmarked-vllmomni-v2` | 5c27c12e | bench(qwen3-omni): refresh vLLM to vllm-omni 3fe3aa359 / vllm 0.22.0 (v2) |
 | 2026-06-30 | `encoders-implemeneted` | 4c33b33a | qwen3-omni encoders: fix ruff lint |
 | 2026-06-30 | `encoders-implemeneted-benchmarked` | 6e91c966 | qwen3-omni encoders: fix ruff lint |
 | 2026-06-30 | `encoder-v4-benchmarked` | ba10df35 | merge encoder-v4 (ruff lint fixes) into encoder-v4-benchmarked |
 | 2026-06-30 | `encoder-v4` | 0099f137 | lint(qwen3-omni): ruff-clean the native encoders + parity tests |
 | 2026-06-30 | `pr/native-encoders-benchmarks-v3` | 47af66f6 | bench: unify mstar_old commit to ae7d173 + drop degenerate old S2T ITL |
 | 2026-06-30 | `bench-v3-codec25` | 85a29565 | docs(bench): codec_chunk_frames 25 for M*-new (reran at 25, matches upstream) |
 | 2026-06-30 | `v3-finalist` | 4bc86578 | qwen3-omni: review fixes on native encoders (#131) |
 | 2026-06-30 | `pr/native-encoders-fixes` | fdd8b38b | qwen3-omni: ship benchmarked encoder defaults + fix dormant bugs (#131) |
 | 2026-06-29 | `bench/qwen3-omni-joint` | 8f163051 | docs(LEARNINGS): correct mixed-walk-piggyback status — implemented & benchmarked |
 | 2026-06-29 | `pr/native-encoders-v3` | fc2c1d81 | qwen3-omni: native audio/vision encoders (#131) |
 | 2026-06-29 | `opt/encoder-gap` | 37248024 | opt(encoder-gap): cut CPU overhead between encoder and Thinker prefill |
 | 2026-06-29 | `opt/combined-lowrisk` | 1f66ce68 | opt(compile-dynamic): torch.compile(dynamic=True) on encoder forward |
 | 2026-06-29 | `opt/vision-sync-elim` | 5045c496 | opt(async-off): default MSTAR_ENCODER_ASYNC off (flaky) |
 | 2026-06-29 | `opt/compile-dynamic` | 87a46f79 | opt(async-off): default MSTAR_ENCODER_ASYNC off (flaky) |
 | 2026-06-29 | `opt/encoder-cudagraph` | 5715a149 | opt(async-off): default MSTAR_ENCODER_ASYNC off (flaky) |
 | 2026-06-29 | `opt/async-off` | 05815833 | opt(async-off): default MSTAR_ENCODER_ASYNC off (flaky) |
 | 2026-06-29 | `integration-mnew-v2` | 7e47ebc0 | integration-mnew-v2: async encoder default ON, vision-only |
 | 2026-06-29 | `exp/spatial-merge-node` | 67c9e371 | exp(qwen3-omni): split spatial merge into its own GraphNode (MSTAR_SPATIAL_MERGE_NODE) |
 | 2026-06-29 | `exp/encoder-cache` | 811d65d3 | exp(qwen3-omni): encoder output cache by content hash (MSTAR_ENCODER_CACHE) |
 | 2026-06-29 | `exp/encoder-chunk-coalesce-with-prefill` | 6457f7ea | exp4: B-only sweep script (encoder coalesce + chunked prefill) |
 | 2026-06-29 | `exp/encoder-async-schedule` | 04a8cc60 | exp: cross-path results — PROMISING on I2T, NEGATIVE on S2T, NEUTRAL on I2S |
 | 2026-06-29 | `exp/encoder-chunk-coalesce` | d5e16b40 | exp: full B-only sweep results — NEUTRAL, park |
 | 2026-06-29 | `exp/encoder-placement-profiling` | 162a2349 | exp: ninja on PATH + full I2T sweep script (placement profile) |
 | 2026-06-29 | `pr/native-encoders-benchmarks-v2` | 907acb2f | bench(qwen3-omni): native encoder serving benchmark data + verification (v2) |
 | 2026-06-29 | `pr/native-encoders-benchmarks` | 8449b0db | bench(qwen3-omni): add sweep.sh entry point for reproducible benchmark runs |
 | 2026-06-29 | `pr/native-encoders` | 5d46b9c3 | qwen3-omni: native audio/vision encoders (#131) |
 | 2026-06-29 | `pr/native-encoders-v2` | 5d46b9c3 | qwen3-omni: native audio/vision encoders (#131) |
 | 2026-06-29 | `opt/combined-vision-opts` | e943d721 | Merge branch 'opt/batch-vision-prefill' into opt/combined-vision-opts |
 | 2026-06-29 | `opt/batch-vision-prefill` | 7aa21abb | opt(qwen3-omni): batched vision prefill (MSTAR_BATCH_VISION_PREFILL) |
 | 2026-06-29 | `opt/vision-cudagraph-align` | 6314d58f | opt(qwen3-omni): align prefill_vision CUDA graph buckets (MSTAR_VISION_GRAPH_ALIGN) |
 | 2026-06-29 | `exp/vision-cudagraph` | 3c578531 | vision: align Thinker prefill_vision CUDA-graph buckets (MSTAR_VISION_GRAPH_ALIGN) |
 | 2026-06-29 | `main` | 9ee13699 | feat(benchmark): standardized sweep entry point (benchmark/sweep.sh) |
 | 2026-06-29 | `bench/spatial_merge` | f58a805c | fix(qwen3-omni): guarantee a Thinker prefill_text step for templated text |
 | 2026-06-29 | `integration-mnew` | f58a805c | fix(qwen3-omni): guarantee a Thinker prefill_text step for templated text |
 | 2026-06-29 | `exp/combined-coalesce-piggyback` | 0671191f | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/compose-textout` | a01b24e1 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/config-knobs` | 6db60c10 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/cuda-graph-bucketing` | 5d8f614a | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/encoder-coalesce` | 79e8e62e | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/fp8-quant` | c2bc6f32 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/mixed-cg-bucketed` | 6bdcb907 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/mixed-cg-coarse` | 6bdcb907 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/mixed-cg-decodeonly` | 6bdcb907 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/mixed-cg-supergraph` | 6bdcb907 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/mixed-walk-piggyback` | 6bdcb907 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/moe-kernels` | 96a2e440 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/parity-mode` | a867353f | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/precision-toggles` | 7257b9bd | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/spec-decode-mtp` | 553bba74 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/talker-batchfill` | 81ac4a03 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/talker-pending-queue` | 6afd018a | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/token-reduction` | c5ac8732 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/vocoder-chunk-adaptive` | e4e96e96 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `qwen3-omni-unified` | 064bbe5c | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/async-audio-pipeline` | 7929e3fc | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/batch-vision-prefill` | 7fb7e456 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `exp/chunked-prefill` | db82a4d5 | revert(benchmark): force Thinker greedy on all paths (match main) |
 | 2026-06-29 | `bench/encoder-coalesce` | 0e3ae26e | bench: A/B encoder coalescing (MSTAR_ENCODER_COALESCE) S2T B=1,4,8 |
 | 2026-06-29 | `exp/vocoder-adaptive-chunk` | 9037a0e0 | feat(qwen3-omni): adaptive vocoder chunk size based on batch level |
 | 2026-06-29 | `exp/encoder-placement` | f407d8aa | feat: encoder placement reshuffle — move encoders to Rank 0 (config-only) |
 | 2026-06-28 | `bench/qwen3-omni-unified` | e54497a3 | bench(qwen3-omni): unified harness — fast targeted runner + reproducible final + scripted charts |
 | 2026-06-28 | `fusion-experiment` | d5626a52 | WIP: fusion experiment — engine/worker/model changes |
 | 2026-06-28 | `codec-chunk` | 41f81375 | feat(qwen3-omni): env knob for codec chunk size (Lever 1) — net-negative, default OFF |
 | 2026-06-28 | `audio-encoder-opt` | d2ef987e | perf(qwen3-omni audio): env-gated GPU log-mel feature extraction (MSTAR_GPU_MEL) |
 | 2026-06-28 | `code2wav-sp` | a8c76034 | test(qwen3-omni): Code2Wav SP parity — cover large chunks (throughput-path compose) |
 | 2026-06-28 | `bench/chunked-prefill-ab` | 0f8fc45b | docs: one-week handoff plan to finish #131 (validated wins, fair baseline, week plan) |
 | 2026-06-28 | `bench/encoder-placement` | 0f8fc45b | docs: one-week handoff plan to finish #131 (validated wins, fair baseline, week plan) |
 | 2026-06-28 | `bench/itl-speech` | 0f8fc45b | docs: one-week handoff plan to finish #131 (validated wins, fair baseline, week plan) |
 | 2026-06-28 | `bench/itl-speech-conc` | 0f8fc45b | docs: one-week handoff plan to finish #131 (validated wins, fair baseline, week plan) |
 | 2026-06-28 | `review/qwen3-omni-prompt-layout` | 0f8fc45b | docs: one-week handoff plan to finish #131 (validated wins, fair baseline, week plan) |
 | 2026-06-28 | `ttft-profile` | 6c3f6636 | bench(ttft-decompose): B=1 per-stage TTFT breakdown for Qwen3-Omni S2T/I2T |
 | 2026-06-28 | `merge-prefill-walks` | 7c7cd553 | bench(merge-prefill-walks): B=1 TTFT A/B — merge is correct but no measurable win |
 | 2026-06-28 | `gpu-img-preprocess` | 0cb3f988 | feat(qwen3-omni): env-gated GPU image preprocessing (resize+patchify) |
 | 2026-06-28 | `vllm-layout` | 09e96b8b | feat(qwen3-omni): token+position parity with vLLM (FIX1 system dup, FIX2 audio M-RoPE) |
 | 2026-06-28 | `bench/qwen3-omni-s2s-mstar-old` | 4616aae0 | bench(s2s-mstar-old): Qwen3-Omni S2S M*-old (HF encoder) closed-loop sweep B=1..32 |
 | 2026-06-28 | `bench/qwen3-omni-i2s-mstar-old` | e1e1f50f | bench(i2s-mstar-old): Qwen3-Omni I2S M*-old (HF encoder) closed-loop sweep B=1..32 |
 | 2026-06-28 | `bench/qwen3-omni-s2s-vllm` | 10c0db46 | bench(s2s-vllm): Qwen3-Omni S2S vLLM-Omni closed-loop sweep B=1..32 |
 | 2026-06-28 | `bench/qwen3-omni-i2s-vllm` | 4a817d6e | bench(i2s-vllm): Qwen3-Omni I2S vLLM-Omni closed-loop sweep B=1..32 |
 | 2026-06-28 | `bench/qwen3-omni-seedtts-2gpu` | 2482f534 | benchmark(qwen3-omni-seedtts-2gpu): switch Figure 5 to closed-loop (canonical protocol) |

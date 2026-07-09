# DOC CATALOG — all markdown, sorted newest-first

Triage guide: **memory/** = current agent findings (highest signal). **benchmarks/** dated
2026-07-06+ = the final push (win-map, refactors, lever-exhaustion, V9 review). Older
benchmarks docs (HANDOFF_V5, NUMBERS, early RESULTS) are superseded. **branch-unique/** =
design notes tied to one experimental branch (relevant only if working that branch).
Dates = git first-add (tracked) or file mtime (memory/home).

| date | relevance | category | source branch/worktree | file | description |
|---|---|---|---|---|---|
| 2026-07-08 | HIGH (current findings) | memory | agent memory (all branches) | `MEMORY.md` | - [Full benchmark metrics](feedback_benchmark_metrics.md) — always capture TTFT/ITL/RTF/JCT/throughput, copy new runner to old codebases |
| 2026-07-08 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_bench_env.md` | Environment quirks when launching mstar Qwen3-Omni servers + benchmarks in /home/tim worktrees (SHM protocol, ninja PATH, offline datasets) |
| 2026-07-08 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_dp_replicas.md` | M* data-parallel Thinker replicas 2026-07 — DP wins decode/tok-s (ITL 3-6ms) but loses short-output req/s (encode-Thinker contention); TP=2  |
| 2026-07-08 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_parity_ideas.md` | M* 5 parity-safe throughput ideas (2026-07-08 code review) + CRITICAL: MSTAR_MOE_AUTOTUNE is NOT byte-identical (shipped parity bug) |
| 2026-07-08 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_ttft_root_cause.md` | M* i2t B32 loss root cause 2026-07 — TTFT (prefill serialization), NOT decode; one-walk-per-batch freezes decodes; ENCODER_ASYNC is measured |
| 2026-07-07 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `AR_LOOP_10_REFACTORS.md` | M* AR decode loop — empirical CPU floor + 10 refactors to make it Python-free / torch.compile-able |
| 2026-07-07 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `B32_LEVER_EXHAUSTION_LOG.md` | B32 i2t gap — exhaustive lever log & verdict (2026-07-07) |
| 2026-07-07 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `BEAT_VLLM_MASTERPLAN.md` | BEAT vLLM — Master Plan: +50% on TTFT, ITL, req/s, tok/s (Qwen3-Omni 30B-A3B, H200) |
| 2026-07-07 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `LOOP_THROUGHPUT_RESULTS.md` | Loop results — i2t throughput push vs vLLM (2026-07-07) |
| 2026-07-07 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_decode_profile.md` | Empirical py-spy floor of M* i2t B32 decode worker 2026-07 — _postprocess_batch 23%, check_stop D->H 18%, zmq 13%, sync 8% |
| 2026-07-07 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_throughput_levers.md` | M* i2t throughput lever results 2026-07 — MoE-autotune washes, naive multi-step regresses; TTFT dominates B32 JCT |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `BENCH_V9_RESULTS.md` | BENCH V9 — fresh i2t baseline (vs vLLM 0.22) + E1 fix A/B |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `FEATURES_SINCE_ENCODERS.md` | FEATURES_SINCE_ENCODERS — complete feature set of the M* code delta |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `HANDOFF_V8.md` | HANDOFF_V8 — M* vs vLLM-Omni campaign (state as of 2026-07-06) |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `PR_DECOMPOSITION.md` | PR_DECOMPOSITION — how to upstream the M* delta as small, decoupled PRs |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `REVIEW_V9_WHY_WE_LOSE.md` | REVIEW V9 — Why M* loses to vLLM-Omni 0.22 on text paths, and how to fix it |
| 2026-07-06 | HIGH (recent) | benchmarks | benchmarks / encoders-…-v2 | `TORCH_COMPILE_FINAL.md` | TORCH_COMPILE_FINAL — final static audit of torch.compile in M* |
| 2026-07-06 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_boot_shm_leak.md` | M* server boot OOM at set_device is an orphaned /dev/shm leak (host RAM), NOT a loader/GPU race |
| 2026-07-06 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_sweep_v5_boot_lottery.md` | 2026-07-06 full 24-cell sweep committed; boot-to-boot variance measured at 18% on identical code — boot determinism is the next lever for i2 |
| 2026-07-06 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_vllm_mrv2.md` | Why M* lost text paths — vLLM core 0.22 shipped Model Runner V2 (zero CPU-GPU sync decode); corrected scoreboard |
| 2026-07-05 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `HANDOFF_V7.md` | HANDOFF_V7 — M* vs vLLM-Omni campaign (state as of 2026-07-05 ~02:00 UTC) |
| 2026-07-05 | MED (workspace) | home | /home/tim (all branches) | `AGENTS.md` | Experimental Discipline |
| 2026-07-05 | MED (workspace) | home | /home/tim (all branches) | `CLAUDE.md` | Benchmarking and GPU workspace conventions |
| 2026-07-05 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_sidecar_checkstop.md` | Sidecar Stage-2 check_stop offload (MSTAR_SIDECAR_CHECKSTOP) — built, branch, scope, A/B recipe |
| 2026-07-05 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_stack_n2.md` | opt/stack-n2 is the new best build (merge + cfgv2 + checkstop); flagship 0.938 pooled with peaks >1.05; variance is the last gap; Law 14 see |
| 2026-07-04 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `GOAL.md` | GOAL — Beat vLLM-Omni v0.22 by ≥5% on every path × every batch |
| 2026-07-04 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `GOAL_MATRIX.md` | GOAL_MATRIX — current best-defensible state of every cell |
| 2026-07-04 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `HANDOFF_V5.md` | HANDOFF_V5 — M* vs vLLM-Omni campaign (state as of 2026-07-04 ~18:45 UTC) |
| 2026-07-04 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `HANDOFF_V6.md` | HANDOFF_V6 — M* vs vLLM-Omni campaign (state as of 2026-07-04 ~23:30 UTC) |
| 2026-07-04 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `VLLM_RELIABILITY.md` | vLLM-Omni reliability ledger |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_24option_board.md` | 2026-07-03 re-research verdict on campaign docs + the 24-option board and phased plan (PLAN_BEAT_VLLM_V4.md) |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_campaign_20260704.md` | Campaign night 2026-07-04 — 20/24 goal cells green live; merge-config promoted; fold/gather/jitter falsified; regime inverted (main<GPU); vL |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_custom_ops_crusade.md` | opt/custom-ops branch — torch.library custom ops removed the compiled-thinker graph breaks (816→~165), MSTAR_CUSTOM_OPS=1; boot-time cache f |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_lab_wrapper_reaping.md` | lab_server.sh wrapper dies when the agent shell tree is reaped between turns — setsid it |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_merged_prefill.md` | M* merged multimodal prefill walk (MSTAR_MERGED_PREFILL, plan B1/B5) — branch, design, key gotcha |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_v2_budget_policy.md` | V2 budgeted chunked-prefill admission (MSTAR_MIXED_BUDGET_TOKENS) built on opt/v2-policy, awaiting GPU smoke |
| 2026-07-04 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_w2_retest.md` | W2 step-txn memoization retest rebased onto opt/custom-ops — branch, flag, gates, prediction |
| 2026-07-04 | LOW (older/superseded) | branch-unique | opt/prefill-gather  (mstar-gather) | `mstar-gather__DESIGN_merged_prefill.md` | Merged multimodal prefill walk (`MSTAR_MERGED_PREFILL`) |
| 2026-07-03 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `BEATING_NEW_VLLM.md` | BEATING NEW vLLM — campaign handoff |
| 2026-07-03 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `NUMBERS_V3.md` | NUMBERS_V3.md — M*-v3 PREVIEW (final stack: v2 + W5 mixed + SLIM_EMIT + FAST_ROUTE + |
| 2026-07-03 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `PLAN_BEAT_VLLM_V4.md` | PLAN_BEAT_VLLM_V4 — re-research, 24-option board, and execution plan |
| 2026-07-03 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `VLLM_022_GRID.md` | vLLM 0.22 vs our engine — what they changed, what we built, in plain language |
| 2026-07-03 | MED (workspace) | home | /home/tim (all branches) | `EXPERIMENTS.md` | Flag decomposition on the sidecar stack (labs, 3 rounds each) — the stack re-baselined |
| 2026-07-03 | HIGH (current findings) | memory | agent memory (all branches) | `feedback_experiment_report_format.md` | Required per-experiment report format — losing-paths table + one-sentence tried + one-sentence next |
| 2026-07-03 | HIGH (current findings) | memory | agent memory (all branches) | `project_moe_bakeoff_24.md` | M* campaign |
| 2026-07-03 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_decode_bottleneck.md` | Measured decomposition of M* qwen3-omni B32/B1 text-decode step (nsys+NVTX, 2026-07-02) and validated/rejected optimizations |
| 2026-07-03 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_talker_code2wav_colocation.md` | Talker+Code2Wav colocation can't be done by a yaml; the default already colocates them; fusing the codec edge needs a code change |
| 2026-07-03 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_v1_async_sched.md` | V1 async-scheduling (deferred postprocess) for M* — built, GPU-smoked, FAILED (non-identical +9% length, tok/s wash); parked |
| 2026-07-03 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__sweep_canon_20260703__README.md` | Canonical-pair (GPUs 6,7) runs of the final stack, 2026-07-03 19:27-19:59 UTC |
| 2026-07-03 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__sweep_mstar_v3__MANIFEST.md` | sweep_mstar_v3 — final-stack preview cells (2026-07-03, GPU pair 0,1) |
| 2026-07-03 | LOW (older/superseded) | branch-unique | opt/prefill-gather  (mstar-gather) | `mstar-gather__SMOKE.md` | SMOKE — merged multimodal prefill (`MSTAR_MERGED_PREFILL`, opt/prefill-merge) |
| 2026-07-03 | LOW (older/superseded) | branch-unique | opt/prefill-gather  (mstar-gather) | `mstar-gather__docs__SIDECAR_DESIGN.md` | SIDECAR_DESIGN — exiling per-token emit/postprocess work from the worker GIL |
| 2026-07-03 | LOW (older/superseded) | branch-unique | opt/speech-floor  (mstar-speech) | `mstar-speech__SMOKE.md` | SMOKE — M* speech host-floor bundle (opt/speech-floor) |
| 2026-07-03 | LOW (older/superseded) | branch-unique | opt/async-sched  (mstar-v1async) | `mstar-v1async__SMOKE.md` | SMOKE — MSTAR_ASYNC_SCHED (V1 async scheduling / GPU-resident sampled ids) |
| 2026-07-03 | LOW (older/superseded) | branch-unique | opt/v2-policy  (mstar-v2pol) | `mstar-v2pol__SMOKE.md` | SMOKE — V2 budgeted chunked-prefill admission (opt/v2-policy) |
| 2026-07-02 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `EXPERIMENTS.md` | Qwen3-Omni optimization — experiment knowledge base |
| 2026-07-02 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `HEADTOHEAD.md` | HEADTOHEAD.md — live same-window A/B vs vLLM 0.22 (supplementary) |
| 2026-07-02 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `NUMBERS_V2.md` | NUMBERS_V2.md — M*-v2 (opt/decode-v2 1e171e1 (+FAST_POSTPROC), fp8 MoE + fused topk + |
| 2026-07-02 | HIGH (current findings) | memory | agent memory (all branches) | `feedback_ingraph_microbench.md` | Decode-size GPU kernel microbenchmarks are invalid outside CUDA graphs — capture a loop in a graph and time replays |
| 2026-07-02 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_mixed_batch_p2.md` | W5-P2 mixed prefill+decode captured batch — code-verified design facts for exp/mixed-batch-p2 |
| 2026-07-02 | HIGH (current findings) | memory | agent memory (all branches) | `project_mstar_vllm_decode_gap.md` | Root-cause decomposition of why M* text-decode throughput trails vLLM 0.22, and which levers actually help |
| 2026-07-02 | LOW (older/superseded) | branch-unique | opt/prefill-gather  (mstar-gather) | `mstar-gather__DESIGN_chunked_prefill_v2.md` | W5 Phase 1 — Chunked Thinker Prefill (V2) |
| 2026-07-01 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__DESIGN_mixed_walk_graph.md` | Graphed mixed replay — implementation plan (the stubbed piece) |
| 2026-06-30 | MED (workspace) | home | /home/tim (all branches) | `MIXED_CG_GROUNDING.md` | Grounding: CUDA-graph capture for the MIXED prefill+decode walk |
| 2026-06-30 | MED (workspace) | home | /home/tim (all branches) | `RESEARCH_qwen3omni_cudagraph_compile_batching.md` | Qwen3-Omni on M*: CUDA Graphs, Batching, and torch.compile — Presentation Brief |
| 2026-06-30 | MED (workspace) | home | /home/tim (all branches) | `REVIEW_encoder_v4.md` | Review: Qwen3-Omni native encoders (issue #131) — encoder-v4-new (b15bfea) + benchmarked-new (a3a25ca) |
| 2026-06-30 | MED (workspace) | home | /home/tim (all branches) | `REVIEW_native_encoders.md` | Review: Qwen3-Omni native encoders (#131) + benchmark showcase |
| 2026-06-30 | MED (workspace) | home | /home/tim (all branches) | `STORY.md` | The Time‑to‑First‑Token / Inter‑Token‑Latency Trade‑off in Image‑to‑Text Serving |
| 2026-06-29 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `LEARNINGS.md` | Qwen3-Omni Serving Optimization: Experiment Log and Learnings |
| 2026-06-29 | MED (workspace) | home | /home/tim (all branches) | `CANONICAL_FLAGS.md` | Canonical "full optimization flag set" — Qwen3-Omni serving (mstar-encoders) |
| 2026-06-29 | HIGH (current findings) | memory | agent memory (all branches) | `feedback_benchmark_metrics.md` | All benchmarks must capture TTFT, ITL, RTF, JCT, and throughput — never run with the old runner that only reports JCT |
| 2026-06-29 | HIGH (current findings) | memory | agent memory (all branches) | `feedback_mstar_new_label.md` | mstar_new on benchmark branch = current shipping-candidate build; label moves forward as optimizations land |
| 2026-06-29 | HIGH (current findings) | memory | agent memory (all branches) | `feedback_mstar_worktree_pythonpath.md` | mstar worktrees need PYTHONPATH=<worktree> on every launch — editable install + spawn silently loads a DIFFERENT worktree's code, invalidati |
| 2026-06-29 | LOW (older/superseded) | branch-unique | (detached)  (bench-v2) | `bench-v2__benchmarks__qwen3-omni-native-encoders__BENCHMARK.md` | Native Encoder Benchmark Evidence |
| 2026-06-29 | LOW (older/superseded) | branch-unique | (detached)  (bench-v2) | `bench-v2__benchmarks__qwen3-omni-native-encoders__NUMBERS.md` | NUMBERS.md -- headline numbers (auto-generated) |
| 2026-06-29 | LOW (older/superseded) | branch-unique | (detached)  (mstar-lowrisk) | `mstar-lowrisk__RESULTS.md` | Exp 2: Async Encoder Scheduling — Cross-Path Results |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `EXPLANATION_GRID.md` | Why M\*-new wins (and where it doesn't): the code-rooted grid |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `FINDINGS_section7.md` | <!-- |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `NUMBERS.md` | NUMBERS.md -- headline numbers (auto-generated by aggregate.py --refine-dir) |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `PR_SUMMARY.md` | <!-- |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `RECHECK_I2S.md` | I2S low-batch recheck (B=1, B=2) — variance vs regression |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `RESULTS.md` | Qwen3-Omni on M* — joint benchmark results (#131) |
| 2026-06-28 | LOW (older/superseded) | benchmarks | benchmarks / encoders-…-v2 | `REVIEWER_START.md` | Reviewer — start here |
| 2026-06-28 | HIGH (current findings) | memory | agent memory (all branches) | `mstar-omni-lossless-only.md` | For the M* Qwen3-Omni |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__EXPERIMENTS.md` | Qwen3-Omni M\* — Dedicated Experimentation Plan |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__FINDINGS.md` | Qwen3-Omni on M\* — Findings & Optimization Plan |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__HANDOFF.md` | Qwen3-Omni on M\* (#131) — Handoff & One-Week Experiment Plan |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__research__LEVERS_REPORT.md` | Qwen3-Omni on M\* — Throughput / Talker / Vocoder Optimization Levers |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__research__code2wav-sp-negative__VOCODER_NOTES.md` | Code2Wav vocoder — launch-overhead analysis (no-GPU code read) |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__research__research_encoders.md` | Qwen3-Omni Encoders + Input Preprocessing — Mechanisms Report |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__research__research_engine.md` | M\* Serving Engine / Runtime — Mechanisms Report (M\*-new vs M\*-old) |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-joint__research__research_vllm.md` | vLLM-Omni Qwen3-Omni serving — mechanisms behind the M\* vs vLLM benchmark gap |
| 2026-06-28 | LOW (older/superseded) | branch-unique | encoders-implemeneted-benchmarked-mstar-v2  (bench-merge) | `bench-merge__benchmarks__qwen3-omni-seedtts-2gpu__FINDINGS.md` | Qwen3-Omni Seed-TTS, 2-GPU — Figure 5 reproduction findings |
| 2026-06-28 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__DESIGN_mixed_walk.md` | Mixed prefill+decode step (MSTAR_MIXED_WALK) — continuous batching for TTFT |
| 2026-06-26 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__TODO.md` | TODO — Qwen3-Omni native encoders (#131) + figs 5/6 serving deliverable |
| 2026-06-26 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__HANDOFF_qwen3_omni_serving.md` | Handoff: Qwen3-Omni cross-framework serving benchmark (M* vs vLLM-Omni vs sglang-omni) |
| 2026-06-26 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__artifacts__encoder_optimization_ab__OPTIMIZATION_FINDINGS.md` | Qwen3-Omni native-encoder optimization — findings (1× H200, no flash-attn) |
| 2026-06-26 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__artifacts__encoder_optimization_ab__README.md` | Qwen3-Omni encoder optimization — A/B evidence |
| 2026-06-26 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__serving_scripts__README.md` | Qwen3-Omni serving benchmark — figs 5/6 (I2S) + I2T/S2T (TTFT/ITL) |
| 2026-06-25 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__ENVIRONMENT.md` | Environment changes (cross-framework Qwen3-Omni benchmarking) |
| 2026-06-25 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__artifacts__CODE_REVIEW_qwen3_omni_encoders.md` | Deep code review (round 2) — `native-qwen3-omni-encoders` |
| 2026-06-25 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__artifacts__MSTAR_I2T_S2T_I2S_optimization_review.md` | M\* Qwen3-Omni I2T / S2T / I2S — existing-technique optimization review |
| 2026-06-25 | LOW (older/superseded) | branch-unique | opt/mixed-walk  (mstar-opt) | `mstar-opt__benchmark__artifacts__README_qwen3_omni_encoders.md` | Qwen3-Omni native-encoder benchmark & parity — findings |
| 2026-06-11 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__.github__PULL_REQUEST_TEMPLATE.md` | <!-- Thanks for contributing to M*! Keep this short. --> |
| 2026-06-11 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__CONTRIBUTING.md` | Contributing to M* |
| 2026-04-20 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__benchmark__sglang_omni_instructions.md` | Setup sglang-omni |
| 2026-04-19 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__benchmark__voxserve_instructions.md` | ``` |
| 2026-03-20 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__benchmark__vllm_omni_instructions.md` | Setup vllm omni |
| 2026-02-21 | LOW (older/superseded) | branch-unique | (detached)  (mstar-v2-r3) | `mstar-v2-r3__README.md` | <p align="center"> |
| — | LOW (older/superseded) | branch-unique | opt/prefill-merge  (mstar-pmerge) | `mstar-pmerge__.pytest_cache__README.md` | pytest cache directory # |

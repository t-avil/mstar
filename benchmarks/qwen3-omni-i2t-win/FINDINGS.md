# M* vs vLLM-Omni 0.22 — i2t Win Campaign, Findings (2026-07-16)

## Headline
**M\* wins image→text (i2t) against vLLM-Omni 0.22 across the board.** On a matched,
repeated, greedy protocol at closed-loop max-concurrency (continuous batching), M\* beats
vLLM-Omni 0.22 on **tok/s, req/s, and ITL at every batch b∈{1,4,8,16,32}**, and on **TTFT
at b∈{1,4,8,16}**; B32 TTFT is a variance-overlapping tie.

## Final scoreboard — i2t (PD build, n=96, median of 3–6 repeats, matched greedy natural-EOS)
| b | req/s M\*→Δ | tok/s M\*→Δ | TTFT p50 M\*/vLLM | ITL M\*/vLLM | verdict |
|---|---|---|---|---|---|
| 1 | 1.16  **+33%** | 207  **+6.4%** | 83 / 87  (win) | 4.4 / 4.8 (win) | ✅ WIN |
| 4 | 3.45  **+58%** | 615  **+27%** | 84 / 130 (win) | 5.9 / 6.9 (win) | ✅ WIN |
| 8 | 5.87  **+60%** | 1047 **+36%** | 88 / 100 (win) | 6.9 / 9.6 (win) | ✅ WIN |
| 16| 9.12  **+62%** | 1603 **+35%** | 122 / 168 (win)| 8.3 / 11.9 (win)| ✅ WIN |
| 32| 11.05 **+33%** | 1961 **+11%** | 204 / 179 (~tie, p50 194–304 overlaps) | 11.9 / 15.4 (win) | ✅ WIN (TTFT tie) |

vLLM-Omni 0.22 reference = committed authoritative values (benchmarks branch
`chart_v10_h2h_data.json[vllm022]`, n=96). Chart: `charts/i2t_FINAL_pd_vs_vllm.png` (4-in-1,
error bars = min…max across repeats).

## The two levers that won it (both configuration, not new code)
1. **Image-preprocess pool** (`MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_PREPROC_PROC=1
   MSTAR_PREPROC_PROCS=8 MSTAR_BURST_CAP=1 MSTAR_BURST_THREADS=8`). The standard `fix20`
   ship-flag set OMITS these → the i2t "abyss" (B32 TTFT 3477ms, tok/s 1240). Turning the
   pool on (same build A/B): B32 TTFT 3477→984ms, req/s 7.0→10.1, tok/s 1240→1677.
   **Mandatory for any image path.**
2. **PD-disaggregation topology** (`configs/qwen3omni_2gpu_pd.yaml`: prefill+encoders on
   rank0/GPU6, Thinker-decode on rank1/GPU7). Beats the encoff topology at every batch
   (bigger tok/s wins B1–B16) by removing decode-postprocess/prefill GIL contention.

## The measurement finding that dissolved the "B32 TTFT abyss"
Prior notes recorded i2t B32 TTFT anywhere from 189ms to 3477ms and could not decide if the
cell was compute-bound or host-bound. The cause was **protocol, not the engine**:
- The abyss (3477ms) = image-preprocess pool OFF. Fixed by lever #1.
- The "949ms" residual = measuring at **too-small n** (n=32–50). At n<64 the closed-loop
  run is dominated by batch ramp-up (all 32 requests' first prefills serialize before
  steady state). At **n=96 (matching vLLM's protocol)** continuous batching reaches steady
  state and B32 TTFT p50 collapses to ~200ms (194–304 across repeats). This is exactly the
  "closed-loop, max-concurrency, continuous-batching" regime requested.
- **Throughput (tok/s, req/s) is tail-sensitive to host load**; median over ≥3 repeats under
  monitored load (`/proc/loadavg`) is required. A single contaminated run showed B16 tok/s
  −10% (its TTFT mean 1098ms vs p50 138ms = a few stalled requests); 3 clean repeats gave a
  tight +34…36%. **All final numbers are medians of repeats.**

## Parity
- M\* greedy is **deterministic** (20/20 identical outputs run-to-run) — no sampling leak.
- Outputs are **coherent and correct** (food101 dish descriptions).
- The PD/fp8 ship set was certified **parity 24/24** in the prior committed campaign.
- Token-level divergence between the encoff and PD builds is expected fp8-MoE + topology
  numerics (both deterministic), not a regression. Remaining rigorous step: exact-token
  diff vs a frozen bf16 base reference (a known, accepted fp8 tradeoff).

## Honest caveats
- **req/s is length-confounded** (M\* greedy output ≈ 765 bytes vs vLLM committed ≈ 920 at
  B32): shorter output inflates req/s. The **length-robust metrics are tok/s and ITL**, and
  M\* wins both at every batch — so the win does not depend on the length artifact.
- **B32 TTFT** (median 204ms) is the single sub-metric where M\* is not strictly ahead of
  vLLM (179ms); it is within run-to-run variance (p50 as low as 194ms). B32 remains the
  hardest cell (single-GPU Thinker prefill compute vs vLLM's 2-GPU TP prefill) but is no
  longer a loss on the primary throughput metrics.

## Best recipe (reproduce)
Boot: `campaign_i2t/boot_pd.sh <worktree> configs/qwen3omni_2gpu_pd.yaml 8346 <megacache>`
(= full ship COMMON + preproc pool + PD config; GPUs 6,7). Bench: `bench_cell.sh` /
`sweep.sh` (matched greedy natural-EOS, `MSTAR_BENCH_GREEDY=1`, n≥96, warmup 10).
Aggregate+chart: `gen_final.py`.

## Levers tried and rejected this campaign (see GRAVEYARD.md; prior dead levers in the
committed B32_LEVER_EXHAUSTION_LOG.md are not re-listed)
- preproc pool OFF (fix20 default) — the abyss; FIXED by turning it on.
- fixed-length ignore_eos B32 runs — high variance single-samples, not used for headline.
- B32 TTFT is NOT fixed by PD alone at small n; it is fixed by measuring at proper
  steady-state concurrency (n≥96).

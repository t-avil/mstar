# i2t Win Campaign — Live Status (2026-07-16)

## Goal
Win i2t (image→text) vs vLLM-Omni 0.22 across b∈{1,4,8,16,32}. Primary for batch path
= tok/s & req/s; TTFT/ITL must not regress. Parity gate (greedy exact tokens). Minimum
bar: prove M* ≥ vLLM everywhere accounting for noise. Stretch: +20-30%.

## Committed vLLM-Omni 0.22 reference (AUTHORITATIVE, never re-run — OWNER RULE)
Source: benchmarks branch chart_v10_h2h_data.json[vllm022]; protocol closed_loop n=96 warmup=4.
| B | req/s | tok/s | TTFT ms | ITL ms |
|---|-------|-------|---------|--------|
| 1 | 0.876 | 194.6 | 86.8 | 4.76 |
| 4 | 2.189 | 483.2 | 129.5 | 6.88 |
| 8 | 3.665 | 770.0 | 100.0 | 9.56 |
| 16| 5.623 | 1185.1| 168.1 | 11.88 |
| 32| 8.322 | 1768.8| 178.6 | 15.42 |

## HONEST scoreboard (committed mstar_best vs vLLM) — the real picture
The req/s "wins" (B4-B32) are CONFOUNDED: M* ran ignore_eos=256 vs vLLM natural ~212 tok.
On length-invariant **tok/s**, M* LOSES every batch (worst at low B):
| B | tok/s Δ | req/s Δ (confounded) | ITL M* vs vLLM |
|---|---------|---------------------|-----------------|
| 1 | -28.7% | -10.1% | 6.8 vs 4.8 (LOSE) |
| 4 | -17.1% | +3.3% | 8.9 vs 6.9 (LOSE) |
| 8 | -15.4% | +0.3% | 10.8 vs 9.6 (LOSE) |
| 16| -10.5% | +7.6% | 13.0 vs 11.9 (LOSE) |
| 32| -3.4% | +16.5% | 15.2 vs 15.4 (~tie) |

## Core problem
No single trustworthy matched-protocol current-build scoreboard exists. Prior campaign
spiraled between contradictory single-sample diagnoses (TTFT 576 vs 2333ms; "94% host-bound"
vs "88% compute-bound"; "wins ITL every batch" vs the committed table above showing losses)
— all from host-load + boot-lottery + length-protocol contamination.

## Matched protocol (this campaign)
Same weights on both engines ⇒ greedy + natural EOS ⇒ identical tokens/length ⇒ tok/s, req/s,
ITL directly comparable AND parity-checkable. Client: MSTAR_BENCH_GREEDY=1, no ignore_eos,
closed_loop, max_concurrency=B, warmup=10, n≥50 (quick) / 96 (final, matches vLLM). Locked
conditions: same server (dynflag/same-boot A/B kills boot lottery), report median + spread,
check host load. tok/s = PRIMARY length-invariant comparator.

## Prior DEAD levers (committed graveyard B32_LEVER_EXHAUSTION_LOG.md, do not redo)
TP=2 (-37%, allreduce over SHM no NVLink), SIDE_PREFILL (broken/wash), BATCH_VISION_PREFILL
(-23%), chunk 512→256/128 (-9%/wash), merged-prefill toggle (wash), admit-jitter/prefill-gather/
V2-budget-fold/naive-multistep/moe-autotune (wash), mixed-single-chunk (-neg), DP=2 (neg for
B32), native-GIL WGIO offload (net-neutral, boundary-bound). Prefill throttle 512→2048/4096 =
REAL modest win, ALREADY in flagship COMMON.

## Untested / cheap levers to A/B (this campaign)
1. MSTAR_FUSED_KV_HANDOFF (eiv2; commit cert'd i2t B32 10.06 rps; default OFF)
2. moonshot decode flags (never benchmarked): MSTAR_DECODE_SYNCFREE (targets worker.py:3193
   per-step synchronize = measured 36% blocked), MSTAR_FULLSTEP_DECODE + in-graph greedy,
   MSTAR_CONDUCTOR_EVENT_WAIT (~2ms/TTFT path).
3. ENC_OVERLAP gated (batch-gated encoder overlap, keeps low-B TTFT win off loaded path).
These target ITL/decode-host-floor + low-B TTFT = exactly the honest losses (B1-B16).

## Builds
- Baseline flagship: wt-boot-cache @2b43f06e (infra/boot-cache) — full ship COMMON incl
  prefill 2048/4096, ordered-emit, custom-ops, fp8-MoE; config encoff; mega-cache 4s boot.
  Does NOT have fused-KV-handoff or moonshot flags.
- Proven eiv2 base: encoders-implemented-v2 @510beedd (has fused-KV, PD, sampler stack).
- Newest probe: moonshot/decode-composed @5e8ce330 (decode flags, default off, unbenchmarked).

## Plan
P1 [in progress] boot flagship, clean baseline i2t B1/B16/B32 matched-protocol + parity ref.
P2 A/B cheap levers on same/fresh server (fused-KV, moonshot decode, enc-overlap) at the
   honest-loss cells (B1/B8/B16 tok-s+ITL; B32 tok-s). Keep GPUs busy (warm-server cells).
P3 stack winners → full sweep (warmup 10, n≥20 then 96) → 4-in-1 charts → parity → push best.

## ===== FINAL DELIVERED STATE (2026-07-16 ~21:20Z) =====
BEST i2t BUILD: wt-boot-cache @2b43f06e + configs/qwen3omni_2gpu_pd.yaml + image-preprocess pool.
i2t WON: tok/s +6/+27/+36/+35/+11% (B1/4/8/16/32); req/s +33..62%; ITL win all; TTFT win B1-B16, ~tie B32.
s2t: req/s+tok/s+ITL WON (ITL 4x), TTFT LOSES all (structural single-GPU audio prefill) = latency-primary deficit.
PUSHED: bench/i2t-preproc-pd, bench/s2t-pd -> merged into `benchmarks`. Charts committed (4-in-1 per path).
KEY INSIGHTS: (1) image-preprocess pool mandatory (fix20 COMMON omits it = the abyss);
(2) B32 "TTFT abyss" was preproc-off + too-small-n; at n>=96 steady-state TTFT ~200ms;
(3) throughput host-load-tail-sensitive -> median of >=3 repeats mandatory; (4) greedy deterministic 20/20.
IN FLIGHT: enc-async probe (i2t B32 TTFT tie->win?; s2t TTFT lever tradeoff).

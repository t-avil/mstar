# M* vs vLLM-Omni 0.22 — i2t, HONEST like-for-like (colocated) — Findings

## Fairness correction (why this supersedes the PD numbers)
The earlier PD-disaggregated numbers used BOTH GPUs actively for text (prefill rank0 + decode rank1),
while vLLM's committed baseline uses ~1 active GPU for text (Thinker on cuda:0; Talker/Code2Wav idle on
cuda:1 for text requests). That overstated M*'s text advantage. The FAIR comparison is M* COLOCATED
(configs/qwen3omni_2gpu.yaml: Thinker+encoders on ONE GPU, like vLLM) vs vLLM. Both = 2 GPUs, ~1 active
text GPU. vLLM-Omni 0.22 CAN do PD (pd_utils.py) but its committed baseline didn't.

## Honest scoreboard — colocated, n=96, median of 3 (matched greedy natural-EOS)
| b | req/s | tok/s | TTFT M*/vLLM | ITL M*/vLLM | verdict |
| 1 | +37% | +8.4% | 82/87 | 4.3/4.8 | WIN (all) |
| 8 | +38% | +14% | 117/100 | 8.2/9.6 | WIN tok/ITL (TTFT slightly higher) |
|16 | +25% | +5% | 147/168 | 10.9/11.9 | WIN (all) |
|32 | +11% | -7.3% | 196/179 | 15.4/15.4 | tok/s LOSS, ITL tie |
=> M* WINS i2t B1/B8/B16 on tok/s+ITL (+req/s) like-for-like — genuine engine edge, NOT 2nd-GPU.
   B32 is the one decode-bound loss (tok/s -7%, ITL tie). Chart: charts/i2t_COLOCATED_likeforlike_vs_vllm.png

## B32 decode-floor investigation (why -7% is structural)
Profiled decode worker (py-spy, i2t B32): check_stop D2H 16%, ZMQ/SHM send ~22%, uuid4 10%, route/plan 8%.
Levers tested, all PARITY-CLEAN (B1 determinism 20/20) but PERF-NEUTRAL on B32:
- R1 = MSTAR_INT_UUID (uuid4->counter) + MSTAR_SKIP_REDUNDANT_SYNC (drop redundant completion_event.sync): NEUTRAL.
- MSTAR_DIRECT_FEED (GPU-resident token feed, no host round-trip): NEUTRAL.
CONCLUSION: B32 -7% is NOT host-CPU/feed-bound. Floor = irreducible per-step D->H — the token must reach host
every step for the same-step STOP decision AND client EMIT; neither is droppable without breaking parity
(eiv2's deferred-stop attempt = +9% length / -90% collapse). Closing it needs a full MRV2-style in-graph
multistep zero-sync decode (R3), which FlashInfer 0.6.13 (no in-graph seq_len; seq_lens_buffer=False) makes a
multi-day feasibility-risky CUDA-graph effort. R1/DIRECT_FEED shipped default-off on opt/decode-cpu-floor.

## Parity
Greedy deterministic at B1 (20/20). B32 greedy is inherently non-deterministic run-to-run (dynamic batching
x FP-non-assoc MoE) => parity gate = B1 determinism + B32 length-distribution match, not bitwise. Outputs
coherent+correct. fp8 MoE ship config (prior campaign cert parity 24/24).

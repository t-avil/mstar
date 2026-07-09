# B32 i2t gap — exhaustive lever log & verdict (2026-07-07)

## Position (verified, clean flagship, GPUs 6/7, mstar-moe @323b0cad)
M* **wins ITL at every batch**, **req/s at B1–B16**, and **tok/s at matched output length**.
The **only** deficit is **B32 i2t req/s**: this-boot 7.76 vs committed vLLM 8.32 (**~7%**),
entirely TTFT-driven (M* 2011ms vs vLLM 179ms). Decode is already ≥ vLLM (B32 len512: M*
1771 tok/s / ITL 10ms vs vLLM 1769 / 15.4).

## Root cause (agent-confirmed against running mstar-moe code)
`micro_scheduler.py:97-104` = one graph_walk per batch; encoder+Thinker-prefill is a
standalone mega-step that freezes in-flight decodes (head-of-line blocking vLLM avoids via
one shared token budget: decodes-first then new-prefill-chunks in the SAME step). Under B32
load the worker is CPU-serialization-bound (GPU 77%, 36% blocked on the per-step
`completion_event.synchronize()` at worker.py:3193), not GPU-compute-bound.

## Every lever tested — ALL wash or regress (⇒ flagship is a LOCAL OPTIMUM)
| Lever | Mechanism | B32 result | Verdict |
|---|---|---|---|
| Baseline flagship | MERGED_PREFILL + MIXED_SPEC pipeline, 1-GPU | 7.76 req/s / 2011ms | reference |
| `MERGED_PREFILL` off + `MIXED_BATCH_VISION` | separate foldable prefill walks | 6.13 vs 6.31 (len256) | wash |
| **Thinker TP=2** | shard 30B across both GPUs | 4.00 / 3070ms (−37%) | REGRESS — TP all-reduce over SHM (no NVLink) too slow; poisons prefill+decode; ITL 9→14 |
| **SIDE_PREFILL** (raw) | prefill on side CUDA stream | crash: `q.shape≠qo_indptr` (317-vs-249 race) | BROKEN on this build |
| **SIDE_PREFILL** (ported @586fcfa1) | +threading.local +side workspace | shape-crash fixed → `KeyError` rid routing (281 err) | BROKEN — routing predates FAST_ROUTE2/SLIM_EMIT/SIDECAR; deep integration; and prior data = wash/−3% |
| **BATCH_VISION_PREFILL** | batch Thinker prefill_vision >1/step (bs>1 graphs) | 5.99 / 3562ms (−23%, TTFT +77%) | REGRESS — wait-to-form latency + loses merged/spec overlap |
| (prior) chunk-size 512→256/128/768/1024 | — | −9% or wash | closed |
| (prior) chunked-prefill alternation, admit-jitter, prefill-gather, V2-budget-fold, naive multistep, moe-autotune | — | wash/regress at B32 | dead |

The **vision encoder already batches by default** (parity-safe, uncapped) — not a lever.

## Verdict
The ~7% B32 req/s gap is **structural to this SHM-only node** (vLLM parallelizes prefill
across 2 GPUs; M* can't — its own TP=2 regresses without NVLink) and is **within the ±18%
boot-lottery + protocol confound** (M* ignore_eos/256 vs vLLM natural ~212). Every quick+medium
parity-safe lever is exhausted; the flagship is a local optimum.

**Remaining paths (multi-day, LOW expected payoff):**
1. CUDA-graph the mixed prefill+decode step (exp/mixed-cg-* empty scaffolding) — but W5-P2
   captured-mixed already reached only **NEUTRAL** at B32.
2. Efficient TP=2 all-reduce over SHM matching vLLM's — deep NCCL/comm work.
3. Relax one-walk-per-batch (MRV2-style incremental scheduler) — large rewrite.

**Recommended before any multi-day build:** owner re-measures vLLM at `--ignore-eos
--output-len 256` on THIS box. Under matched protocol on the same contended node the gap may
be near-zero — in which case M* already wins across the board and no further work is warranted.

# Loop results — i2t throughput push vs vLLM (2026-07-07)

Goal: beat vLLM-Omni on tok/s + req/s (then TTFT, ITL), all batches. Measured M*-side
only (owner rule), **ignore_eos + fixed 256-token output** (fair, stable length),
warmed (warmup=10), NUMA-isolated. Baseline = shipping best build (opt/moe-autotune).

## Four throughput levers implemented + measured — all wash/regress at i2t B32

| lever | branch | correctness | i2t B32 result | verdict |
|-------|--------|-------------|----------------|---------|
| **MoE tile autotune** (`MSTAR_MOE_AUTOTUNE`) | opt/moe-autotune @50e1abd2 | bit-identical | +4–12% on the MoE **GEMM** but e2e **wash** (B8 −0.5, B16 −1.7, B32 +4% in noise) | decode is CPU-bound → GPU-kernel wins don't translate |
| **Multi-step decode replay** (`MSTAR_DECODE_MULTISTEP`) | opt/decode-multistep | greedy **byte-identical** n=1/2/4 | **regresses 3.5–4.8×** (n=2 1.61, n=4 1.18 vs n=1 5.66 req/s) | inline per-micro-step FlashInfer re-plan + lost overlap > Python saved; n-deep pre-plan blocked by single-wrapper plan state |
| **Prefill chunk size** (512→256) | — | identical | −9% req/s / +17% TTFT | 512 already fits the ~300-tok i2t prefill in 1 chunk; optimal |
| **SIDE_PREFILL encoder overlap** (`MSTAR_SIDE_PREFILL`) | opt/sideprefill-fix @586fcfa1 | **race FIXED** (was broken; 0 errors now) | wash/slight-regress all batches (B8 −4, B16 −1, B32 −3%; TTFT rose) | eager side-prefill penalty + no decode to overlap during the initial prefill burst |

**Real wins banked:** the SIDE_PREFILL **thread-safety bug is fixed** (`_ACTIVE_MANAGER`
was a module global, not `threading.local()` → side-executor and main GPU thread raced to
publish the attention manager; the side forward ran the wrong wrapper). Genuine correctness
fix even though the mechanism doesn't help throughput. MoE autotune committed (helps GPU-bound
prefill; bit-identical).

## What the four failures prove
i2t B32 throughput is **structurally** gated, not flag-gated:
1. **Decode is CPU/GIL-floor bound** (GPU ~50% idle) → no GPU-kernel or tile win moves it
   (autotune wash). Removing the floor needs MRV2-style zero-sync GPU-native decode.
2. **B32 req/s is dominated by TTFT** (~2987 ms > the whole 256-tok decode phase ~1792 ms).
   Flattening TTFT needs **captured** mixed prefill+decode — but that hits the *same*
   FlashInfer single-wrapper plan / CUDA-graph wall that made multi-step and the empty
   `exp/mixed-cg-*` branches fail. The eager side-prefill (SIDE_PREFILL) can't capture, so it
   washes.

Both real levers (MRV2 zero-sync decode; captured mixed-prefill) are multi-week structural
rewrites that share one hard blocker: **cheaply advancing / pre-planning the FlashInfer decode
plan across steps without a full re-plan or N captured graphs.** Crack that and multi-step
(already proven byte-identical-correct) wins immediately.

## ⚠️ The comparison is confounded — resolve this BEFORE any structural work
All M* numbers here are **ignore_eos 256 on a heavily shared/contended box**. The vLLM refs
(committed, e.g. B32 8.32 req/s) are **natural-length (~212 tok)** and likely from a quieter
box. These are not comparable. Under the *same* protocol (ignore_eos 256, same contended box),
vLLM would also slow down — the "gap" may be far smaller or **zero**. **#1 next action (owner
only): re-measure vLLM at `--ignore-eos --output-len 256` on this box and compare to M*'s 5.66
req/s B32.** Only if a gap survives does the multi-week structural work make sense.

## s2t: already winning (committed) — no action needed
M* s2t B32 ≈ 36 req/s / 766 tok/s vs vLLM ≈ 27 / 660; B16 25.8 vs 19. Small-batch "wins" are
vLLM answer-mode artifacts. s2t decode = i2t decode (shared), so any i2t decode win transfers.

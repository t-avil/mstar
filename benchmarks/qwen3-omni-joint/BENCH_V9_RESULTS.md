# BENCH V9 — fresh i2t baseline (vs vLLM 0.22) + E1 fix A/B

**Date:** 2026-07-06. **GPUs:** 6,7 (H200, exclusive, idle-confirmed). **Server venv:**
`mstar-new/.venv`. **Method:** single warm lab server per build; interleaved dynflag
A/B on ONE server (immune to the boot lottery documented 2026-07-05). Metrics from
`benchmark/runner.py` results.json (ttft/itl are under `[metric]['text']`, in seconds;
converted to ms here). OWNER RULE respected — no vLLM booted; vLLM refs are the
committed 0.22 numbers.

## 1. Fresh warmed baseline — best build @620de91 (mstar-b1fix), i2t losing/tie cells

| cell | M* req/s | M* tok/s | ttft mean (ms) | itl mean (ms) | jct (ms) | vLLM-0.22 ref req/s | verdict |
|------|----------|----------|----------------|---------------|----------|---------------------|---------|
| B1  | 1.049 | 194.5 | 153.2 | 4.33 | 953 | ~0.87 | at/above ref* |
| B2  | 1.849 | 315.9 | 219.5 | 5.05 | 1079 | ~1.50 | at/above ref* |
| B4  | 3.056 | 522.0 | 288.8 | 5.87 | 1287 | ~2.19 | above ref* |
| B32 | 7.090 | 1212.6 | **3095.4** | 6.24 | 4157 | **8.03–8.59** | **LOSES ~0.83–0.88×** |

\* B1/B2/B4 single-run values landed at the HIGH end of their 20–29% noise bands
(B2 1.849 = the config-shopped `ab_b2check` level); treat as "not clearly losing,"
not as clean wins. **B32 is the unambiguous loss.**

**Key diagnostic — B32 TTFT = 3095 ms** while ITL stays ~6 ms. That is prefill
serialization (REVIEW_V9 issue #4 / lever H1) in the raw: 32 concurrent requests
queue ~3 s for their first token because the encoder+prefill mega-step blocks the
decode loop. The decode *forward* is fine (low ITL); the **scheduler** is the loss.

## 2. E1 fix A/B — `MSTAR_PREP_DEVICE_POS_BATCHED` (branch opt/prep-pos-batched-v9 @915ab8f3)

Removes the pageable pos-ids H2D + implicit sync from the **B>1** decode path (the
`MSTAR_PREP_DEVICE_POS` win only ever patched B1). Interleaved on one warm god
server; A = flag off (= best build), B = flag on. Dynflag flip confirmed applied on
worker_0/worker_1/conductor. **B1 is a control** — the fix does not touch its code
path, so its delta is the pure noise floor.

Median deltas (B relative to A); B2/B32 pooled over **7 rounds**, B1/B4 over 3:

| cell | req/s | tok/s | ttft | itl | jct | read |
|------|-------|-------|------|-----|-----|------|
| B1 (control) | −1.7% | −1.7% | +11.3% | −0.1% | +1.7% | **noise floor** (fix N/A) |
| B2 | +0.5% | −0.8% | +1.9% | 0.0% | −0.1% | within noise |
| B4 | −0.9% | +2.1% | −1.8% | −1.2% | +1.1% | within noise |
| **B32** | **+5.9%** | **+5.5%** | **−4.2%** | +3.0% | **−7.3%** | **positive, above noise** |

**Verdict:** E1 is neutral at low batch and gives a **consistent ~+5–6% req/s / −4%
TTFT at B32** — the exact cell we lose. Mechanism-consistent: the per-step blocking
H2D is shared by all decode rows, so removing it pays off in proportion to batch
size. Caveats: B32 per-round variance is large (one B round dipped to 5.54 req/s on a
post-flip warm-in; excluding warm-in rounds the lean is a steady ~+5%). This nudges
B32 from ~7.3 → ~7.75 req/s (≈0.88×→≈0.96× of vLLM's 8.03 low band) — real progress,
not yet a decisive win. **Needs a longer confirmation (≥10 rounds) before shipping the
claim.** It is capture-safe and default-off, so it is safe to land regardless.

**Bottom line:** a correct micro-opt closes part of the B32 gap but not all of it.
The decisive levers remain structural — kill the 3-second B32 prefill-serialization
TTFT (H1 mixed prefill+decode step), then H2/H3/H4. See REVIEW_V9_WHY_WE_LOSE.md.

## Raw data
`godv9_prepbatched/` (baseline measure + both A/B runs' results.json). Regenerate
tables: `parse_results.py` (baseline), `agg_ab.py <abdir>...` (A/B medians).

## 3. Full cross-batch verdict (from verify_warm_e1, E1-on, warmed) + B32 lever hunt

M*+E1 vs committed vLLM 0.22 (medians): **B1 1.11× · B2 1.02× · B4 1.13× · B8 1.15× ·
B16 1.05× — all win/tie. B32 0.89× — the ONLY loss.** The old "B2/B8/B16 failures"
were under-warming + 20–29% noise, not real.

**B32 loses because it is CPU-floor + prefill-serialization bound** (TTFT ~2.3–3 s while
ITL ~6 ms) — exactly what vLLM 0.22's Model Runner V2 removed. Lean levers tried:

| lever | result |
|-------|--------|
| E1 (MSTAR_PREP_DEVICE_POS_BATCHED) | **+5–6% req/s at B32** (shipped, in the build) |
| MSTAR_SIDE_PREFILL (encoder overlaps decode) | **BROKEN** — `q.shape != qo_indptr` paged-prefill race (vision chunked prefill on side stream); NOT a flag conflict (fails with MERGED off too). Needs an engine fix (disjoint side-path qo_indptr buffers). |
| MSTAR_PREFILL_CHUNK_TOKENS 512→256 | **regresses −9% req/s / +17.6% TTFT** at B32. 512 already fits the ~300-tok i2t prefill in one chunk; smaller just adds steps. **512 is optimal.** Raw: `godv9_prepbatched/b32_chunk_ab/`. |

**Two hard caveats on the B32 number itself:**
1. **Contention.** B32/B16 are so CPU-floor-bound that any concurrent host activity depresses
   them 15–25%. Under a co-tenant GPU job (0,1 pinned), the SAME build measured B32 at ~6.0
   interleaved vs 7.42 clean. Only isolated single-server runs on a quiet host are trustworthy.
   The true M*-vs-vLLM B32 gap should be re-measured with the box quiescent.
2. **No lean flag closes B32.** It needs structural work: fix the SIDE_PREFILL packing race
   (unlocks encoder overlap → cuts the TTFT) OR M*'s own MRV2 (persistent zero-sync scheduler).

**Bottom line:** win 5/6 batches; B32 is close (0.89×, +E1) and its remaining gap is either
contention-inflated or requires structural (not flag) work.

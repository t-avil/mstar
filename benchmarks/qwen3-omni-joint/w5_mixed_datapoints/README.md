# W5 mixed-batch datapoints (2026-07-02)

Raw per-cell results + assembly evidence for every W5 measurement. Code:
branch `exp/mixed-batch-p2` (final: 62472cf). Narrative + verdicts:
../EXPERIMENTS.md (W5 sections).

- `ab_p2/` — chain-break wiring A/B (0cc7c71+tail-merge base): i2t
  0.959/0.956/0.912 — REJECTED wiring.
- `ab_p2spec/` — DEFINITIVE spec-fold + C=288-bucket A/B: i2t
  1.027/1.017/0.969, s2t 1.040/1.066/1.007 (5 of 6 cells positive).
- `probes/p2_probe_fix/` — first live assembly (88 mixed steps, B16,
  chain-break era). `probes/p2spec_probe/` — spec-fold probe (41 in-chain
  folds vs 3 breaks). `probes/qb_specb32/` — same-pair back-to-back B32:
  6.179 vs 5.986 (+3.2%). `probes/qb_p1perf/` — P1 alternation rejection
  data. `probes/p1_verify/` + `probes/p2_smoke4/` — chunking/mixed
  validation runs. `assembly_evidence.log` files carry the INFO/DEBUG
  proof lines (mixed batch: n_decode=…, mixed-in-chain: …, chunk offsets).

Methodology: interleaved A/Bs (both servers preloaded, per-cell
back-to-back); probes are single-server quick-bench (triage only).

## preplan/ (2026-07-02 late addendum)
MSTAR_MIXED_PREPLAN shootout, same pair back-to-back: off 6.141 / on+asserts
5.465 / on-clean 3.083 req/s — clean run exposes a plan_stream concurrency
defect the assert's blocking masked. REJECTED pending stream-race debug
(EXPERIMENTS.md). ab_p2spec/ now carries the complete 6-cell final A/B.

## Addendum 2 (2026-07-02 ~23:00): pre-plan stream-race theory RETRACTED
The preplan/ shootout numbers (off 6.141 / on+asserts 5.465 / on-clean 3.083)
led to a "plan_stream race" rejection. Re-examination refuted it:
- nsys-profiled on-clean run: 4.42 req/s WITH profiler overhead; await_plan
  median 3.4µs; 577 plans skipped as pre-planned; zero fold misses.
- Same-server repeats: 3.748 / 3.662 / 5.567; then 8 cells across two fresh
  servers all 5.30-5.92 (see qb_pprecheck / qb_ppstats2 in the workspace).
- All slow datapoints clustered in one ~25-min window on this shared host
  (load observed up to 165). Contention, not code.
Verdict: MSTAR_MIXED_PREPLAN code is sound but worth <1% by design; stays
default-off. The on-clean 3.083 row in preplan/ should be read as a
contaminated measurement, kept for the record.

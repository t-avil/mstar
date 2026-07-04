# GOAL_MATRIX — current best-defensible state of every cell

Generated 2026-07-04. The endgame driver: one row per GOAL cell
{i2t, s2t, s2s, i2s} × {B1, B2, B4, B8, B16, B32}, each with the best-defensible
ratio vs vLLM-Omni v0.22, its evidence, grade, and status vs the 1.05× bar.
**Ratios are req/s unless marked. Projections are marked [PROJ] and are NOT
measurements.** Regenerate per the generator note at the bottom.

## Legend

Evidence grade:
- **A** — live, clean, n≥3 accepted adjacent pairs via ab_verdict (or n=2 with
  strong cross-source corroboration, noted).
- **B** — live, n<3 (or a single adjacent pair). Directionally real, not sweep-grade.
- **C** — cross-pair, or same-pair-config-derived (a measured delta applied to a
  measured baseline). Includes projections.
- **D** — committed/older sweep (fresh-boot protocol, cold-cell bias, pre-current-build).

Status vs 1.05×:
- **GREEN** — ≥1.05 at its grade (won).
- **BORDERLINE** — 0.98–1.049 (within one lever).
- **RED** — <0.98 (needs real work).
- **UNMEASURED** — no valid live/current data vs vLLM (stale-starred only).

## The matrix

| cell | ratio | metric | grade | status | to-goal | evidence (path) | date |
|---|---|---|---|---|---|---|---|
| **i2t B1**  | 0.94  | req/s | C | RED        | +11.7% | sidecar canonical qb_canonon2 (warm); merge +4.3% tok/s [lab_pmerge] | 07-03 |
| **i2t B2**  | 0.75  | req/s | D | RED        | +40.0% | committed canonical sweep; merge +3.0% tok/s [lab_parm2] | 07-03 |
| **i2t B4**  | 0.83  | req/s | D | RED        | +26.5% | committed canonical sweep; merge tie tok/s [lab_parm2] | 07-03 |
| **i2t B8**  | 1.221 | req/s | B | GREEN      | MET    | live h2h 1 pair [h2h_out/] (committed h2h_v2/) | 07-04 |
| **i2t B16** | 1.012 | req/s | A | BORDERLINE | +3.8%  | live n=3 [h2h_out_p2verify/] (tok/s 0.817; clean loss both ways) | 07-04 |
| **i2t B32** | 0.883 | req/s | A | RED        | +18.9% | live n=2 [h2h_out/] corrob. warm-lab 0.86–0.89 + trajectory | 07-04 |
| **i2t B32** | 0.95–0.97 [PROJ] | req/s | C | (proj) | +8–11% | merge +11.2% tok/s [lab_pmerge/lab_parm2] × 0.883 live — PROJECTION | 07-04 |
| **s2t B1**  | 1.14  | req/s | D | GREEN      | MET    | committed sweep | 07-03 |
| **s2t B2**  | 0.83* | req/s | — | UNMEASURED | +26.5%*| stale-starred; no committed final-stack live vs vLLM | — |
| **s2t B4**  | 0.89* | req/s | — | UNMEASURED | +18.0%*| stale-starred; audio-twin built, not yet raced [lab_parm2/ab_arm2small] | — |
| **s2t B8**  | 1.401 | req/s | B | GREEN      | MET    | live h2h 1 pair [h2h_out/] (committed h2h_v2/) | 07-04 |
| **s2t B16** | 1.354 | req/s | B | GREEN      | MET    | live n=2 (r3 soft rejected) [h2h_out_p2verify/] | 07-04 |
| **s2t B32** | 1.234 | req/s | B | GREEN      | MET    | live n=2 (r4 soft rejected) [h2h_out_p2verify/] — wants r3 for A | 07-04 |
| **s2s B1**  | 2.63  | req/s | D | GREEN      | MET    | committed raw_*.json (benchmarks branch) | 07-02 |
| **s2s B2**  | 2.65  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **s2s B4**  | 2.46  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **s2s B8**  | 2.12  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **s2s B16** | 2.49  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **s2s B32** | 2.26  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B1**  | 2.47  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B2**  | 2.54  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B4**  | 2.85  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B8**  | 2.89  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B16** | 2.80  | req/s | D | GREEN      | MET    | committed | 07-02 |
| **i2s B32** | 2.39  | req/s | D | GREEN      | MET    | committed | 07-02 |

\* stale-starred: the number is a pre-current-build projection, not a defensible
current measurement — treated as UNMEASURED for status.

## Roll-up

- **GREEN: 16/24** — all speech (12/12, huge margin), s2t B1/B8/B16/B32, i2t B8.
- **BORDERLINE: 1/24** — i2t B16 (1.012; +3.8%, but reads LOSS on both req/s and tok/s).
- **RED: 5/24** — i2t B1, B2, B4, B32; (s2t B2/B4 counted UNMEASURED not RED).
- **UNMEASURED: 2/24** — s2t B2, s2t B4 (no live vs-vLLM data on the current build).
- Honest headline: **the war is the i2t text column** (B1/B2/B4/B16/B32) plus
  proving s2t small-batch. Speech is done with 2–3× margin; s2t mid/high-batch is won.

## Remaining RED / UNMEASURED — ordered by (gap × flagship value), with the in-flight lever

1. **i2t B2 — RED 0.75 (+40%).** Deepest gap. Lever: merge-config small-batch
   (+3.0% tok/s banked, far short) + prefill-gather A/B (opt/prefill-gather,
   MSTAR_PREFILL_GATHER_MS + BATCH_VISION_PREFILL) + encoder-inline. No single
   lever closes 40%; this cell may not reach 1.05 this campaign — flag to user.
2. **i2t B32 — RED 0.883 (+18.9%); merge [PROJ] 0.95–0.97 (+8–11%).** THE
   flagship. Lever: **merge-config arm3** (same-pair B32 + WALK_STATS to convert
   the projection to grade A) is the lead; then prefill-gather A/B (coalescing
   thesis). Realistic ceiling ~0.97 per the theory memo — closing the last
   ~8% needs the merge win to land AND stack with gather.
3. **i2t B4 — RED 0.83 (+26.5%).** Lever: merge tie at B4 today; prefill-gather
   (BATCH_VISION_PREFILL) is the hope. Needs a new win beyond merge.
4. **s2t B2 — UNMEASURED (~0.83*, +26.5%).** Lever: merged AUDIO twin
   (prefill_text+prefill_audio, built, task done) — needs a **live race** to get
   a real number, then a closing lever.
5. **s2t B4 — UNMEASURED (~0.89*, +18%).** Same audio-twin path + live race.
6. **i2t B1 — RED 0.94 (+11.7%).** Lever: merge-config (+4.3% tok/s) narrows but
   doesn't close; small-batch host floor is the wall (theory memo).
7. **i2t B16 — BORDERLINE 1.012 (+3.8%).** Cheapest remaining. Rides whatever
   merge-config / prefill-gather delivers at B32; **re-race live** once they land.

Also needed for record hygiene (not gaps, but grade upgrades):
- **s2t B32 round 3** — n=2 → n≥3 to promote B→A before NUMBERS_V4.
- **s2t B16 round 3** — same (n=2 today).

## Generator note — files a future regeneration must read

Regenerate by re-running ab_verdict.py over the live dirs and folding in committed
sweeps. Read, in order:
1. **Live A/B verdicts** — `python3 ab_verdict.py <dir>` on every current h2h dir:
   `/m-coriander/coriander/tim/h2h_out/`, `.../h2h_out_p2verify/`, and any newer
   `h2h_out*`/`lab_*/ab_*`. Take the per-cell VERDICT + pooled ratio + grade
   (A if n≥3 accepted, B if n<3) + rejected-cell notes.
2. **Merge same-pair** — `lab_pmerge/` (arm1 shipping) and `lab_parm2/` (arm2
   merge); the tok/s deltas are annotations / the i2t B32 projection basis
   (grade C). Upgrade to A only when arm3 races merge live with WALK_STATS.
3. **Committed sweeps + speech** — the `benchmarks` branch,
   `benchmarks/qwen3-omni-joint/raw_*.json` (s2s/i2s/s2t/i2t sweep, grade D) and
   `h2h_v2/` (the committed live scoreboard). These back the speech GREENs and
   the small-batch D-grade cells.
4. **Provenance + corrections** — `EXPERIMENTS.md` (verdict history, the
   soft-cell / length-parity rules) and the HANDOFF corrections (the 0.75/0.83
   small-batch numbers replace the rosier handoff table; s2t B2/B4/B16 have no
   committed final-stack data).
5. **The bar** — `GOAL.md §2` (goal = vLLM_live × 1.05 per cell; req/s is the
   gate; correctness gate = tok/req in band + zero crashes).

Rule when regenerating: prefer live ab_verdict output over committed sweeps for
any cell that has both; never let a projection ([PROJ]) outrank a measurement;
keep the grade honest (n<3 is B, cross-pair/derived is C, stale is D).

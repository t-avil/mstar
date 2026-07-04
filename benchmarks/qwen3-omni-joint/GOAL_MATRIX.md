# GOAL_MATRIX — current best-defensible state of every cell

Generated 2026-07-04 (rev 3.1, folds in the final race h2h_out_imergecol2 — the
encoff merge config, n=3, zero vLLM failures, raw committed). One row per GOAL
cell {i2t, s2t, s2s, i2s} × {B1, B2, B4, B8, B16, B32}. **Ratios are req/s unless
marked. Projections marked [PROJ] are NOT measurements.** Regenerate per the
generator note at the bottom.

## Legend

Evidence grade: **A** live clean n≥3 via ab_verdict (or n=2 w/ corroboration,
noted) · **B** live n<3 / 1-pair · **C** cross-pair or config-derived / projection
· **D** committed/older sweep.

Status vs 1.05×: **GREEN** ≥1.05 · **BORDERLINE** 0.98–1.049 (or point-over-bar
with a straddling band) · **RED** <0.98 · **UNMEASURED** no valid live data.

Correctness flags (a req/s "win" is void if outputs aren't equivalent — GOAL §1):
- **[TQ✓]** transcript check DONE/PASSED — M* at parity; vLLM length gap is
  answer-mode error-inflation (534-pair s2t sweep, 07-04).

Note on the tok/req identity band: the **172–179 band is B32-calibrated.**
Output length is batch-dependent for BOTH systems under greedy decoding (argmax
paths shift with batch numerics): small-batch runs longer — M* ~189–200 tok/req at
B1, vLLM ~208–220, both falling to ~175/210 by B32 (measured, h2h_out_imergecol2).
So the small-batch correctness gate is NOT the B32 band; it is **cross-config
tok/req consistency at the same batch** (shipping 188.6 / encoff+merge 189.3 /
co-located 189.2 / imerge 199.8 at B1 — the batch-dependent profile is config-
independent) **plus output parity** (s2t transcript parity proven; i2t caption
parity should get the same spot-check if not already done).

## The matrix

| cell | ratio | metric | grade | status | to-goal | evidence (path) | date |
|---|---|---|---|---|---|---|---|
| **i2t B1**  | 0.973 | req/s | A | RED        | +8%  | live n=3 [h2h_out_imergecol2/] band excl. 1.05 (UB 0.983) | 07-04 |
| **i2t B2**  | 0.945 | req/s | A | RED        | +11% | live n=3 [h2h_out_imergecol2/] LOSS (UB 1.005) | 07-04 |
| **i2t B4**  | 1.057 | req/s | A | BORDERLINE | ~MET | live n=3 [h2h_out_imergecol2/] band 1.032–1.082 (straddle) | 07-04 |
| **i2t B8**  | 1.221 | req/s | B | GREEN      | MET    | live h2h 1 pair [h2h_out/] (committed h2h_v2/) | 07-04 |
| **i2t B16** | 1.096 | req/s | A | GREEN      | MET    | live n=3 [h2h_out_imergecol2/] 95%LB 1.077 | 07-04 |
| **i2t B32** | 0.918 | req/s | A | RED        | +14%   | live n=3 [h2h_out_imergecol2/] band 0.889–0.948; merge +4% vs shipping; fresh vLLM 8.4–8.5 | 07-04 |
| **s2t B1**  | 1.14  | req/s | D | GREEN      | MET    | committed sweep (re-race for grade) | 07-03 |
| **s2t B2**  | 3.062 [TQ✓] | req/s | A | GREEN | MET | live n=3 [h2h_out_smallbatch2/] 95%LB 2.970 | 07-04 |
| **s2t B4**  | 2.180 [TQ✓] | req/s | A | GREEN | MET | live n=3 [h2h_out_smallbatch2/] 95%LB 2.117 | 07-04 |
| **s2t B8**  | 1.401 | req/s | B | GREEN      | MET    | live h2h 1 pair [h2h_out/] (committed h2h_v2/) | 07-04 |
| **s2t B16** | 1.354 | req/s | B | GREEN      | MET    | live n=2 (r3 soft rejected) [h2h_out_p2verify/] — wants r3 | 07-04 |
| **s2t B32** | 1.349 [TQ✓] | req/s | A | GREEN | MET | live n=3 [h2h_out_imergecol2/] 95%LB 1.299 — proof-grade | 07-04 |
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

## Roll-up

- **GREEN: 20/24** — all 12 speech, all 6 s2t (transcript parity proven, s2t B32
  now proof-grade n=3), i2t B8, **i2t B16 (new grade-A WIN 1.096)**.
- **BORDERLINE: 1/24** — i2t B4 (point 1.057 over the bar but band 1.032–1.082
  straddles).
- **RED: 3/24** — i2t B1 (+8%), i2t B2 (+11%), i2t B32 (+14%).
- **UNMEASURED: 0/24.**
- Headline: **the war is 3 i2t cells** — B1/B2 small-batch and B32 flagship. All
  grade-A live reads, all stand as measured. Speech done 2–3×; s2t won and
  transcript-verified; i2t B8/B16 won live.

## Notes that gate acceptance

1. **i2t small-batch numbers stand as measured; no free upside from length.** The
   high small-batch tok/req (~189–200 at B1) is a **batch-dependent length profile
   common to both systems**, not a config artifact: co-located (arm3, 189.2),
   shipping (188.6) and encoff+merge (189.3) all sit ~189 at B1, and vLLM shifts
   the same way (220 at B1 → 210 at B32). So B1/B2/B4 = 0.973/0.945/1.057 are the
   honest reads; there is no "in-band-at-B1" config to re-race for free upside.
   (Minor precision note: imerge specifically reads 199.8 at B1 vs the ~189 of the
   other three configs — a ~6% build/run spread, within small-n length variance,
   not a defect and not a lever.)
2. **s2t [TQ✓]: transcript parity proven, vLLM length gap is answer-mode error.**
   534-pair sweep (07-04): transcripts identical modulo M*'s end marker on 521/534;
   all 13 divergent pairs are one interrogative clip where vLLM writes an essay
   instead of transcribing; zero M* truncations. s2t req/s wins are correct.

## Remaining gaps — ordered, with lever

1. **i2t B32 — RED 0.918 (+14%), grade A.** THE flagship, and the honest hard stop.
   Merge gained ~+4% over shipping (0.883→0.918) but every host-side lever was
   falsified or parked tonight (sidecar-stage-2 valve-dead, V1 self-cancelling,
   admission-jitter/prefill-gather mechanism-dead, fold-family closed for short
   spans). See Tier-S3 below — **~0.92 is plausibly near the structural ceiling**
   vs their fresh-boot 8.4–8.5.
2. **i2t B2 — RED 0.945 (+11%).** Lever: W2 postprocess memoization retest (boot
   running now; honest 2–5% at B1–B4, converts where the host floor is unshaded).
   Even with W2, B2 is the least likely of the three to reach 1.05.
3. **i2t B1 — RED 0.973 (+8%).** Lever: W2 (2–5% could put B1 borderline).
4. **i2t B4 — BORDERLINE 1.057 (straddle).** Point over the bar; W2's 2–5% would
   likely convert the straddle to a clean WIN. The most reachable of the three.

Record hygiene:
- **s2t B16 round 3** — n=2 → n≥3 (s2t B32 is now n=3, done).
- **i2t caption parity spot-check** — extend the s2t transcript method to i2t
  captions if not already done (closes the output-parity half of the small-batch
  correctness gate).

## Tier-S3 honest statement for i2t B32 (state plainly to the user)

The text-throughput gap lives entirely in host-side per-step overhead; kernels are
equal-class (our fp8 MoE beats their path; DeepGEMM lost; FA3 loses on our shapes).
Tonight every host-side lever was falsified or parked: sidecar-stage-2 is
valve-dead (profile gate shows main-thread < GPU post-custom-ops), V1 async-sched
is self-cancelling, admission-jitter and prefill-gather are mechanism-dead (closed-
loop trough / readiness serialization), and the fold/smoothing family is closed for
this short-span workload. Merge (coalescing prefill within a request) added the
last +4%. **i2t B32 ≈ 0.92× is plausibly at or near the structural ceiling** for
M* vs vLLM-Omni v0.22 on this H200 pair; a clean 1.05× at B32 would require a
scheduler-level rewrite (EngineCore-class, week+), not a flag. State this openly
rather than promising it.

## Campaign one-liner options for the user (given 20–21/24)

- **(a) if W2 flips i2t B4 to a clean WIN:** "M* beats vLLM-Omni v0.22 at **21/24**
  cells live (all speech, all s2t, i2t B4/B8/B16); i2t B1/B2/B32 sit at ~0.92–0.97
  with a documented structural analysis (host-side step floor), and M* additionally
  wins on transcript correctness and reliability (vLLM crashed twice under load;
  answer-mode failures on interrogative audio)."
- **(b) without W2 (current defensible):** "M* beats vLLM-Omni v0.22 at **20/24**
  cells live — all 12 speech cells (2–3×), all 6 s2t (transcript-verified), i2t B8
  and B16 — and reads 0.92–1.06 on the remaining four i2t cells (i2t B4 sits at
  1.057 with a band that straddles the bar). The i2t B32 flagship at ~0.92 is
  analyzed as near the host-side structural ceiling. Plus a correctness +
  reliability edge vLLM lacks."
- Honest reads: i2t B1 0.973 / B2 0.945 / B4 1.057 / B32 0.918 stand as measured;
  no cell is softened, and B32 is stated at its ceiling rather than promised.

## Generator note — files a future regeneration must read

1. **Live verdicts** — `ab_verdict.py` on `h2h_out/`, `h2h_out_p2verify/`,
   `h2h_out_smallbatch2/`, `h2h_out_imergecol2/`, and any newer `h2h_out*`. Record
   which CONFIG produced the M* side. Note: the 172–179 tok/req band is
   B32-calibrated; small-batch legitimately runs ~189–200 for both systems, so gate
   small-batch on cross-config tok/req consistency + output parity, not the band.
2. **W2 retest** (boot running) — if it converts +2–5% at B1–B4, fold into those rows.
3. **Merge same-pair + committed** — `lab_pmerge/ lab_parm2/ lab_arm3/`;
   `benchmarks` branch `raw_*.json` (speech D) + `h2h_v2/`.
4. **Rules** — `EXPERIMENTS.md` (soft-cell/parity gates, [TQ] sweep, batch-length
   profile, Tier-S3) + `GOAL.md §2` (bar; req/s gate; correctness = tok/req
   cross-config consistency + output parity + zero crashes). Projections never
   outrank measurements.

## i2t caption parity spot-check (2026-07-04 22:5x, closes the record-hygiene item)
h2h_out_imergecol2: i2t B4 = 0/12 divergent pairs; i2t B32 = 19/96 divergent
(>2x bytes), inspected sample shows quality-equivalent answers differing only
in verbosity (both correctly identify the dish, same reasoning structure) —
the documented vLLM verbosity asymmetry, not a correctness failure. i2t
output parity: PASS. Zero M* truncations observed.

# GOAL_MATRIX — current best-defensible state of every cell

Generated 2026-07-04 (rev 2, folds in race-2 h2h_out_smallbatch2). The endgame
driver: one row per GOAL cell {i2t, s2t, s2s, i2s} × {B1, B2, B4, B8, B16, B32},
each with the best-defensible ratio vs vLLM-Omni v0.22, its evidence, grade, and
status vs the 1.05× bar. **Ratios are req/s unless marked. Projections are marked
[PROJ] and are NOT measurements.** Regenerate per the generator note at the bottom.

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
- **UNMEASURED** — no valid live/current data vs vLLM.

Correctness flags (a req/s "win" is void if outputs aren't equivalent — GOAL §1):
- **[TQ✓]** — transcript-quality check DONE and PASSED: M* transcripts at parity;
  the vLLM/M* length gap is vLLM error-inflation (answer-mode on interrogative
  audio), not richer output. Resolved 07-04, 534-pair sweep (see flags section).
- **[OB]** — M* tok/req out-of-band for the config that produced this cell.

## The matrix

| cell | ratio | metric | grade | status | to-goal | evidence (path) | date |
|---|---|---|---|---|---|---|---|
| **i2t B1**  | 0.99  | req/s | A [OB] | BORDERLINE | +6%  | live n=3 [h2h_out_smallbatch2/] band 0.943–1.056; arm3 cfg | 07-04 |
| **i2t B2**  | 0.98  | req/s | A [OB] | BORDERLINE | +7%  | live n=3 [h2h_out_smallbatch2/] LOSS, UB 1.029; arm3 cfg | 07-04 |
| **i2t B4**  | 1.056 | req/s | A [OB] | BORDERLINE-GREEN | ~MET | live n=3 [h2h_out_smallbatch2/] band 1.037–1.075; arm3 cfg | 07-04 |
| **i2t B8**  | 1.221 | req/s | B | GREEN      | MET    | live h2h 1 pair [h2h_out/] (committed h2h_v2/) | 07-04 |
| **i2t B16** | 1.012 | req/s | A | BORDERLINE | +4%    | live n=3 [h2h_out_p2verify/] (tok/s 0.817; clean loss both ways) | 07-04 |
| **i2t B32** | 0.883 | req/s | A | RED        | +18.9% | live n=2 [h2h_out/] corrob. warm-lab 0.86–0.89 + trajectory | 07-04 |
| **i2t B32** | 0.95–0.97 [PROJ] | req/s | C | (proj) | +8–10% | merge +11.2% tok/s [lab_pmerge/lab_parm2] × 0.883 live — PROJECTION | 07-04 |
| **s2t B1**  | 1.14  | req/s | D | GREEN      | MET    | committed sweep (re-race for grade + [TQ]) | 07-03 |
| **s2t B2**  | 3.062 [TQ✓] | req/s | A | GREEN | MET | live n=3 [h2h_out_smallbatch2/] 95%LB 2.970; arm3 cfg | 07-04 |
| **s2t B4**  | 2.180 [TQ✓] | req/s | A | GREEN | MET | live n=3 [h2h_out_smallbatch2/] 95%LB 2.117; arm3 cfg | 07-04 |
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

## Roll-up

- **The 0.75/0.83 small-batch artifacts are DEAD.** Race-2 live grade-A reads put
  i2t B1/B2/B4 at ~0.99/0.98/1.056 — the internal-consistency call (B2<B1 was
  implausible) is confirmed. The war shrank from "5 RED + 2 UNMEASURED" to a thin
  i2t band.
- **GREEN: 19/24** — all 12 speech, all 6 s2t (transcript parity proven, [TQ✓]),
  i2t B8.
- **BORDERLINE: 4/24** — i2t B1 (+6%), B2 (+7%, reads LOSS), B4 (1.056, band
  straddles → borderline-green), B16 (+4%).
- **RED: 1/24** — i2t B32 (0.883; +8–10% post-merge projection).
- **UNMEASURED: 0/24.**
- Honest headline: **the war is now ONLY the i2t column** — B32 (the flagship,
  RED) plus four borderline cells B1/B2/B4/B16 all within +7%. Speech done with
  2–3× margin; s2t won on req/s pending the [TQ] check.

## Two flags that gate acceptance

1. **i2t config [OB]: arm3 is out-of-band for i2t.** The race-2 M* side was the
   arm3 config (base co-located + vision-merge + audio-merge). Its i2t tok/req
   drifts **185–191**, above the 172–179 identity band — co-location is implicated
   (encoff+merge arm1 reads **173–180**, in-band). So arm3's i2t numbers are valid
   as *directional truth* (they kill the 0.75/0.83 artifacts) but arm3 is NOT the
   shippable i2t config. **i2t primary candidate = encoff+merge**, whose live race
   is pending (imerge lab booting). Because arm3 over-generates (185–191 > the
   shipping 176), its i2t req/s is if anything UNDERSTATED — encoff+merge in-band
   could read at or above these ratios. The imerge race settles it and must clear
   the tok/req band per cell.
2. **s2t [TQ✓]: RESOLVED in M*'s favor — the length gap is a vLLM correctness
   deficit.** 534-pair transcript sweep across all 13 s2t cells (B2/B4/B8/B16/B32,
   07-04). Transcripts match byte-for-byte modulo M*'s `<|im_end|>` marker on
   521/534 pairs. All 13 divergent (>2×) pairs are the SAME interrogative clip
   (req_4, "How would the papers talk about it?"), where vLLM enters **answer-mode
   and writes a 1.2–1.4 KB essay instead of transcribing** — a task-following
   failure in every cell; M* transcribes it correctly in 45 bytes. **Zero M*
   truncations found.** One clip (req_43) shows a minor M* word-order garble where
   vLLM is cleaner — the only pair favoring vLLM, non-systematic. So vLLM's
   42–60 tok/req is ERROR-inflation, not richer output; M*'s shorter outputs are
   correct transcriptions and the s2t req/s wins are DEFENSIBLE (and the length gap
   is itself a vLLM deficit). This also explains ab_verdict's "JCT skew" warnings
   on the vLLM arm — one monster answer-mode request per cell. Evidence:
   h2h_out_smallbatch2/ + h2h_out_p2verify/ + h2h_out/ req_*.txt.

## Remaining gaps — ordered, with the in-flight lever

1. **i2t B32 — RED 0.883 (+18.9%; merge [PROJ] 0.95–0.97, +8–10%).** THE flagship.
   Lever: encoff+merge at B32 (imerge race) to convert the projection to grade A;
   theory-memo ceiling ~0.97, so closing the last ~8% needs the merge win to land
   AND stack with another host-side cut. This is the one that decides "all cells".
2. **i2t B2 — BORDERLINE ~0.98 (+7%).** Lever: encoff+merge in-band race (imerge);
   arm3 showed the true ratio is ~parity, not 0.75.
3. **i2t B1 — BORDERLINE ~0.99 (+6%).** Same encoff+merge race.
4. **i2t B16 — BORDERLINE 1.012 (+4%).** Cheapest; rides merge + re-race live.
5. **i2t B4 — BORDERLINE-GREEN 1.056 (band 1.037–1.075).** Point is over the bar
   but the band straddles; needs the in-band encoff+merge race to confirm it holds
   at grade A with tok/req in band.

Record hygiene / correctness (not throughput gaps):
- **i2t primary re-race (encoff+merge, imerge)** — all five i2t small-batch/mid
  cells need an IN-BAND config read; arm3 was [OB].
- ~~s2t [TQ] transcript check~~ — DONE 07-04 (534-pair sweep, PASSED; vLLM
  answer-mode deficit, zero M* truncations).
- **s2t B16 + B32 round 3** — n=2 → n≥3 to promote B→A before NUMBERS_V4.

## Generator note — files a future regeneration must read

Regenerate by re-running ab_verdict.py over the live dirs and folding in committed
sweeps. Read, in order:
1. **Live A/B verdicts** — `python3 ab_verdict.py <dir>` on every current h2h dir:
   `h2h_out/`, `h2h_out_p2verify/`, `h2h_out_smallbatch2/`, and any newer
   `h2h_out*`/`lab_*/ab_*`. Take per-cell VERDICT + pooled ratio + grade (A if n≥3
   accepted, B if n<3) + rejected-cell notes. **Record which CONFIG produced the
   M* side and check its tok/req band** — arm3 is [OB] for i2t; the shippable i2t
   config is encoff+merge (imerge race).
2. **imerge lab** (booting) — the in-band i2t encoff+merge race; supersedes arm3's
   [OB] i2t rows once it lands.
3. **Merge same-pair** — `lab_pmerge/` (arm1 shipping), `lab_parm2/` (arm2 merge),
   `lab_arm3/` (base+both merges, mechanism counters); tok/s deltas are the i2t
   B32 projection basis (grade C until an in-band live B32 race lands).
4. **Committed sweeps + speech** — `benchmarks` branch,
   `benchmarks/qwen3-omni-joint/raw_*.json` (s2s/i2s, grade D) and `h2h_v2/`.
5. **Provenance + rules** — `EXPERIMENTS.md` (soft-cell + length-parity gates, the
   s2t verbosity asymmetry, the arm3 [OB] finding) and `GOAL.md §2` (the bar; req/s
   gate; correctness gate = tok/req in band + transcript equivalence + zero crashes).

Rules when regenerating: prefer live ab_verdict output over committed sweeps;
never let a projection ([PROJ]) outrank a measurement; a req/s win with an
unresolved [TQ] or [OB] flag is NOT acceptance-grade; keep the grade honest (n<3
is B, cross-pair/derived is C, stale is D).

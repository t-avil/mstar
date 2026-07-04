# GOAL — Beat vLLM-Omni v0.22 by ≥5% on every path × every batch

Defined 2026-07-04. This is the campaign's contract: what "done" means, how
it's measured, what order we attack, and the stretch tiers beyond it.

## 1. The objective, precisely

**PRIMARY GOAL:** M* (one shippable build + at most two documented
per-workload configs) beats vLLM-Omni v0.22 by **≥1.05× in req/s** on
**every cell** of the matrix {i2t, s2t, s2s, i2s} × {B1, B2, B4, B8, B16,
B32}, proven by the acceptance protocol in §4.

Definitions that keep the goal honest:
- **Metric:** req/s (tokenizer-free; vLLM generates ~20% longer text, so
  tok/s flatters neither side consistently). Latency (jct/TTFT) is tracked
  but is a stretch axis, not the gate.
- **Baseline:** vLLM-Omni v0.22 measured LIVE in the same window on the
  same box (HEADTOHEAD protocol), not the stale committed table. Their
  committed 8.210 at i2t B32 vs live 8.24–8.46 shows why: live-only.
- **Build rule:** one primary build (opt/custom-ops lineage). A second
  config is allowed per workload class (e.g. merged-prefill config for
  small-batch/latency; audio-optimal placement for speech) — but each cell
  must be won by a config we'd actually ship for that workload, and the
  config→workload mapping must be documented.
- **Correctness gate per cell:** tok/req in the established band (M*
  172–179 for i2t), zero errors, stop-parity. A "win" with corrupted
  outputs is a loss (the V1 lesson).
- **Reliability rider:** during acceptance, zero M* self-inflicted crashes.
  (vLLM died twice under our benchmark load on 2026-07-04 — reliability is
  a real differentiator; we claim it only if we keep earning it.)

## 2. Current state → gap-to-goal (goal per cell = vLLM_live × 1.05)

Ratios: best defensible today (live where raced, else committed/warm; * =
stale-build cell needing re-measurement; banked = validated but not in the
racing build's config).

| cell | i2t | gap to 1.05 | s2t | gap | s2s | i2s |
|---|---|---|---|---|---|---|
| B1 | 0.94 | **+12%** | 1.14 | ✓ | 2.63 ✓ | 2.47 ✓ |
| B2 | 0.82 (+merge→~0.87) | **+21–28%** | 0.83* | **+27%*** | 2.65 ✓ | 2.54 ✓ |
| B4 | 0.88 (+merge→~0.92) | **+14–19%** | 0.89* | **+18%*** | 2.46 ✓ | 2.85 ✓ |
| B8 | **1.22 live ✓** | — | **1.40 live ✓** | — | 2.12 ✓ | 2.89 ✓ |
| B16 | 1.03–1.07 | **+0–2%** (borderline) | 1.26* | ✓* | 2.49 ✓ | 2.80 ✓ |
| B32 | **0.88 live** | **+19%** | 1.01–1.11 | **+0–4%** | 2.26 ✓ | 2.39 ✓ |

Read: speech is done everywhere (2.1–2.9× ≥ goal with huge margin). B8 is
won live on both text paths. The war is **five text cells**: i2t B32 (the
flagship), i2t/s2t B2+B4 (the small-batch hole), and two borderline
verifications (i2t B16, s2t B32).

## 3. Priorities (ordered by gap × lever-availability × flagship value)

**P0 — i2t B32 to ≥1.05× (need +19% from 0.88).** The flagship cell, the
one everyone quotes. Levers, in order:
  a. Sidecar stage-2: exile route_outputs (~1.8ms) + check_stop consumption
     (~1.1–2.1ms) to the sidecar process — the identical move that already
     paid +9%. Est. +8–15%. THE next build.
  b. V1 async-sched redesigned: the compile work satisfied its precondition
     (main-thread < GPU time is now close); fix the identity failure by
     deferring ONLY emit/transport, keeping stop-state sync. Est. +3–8%.
  c. Custom-ops magnitude polish: step-5 (talker dense-attn op) + budget
     tuning (MSTAR_MIXED_BUDGET_TOKENS sweep 256/512/1024) + variance work
     (our soft cells cost ~0.05× of pooled ratio — find the admission-wave
     cause; vLLM budgets every step, we still wave).
  a+b+c stacked ≈ +12–25% → covers the +19% with margin if two of three
  convert.

**P1 — small-batch text (B2/B4 both paths; need +14–28%).** Levers:
  a. Merged prefill is banked (+4–6% i2t) — build the AUDIO twin
     (prefill_text+prefill_audio merge; same span-concatenation design,
     ~1 day) to cover s2t B2/B4.
  b. Sidecar stage-2 helps here MORE than at B32 (the host floor is naked
     at small batch).
  c. Encoder-inline: fold the encoder walk into the merged prefill walk
     (removes the remaining admission round-trip; the B2/B4 deficit is
     prompt-path latency + naked host floor, proven).
  d. W2 memoization retest at B2/B4 (queued; theory says it converts
     exactly here).

**P2 — borderline verifications (cheap).** i2t B16 (1.03→ needs +2%: will
ride P0's levers; verify), s2t B32 (1.01 committed vs 1.11 canonical —
race it live; V2 budget already helps here), s2t B16 (re-measure on the
current build; 1.26* is stale).

**P3 — hold speech + reliability.** No new speech work (user directive)
except regression sentinels in every sweep; their shm-transport fix is one
release away, so re-race speech when they ship 0.23.

## 4. Acceptance protocol (two stages, as proposed)

**Stage 1 — targeted wins.** Attack P0/P1 with the sloppy-fast loop
(adjacent-pair ratios, 2-round POCs). A cell is provisionally won when a
live adjacent-pair race shows ≥1.05× twice.

**Stage 2 — the proof sweep (only after Stage-1 cells are green).**
1. Full live HEADTOHEAD: all 4 paths × all 6 batches, M* on canonical 6,7,
   vLLM on 2,3, alternating per-cell, ×3 rounds, warm protocol both sides,
   solo box window (or accept-and-note contention symmetrically).
2. Every cell ≥1.05× in the round-pooled ratio; correctness gates green;
   zero M* crashes for the sweep's duration.
3. Committed artifacts: raw results.json per cell per round on the
   benchmarks branch, NUMBERS_V4.md generated from raw, HEADTOHEAD_V3.md
   with the method. The claim lives only if a third party can recompute it
   from committed files.

## 5. Stretch tiers (the "4×" ideation)

- **Tier S1 — 4× on speech:** i2s B8 already reads 2.89×; the un-shipped
  levers are the K-step talker graph (collapse K host rounds per K frames),
  code2wav sequence-parallel for long audio (parity-proven, never
  e2e-raced), and their per-chunk flock transport (which THEY may fix —
  4× is a race against their 0.23). Realistic: 3.5–4.5× at i2s B4–B8
  latency-bound cells. Worth one build-week only after P0/P1.
- **Tier S2 — latency 2–3×:** TTFT at small batch (their 0.16s vs our
  0.43s at B32 today INVERTS the throughput story) — merged prefill +
  encoder-inline + chunked admission could take i2t B1–B4 TTFT under
  theirs; a "wins TTFT too" claim doubles the small-batch story. Track
  TTFT in every P1 A/B.
- **Tier S3 — 4× is NOT physical on text throughput:** kernels are
  equal-class (proven: our fp8 beats their path; DeepGEMM lost); the
  entire text gap lives in host-side step overhead, bounded ~1.3–1.5×
  total. Anyone promising 4× text throughput on this hardware pair is
  selling something. State this openly.
- **Tier S4 — reliability as a headline:** publishable differentiator if
  acceptance logs N days of benching with 0 M* crashes vs their measured
  MTBF under identical load. Free to claim if we keep the logs.

## 6. Definition of DONE
The Stage-2 proof sweep is committed and pushed; every cell of the matrix
reads ≥1.05× live; the manager-facing one-liner is: "One build (plus two
documented workload configs) beats vLLM-Omni v0.22 by ≥5% on every path at
every batch size, live on the same hardware, with zero crashes during the
proof — raw data in the repo."

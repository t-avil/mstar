---
name: mstar-campaign-20260704
description: Campaign night 2026-07-04 — 20/24 goal cells green live; merge-config promoted; fold/gather/jitter falsified; regime inverted (main<GPU); vLLM 5 crashes
metadata: 
  node_type: memory
  type: project
  originSessionId: b5e26f29-b1c0-4cee-8a6f-c7d6b5b6ef24
---

State after 2026-07-04 night session (supersedes parts of [[mstar-decode-bottleneck]]
and [[mstar-v2-budget-policy]]; full detail in HANDOFF_V6.md at pool root +
benchmarks/qwen3-omni-joint/ on the docs branch):

- **Scoreboard**: 20/24 GOAL cells ≥1.05× live (all speech, all s2t, i2t B8/B16).
  RED: i2t B1 0.973 / B2 0.945 / B32 0.918 (bands committed); i2t B4 1.057 straddle.
- **Configs (two now)**: primary candidate = encoff + MSTAR_MERGED_PREFILL=1
  (MINUS both _VISION flags) — +11.2% tok/s at i2t B32, flipped B16 to WIN;
  s2t-small config = base yaml + both merges (opt/prefill-merge-audio @ a2d2f47).
- **Regime inverted post-custom-ops**: main-thread 32% vs GPU 65% at B32 →
  ALL wait-removal levers (checkstop stage-2, V1) parked by valve law.
- **Falsified with counters**: fold family (100% of food101/libri prefills are
  unchunkable ≤256-tok spans; V2 budget produces 0 folds on flagship), admission
  jitter (guard self-suppresses at trough), prefill-gather (readiness serializes
  through KV/encode pipeline — cross-request coalescing structurally dead;
  within-request merged walk is the only coalescing that works).
- **vLLM-Omni 0.22 reliability**: 5 failure events in one session (incl. idle
  death, zombie EngineCore behind live API, release-unlocked-lock teardown
  defect); MTBF ~30-60 min under bench load; ledger committed. M* ~7 boots,
  0 self-inflicted. ALWAYS anti-zombie probe (real completion, not /v1/models)
  before racing vLLM.
- **s2t correctness**: transcript parity proven on 534 pairs; vLLM answer-modes
  on interrogative audio (13/13 cells) — their small-batch s2t "length" is error.

**Why:** avoids re-running tonight's falsifications and re-measuring settled cells.
**How to apply:** start from HANDOFF_V6; use ab_verdict.py for every verdict;
respect max-3-concurrent-M*-labs (RAM 1.5TB box) and never boot during races.

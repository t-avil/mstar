---
name: experiment-report-format
description: Required per-experiment report format — losing-paths table + one-sentence tried + one-sentence next
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

For EVERY experiment report, use this structure (user demanded it become a habit, 2026-07-03):

1. "We are losing at these paths" — a compact table of current ratios vs vLLM
   (only the paths/batch cells relevant to the experiment, losing cells flagged).
2. "What we tried" — ONE sentence.
3. "What's next" — ONE sentence.

**Why:** the user relays results upward fast and wants scannable state, not prose.
**How to apply:** end every benchmark/experiment turn with this block; keep the
table to committed/on-disk numbers; mark projections explicitly. Also: user
wants iteration speed maximized — prefer the persistent lab server
(lab_server.sh / lab_ab.sh in /m-coriander/coriander/tim/) over fresh boots;
signal target ≈5 min per A/B pair. Related: [[mstar-24option-board]]

MODE (2026-07-03 late, user directive): SLOPPY-FAST POC iteration — quick
ratio A/Bs only (1-2 rounds, small cells OK), validate direction, merge
winners into one integration branch; NO absolute-number sweeps until the
experiment queue is exhausted, then ONE full sweep at the end. Never leave
GPUs idle; parallel streams on both NUMA pairs (ratios tolerate concurrent
labs; absolutes don't).

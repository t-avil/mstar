---
name: mstar-stack-n2
description: opt/stack-n2 is the new best build (merge + cfgv2 + checkstop); flagship 0.938 pooled with peaks >1.05; variance is the last gap; Law 14 seesaw
metadata: 
  node_type: memory
  type: project
  originSessionId: b5e26f29-b1c0-4cee-8a6f-c7d6b5b6ef24
---

As of 2026-07-05 ~06:45 (supersedes the ceiling claim in
[[mstar-campaign-20260704]]; full state in HANDOFF_V7.md):

- **Best build: opt/stack-n2** = merge-config + MSTAR_SAMPLER_CFG_CACHE_V2=1
  (+5.2% B32 — the V1 cache's tuple-of-rids key THRASHED under admission
  churn, silently reinstating the 9ms six-sync penalty; slot-tensor fix)
  + MSTAR_SIDECAR_CHECKSTOP=1 (+2.3%, shadow-gated clean).
- **Flagship i2t B32: 0.938 pooled n=6 [0.885-0.994], peaks 1.048/1.031** —
  first-ever live wins (9.008 vs 8.592); M* absolutes entered vLLM's band.
  Remaining gap = M* CELL VARIANCE (7.1-9.0 vs their 8.2-8.6), not speed.
- Final matrix: 21/24 ≥1.05; B1 0.972 / B2 0.999 / B32 0.938 = parity class.
- **Law 14 (seesaw):** every landed host-side win re-flips the wall between
  main-thread and gpu-thread — RE-PROFILE (profile_gate.sh) after every win
  before trusting any parked-lever verdict; checkstop was parked-then-shipped
  in one night this way.
- vLLM-Omni 0.22: 7 failure events in ~12h (ledger committed); always
  anti-zombie probe (real completion) before racing.
- Ops: pair 4,5 quarantined (4 straight boot deaths); pair 0,1 reads ~10%
  low — canonical comparisons on 6,7 only; verify git ls-files after
  committing raw (a *.json gitignore swallowed 4 commits).

**How to apply:** next flagship item is the variance/soft-cell cause; start
from HANDOFF_V7 §4.

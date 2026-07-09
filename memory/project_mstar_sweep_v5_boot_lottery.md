---
name: mstar-sweep-v5-boot-lottery
description: 2026-07-06 full 24-cell sweep committed; boot-to-boot variance measured at 18% on identical code — boot determinism is the next lever for i2t B2/B4
metadata: 
  node_type: memory
  type: project
  originSessionId: b5e26f29-b1c0-4cee-8a6f-c7d6b5b6ef24
---

2026-07-06: full one-boot 24-cell sweep (sweep_mstar_v5, committed with env/
requirements/sentinel) + criterion-warmed re-verifications across 3 boots of
the SAME build (opt/prep-h2d 620de91 via launch_mstar_best.sh). Final matrix
vs committed vLLM refs: 17/24 ≥1.05x (speech 12/12 at 2.3-3.1x, s2t B1 1.68x
confirmed — the sweep's 0.71 was a cold-cell artifact, s2t B2/B8/B16, i2t
B8 1.34/B16 1.28), s2t B4/B32 + i2t B1 parity, i2t B32 band-parity (median
8.56, peaks 8.67).

**Key finding: i2t B4 measured 0.90 on boot-2 and 1.058 on boot-3 — an 18%
boot-to-boot split on identical code+flags (largest directly measured).**
i2t B2 0.94→0.98 likewise. Pooled both parity-class (0.96-0.98). These cells
are NOT code regressions; they sample a boot distribution straddling the ref.

**Why:** and next lever = boot-variance reduction (autotune/capture
determinism), not throughput code. **How to apply:** never grade small-batch
i2t from one boot; pool across boots or fix the boot lottery first. Charts:
gen_v4_charts.py layers v2→v3→v4→v5→sweep_mstar_v5_verified per-metric.

SUPERSEDED IN PART by REVIEW_V9_WHY_WE_LOSE.md (2026-07-06, parallel audit
session): (1) the afternoon boot failures were NOT bring-up-internal — root
cause was 289GB of tim-owned orphaned /dev/shm segments starving NUMA node 1
(14.6GB free) + launcher membind=1 fallback → CUDA context OOM; cleaned, boots
again; ALWAYS check per-NUMA-node free + /dev/shm, not just total RAM.
(2) MSTAR_PREP_DEVICE_POS only fixed the B1 per-request path — the BATCHED
pos_ids path (submodules.py:609) still does pageable fp32 H2D every step at
B>1 = the losing cells; fix = "E1 god branch", top priority. (3) vLLM 0.22
gains come from core Model Runner V2 (GPU-native input prep, zero-sync decode
loop). (4) Distrust pre-sweep "green" claims for i2t B1/B2/B16 — races could
average in vLLM crash runs (thr=0.0); filter inference_system=="ours", drop
thr<=0, quote medians+spread of >=5 warmed repeats.
Related: [[mstar-stack-n2-build]] ("variance is the gap"), [[mstar-decode-bottleneck-2026-07]].

UPDATE 2026-07-06 evening: E1 (opt/prep-pos-batched-v9 @915ab8f = prep-h2d +
MSTAR_PREP_DEVICE_POS_BATCHED, the V9 batched pos_ids fix) CERTIFIED on the
warmed 11-cell text matrix: 9/11 win-or-parity, s2t B8 1.30x/B16 1.35x/B32
1.17x new bests; i2t B32 0.90-this-boot (lottery; same-server A/B +5.9%
stands). PROMOTED: launch_mstar_best.sh now boots mstar-godv9 @915ab8f —
the shipping-candidate label moved to E1. Raw: sweep_mstar_e1/ on the bench
branches. First post-shm-fix boot was clean (12 min) — V9 root cause holds.

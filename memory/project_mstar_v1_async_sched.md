---
name: project_mstar_v1_async_sched
description: "V1 async-scheduling (deferred postprocess) for M* — built, GPU-smoked, FAILED (non-identical +9% length, tok/s wash); parked"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

V1 "async scheduling / GPU-resident sampled ids" for the M* engine (branch
`opt/async-sched`, from `opt/sched-pack`; flag `MSTAR_ASYNC_SCHED`, requires
`MSTAR_DIRECT_FEED`). Idea: defer the whole per-step postprocess (check_stop /
route+store / emit / WGD) one loop iteration so it consumes tokens whose D→H
already finished, removing the completion_event/check_stop wait from the
critical path. Built 2026-07-03, CPU-validated, GPU-smoked on GPUs 4,5.

**Verdict: FAILED as-built. Parked (flag default-OFF, off-path byte-identical).**
- NON-IDENTICAL: async ON systematically lengthens i2t B32 generations +9%
  (tok/req OFF 178.3 vs ON 194.3, same worktree, 4 cells each). Fails the
  byte-identity bar.
- NO SPEED WIN: per-token tok/s a wash (OFF-healthy 1287 vs ON ~1250). The
  apparent req/s drop is just the length inflation (more tok/req at equal
  tok/s). Matches SIDECAR_DESIGN §0 "+0-5%, real value post-sidecar", the W3
  skip, and the GIL-valve law.

**Why (mechanism, confirmed by elimination):** NOT deferred sampler state — the
philox RNG `_step_offset` and rep-penalty `seen_mask` both update inside
`sample()` on the GPU thread (sampling.py:489-490), untouched by the postprocess
deferral. It's the DEFERRAL shifting scheduling: async carries a 2nd overrun row
per stopping request + lags admission/routing one iteration → decode batches have
systematically different composition → with nondeterministic fp8-MoE numerics the
sampled distribution drifts ~9% longer. The `_async_trim` keeps overrun tokens out
of the emitted stream (unimodal length dist, no missed-EOS) but can't undo the
numeric footprint the overrun rows inject into the shared forward.

**Fundamental tension:** the win needs deferring check_stop (to drop the token
wait); deferring check_stop is what lags the stop → overrun → composition shift →
non-identity. Identity and the-win are opposed for this wait. Deferring only
emit/transport preserves identity but captures ~none of the wait (the dominant
cost is waiting for the forward to COMPLETE, not the D2H copy). Don't re-attempt
V1 naively; revisit only if main-thread Python drops well below GPU time.

Commits: 0750145 (impl), 1efaf37 (CPU tests), d153777 (SMOKE.md), 18b1244 (fix:
a real KeyError deferred-remove race the smoke caught — `_remove_request` must
defer removal of `_deferred_pp` rids, not just `_in_flight_rids`). Relates to
[[project_mstar_decode_bottleneck]], [[feedback_ingraph_microbench]].

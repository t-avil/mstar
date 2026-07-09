---
name: project_mstar_sidecar_checkstop
description: "Sidecar Stage-2 check_stop offload (MSTAR_SIDECAR_CHECKSTOP) — built, branch, scope, A/B recipe"
metadata: 
  node_type: memory
  type: project
  originSessionId: b5e26f29-b1c0-4cee-8a6f-c7d6b5b6ef24
---

Sidecar STAGE-2 = check_stop offload ONLY (route/store CANNOT move — it drives next-step scheduling readiness + touches CUDA tensor lifecycle; that's Stage-3 EngineCore-rewrite "expect NO" per SIDECAR_DESIGN §3.2/§8). The HANDOFF_V5 "exile route_outputs" framing was WRONG; verified against code.

Built on branch **opt/sidecar-checkstop**, worktree /m-coriander/coriander/tim/mstar-stage2 (forked from opt/custom-ops), pushed to fork t-avil/mstar. Files: mstar/worker/worker.py + test/modular/test_sidecar_checkstop.py (11 CPU tests pass).

**What MSTAR_SIDECAR_CHECKSTOP=1 does (static flag, CUDA-only):** deferred-consume of the check_stop D→H — `_checkstop_barrier` records a side-stream event instead of blocking `side.synchronize()`; reordered postprocess so dynamic-loop-iter Python overlaps the copy; `_await_checkstop` polls event.query() (ready→no-wait `checkstop_deferred_consume`; not-ready→counted blocking `checkstop_sync_fallback`). Decision is ALWAYS same-step (rule: no deferred decision = avoids V1 identity failure). `_compute_new_stops` extracted verbatim for shadow. Shadow `MSTAR_SIDECAR_CHECKSTOP_SHADOW=1` recomputes from forced-sync D→H, legacy authoritative, mismatch→WARNING+`checkstop_shadow_mismatch`.

**A/B recipe:** two-server alternation (static flag, NOT dyn_ab). Arm A = winning build; Arm B = +MSTAR_SIDECAR_CHECKSTOP=1. Both need full winning stack incl MSTAR_EMIT_SIDECAR=1. PYTHONPATH=<mstar-stage2>. Run shadow FIRST, require checkstop_shadow_mismatch==0 over thousands of steps, then drop shadow for perf. Mechanism-alive: checkstop_deferred_consume must dominate; if sync_fallback dominates it's a correct no-op (wait was load-bearing graph-tail), not a regression.

**Risk:** GIL-valve (Law 2) — converts only if main thread still the wall after emit sidecar; FAST_SEND is the −3% counter-precedent. Predict +3–10%, kill <+2%.

**2026-07-05 UN-PARKED + STACKED.** Original profile gate parked it (main 32% < GPU 64.8%). But cfgv2 (MSTAR_SAMPLER_CFG_CACHE_V2, the sampler churn-proof cache, WON +5.2% B32) flipped the regime BACK: cfgv2-on profile = MainThread 55% active = the wall again (GPU 85.7% busy, gpu-worker idle). So checkstop is viable again. Built **opt/stack-n2** = cfgcache-v2@ded928d + checkstop cherry-pick (7ef5150), worktree /m-coriander/coriander/tim/mstar-stack, pushed. Zero-conflict (checkstop=worker.py, cfgv2=sampling.py). CPU tests pass (11/11 checkstop + 2/2 test_sampling_flashinfer_seed with ninja on PATH). A/B: reboot arm, V2=1+CHECKSTOP=1+SHADOW=1 first (gate shadow_mismatch==0 + deferred_consume dominant), then drop SHADOW for perf vs cfgv2 lab B-side. NOTE cfgv2 identity bug I caught (off→on dynflag flip left slot rand stale) was FIXED in ded928d — but only matters for dyn_ab; the stack A/B is boot-static.

**NOT built (deliberate, team-lead-approved rule 4):** the fuller §6.2 offload — EOS-in-sidecar + StopFeedback reverse channel + persistent multi-step overstay set. Reason: reverse channel breaks the one-way data-flow invariant (design's central safety property §6.1); redundant under same-step decision; needs GPU shadow validation. See [[project_mstar_v1_async_sched]] for the deferred-decision identity trap this avoids. Builds on [[project_mstar_custom_ops_crusade]] emit-sidecar Stage-1.

---
name: project_mstar_w2_retest
description: "W2 step-txn memoization retest rebased onto opt/custom-ops — branch, flag, gates, prediction"
metadata: 
  node_type: memory
  type: project
  originSessionId: b5e26f29-b1c0-4cee-8a6f-c7d6b5b6ef24
---

W2 = MSTAR_STEP_TXN (decode-step transaction): memoized replay of the WHOLE
per-rid _postprocess_batch derivation (node-complete loop advance + ready-slot
bookkeeping + output routing + ref-count fanout) for a uniform *continuing*
thinker_decode step. Composes ABOVE W1 (MSTAR_FAST_POSTPROC): fast_execute
`continue`s past the entire slow block incl the newer FAST_ROUTE2. Original
verdict (old base): CORRECT (zero shadow-verify mismatches over thousands of
B8/B32 steps) but WASH at B32 (postprocess overlaps GPU). Theory: converts at
B2/B4 where the host floor is naked.

RETEST (2026-07-04): rebased single commit b98de66 (from exp/step-txn,
worktree mstar-w2) onto opt/custom-ops -> NEW branch opt/w2-retest, NEW worktree
/m-coriander/coriander/tim/mstar-w2retest, commit a2788a9, pushed to fork.
Rebase was clean: diff vs opt/custom-ops = ONLY step_txn.py (new, 479L) +
worker.py (+189L, ZERO deletions => flag-off byte-identical to custom-ops). 3
conflicts, all additive (kept both sides): __init__ flag inits + the 2
invalidate-on-rid-drop sites now call BOTH invalidate_route_plan AND
_step_txns.invalidate. No CPU tests exist for step_txn (validated by runtime
shadow-verify); modular test failures (worker_speculation 10/13, vjepa2) are
PRE-EXISTING identical on the base, not from the merge.

Flag BOOT-STATIC (not in _refresh_dynamic_flags) => A/B is two-server / reboot,
NOT dyn_ab. Added Law-4 counters step_txn_fast_hits + step_txn_captures
(WALK_STATS WARNING dump; original had NO positive hit counter).

A/B RECIPE: arm A = winning i2t/s2t flags, MSTAR_STEP_TXN=0, booted from
opt/w2-retest (byte-identical off); arm B = +MSTAR_STEP_TXN=1. Cells i2t:2,
i2t:4, s2t:2, s2t:4 (W2 only fires on thinker_decode, so text-output cells get
full decode coverage). GATES: (1) PRE: MSTAR_STEP_TXN_VERIFY=1 must show ZERO
mismatches on THIS base first — the custom-ops postprocess flow changed vs W2's
original, so the hand-mirrored slow-path reproduction must be re-verified before
any perf read; a mismatch => W2 incorrect on this base, fix-first. (2) mechanism:
step_txn_fast_hits>0, high hit-rate. (3) spread gate (Law 3/6) — boot-static so
≥3 rounds; original park was an 18% B2 same-config spread failure.

PREDICTION (revised DOWN from the old 5-10%): arm A already runs W1+FAST_ROUTE2,
so W2's MARGINAL = only the residual it uniquely covers (mark_node_complete loop
advance + LoopStateRegistry reset_for_iter + ref-count fanout). Honest estimate
2-5% at B2/B4, plausibly wash if FAST_ROUTE2 already shaved the LoopStateRegistry
churn — Law 7 says microbench/arithmetic the residual host-Python slice if the
spread gate can't resolve it. KILL: fast_hits confirms firing AND B2/B4 ON within
OFF spread over 3 rounds => W2 confirmed dead even at naked host floor after
custom-ops (theory falsified, close the lever). Related:
[[project_mstar_decode_bottleneck]] [[project_mstar_custom_ops_crusade]].

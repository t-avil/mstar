# showcase/dual-goal — clean feature-by-feature history (status)

Fork point: `encoders-implemeneted` @4c33b33a (the baseline). Goal: one clean,
gated, single-feature commit per winning feature, PD/DP NEVER present, no leftover
experimental code — for academic presentation of the journey baseline -> winning.

## UPDATE 2026-07-19 (multi-day completion in progress)
8 clean feature commits now done (added: custom-op route MSTAR_CUSTOM_OPS [4 primaries,
side-prefill fix 586fcfa1 excluded as out-of-scope], sampler-config cache
MSTAR_SAMPLER_CFG_CACHE [primary]). STRATEGY for the rest: apply remaining feature
intro-commits in godv9 TOPOLOGICAL ORDER (distance from baseline) so shared-file changes
(sampling.py, worker.py, submodules.py) stack correctly instead of colliding. Remaining
order: 9175055b sidecar(15) -> 76f3afec/6e39106b chunked-prefill(18,20) -> mixed-batch
cluster 1093527a..e31f39ed(22-32) -> route 267e5cf0/fd8eae5f/333b7ddd(58-68) ->
55faf299 checkstop-talker(76) -> 01856e80 codec-emit(78) -> 0cea3e8d mixed-budget(81) ->
35a9358c/915ab8f3 prep-device-pos(95-97) -> 92b31307/07eba5bb merged-audio(107-109) +
655f15ca auto-gate -> 18330f4d/a6f55ba1 preproc-pool(115). Squash multi-commit features.
Resolve conflicts referencing the winning-branch final code. Excluded fixes are noted per
commit. FINAL: rebench the completed showcase (when GPUs free) to confirm it reproduces
winning numbers; then push + note in benchmarks README.

## DONE — 8 verified clean commits (pushed: 6 below + custom-ops + sampler)
1. remove redundant HF-CPU image-preprocess fallback + dead guards  (cleanup-first)
2. fp8 MoE grouped GEMM (MSTAR_MOE_FP8)
3. ordered emit (MSTAR_ORDERED_EMIT)
4. batched vision prefill + capture-grid overrides (MSTAR_BATCH_VISION_PREFILL, grids)
5. grid-parser review fixes (MSTAR_PREFILL_BUCKETS, dup-token fix)
6. decode host-floor stack (FAST_POSTPROC + BATCH_EMIT + SLIM_EMIT)
(all cherry-picked clean from eiv2 `encoder-implemented-v2`, syntax-verified)

## REMAINING — planned, needs careful conflict resolution (NOT rushed)
Each of these winning features has a clean PRIMARY commit in godv9's history that
cherry-picks cleanly, but their FOLLOW-UP FIX commits + worker.py-heavy features
conflict against godv9's intervening refactors. Extracting them cleanly (primary +
fix, squashed, conflict-resolved, each reproducing winning behavior) is delicate
git surgery best done with review, not force-resolved autonomously.

Ordered plan (dependency-aware), with source commits (godv9 line):
 7. custom-op route (MSTAR_CUSTOM_OPS)         b7fb94e8+f152e3bd+be16eebe+12ce776d (+586fcfa1 fix*)
 8. on-device batched pos-ids (PREP_DEVICE_POS/_BATCHED)  35a9358c+915ab8f3(+620de912)
 9. sampler-config cache (SAMPLER_CFG_CACHE/_V2)  f8e98d1a(+d21c4973*+fixes)
10. faster route + send (FAST_ROUTE/_ROUTE2, FAST_SEND)  267e5cf0+fd8eae5f+333b7ddd
11. integer stop check + Talker (FAST_CHECKSTOP/_TALKER)  3b5e390a*+55faf299
12. emit sidecar + sidecar checkstop (EMIT_SIDECAR/_SIDECAR_CHECKSTOP)  9175055b*
13. chunked codec-token emit (CODEC_CHUNK_EMIT)  01856e80
14. off-process preproc pool (PREPROC_PROC/PROCS, BURST_CAP/THREADS)  18330f4d*+a6f55ba1  *** i2t win ***
15. chunked Thinker prefill + budgets (CHUNKED_PREFILL_V2, PREFILL_CHUNK_TOKENS, MIXED_BUDGET_TOKENS)  76f3afec(+6e39106b*+0cea3e8d)
16. captured mixed batch (MIXED_BATCH[_VISION])  1093527a+b265cd1c+7c09d101+f45cbbdf+a599bb63+864d4925
17. mixed step rides decode spec chain (MIXED_SPEC)  e31f39ed+helpers
18. s2t merged audio prefill + occupancy auto-gate (MERGED_PREFILL_AUDIO/_MAX_BS)  92b31307+07eba5bb+655f15ca  *** NEW ***
(* = commit(s) that conflict on cherry-pick and need manual resolution)

Decisions already made: KEEP FAST_SEND (in winning flag string, fidelity over the
-3% wash); EXCLUDE R1 (INT_UUID/SKIP_REDUNDANT_SYNC — default-off, not in any enable
list). EXCLUDE all PD/DP (9a3ffd1f, e79cc825, 510beedd FUSED_KV_HANDOFF), INGRAPH_GREEDY,
PINNED_SAMPLE_PARAMS, ENC_OVERLAP, WALK_STATS counter.

The COMPLETE winning feature set + how-to-enable is documented in FEATURES.txt on
branch winning/dual-goal (the validated build these commits reconstruct). eiv2
(`encoder-implemented-v2` @510beedd) is the style model + cherry-pick source.

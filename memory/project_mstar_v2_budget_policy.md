---
name: project_mstar_v2_budget_policy
description: "V2 budgeted chunked-prefill admission (MSTAR_MIXED_BUDGET_TOKENS) built on opt/v2-policy, awaiting GPU smoke"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

V2 "budgeted chunked-prefill admission" for M* mixed batching, built code-only
on branch **opt/v2-policy** (off opt/integration-v4), default OFF, pushed to
fork. Commits 0cea3e8 (impl) / a27f19e (tests) / 622f7e5 (SMOKE.md). Worktree
/m-coriander/coriander/tim/mstar-v2pol.

**MSTAR_MIXED_BUDGET_TOKENS=N** (0=off): fold a ready mixable chunk into the
decode spec chain on EVERY step (capped at N total tokens = n_decode + C),
instead of only at must_yield_away yield boundaries (~8% of steps). Reuses the
existing fold machinery; eager_probe is now `mixed_single_chunk OR budget>0`.
Also MSTAR_MIXED_BUDGET_MIN_DECODE (default 24, the inherited occupancy floor).
Counters: budget_folds / budget_fold_tokens / budget_skips_floor. Dynflags-
refreshable (bakes nothing into capture).

**Why it is NOT the closed MSTAR_MIXED_SINGLE_CHUNK** (see [[project_mstar_decode_bottleneck]]
graveyard): SINGLE_CHUNK ALSO routed short standalone prefills through the chunk
planner (allow_single_chunk) → occupancy tax −10%. V2 leaves allow_single_chunk
and the mixable gate (_chunk_entry_passes_gates needs prefill_chunk_len)
untouched — only ALREADY-EXISTING chunks (long/vision prefills) are accelerated.
Standalone admission is byte-identical.

Expected value = TTFT/admission at B2-B8; a fold is ~compute-neutral so **B32 is
a no-regression sentinel, not a win**. SMOKE.md: 3 arms (off / budget-512 /
budget+split-attn+preplan), i2t B2/4/8 primary + B32 sentinel, one-server dyn_ab
for off-vs-budget. Risk: inert if the workload produces no chunks (budget_folds
stays 0 — check vision chunking / long prompts first). Ties into
[[project_mstar_mixed_batch_p2]]. Awaiting main's GPU smoke.

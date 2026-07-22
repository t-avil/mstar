# Merge strategy — landing #131 on `upstream/main`

**Status date:** 2026-07-22. Verified with `git merge-tree --write-tree` (real
`<<<<<<<` conflict markers in the merged tree, not the noisy name-only list).

## 1. Parity: you are 14 commits behind upstream

- Base of this work = `origin/main` `9ee13699`. That is **14 commits behind
  `upstream/main` `d7e79890`**, and only **5 commits ahead** — and those 5 are
  *docs + `benchmark/sweep.sh`*, **no product code**. So the fork's `main` has not
  diverged from upstream in code; it has simply not kept up.
- Merge-base(upstream, submission) = `ae7d1736`. `submission/single-config` is 144
  commits past it; `encoders-implemented-v2` is 24 past it.
- The 14 upstream commits that moved under us, biggest conflict drivers first:
  | commit | PR | what it rewrote |
  |---|---|---|
  | `a3baaa5c` | #121 cosmos3 | **HUGE** — `cache_manager.py +1210`, `cuda_graph_runner.py +202`, `kv_cache_engine.py`, `data_worker.py`, `conductor.py +38`. #1 driver. |
  | `78995a22` | #154 | piecewise runner model-agnostic — `cuda_graph_runner.py +559/-413` rewrite |
  | `52ae1c00` | #149 | remove `conductor_new_token` — `conductor.py`, `worker.py -90` |
  | `8a107f39` | #164 | vendored Rust transport — `communicator.py`, `data_worker.py`, `worker.py` |
  | `d7e79890`/`07b39f7c`/`e0cd8606` | #181/#177/#180 | `data_worker.py`, `worker.py`, `tensors.py` |
  | `d198f8b3`/`59fe435c`/… | #157/#159/… | MoE-only / ASR (additive), trivial |

## 2. Conflict matrix (exact files, verified hunk counts)

| branch vs `upstream/main` | files | hunks | touches core mechanism? |
|---|---|---|---|
| `submission/single-config` | 11 | ~29 | conductor.py auto-merges **clean** |
| `encoders-implemented-v2` | 5 | 15 | does **not** touch conductor / cache_manager / cuda_graph_runner |
| `encoders-implemented-md` | — | — | **orphan docs branch, no shared history, no product code — exclude** |

`submission` conflicts: `worker.py` 8 · `engine/cache_manager.py` 5 ·
`api_server/data_worker.py` 4 · `engine/cuda_graph_runner.py` 2 · `entrypoint.py` 2 ·
`communication/communicator.py` 2 · `engine/kv_cache_engine.py` 2 · `distributed/base.py`
1 · `qwen3_omni/components/attention.py` 1 · `utils/sampling.py` 1 · `micro_scheduler.py` 1.

**All 29 conflicts are in the default-off flag machinery** (emit path, cache manager,
cuda-graph runner). **None are in the core single-config mechanism.**

## 3. The key result: the core win is one clean commit

The core #131 win is a **single commit — `4216adc9`** "feat(single-config):
replicated-encoder + output-modality routing" — touching only
`mstar/conductor/conductor.py` + `configs/qwen3omni_2gpu_dpenc.yaml`. In the
test-merge against upstream, **`conductor.py` has zero conflict markers** even though
both sides edited `_assign_worker_graphs_to_workers`. So the actual feature rebases
essentially clean; the 29 conflicts belong to the *perf-flag stack*, which is
separable.

## 4. Recommended PR split (decoupling)

1. **PR-1 — core (this is #131 / #150):** cherry-pick `4216adc9` alone onto fresh
   `upstream/main`. Replicated-encoder + output-modality routing + `dpenc.yaml`.
   Rebases clean modulo the one hazard in §5. **This is the PR to open first.**
2. **PR-2 — `MSTAR_ORDERED_EMIT` first-token-race correctness fix.** Self-contained,
   ship-critical (fixes a real dropped/reordered first token under concurrency),
   defensible on its own merits independent of the throughput story.
3. **PR-3 — image multiprocess preprocess pool.** `api_server/data_worker.py`; only
   mechanical collisions with #181/#177. This is the i2t TTFT fix.
4. **PR-4+ — host-floor decode flag stack (defer).** `FAST_SEND`, `SIDECAR_CHECKSTOP`,
   `SLIM_EMIT`, batched emit, cuda-graph split-attn, fp8/custom-ops. These are what
   collide head-on with #149/#154/#180 in `worker.py` + `cuda_graph_runner.py`.
   **Rebase these AFTER PR-1/2/3 and RE-BENCHMARK on upstream's rewritten piecewise
   runner** — #154 changed the capture path these flags optimize, so their measured
   deltas must be re-earned, not assumed.

`encoders-implemented-v2` is the cleanest base for the flag-stack work (5 files/15
hunks, dodges the cosmos3 and piecewise rewrites entirely) — but it does **not**
contain the core single-config win, so it is not the base for PR-1.

## 5. The one rebase hazard to hand-check (core mechanism)

Both sides refactored `_assign_worker_graphs_to_workers`; git 3-way-merges it without
a marker, but the result needs a human check + retest:

- base `ae7d1736`: `def _assign_worker_graphs_to_workers(self)` — plain DP random pick.
- **submission** (`conductor.py:472`): adds param `output_modalities=None` + the
  modality route (`_speech_out` → rank 1, else rank 0); call site `_do_ingest_request`
  passes `body.initial_output_modalities`.
- **upstream** (`conductor.py:451`): kept the **no-arg** signature and added a new
  `if wg._instance_ranks:` whole-instance-lockstep branch (#176/#154).
- Auto-merge yields a function carrying BOTH the new param AND upstream's
  `_instance_ranks` branch. **HAZARD:** verify the modality override actually fires on
  the new `_instance_ranks` path and was not merged only onto the old flat `else`
  branch (which would silently disable routing for instance-based worker graphs).
  **Action:** after PR-1 rebases, run the CPU routing assertion (text→rank0,
  speech→rank1) on the upstream conductor to confirm the route still fires, then a
  live i2t + i2s smoke to confirm placement.
- Call site is **not** a hazard: upstream left `initial_output_modalities` intact; the
  default-None param reconciles.

## 6. Two-minute reproduction of this analysis

```bash
git fetch upstream
git merge-tree --write-tree upstream/main submission/single-config | grep -c '^<<<<<<<'   # hunks
git log --oneline upstream/main ^origin/main                                              # the 14
git show --stat 4216adc9                                                                  # the core commit
```

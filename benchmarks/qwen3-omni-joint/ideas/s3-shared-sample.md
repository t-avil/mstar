# Idea s3: shared sample+unpack (fix-20 item #10)

Branch: `idea/s3-shared-sample`, worktree `/m-coriander/coriander/tim/wt-s3-shared-sample`.
Flag: `MSTAR_SLIM_SAMPLE` (default OFF, dynflag-toggleable, per-call env read
— no `register_cache_clear` needed since nothing in the new code is cached).

## Per-step Python bodies found (file:line, before this change)

1. `mstar/utils/sampling.py:731-739` (`Sampler.sample`, ARGMAX_FAST check) —
   `all(c.temperature == 0 for c in configs)` + `any(c.repetition_penalty !=
   1.0 for c in configs)`: 2 generator scans over the batch, every decode
   step, before even reaching the "full" path.
2. `mstar/utils/sampling.py:799-802` (`Sampler.sample`, post-ARGMAX-FAST-check)
   — `any_rep_pen` / `any_greedy` / `any_top_k_zero` / `all_top_k_zero`: 4
   more independent scans over the same `configs` list, run on every
   non-all-greedy step (i.e. whenever ARGMAX_FAST doesn't fire, which is
   most mixed-temperature production traffic).
   Total: up to **6 separate Python-level scans of the same B-sized list**
   per `sample()` call, B <= 32 in practice.
3. `mstar/engine/kv_cache_engine.py:509-516` (`_execute_batched`, eager
   batched-logits fast path) — slice `batched_logits[:len(request_ids)]`,
   `sampler.sample(...).clone()`, `.split(1)`, build
   `{rid: {"new_token": [view]}}` via a for-loop that also merges into any
   pre-existing per-rid dict and pops `"logits"`.
4. `mstar/engine/cuda_graph_runner.py:2191-2218` (`_sample_and_remap`,
   `__batched_logits__` fast path, CUDA-graph post-replay) — same
   slice/index_select + `sampler.sample(...).clone()` + `.split(1)` +
   `{rid: {"new_token": [view]}}` dict comprehension, plus `slot_map`
   handling for `MSTAR_MIXED_SPLIT_ATTN`.
   Bodies 3 and 4 had already drifted into two independently-maintained
   copies of the identical operation — the FlashInfer output-buffer-alias
   `.clone()` rationale comment is duplicated **verbatim** in both files,
   which is the exact drift risk item #10 called out.

## What was fused / hoisted

- **`_scan_sampling_configs(configs)`** (`mstar/utils/sampling.py`): one
  Python loop over `configs` computing `all_greedy`, `any_greedy`,
  `any_rep_pen`, `any_top_k_zero`, `all_top_k_zero` together, replacing up
  to 6 separate `any()`/`all()` scans (items 1+2 above) with 1. Wired into
  `Sampler.sample` behind `MSTAR_SLIM_SAMPLE`; flag off runs the original
  `any()`/`all()` calls completely untouched, in the same order, so there
  is zero behavior or performance change when off.
- **`sample_batched_and_unpack(sampler, request_ids, batched_logits,
  slot_map=None)`** (`mstar/utils/sampling.py`): the shared body for items
  3 and 4 — slice/`index_select`, `sampler.sample(...).clone()`,
  `.split(1)`, zip into `{rid: {"new_token": [view]}}`. Returns
  `(sampled, new_token_map)` so `MSTAR_DIRECT_FEED`-style callers get the
  whole-batch tensor without a second sample/clone. Both engine call sites
  now call this behind `MSTAR_SLIM_SAMPLE`; each keeps its own
  call-site-specific merge logic on top (kv_cache_engine merges into any
  pre-existing per-rid dict and pops `"logits"`; cuda_graph_runner builds
  `MSTAR_DIRECT_FEED` and non-logit-output entries) — only the genuinely
  duplicated slice+sample+clone+split core moved, not the divergent
  surrounding logic, to keep the change low-risk.

## Design notes / what was deliberately left alone

- `CudaGraphableSampler.sample` (the actual in-graph-capturable sampler,
  used by Orpheus's `_forward_prefill_batched`/`_forward_decode_batched`
  and Qwen3-Omni's Talker code-predictor) is untouched — both engine call
  sites that got the shared helper call the **eager** `Sampler`, not
  `CudaGraphableSampler`; sampling itself always happens in eager Python
  even on the CUDA-graph path (the graph only replays the forward, not the
  sample). No CUDA-graph capture semantics are touched by this change.
- `mstar/model/orpheus/submodules.py:233-263` and
  `mstar/model/qwen3_omni/submodules.py:2550` have a structurally similar
  `{rid: {"new_token": [...]}}` pattern but call `CudaGraphableSampler.sample`
  (no `.clone()` — different aliasing contract, likely intentional since
  they run inside the capture region) — deliberately NOT folded into the
  shared helper; unifying would risk changing capture-region semantics for
  a helper meant to be a pure eager-code convenience.
- `kv_cache_engine._sample_decode_outputs` (the non-batched per-rid decode
  fallback, only reached when a submodule doesn't emit `__batched_logits__`)
  was left alone: it's a materially different loop shape (per-rid
  `sampler.sample([rid], ...)` calls, not one batched call), not the
  same duplicated body as items 3/4, and changing it isn't part of this
  idea's blast radius.

## Risks

- The `all_greedy if slim_sample else all(...)` / `any_rep_pen if
  slim_sample else any(...)` ternaries in `Sampler.sample` rely on Python
  only evaluating the taken branch of a conditional expression (no
  `UnboundLocalError` risk from referencing `all_greedy`/`any_rep_pen`
  before assignment when `slim_sample` is False) — verified by the CPU
  test below exercising both flag states through the real `Sampler`.
- `sample_batched_and_unpack`'s `slot_map` branch
  (`MSTAR_MIXED_SPLIT_ATTN`) is only exercised by `cuda_graph_runner`
  today; `kv_cache_engine`'s call always passes `slot_map=None`. Covered
  by a synthetic-slot_map unit test since a live `MIXED_SPLIT_ATTN` GPU
  repro wasn't in scope (no-GPU constraint on this task).
- No change to sampling math, RNG offsets, or seen-token-mask bookkeeping
  in either branch — only which Python statements assemble the already-
  identical booleans/dict, so token-identity risk is limited to the ternary
  short-circuit behavior above.

## Validation done (this task, CPU-only, no GPU touched)

- `python -m compileall mstar` — clean.
- `PYTHONPATH=.../wt-s3-shared-sample <mstar-new venv python> -c "import
  mstar"` — clean.
- `/m-coriander/coriander/tim/tmp/test_slim_sample.py` (CPU-only,
  `CUDA_VISIBLE_DEVICES=""`):
  - `_scan_sampling_configs` vs the legacy 5-line any()/all() computation,
    across 5 hand-picked batches (all-greedy, all-sampled, mixed
    greedy/sampled, greedy+rep-penalty, mixed top_k) plus 20 randomized
    batches up to bs=32 — exact match on all 5 booleans, every case.
  - `sample_batched_and_unpack` vs the legacy inline slice+sample+clone+
    split+dict-build, using a `_FakeSampler` (records call args, returns
    argmax) — both no-`slot_map` and `slot_map` cases: identical sampled
    tensor, identical per-rid `new_token` views, identical
    `sampler.sample()` call args (same request_ids order, same logits
    tensor by value).
  - End-to-end through the real `mstar.utils.sampling.Sampler.sample` with
    `MSTAR_ARGMAX_FAST=1` (the only CPU/no-FlashInfer-JIT-safe branch) and
    `MSTAR_SLIM_SAMPLE` toggled 0/1: token-identical output, matching
    `logits.argmax(dim=-1)` directly.
  - All 4 checks: PASS.
- Did not attempt the mixed-temperature FlashInfer path or a live
  `cuda_graph_runner`/`kv_cache_engine` integration run — both need a GPU
  and this task is explicitly no-GPU. That is the gap the A/B recipe below
  closes.

## A/B recipe (for the next GPU session)

1. Quiet-box (load < 25) paired cells, i2t B32 food101, `n>=96`, per
   LEARNINGS_TTFT.md protocol lesson #15. Compose with the fix-20 winner
   set (`MSTAR_EMIT_RID_INDEX` + `MSTAR_EMIT_INLINE_FASTPATH` +
   `MSTAR_ARGMAX_FAST` already promoted).
2. Cell A: winner set only. Cell B: winner set + `MSTAR_SLIM_SAMPLE=1`.
   2 pairs minimum (per LEARNINGS_FIX20.md's "2/2 pairs" bar), alternating
   order to cancel boot-lottery drift.
3. Metrics: ITL (this is a per-decode-step Python-body cut, so ITL is the
   primary signal, not TTFT) and tok/s at B32; also run one B1 sequential
   identity check (flag off vs on, same seed, greedy) to confirm token
   sequences match before trusting any throughput delta.
4. Also smoke `MSTAR_MIXED_SPLIT_ATTN=1` + `MSTAR_SLIM_SAMPLE=1` together
   once — the only live exerciser of `sample_batched_and_unpack`'s
   `slot_map` branch.
5. Expected effect size: small. This is pure per-step CPU-interpreter
   overhead reduction (a handful of scans over B<=32 items, one fewer
   Python function's worth of duplicated bytecode at two call sites) — not
   a kernel or scheduling change. Given the fix-20 py-spy floor
   (`_postprocess_batch` 23%, `check_stop` D→H 18%, zmq 13%, sync 8%),
   config-boolean scanning inside `sample()` is a sliver of the remaining
   Python-body budget; expect low-single-digit-percent ITL at best, likely
   within noise at B32 under any real load. Ship if 2/2 pairs are
   non-negative and B1 identity holds; otherwise park like #7/#3.

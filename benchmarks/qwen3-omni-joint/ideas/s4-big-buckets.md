# S4 — Grow the captured mixed-step chunk grid (MSTAR_MIXED_CHUNK_SIZES)

Branch/worktree: `idea/s4-big-buckets` @ `/m-coriander/coriander/tim/wt-s4-big-buckets`.
Status: implemented, default OFF (unset = byte-identical), boot-time flag. No GPU
touched, no server started. Commit: see `git log -1` on the branch.

Scope: `mstar/model/qwen3_omni/submodules.py` (capture grid), `mstar/worker/micro_scheduler.py`
(G1 gate mirror), `mstar/worker/worker.py` (COADMIT budget clamp), plus a new CPU-only
test file. Nothing else touched.

---

## The problem (from fix1_coadmit.md / LEARNINGS_FIX20.md finding #1)

The `thinker_mixed` step (decode rows + one prefill chunk row) is a captured CUDA
graph. Its only chunk buckets are `MIXED_BATCH_CHUNK_SIZES = [256, 288, 512]`
(`submodules.py:1600` before this change). A chunk larger than 512 tokens can
**never** fold into a decode step, no matter how large `MSTAR_COADMIT_BUDGET_TOKENS`
is set — routing it into an uncaptured shape is the UNCAP-IMA failure. Ship runs
`MSTAR_PREFILL_CHUNK_TOKENS=2048`, so i2t **text** prefill chunks are already up to
2048 tokens; they simply never fold. This is P3 from the fix1 report: "grow the
captured mixed bucket at boot" is the only path to real one-step co-admission
beyond the current 512-token ceiling.

## What I found while implementing it

**Two bugs, not one.** The obvious fix is "make `MIXED_BATCH_CHUNK_SIZES`
env-overridable." I did that, but tracing every consumer turned up a second,
independent bug the fix1 report's own claim papers over:

`mstar/worker/micro_scheduler.py` kept a **hardcoded duplicate**,
`_MIXED_MAX_CHUNK_TOKENS = 512` (old line 338), with a comment saying it "mirrors
`ThinkerSubmodule.MIXED_BATCH_*`... if those change, change these" — a manual-sync
constant, by design (the scheduler must not import the model submodule). This
constant gates G1 in `_chunk_entry_passes_gates` (`micro_scheduler.py:457`, was
`:408`) AND is read by `worker.py`'s `_compute_coadmit_budget`
(`worker.py:2116-2148`, was `:2116-2143`) to compute the COADMIT budget clamp.
fix1's report claims (its `worker.py:2127` docstring, verbatim): *"Derived from
the scheduler's capture-mirror constants so it tracks any P3 grid growth
automatically."* **That claim was false as written** — the constant was a plain
`int` literal; growing `ThinkerSubmodule.MIXED_BATCH_CHUNK_SIZES` at boot would
silently leave the scheduler's gate and the COADMIT clamp both pinned at 512,
so G1 would keep rejecting exactly the chunks the new capture grid was built to
admit. I verified this was live (not just theoretical) before fixing it: see
"Validation" below, `test_gate_rejects_1024_chunk_by_default` /
the pre-fix state where the grid grows but the gate constant doesn't move.

## Design

1. **`ThinkerSubmodule.MIXED_BATCH_CHUNK_SIZES`** (`submodules.py:1595-1637`)
   converted from a plain class-attribute list to a `@property` (same pattern
   already used for `PREFILL_VISION_TOKEN_BUCKETS`, `submodules.py:1619-1622`),
   reading `MSTAR_MIXED_CHUNK_SIZES` (comma-separated ints). Parsed values are
   **union'd with the default** `[256, 288, 512]`, never replacing it — an
   override can only *grow* the grid, so a caller who sets
   `MSTAR_MIXED_CHUNK_SIZES=1024` doesn't accidentally drop the 288 bucket (the
   measured fix for the 258-token vision-span tail-merge, `submodules.py:1596-1599`).
   Malformed input / all-non-positive input falls back to the untouched default.
   Unset/empty → byte-identical to before this change (verified: default resolves
   to exactly `[256, 288, 512]`).

2. **`get_cuda_graph_configs`** (`submodules.py:1898-1913`, unchanged logic) already
   iterates `self.MIXED_BATCH_CHUNK_SIZES` to build `mixed_packed`, one capture
   bucket per `c` at key `MIXED_BATCH_BS + c`. Because this is a property now
   instead of a literal, **no other line in that method needed to change** — new
   sizes just appear in the dict the existing capture loop walks.

3. **Bucket selection at replay** needed **zero changes**. `mstar/engine/cuda_graph_runner.py`'s
   `_get_padded_num_tokens` (`:908-919`) already does a generic
   `bisect.bisect_left` over `config.get_total_tokens(bs)` (which for
   `FlashInferPackedCudaGraphConfig` is just `list(packed_seq_len_to_inputs.keys())`,
   `cuda_graph_config.py:102-103`) and pads UP to the nearest captured bucket —
   this mechanism already exists precisely to let a bucket grid grow (see the
   `_env_buckets` docstring at `submodules.py:1539` et seq., written for the
   prefill-token grid, with the same "raising the ceiling is correctness-neutral,
   just more capture memory" reasoning I'm relying on here for the mixed grid).

4. **`MSTAR_PREFILL_CHUNK_TOKENS` interplay** — checked, no change needed. The
   planner's own bucket list, `qwen3_omni_model.py:_PREFILL_CHUNK_BUCKETS =
   [128, 256, 512, 1024, 2048]` (`:337`), **already** supports 1024/2048-token
   chunks; only `MSTAR_PREFILL_CHUNK_TOKENS`'s value (default 512, ship runs 2048,
   `qwen3_omni_model.py:317-331`) gates whether the planner ever produces one. So
   a chunk up to 2048 tokens already exists in the pipeline today for standalone
   (non-mixed) prefill; it just couldn't fold. This change doesn't touch the
   planner at all — it only widens what the mixed-fold gate (G1) and the mixed
   capture will *accept*. To exercise a 1024/2048-token fold, both flags must be
   set together: `MSTAR_PREFILL_CHUNK_TOKENS=1024` (or 2048) so the planner
   *produces* a big-enough chunk, and `MSTAR_MIXED_CHUNK_SIZES=...,1024,...` (or
   `...,2048`) so the mixed capture and G1 gate *accept* it. Neither alone is
   sufficient — this composition is the whole point of the fix.

5. **`MicroScheduler._max_chunk_tokens()`** (`micro_scheduler.py:377-388`, new
   `@classmethod`, replacing the old `_MIXED_MAX_CHUNK_TOKENS = 512` constant)
   + module-level `_resolve_mixed_chunk_sizes()` (`micro_scheduler.py:29-52`) —
   a **duplicate** parse of the identical env var (union-with-default, same
   semantics), kept scheduler-local for the same "don't import the model
   submodule" reason `qwen3_omni_model._PREFILL_CHUNK_BUCKETS` already documents
   for its own duplication of `ThinkerSubmodule.PREFILL_TOKEN_BUCKETS`. Callable
   as both `self._max_chunk_tokens()` (used at the G1 gate,
   `micro_scheduler.py:457`) and `MicroScheduler._max_chunk_tokens()` (used from
   `Worker.__init__`, `worker.py:2148`, before a scheduler instance exists —
   this is why it's a `classmethod`, not an instance property).

6. **`Worker._compute_coadmit_budget`** (`worker.py:2116-2148`) now calls
   `MicroScheduler._max_chunk_tokens()` instead of the old class attribute — the
   COADMIT clamp (`bs + max_captured_chunk`) now **actually** auto-widens with
   the grid, making fix1's docstring claim true instead of aspirational.

Boot-time-only, process-lifetime cache: `_resolve_mixed_chunk_sizes()` resolves
the env var once and caches the tuple at module scope (not per-instance, not a
dynflag) — matches every other boot-time flag in this file (e.g.
`_split_attn_env_snapshot` in `cuda_graph_runner.py:396-408`). No walk
registration changed; no change to any dynflag-refreshable behavior.

## Memory cost — the actual numbers, from reading the capture code

The naive worry ("N new buckets = N new full-size activation captures") is wrong
for two independent reasons visible in `mstar/engine/cuda_graph_runner.py`:

**1. Static packed-input tensors are interned at the MAX bucket size only.**
`_intern_static_buffer` (`cuda_graph_runner.py:434+`) allocates a shared buffer
per `(config_idx, tensor_key)` **on first encounter**, and `warmup_and_capture`
iterates `reversed(sizes) × reversed(sorted(...))` (`:257-259`) — **largest
bucket first**. Every smaller bucket then re-slices the same underlying storage
(`shared_static_buffers`, `:215`). So `input_embeds`/`cos_3d`/`sin_3d` (and
`deepstack_<i>` under vision-mix) for the `thinker_mixed` config are allocated
**once, at the size of the new largest bucket** — not once per bucket.

Exact math (`thinker_hidden_size=2048`, `head_dim=128`, bf16, from the real
Qwen3-Omni-30B-A3B-Instruct `config.json`):
per-token bytes = `input_embeds(2048*2) + cos_3d+sin_3d(128*2*2)` = 4608 B/token
(text mode); `+ 3× deepstack_i(2048*2)` = 12288 B/token more under
`MSTAR_MIXED_BATCH_VISION` (`deepstack_visual_indexes` default has 3 entries).

| grid max C | tokens (32+C) | text-mode buffer | vision-mode buffer |
|---|---|---|---|
| 512 (today) | 544 | 2.5 MB | 9.2 MB |
| 1024 | 1056 | 4.9 MB | 17.8 MB |
| 2048 | 2080 | 9.6 MB | 35.1 MB |

Growing to a 2048 top bucket costs **~7 MB (text) / ~26 MB (vision) more**
static-buffer memory, total, not per-added-bucket — negligible on a 141 G H200.

**2. FlashInfer workspace buffers are fixed-size and shared, not per-bucket.**
`_create_persistent_wrappers` (`:302-390`) keys its workspace via
`self.buffer_manager.get(ws_label)` where `ws_label = f"{label}_cugraph_slot{slot_idx}"`
— **no bs/num_tokens in the key**. `WorkspaceBufferManager.get`
(`cache_manager.py:56-61`) allocates a fixed `self.size`-byte buffer once per
label and caches it in a dict. Every bucket of every config (`thinker_decode`,
`prefill_text`, `prefill_vision`, `thinker_mixed`, all bs/token buckets) reuses
the SAME two physical workspace buffers ("main_cugraph_slot0/1"). Adding mixed
buckets adds **zero** new workspace-buffer memory. Each new bucket does still
allocate its own wrapper *object* + a `static_pos_ids` int64 buffer
(`num_tokens * 8` bytes — 16.6 KB at 2080 tokens) + small internal FlashInfer
plan-state scaling with `(bs, max_num_pages)`, not `num_tokens` — KB-scale, not
MB-scale, per added bucket.

**3. The captured graph's activation memory shares ONE pool across the whole
submodule.** `warmup_and_capture` creates exactly one
`torch.cuda.graphs.graph_pool_handle()` per `CudaGraphRunner`
(`:253`) and passes `pool=self.memory_pool` to **every** `torch.cuda.graph(...)`
capture for that submodule (`:658`) — `thinker_decode` + `prefill_text` +
`prefill_vision` + `thinker_mixed` all share it. `prefill_vision` **already**
captures up to **16384 tokens** at bs=1
(`_PREFILL_VISION_TOKEN_BUCKETS_BASE`, `submodules.py:1609`); `prefill_text`
already captures up to 2048 tokens (`PREFILL_TOKEN_BUCKETS`, `:1535`). A new
`thinker_mixed` bucket at C=2048 (2080 total tokens, bs=32) has materially less
token volume than the 16384-token vision capture already resident in the same
pool. Because CUDA-graph private pools reuse freed intermediate blocks across
captures sharing one `pool=` handle, the marginal activation memory from
growing the mixed grid to 1024/2048 should be **small — bounded by, not
additive to, the pool's existing high-water mark** — but this is inference from
the pool-sharing mechanism, not a measured byte count, and is the one part of
this estimate I could not verify without a GPU.

**Recommended empirical check** (next GPU session, not done here per the
no-GPU constraint on this task): `warmup_and_capture` already logs an exact,
instrumented number for free —
`"shared_static_buffers: %d entries, %.2f MB resident ... Total cuda alloc delta
during warmup: %.2f MB"` (`cuda_graph_runner.py:283-297`). Boot once with
`MSTAR_MIXED_CHUNK_SIZES=256,288,512,1024,2048` and diff that log line's "Total
cuda alloc delta" against a default boot — this gives the real number in one
line of log output, no separate instrumentation needed.

**KV cache dominance / OOM framing**: on a 141 G H200 the numbers above (single-
or double-digit MB) are noise next to KV cache, which is sized from remaining
free memory after all captures. The one way this idea *could* meaningfully eat
into KV budget is if the activation-memory-sharing assumption in point 3 is
wrong for this specific shape (bs=32 packed mixed step is a different attention
pattern than the bs=1 long-sequence vision capture) — worth confirming with the
log line above before shipping a large grid to a memory-constrained deployment.

## Boot-time cost (the more likely real cost, separate from GPU memory)

Every `FlashInferPackedCudaGraphConfig` here is captured with `compile=True`
(`submodules.py:1939` — same as the other prefill configs). Each NEW
`(bs=32, num_tokens)` shape is a distinct CUDA-graph capture and, if it hits an
uncached inductor shape, a distinct compile. Per the fix20 loop notes
(`LEARNINGS_FIX20.md` / custom-ops-crusade memory), a cold inductor cache under
load can take **>20 minutes** for one new compiled shape; a warm
`TORCHINDUCTOR_FX_GRAPH_CACHE` + `CACHE_DIR` cuts this to a normal capture pass.
Growing `MSTAR_MIXED_CHUNK_SIZES` by N new sizes means N more capture+compile
passes at boot. This is a boot-time tax to budget for on the FIRST boot with a
new grid (cache miss), not a steady-state cost (subsequent boots hit the warm
cache) — but it should be measured before treating this as "free."

## Risks

- **Boot time** on first boot with a new/larger grid (cold inductor cache per
  new shape) — see above. Mitigate: warm the fx-graph cache once, then reuse.
- **Activation-memory assumption unverified on GPU** (point 3 above) — the
  static-buffer and workspace math is exact (code-derived); the shared-pool
  activation-reuse claim is architecturally sound but not measured here.
- **Bucket alignment / waste**: a captured bucket only helps if a real chunk
  length lands near it. `_PREFILL_CHUNK_BUCKETS` produces {128,256,512,1024,2048};
  matching `MSTAR_MIXED_CHUNK_SIZES` to those same values (as the example env
  string does) keeps padding waste bounded the same way the existing 288 bucket
  bounds the 258-token tail-merge case. An arbitrary grid (e.g. `999`) still
  works correctness-wise (generic bisect pads up) but wastes more per fold.
  Structural ceilings the fix1 report identified as NOT moved by this change:
  G5 (vision boot-time capability, separate flag), G7 (one chunk row per
  step — still true at any C), G2/G3/G4 (occupancy floor / eager-probe —
  COADMIT's territory, unaffected by grid size).
- **Composing with COADMIT**: `MSTAR_COADMIT` (opt/fix20) must be ON for the
  wider clamp to matter in practice — the V2 budget alone was already shown
  non-binding (fix1's headline finding); this idea only removes the G1 wall,
  COADMIT is still what relaxes the timing gates (G2/G3/G4) that decide *when*
  a fold is attempted.

## A/B recipe (target: i2t B32 TTFT via real co-admission)

Requires `opt/fix20`'s `MSTAR_COADMIT` (this idea is orthogonal but composes —
grid growth is boot-time, COADMIT is a dynflag on top). Paired, load-gated
(<25), n>=96, same boot for all cells except the grid change (boot-time, needs
a reboot to flip).

1. Baseline: today's grid (unset `MSTAR_MIXED_CHUNK_SIZES`), `MSTAR_COADMIT=1`,
   `MSTAR_PREFILL_CHUNK_TOKENS=2048` (ship value) — this is exactly fix1's
   already-measured "wash" cell (G1 still walls at 512).
2. **Boot with** `MSTAR_MIXED_CHUNK_SIZES=256,288,512,1024,2048`,
   `MSTAR_PREFILL_CHUNK_TOKENS=2048`, `MSTAR_COADMIT=1` (dynflag, no reboot
   needed once booted with the bigger grid). Now a full 2048-token text chunk
   CAN fold in one step.
3. Same boot as (2), `MSTAR_COADMIT=0` — isolates the grid-growth effect from
   COADMIT's timing relaxation.

Metrics: i2t B32 TTFT p50/p99, req/s, plus the existing WALK_STATS
`_coadmit_probe`/`_coadmit_fold`/`_fold_ok`/`_fold_miss` counters from fix1 —
watch specifically whether folds now land at C=1024/2048 buckets (new counter
recommended: bucket histogram of folded C, if not already present) rather than
falling back to the 512 wall. Capture the `warmup_and_capture` memory-delta log
line from both boots per the empirical-check note above.

## Files touched

- `mstar/model/qwen3_omni/submodules.py` (+~39): `MIXED_BATCH_CHUNK_SIZES`
  literal → `@property` (`:1595-1637`).
- `mstar/worker/micro_scheduler.py` (+~55): module-level
  `_resolve_mixed_chunk_sizes()` + cache (`:16-52`), `_MIXED_MAX_CHUNK_TOKENS`
  constant → `_max_chunk_tokens()` classmethod (`:377-388`), G1 call site
  (`:457`), docstring (`:602`).
- `mstar/worker/worker.py` (+~30 net, mostly docstring): `_compute_coadmit_budget`
  now calls `MicroScheduler._max_chunk_tokens()` instead of the stale hardcoded
  attribute (`:2116-2148`), comment at `:2165`.
- `test/modular/test_mixed_chunk_grid.py` (new, 23 tests): grid parsing
  (submodule + scheduler agreement, grow-only union, malformed input), G1 gate
  before/after grid growth (including a regression test that would have caught
  the stale-clamp bug), COADMIT clamp auto-widening.

Not touched: `mstar/api_server/*`, walk registration, any capture/bucket
selection code in `cuda_graph_runner.py` (confirmed unnecessary — the generic
bisect-pad-up mechanism already handles new dict keys).

## Validation (no GPU, no server)

- `python -m compileall` on all four touched/read files → OK.
- `PYTHONPATH=<worktree> <mstar-new venv>/python -c "import mstar; ..."` → OK;
  confirmed default grid resolves to exactly `[256, 288, 512]` (byte-identical),
  `MIXED_BATCH_CHUNK_SIZES` is a `property` on the class, grown-grid env var
  produces the union, scheduler and submodule grids agree
  (`MicroScheduler._max_chunk_tokens() == max(submodule.MIXED_BATCH_CHUNK_SIZES)`)
  in a fresh process.
- Manually reproduced the pre-fix bug in a fresh interpreter (grid grown, gate
  constant still 512) before applying the `worker.py`/`micro_scheduler.py` fix,
  to confirm the bug was real and not a misreading of fix1's report.
- `test/modular/test_mixed_chunk_grid.py`: 23/23 pass — grid parsing (7 cases ×
  2 call sites), G1 gate rejects an out-of-grid chunk both before and after
  growth (still walls beyond the new ceiling), rejects unchunked prefills
  regardless of grid, COADMIT clamp default (544) and grown (2080).
- `test/modular/test_qwen3_omni_mixed_batch_vision.py`: 17/17 still pass
  unmodified (no regression to existing mixed-batch-vision gating), run
  together with the new file (40/40) to rule out module-cache leakage across
  test files.
- `test/modular/test_phase1.py`: skipped (GPU-gated), unaffected.

# Idea o1 — EAGER FOLD FALLBACK (`MSTAR_EAGER_FOLD`)

Worktree `/m-coriander/coriander/tim/wt-o1-eager-fold`, branch `idea/o1-eager-fold`,
commit `578505e8`. Default OFF, byte-identical off, per-call env read. ~+400 LoC
(incl. CPU tests). No GPU touched, no server started.

## Premise (from fix1_coadmit.md / LEARNINGS_FIX20 finding 1)

The captured mixed fold (MIXED_BATCH/SPEC/COADMIT) is hard-capped at chunk
`C <= 512` because its only mixed CUDA-graph buckets are `{256, 512}` — a larger
chunk would route to an uncaptured graph (the UNCAP-IMA hazard). On the ship
config (`MSTAR_PREFILL_CHUNK_TOKENS=2048`) an i2t text prefill chunk is up to 2048
tokens, so it **can never fold** and runs STANDALONE — freezing the concurrent
decodes for that step. vLLM avoids exactly this by folding a full prefill into the
decode step, which it can do only because **its prefill runs EAGER** (dynamic
per-step shape). This idea gives M* the same escape hatch.

## The mechanism, and why it needs almost no new execution code

The runner already has an eager path for uncaptured shapes, and — critically — it
already handles the `thinker_mixed` graph_walk eagerly. So the whole idea reduces
to: *get a large-chunk `thinker_mixed` batch ASSEMBLED and routed to the non-spec
path*; execution then falls to eager automatically.

Dispatch chain for a `thinker_mixed` batch with `C > 512`
(`mstar/engine/kv_cache_engine.py`):
- `execute_forward` → `_can_use_cuda_graph` (`:611`): `submodule.can_use_cuda_graphs`
  returns True (`thinker_mixed` is a registered walk) BUT `runner.can_run`
  (`cuda_graph_runner.py:836`) returns False — `_get_key_for` finds no token bucket
  ≥ `32+C` → **False → eager**.
- `submodule.can_batch` (`qwen3_omni/submodules.py:1532`) = `len(inputs) > 1` = True
  → `_execute_batched` (`:965`), NOT `_execute_sequential`.
- `_execute_batched` → `submodule.forward_batched(graph_walk="thinker_mixed")`
  (`submodules.py:1963`): `thinker_mixed` is in the `is_prefill` packed-output set
  (`:2026-2029`) → emits `__batched_logits__` gathered at `qo_indptr[1:]-1`.
- `preprocess` for `thinker_mixed` (`submodules.py:1277`) builds the **real** packed
  layout (concat embeds, `plan_attention(seq_lens=[1]*n+[C])`); bucket padding is a
  RUNNER-only concern, so eager has no 512 limit — FlashInfer plans varlen for the
  real `C`.
- `reserve_replay_slot` (`kv_cache_engine.py:1095`) → `reserve_slot` → key miss →
  returns None (eager fallback) — no crash. (Same as how a `C<=512` non-spec mixed
  batch already works today: reserve returns None, `run()` self-advances the slot
  for the captured replay; here `can_run` is False so it goes eager instead.)

## Design decisions (a–e)

**(a) Where the eager path forks.** It does not fork in new code — it reuses
`KVCacheEngine.execute_forward`'s existing captured-vs-eager auto-select. The only
new work is at the SCHEDULER/WORKER layer: allow a `C>512` chunk to be *assembled*
into a `thinker_mixed` batch on the NON-spec path (`get_next_batch` →
`_try_assemble_mixed`), where execution auto-selects eager. This is deliberately
**not** the captured spec-fold path (`pop_mixed_chunk_for_spec` → reserved slot →
replay), which is captured-only and would IMA on `C>512`. (This is the UNCAP-IMA
distinction the brief flagged: UNCAP routed a big chunk into an uncaptured CAPTURED
bucket; here we route to genuine EAGER execution, which is safe.)

**(b) Attention plan for the eager mixed step.** Reuses the existing packed-prefill
plan machinery: `preprocess(thinker_mixed)` calls
`cache_manager.plan_attention(seq_lens=[1]*n_decode + [C], is_causal=True)` and
`forward_batched` gathers per-request last-token logits from `qo_indptr`. Identical
assembly to a captured mixed step, minus the capture/replay — FlashInfer plans the
real varlen shape.

**(c) KV writes & position handling / byte-identity.** Each request's numerics are
the standard path. The chunk row's attention is per-request causal via `qo_indptr`
(block-diagonal), so it attends only to its own KV whether alone or packed with
decode rows; decode rows likewise. Logits gather (`qo_indptr[1:]-1`), KV flush
(`cache_manager.flush_to_store`), and MRoPE advance are the same calls a standalone
`prefill_text` makes. This is the same argument that makes the captured mixed batch
byte-identical; eager fold only removes the 512/graph-capture constraint. **Same
tokens, different schedule** (expected and fine).

**(d) Cost control.** The eager step is slower per-step than a replay, so:
- Gated to **brand-new requests** (`fwd_index==0`, checked inside
  `has_eager_fold_opportunity`) — the arrival case, where the alternative is a
  decode-freezing standalone prefill anyway.
- **Frequency cap** `MSTAR_EAGER_FOLD_MIN_GAP` (default 8): at most one eager fold
  per N spec-decision steps (worker-local `steps_since_eager_fold`). The gap is
  checked BEFORE the ready-scan, so the peek runs at most once per N steps.
- **Chunk bound** `MSTAR_EAGER_FOLD_MAX_CHUNK` (default 2048 = largest prefill
  bucket): a chunk above this stays on today's standalone path, so no single
  pathological eager step and the FlashInfer prefill workspace stays within the
  envelope a standalone prefill already uses.

**(e) Fallback / vision.** Restricted to **`prefill_text` chunks** via the existing
`_mixed_chunk_walks()` (which returns `{prefill_text}` normally). A `prefill_vision`
chunk needs per-layer deepstack, which `preprocess(thinker_mixed)` assembles ONLY
under the boot-time `MSTAR_MIXED_BATCH_VISION` flag (`submodules.py:1333`); with it
off, a vision chunk's deepstack would be silently dropped → wrong logits, so it is
excluded. If a process IS booted vision-capable, `_mixed_chunk_walks()` adds
`prefill_vision` and eager fold composes (eager removes the 512 window, so even a
large image could fold) — but that is a bonus gated on the boot flag, not the
default. For default i2t this means eager fold co-admits the **text** portion after
encode — the i2t TTFT tail — while the vision prefill still runs standalone.

## Exactly which step types can eager-fold

- YES: a brand-new (`fwd_index==0`) request's `prefill_text` chunk with
  `512 < C <= MSTAR_EAGER_FOLD_MAX_CHUNK`, when a `thinker_decode` chain is in
  flight on the same node, `repetition_penalty==1.0`, non-TP node, `MSTAR_MIXED_BATCH`
  on (the assembly substrate).
- NO (unchanged): `C<=512` (captured/spec/coadmit path), `C>` cap (standalone),
  unchunked prefills, `fwd_index>0`, vision chunks without the boot flag, TP nodes,
  rep-penalty != 1.0.

## Expected TTFT math (arrival → first token)

Ship stack, long-text i2t arrival, `C≈2048` text chunk, `n≈31` decodes in flight:
- **Before:** the 2048 chunk can't fold (>512) → runs standalone. That step freezes
  all 31 decodes (~one prefill-step of GPU, decodes get 0 tokens), then decodes
  resume next step. Arrival's first token appears after the standalone prefill step.
- **After:** one eager `thinker_mixed` step runs the 31 decodes + the 2048 chunk
  together. The 31 decodes advance one token DURING the prefill step (not frozen),
  and the arrival's first token comes out of that same step (`qo_indptr` last-token)
  — i.e. arrival first-token ≈ one step, vLLM's flat-TTFT shape, instead of
  standalone-prefill + a stalled decode window. Net: removes the decode freeze and
  collapses the arrival's first-token wait to ~one step. Magnitude is bounded by how
  often large text chunks arrive (frequency cap) — a real but bounded lever, same
  class as COADMIT but for the `>512` chunks COADMIT structurally cannot touch.

## Risks

- **Perf cliff if fired too often.** An eager mixed step is slower than a captured
  decode replay; back-to-back eager folds would erode decode throughput. Mitigated
  by the `MIN_GAP` frequency cap + `fwd_index==0` arrival gate. Tune `MIN_GAP` up if
  a decode-tok/s regression appears in the A/B.
- **IMA.** Structurally avoided: the large chunk is admitted ONLY on the non-spec
  eager path (`eager_ok=True` is passed ONLY by `_try_assemble_mixed`, never by
  `pop_mixed_chunk_for_spec`), and eager execution has no capture bucket to overflow.
  The one-shot `_eager_fold_armed` (read-and-cleared in `_try_assemble_mixed`)
  guarantees exactly one assembly relaxes the cap.
- **Workspace overflow on a huge chunk.** Bounded by `MSTAR_EAGER_FOLD_MAX_CHUNK`
  (default 2048), same class as the `MSTAR_UNCAP_PREFILL` bounded-workspace lesson.
- **Requires `MSTAR_MIXED_BATCH` at boot** (the assembly substrate). Documented
  dependency; `eager_fold_enabled()` ANDs `mixed_batch_enabled()`.

## A/B recipe (for the user's GPU session — I did NOT run it)

Paired, load-gated (loadavg < 25), n≥96 at i2t B32, dynflags toggle, same boot.
Boot with `MSTAR_MIXED_BATCH=1` (+ ship flags), `MSTAR_PREFILL_CHUNK_TOKENS=2048`
so text chunks actually exceed 512. Use prompts with a **long text** portion (the
lever is inert on short food101 labels, whose text chunk is <512 and already folds
via COADMIT/captured).
1. Baseline: `MSTAR_EAGER_FOLD=0`.
2. `MSTAR_EAGER_FOLD=1` (defaults: max_chunk 2048, min_gap 8).
3. Optional: sweep `MSTAR_EAGER_FOLD_MIN_GAP` (4/8/16) to trade TTFT vs decode tok/s.
Metrics: i2t B32 TTFT p50/p99, req/s, decode tok/s (watch for a tok/s regression =
too-frequent eager steps). WALK_STATS counter `_eager_fold` = eager folds fired;
compare against `_coadmit_fold`/`_fold_ok`. Identity check: B1 token-identity
flag-off vs flag-on (same tokens required).

## Validation done (no GPU)

- `python -m compileall` on all four edited files → OK.
- `PYTHONPATH=<worktree> <mstar-new venv>/python -c "import mstar; ..."` → imports
  from the worktree; `eager_fold_enabled` (ANDs mixed_batch), `eager_fold_max_chunk`
  (default 2048, env override), `has_eager_fold_opportunity` present, and
  `_chunk_entry_passes_gates(eager_ok=False)` default confirmed.
- `test_eager_fold_cpu.py` (committed): chunk-cap gate (1024 rejected captured /
  accepted eager; 4096 > cap rejected; rep-penalty rejected) and the arrival scan
  (flag-off False; large new text chunk True; C=512 / fwd_index>0 / engine-not-ready
  / wrong-walk all False). All PASS.

## Files touched (my scope)

- `mstar/model/qwen3_omni/qwen3_omni_model.py`: `eager_fold_enabled()`,
  `eager_fold_max_chunk()`.
- `mstar/worker/micro_scheduler.py`: `_eager_fold_armed` init; `eager_ok` on
  `_chunk_entry_passes_gates`; new `has_eager_fold_opportunity`; arm read-and-clear
  + large-chunk-preferring pick in `_try_assemble_mixed`.
- `mstar/worker/worker.py`: `steps_since_eager_fold` loop state + the eager-fold
  break-chain branch in the speculation decision.
- `test_eager_fold_cpu.py` (new).
Not touched: any engine/runner execution code (the eager path is reused as-is),
`api_server/`, capture/registration code.

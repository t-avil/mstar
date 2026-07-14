# s2: MSTAR_SIDECAR_I2T — extend the emit sidecar's walk-gate to i2t (board #12)

## Why i2t was excluded

`SIDECAR_WALKS = {"thinker_decode", "prefill_text", "thinker_mixed"}`
(`mstar/worker/emit_sidecar.py:91`) is the Stage-1 admission gate: a rid is
scoped into the sidecar only if *every* walk its worker graphs could ever run
on a given worker is in this set (`Worker._add_new_request`,
`mstar/worker/worker.py:838-849`, whole-rid decision, once, per worker —
design invariant, split-brain protection). `prefill_vision` (i2t) and
`prefill_audio` (s2t) are absent, so any rid whose graph touches them never
gets admitted on that worker, for its whole life — not just for the prefill
step, for its *decode* steps too, if prefill and decode share a worker.

I read the code (not just the SIDECAR_DESIGN.md prose) to find out why, since
the doc's stated reason — "E4b lesson: ungated fast paths taxed Talker steps
17%" (`docs/SIDECAR_DESIGN.md:219-222`) — is about Talker/Code2Wav's
**streaming audio-output** edges, not about vision or audio *input* prefill.
Confirmed via `qwen3_omni_model.py:869-903`: `prefill_vision`'s Thinker node
emits `outputs=[GraphEdge(EMIT_TO_CLIENT, "new_token", output_modality="text",
persist=True), StreamingGraphEdge(Talker, "thinker_states"),
StreamingGraphEdge(Talker, "thinker_mask")]` — **byte-identical in shape** to
`prefill_text`'s Thinker node (`qwen3_omni_model.py:810-834`). There is no
audio/streaming edge anywhere in the vision walk's client-bound emit. So the
exclusion isn't structural — it's that Stage 1's validated coverage
(`SIDECAR_DESIGN.md §8 recipe 5`: "s2t B8 + i2s B8 coverage guard... audio
paths must be exactly flat") only ever exercised text-only walks and left
vision/audio prefill for a later wave (`LEARNINGS_TTFT.md` item #12, "wave-2
remainder" in `LEARNINGS_FIX20.md`). The item-processing code
(`Worker._send_outputs_sidecar`, `SidecarState._handle_step`) is already
walk-agnostic — it only reads `outputs.emit_to_client` /
`outputs.new_token_outputs` / `outputs.to_workers`, never `graph_walk` itself
— and a prefill-shaped (non-inline, prem-less) row is *already* exercised by
the existing suite via `thinker_mixed`'s prefill-chunk rows
(`test_mixed_walk_rows_byte_identical`'s `rid-p`). Only the admission gate
needed widening; no new item-processing branch.

## Design

`SIDECAR_WALKS_I2T_EXTRA = frozenset({"prefill_vision", "prefill_multimodal",
"encode_vision"})` in `mstar/worker/emit_sidecar.py`. `prefill_multimodal` is
the merged text+vision walk (`MSTAR_MERGED_PREFILL`); `encode_vision` is the
standalone vision-encoder walk under `MSTAR_CHUNKED_PREFILL_V2_VISION` — it
has *no* `EMIT_TO_CLIENT` edges at all (persist-only, routed to
`EMPTY_DESTINATION`), so it can never produce a sidecar item, but it still
must be listed: the admission check is whole-rid, so a chunked-vision i2t
rid's `my_walks` includes `"encode_vision"` alongside `"prefill_vision"` and
both must clear the subset test. Kept as a *separate* frozenset, not folded
into `SIDECAR_WALKS`, so `MSTAR_SIDECAR_I2T=0` leaves `SIDECAR_WALKS` — and
every admission decision — untouched.

`worker.py` (`mstar/worker/worker.py`):
- Flag read once at init (~line 408-434), same pattern as `MSTAR_EMIT_SIDECAR`
  itself: `self._sidecar_i2t = os.environ.get("MSTAR_SIDECAR_I2T", "0") ==
  "1"`, refused (with a `logger.critical`) if `MSTAR_EMIT_SIDECAR` is off.
  `self._sidecar_walks = SIDECAR_WALKS | SIDECAR_WALKS_I2T_EXTRA if
  self._sidecar_i2t else SIDECAR_WALKS`. Not touched by
  `_refresh_dynamic_flags` (boot-time, same reason the sidecar itself is:
  it's a spawned process, not dynflag-able).
- `_add_new_request`'s admission check (~line 849) changed from `my_walks <=
  SIDECAR_WALKS` to `my_walks <= self._sidecar_walks` — the only behavioral
  line in the whole change.

Composes for free with `MSTAR_ORDERED_EMIT` / `MSTAR_EMIT_SEQNUMS` /
`MSTAR_EMIT_RID_INDEX`: none of them branch on walk name, only on
rid/name/step ordering, which is unchanged by widening admission.

## Files:lines

- `mstar/worker/emit_sidecar.py:91-134` — new `SIDECAR_WALKS_I2T_EXTRA` +
  design/risk comment.
- `mstar/worker/worker.py:53-58` (import), `:408-434` (flag setup),
  `:838-849` (admission check uses `self._sidecar_walks`).
- `test/modular/test_emit_sidecar_identity.py` — 4 new tests (below) + a
  2-line pre-existing-drift fix to `_StubWorker` (`_sched_pack` /
  `_inline_dual` attributes the stub never got after those flags landed;
  unrelated to this change, was blocking the whole file before I could even
  validate).

## Flag

`MSTAR_SIDECAR_I2T` (default `"0"`). Byte-identical off: `self._sidecar_walks
== SIDECAR_WALKS` when unset, so `_add_new_request`'s admission decision, and
therefore every downstream byte, is unchanged from before this flag existed.
Requires `MSTAR_EMIT_SIDECAR=1` (boot-time, spawned-process constraint,
enforced with a loud refusal, not a silent no-op).

## Risks

1. **Topology gotcha — the widening may not engage on the shipped PD-disagg
   configs as-is.** The admission check is per WORKER using the *full* set
   of walks that could ever land there (`all_worker_graph_ids_to_graph_walks`
   across the whole model, not just this request's actual modality — verified
   by reading `Conductor._assign_worker_graphs_to_workers`,
   `mstar/conductor/conductor.py:445-472`, which has no `TODO: merge
   identical worker graphs from different graph walks` fix yet,
   `mstar/model/base.py:256`, so every walk that touches a shared node stays
   a separate `WorkerGraph` but all land via the SAME `_group_id` -> same
   worker). Both `configs/qwen3omni_2gpu_pd.yaml` and
   `configs/qwen3omni_pd_disaggregated.yaml` put the prefill-Thinker
   node_group's `graph_walks:` as `[prefill_text, prefill_audio,
   prefill_vision, (prefill_multimodal)]` — text+audio+vision **bundled on
   one rank**. On that rank, `my_walks` always includes `prefill_audio`,
   which this flag deliberately does NOT add (audio is a different idea's
   territory, and I didn't want to touch a walk outside my assigned scope on
   a file other agents may also be editing) — so the PREFILL rank's subset
   check still fails there even with this flag on, for every modality. The
   flag *does* engage on: (a) the DECODE rank of those same configs
   (`my_walks == {thinker_decode}` already, clean subset, no dependency on
   what walk prefill used — decode-phase i2t was never actually excluded by
   `SIDECAR_WALKS` in PD-disagg, only prefill was), and (b) any topology that
   isolates `prefill_vision` on its own node_group (none shipped today).
   Actionable follow-up: either split the PD configs' prefill node_group by
   modality, or land a matching audio-safe widening alongside this one.
2. **Pre-existing (not introduced by this change) Stage-1 test-harness gap,
   found while validating**: the message that ships the first inline
   template immediately after an earlier non-inline row *for the same (rid,
   name)* is value-identical but not byte-identical between the legacy and
   sidecar paths — the interned name string ships in the earlier record's
   `new_names`, so the later record's template `GraphEdge` references a name
   reconstructed from a separate pickle round-trip and loses the cross-field
   object identity legacy gets for free in one process. Reproduced with
   *zero* i2t/vision code involved (plain `prefill_text`-shaped row -> decode
   — `test_prefill_shaped_then_inline_template_pickle_gap_is_preexisting`),
   so it predates this change; it just was never exercised because every
   original Stage-1 scenario either keeps a rid in one population from its
   first step, or never continues a non-inline row's rid into a later inline
   step. It affects the prefill-row -> first-decode-token transition for
   *every* modality already on the sidecar (t2t via `prefill_text`,
   `thinker_mixed`'s prefill chunks), not something new to i2t. Not fixed
   here (real fix means preserving name identity across two independent ZMQ
   hops, out of scope); pinned as a documented, value-checked exception so a
   real regression (wrong values) still fails loudly.
3. Ordinary Stage-1 risks apply unchanged (split-brain WGD accumulation if
   ownership were ever split — not touched here — and the sidecar's
   fail-fast/permanent-fallback policy on any processing exception).

## A/B recipe (target: i2t B32 ITL + tok/s)

Not run — no GPU access in this worktree (task explicitly said CPU-only,
don't touch GPUs/servers). For whoever runs it:

1. Confirm topology first per risk #1 — use a config where `prefill_vision`
   is NOT bundled with `prefill_audio` on the same node_group, or accept this
   only lands the DECODE-side effect on the shipped PD configs (still real:
   i2t B32's ITL/tok-s during decode currently pays legacy emit cost whenever
   `MSTAR_EMIT_SIDECAR` alone doesn't scope it — verify with a quick
   `_sidecar_rids` size check via logs/counters before assuming decode was
   already covered).
2. Boot pair: `MSTAR_EMIT_SIDECAR=1` alone vs `MSTAR_EMIT_SIDECAR=1
   MSTAR_SIDECAR_I2T=1`, same commit, same load-gated (<25) window, i2t B32
   food101, n>=96, paired A/B per `LEARNINGS_FIX20.md` protocol lessons
   (contention lottery — don't trust a single pair).
3. Token-identity smoke first (tok/req exact match vs the flag-off baseline)
   before any perf cell — same Stage-1 validation recipe as the original
   sidecar.
4. Metrics: ITL p50/p99, tok/s, TTFT (prefill-row emit is now sidecar-owned
   too if risk #1's topology caveat is addressed, so TTFT may move as well as
   steady-state ITL).

## Validation done (CPU-only, this worktree)

- `PYTHONPATH=<worktree> python -c "import mstar"` — OK.
- `python -m compileall` on the 3 changed files — OK.
- `pytest test/modular/test_emit_sidecar_identity.py` — 14/14 pass (10
  pre-existing + 4 new: walk-set composition, admission subset-check
  semantics including the i2s/s2t safety invariant, a prefill_vision-shaped
  row through the real item-processing code with its peer-worker Talker
  signal, and the pre-existing-gap pin).
- `pytest test/modular/test_sidecar_checkstop.py` — 11/11 pass, untouched.
- `test_worker_message_handling.py` — 2/2 pass, untouched.
- `test_fast_send_emit.py` / `test_worker_graphs_manager.py` fail identically
  on a clean `git stash` checkout (unrelated stub/signature drift, confirmed
  pre-existing, not touched by this change).

## Commit

Branch `idea/s2-sidecar-i2t`, worktree
`/m-coriander/coriander/tim/wt-s2-sidecar-i2t`.

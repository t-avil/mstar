# MSTAR_ENC_OVERLAP_V2 — encoder→Thinker handoff overlap (idea o3)

Worktree: `/m-coriander/coriander/tim/wt-o3-enc-overlap`, branch `idea/o3-enc-overlap`,
commit `f88238a9`. Base: `1c4c25c3` (fix20 wave3, ENCODER_ASYNC ported at `1e7f10da`).
Flag: **`MSTAR_ENC_OVERLAP_V2`** (default off, byte-identical off, per-call env read
for behavior; substrate read once at init).

## Dependency map — what serializes encoder(rank0) → Thinker decode(rank1) with ENCODER_ASYNC on

Topology `configs/qwen3omni_2gpu_encoff.yaml`: `vision_encoder`/`audio_encoder` on rank 0,
`Thinker` on rank 1 — different GPUs. An i2t `prefill_vision` walk spans partitions
(encoder partition → Thinker partition); the cross-partition handoff is conductor-driven.

Per-request path (file:line):

1. Conductor sends encoder inputs to rank 0 — `mstar/conductor/conductor.py:995` `_send_partition_inputs`.
2. Rank 0 schedules the encoder. With ENCODER_ASYNC the micro-scheduler prioritizes the
   encoder node — `mstar/worker/micro_scheduler.py:_maybe_pick_async_encoder` (added at `1e7f10da`,
   ~L147) — and runs it on a low-priority side stream — `mstar/worker/worker.py:2445-2482`
   (`use_side_stream` branch). **Note:** on i2t rank 0 is otherwise idle, so ENCODER_ASYNC's
   *side-stream overlap* is largely moot in encoff; the real win there is cross-GPU isolation
   from decode.
3. Rank 0 `_postprocess_batch` **blocks on `output.completion_event.synchronize()`** —
   `mstar/worker/worker.py:3426` — before routing the encoder embeds. (Host-sync; on rank 0
   it only delays the handoff, it does not block decode.)
4. Rank 0 routes embeds and sends `WORKER_GRAPHS_DONE` to the conductor.
5. **Conductor round-trip**: `run` loop `conductor.py:1090-1165`, per-hop wait
   `communicator.wait_for_work(timeout_ms=50)` at `conductor.py:1163`;
   `_process_worker_graphs_done` (`:839`) → `_process_done_forward` (`:918`) →
   `get_partition_forward_pass_args` → `_send_partition_inputs` (`:995`) sends the embeds to
   rank 1. (This is the documented CONDUCTOR_POLL hop; risky to touch — it wedged before, see
   `RESEARCH_MSTAR_TTFT_PATH.md` §1. Left alone.)
6. Rank 1 receives the tensor — `worker.py:4391` `_check_ready_tensors()` — and the Thinker
   `prefill_vision` node becomes **ready**. So `has_new_request_ready`
   (`micro_scheduler.py:1108`) flips True **exactly at encoder completion + embed arrival** —
   its `graph_walk.startswith("prefill")` + `fwd_index==0` + `engine.check_ready` filters only
   pass once the embeds are in. **This is the encoder-completion signal, already in the base.**
7. **The residual serialization (rank 1, the target):** once the prefill is ready, the decode
   loop admits it. Default path — `worker.py:4440` `must_yield_for_fairness = ... has_ready_excluding(...)`
   → `must_yield_away` (`:4463`) → the `if not must_yield_away` speculate gate (`:4656`) is
   skipped → yield-away branch (`:4714-4737`) runs the **prefill STANDALONE on the default
   stream**, freezing the in-flight decodes for that step. ADMIT_FASTPATH (`:4552`) forces the
   same standalone yield sooner. COADMIT (`:4575`+) folds instead, but is capped at the captured
   mixed bucket (chunk ≤ 512 tokens) — a full vision prefill (~1.3k+ tokens) exceeds it and falls
   back to standalone. MSTAR_SIDE_PREFILL *can* overlap a prefill on the side stream
   (`worker.py:4977` `_maybe_dispatch_side`), **but it is gated `if not speculation.is_yield_away`** —
   and the fairness yield makes `is_yield_away=True`, so a freshly-arrived prefill is pulled
   standalone before the side dispatch can ever take it.

**Conclusion:** with ENCODER_ASYNC on, the *encoder* stops contending, but the *Thinker prefill*
still freezes decode because the fairness machinery breaks the spec chain to run it standalone,
and it is too big to fold into the captured mixed bucket. The side-stream substrate that could
overlap it is defeated by the same fairness yield.

## What I cut

The fairness-driven spec-chain break on encoder completion. When a brand-new prefill is ready
(= encoder embeds arrived) **and** a side slot is free, V2 undoes the fairness yield, so:
- the decode chain keeps speculating on the default stream (no freeze), and
- the existing section-3b side dispatch (`worker.py:4977`) runs the just-arrived prefill on the
  side stream, overlapping it with decode.

Result: encode overlaps on rank 0 (ENCODER_ASYNC / cross-GPU) **and** prefill overlaps decode on
rank 1 (side stream) — the handoff no longer freezes decodes.

## Design (file:line, all in `mstar/worker/worker.py`)

- **Substrate activation** (init, ~L657): `self._enc_overlap_v2 = env==1`;
  `self._side_prefill = SIDE_PREFILL or _enc_overlap_v2`. V2 rides the entire SIDE_PREFILL
  substrate (side executor + side stream + `_reap_side_if_done` / `_drain_side` KV-ordering gate),
  so no new stream/thread/correctness machinery is introduced.
- **Peek + suppression** (decode loop, ~L4523-4581): `_enc_overlap_v2_on` requires the substrate
  live **and a free side slot** (`pending_side is None`). `_v2_peek` also fires when we are
  *already* yielding for fairness (COADMIT/ADMIT_FASTPATH only peek when not yielding). If a new
  prefill is ready, set `must_yield_for_fairness=False` and recompute `must_yield_away` from the
  ceiling only — never suppressing the consecutive-spec ceiling.
- Reuses `has_new_request_ready` (the encoder-completion trigger) and the ADMIT_FASTPATH/COADMIT
  hook site, as specified. No new walk types, no boot-time registration.

Diff: `mstar/worker/worker.py`, +76/−4.

## Safety / composition

- **Byte-identical off**: `MSTAR_ENC_OVERLAP_V2` unset → `_enc_overlap_v2=False`,
  `_side_prefill` governed only by `MSTAR_SIDE_PREFILL`, `_v2_peek=False`, suppression block never
  runs. Verified by construction + a truth-table CPU test.
- **Composes with ENCODER_ASYNC on/off**: ENCODER_ASYNC lives on rank 0 (micro-scheduler +
  encoder side stream); V2 lives in the rank-1 Thinker decode loop. Disjoint, both default off.
- **PD-disagg unaffected when off**: off = byte-identical, so `qwen3omni_2gpu_pd.yaml` is untouched.
- **Token-identical**: inherits SIDE_PREFILL's invariants — per-request KV + per-request seeded
  sampling (`conductor.py:985-986`), disjoint FlashInfer workspace for the eager side path
  (`worker.py:2742-2751`), and the chain-break drain gate (`worker.py:5000` `_drain_side`) that
  orders side-stream KV writes before any new default-stream work. Only prefill timing/stream
  changes, not any request's token sequence.
- **No starvation**: consecutive-spec ceiling is never suppressed (stays the backstop); the
  free-slot guard means at most one prefill overlaps at a time and any extra falls back to the
  normal standalone yield. Caveat: relies on the Thinker prefill being side-eligible (KV_CACHE +
  `prefill*` walk — true for i2t/s2t); a non-side-eligible ready node would be pushed back and wait
  for the ceiling. Documented in-code.

## Risks

- The conductor hop (step 5, `wait_for_work(timeout_ms=50)`) is **not** addressed — deliberately
  (it wedged historically; out of scope for a safe cut). If that poll eats timeouts, it caps the
  benefit; V2 only removes the rank-1 freeze once embeds arrive.
- Side-stream prefill contends for SMs with decode. The side stream is least-priority so decode
  should win arbitration, but under a fat vision prefill decode ITL may rise slightly during the
  overlap window (still far better than a full standalone freeze). Measure ITL-during-prefill-wave.
- Benefit is topology-dependent: largest in encoff/PD (encoder off rank 1). In the colocated
  single-GPU config the prefill and decode share one GPU, so the side overlap is SM-limited (same
  as plain SIDE_PREFILL); V2's chain-preservation still helps but less.

## A/B recipe (GPU — for the owner to run; I did not touch GPUs)

Config `qwen3omni_2gpu_encoff.yaml`, ENCODER_ASYNC on for all cells; i2t B32, load-gated <25,
n≥96, paired A/B (per LEARNINGS_TTFT protocol #15):

- **Baseline**: `MSTAR_ENCODER_ASYNC=1` (V2 off).
- **V2**: `MSTAR_ENCODER_ASYNC=1 MSTAR_ENC_OVERLAP_V2=1`.
- Optional 3rd cell `MSTAR_SIDE_PREFILL=1` (no V2) to isolate V2's fairness-suppression from the
  bare side substrate — expect it ~= baseline because the fairness yield defeats side dispatch.

Metrics: **i2t B32 TTFT** (p50/p99) and **ITL during the prefill wave** (the freeze this targets),
plus req/s and length-normalized tok/s. WALK_STATS counter `_enc_overlap_v2_defer` counts how many
fairness yields were converted to side overlaps (0 ⇒ trigger never fired ⇒ check the free-slot
guard / that prefills are actually arriving mid-decode).

## Expected win

Removes the per-admission standalone-prefill freeze on rank 1 (the residual TTFT/ITL spike after
ENCODER_ASYNC). Directionally: ITL-during-prefill-wave should flatten (decode chain never breaks
for admission) and B32 TTFT should drop by roughly the standalone-prefill step cost per admission.
Modest-to-moderate; bounded by the un-addressed conductor hop and by side-stream SM contention.
Not validated on GPU (no GPU access in this task).

# R1: MSTAR_DECODE_EXILE_POSTPROC — design (from forensic blueprint)
Branch opt/decode-cpu-floor, worktree mstar-decodefloor, all in mstar/worker/worker.py.

## Floor root cause
Decode step: await GPU(N) @5031, thread N->N+1 inputs @5080 (GPU->GPU, no D2H), submit GPU(N+1) @5156,
postprocess(N) @5201. postprocess ALREADY runs after submit(N+1) BUT on the same main GIL thread (1-worker
gpu_executor @4349) -> GPU(N+2) can't submit until _postprocess_batch(N) returns = the ~26ms floor.

## _postprocess_batch (body @3463) split
MUST-head (INLINE, feeds next batch composition; do NOT defer — eiv2/ASYNC_SCHED broke here):
  3469 cleanup? (defer), 3488-3510 overstay dedup (MUST), 3537 completion_event.synchronize (MUST),
  3557 check_stop D2H prematerialize (MUST), 3580 _compute_new_stops (MUST),
  3612-3644 stop_loops + _pending_loop_stops.update (MUST — read by next spec @3048)
DEFERRABLE-tail (submit to 1-worker postproc_executor):
  3673-3743 route/store (continuing rids threaded GPU->GPU @5080, so deferring store safe),
  3782 _register_outputs, 3833-3869 _send_outputs/_send_outputs_sidecar emit+ZMQ, 3469 cleanup, 3523 LRU

## Hook
- Add postproc_executor = ThreadPoolExecutor(max_workers=1) beside gpu/plan/side (@4349/4372/4399), gated.
- At 5201: if EXILE: run head inline, submit tail to postproc_executor (FIFO preserves ORDERED_EMIT).
- BARRIER: REMOVE (@4534, _apply_pending_removes_safe_to_drop @5208) must join prior tail future before
  freeing a rid's buffers (emit must not race a REMOVE). Tail is host-only (cpu_output pinned ints @3557,
  routing_per_request) - hand OWNED copies to tail; never touch a live GPU slot (NUM_SLOTS=2 double-buffers).
- Flag OFF: call _postprocess_batch unchanged = byte-identical.

## Reuse
gpu/plan/side executor pattern @4349/4372/4399; MSTAR_EMIT_SIDECAR (@419, _send_outputs_sidecar @1974) =
cleanest deferrable-tail precedent; SIDECAR_CHECKSTOP _d2h_stream/pinned @690; FAST_POSTPROC @231 composes.

## Parity invariant
check_stop DECISION stays same-step (never deferred). Only emit/route/cleanup tail moves off-thread.
Gate: determinism 32/32 (i2t B1 x2) + coherence BEFORE any perf claim.

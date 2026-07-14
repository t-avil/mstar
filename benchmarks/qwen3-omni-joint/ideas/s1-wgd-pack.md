# s1: MSTAR_WGD_PACK — batched control-plane pack (board #11)

## Idea

Board item #11: "Batched WGD/control-plane pack (msgspec, one send per step)
— conductor hops are pure host latency on every walk transition." The py-spy
decode profile shows zmq/control-plane at ~13% of the serve-process CPU
floor; each of those hops is a separate `send_pyobj`/`recv_pyobj` (pickle
serialize + ZMQ syscall + dict-dispatch on receipt), and today the code pays
one such hop **per rid, per direction**, not per step.

Two hot spots, both confirmed by reading the code (not assumed):

1. **Worker → conductor**: `Worker._send_outputs` (`mstar/worker/worker.py`)
   sends one `ConductorMessage(WORKER_GRAPHS_DONE)` per rid whose graph
   completed this step, from inside a `for rid, routing in
   routing_per_request.items(): self._send_outputs(...)` loop
   (`worker.py:3785` call site). At i2t B32 with 32 concurrently-completing
   requests, that's 32 separate `communicator.send("conductor", ...)` calls
   for what is logically one step's worth of completions.
2. **Conductor → worker**: `Conductor._send_partition_inputs` /
   `_send_producer_done` (`mstar/conductor/conductor.py:1016-1096`) each send
   one `WorkerMessage(INPUT_SIGNALS)` per worker per partition. The main
   loop's `run()` can process several requests' `WORKER_GRAPHS_DONE` in one
   iteration (`done_partition_forwards`, `conductor.py:1119`), each of which
   can call `_send_partition_inputs` and fan out to the same worker — so one
   iteration can re-send to the same worker N times.

Both are exactly "one conductor hop per walk transition" host latency named
in `LEARNINGS_TTFT.md` fix #11 and `RESEARCH_MSTAR_TTFT_PATH.md`'s finding
that every prefill_vision walk pays its own hop (32 serialized bs=1 walks at
B32).

## Codec check

`pyproject.toml` has no `msgspec` or `orjson` dependency, and neither is
installed in `mstar-new/.venv`. The message bodies
(`WorkerGraphsDone`, `InputSignals`, etc., in `mstar/utils/ipc_format.py`)
carry arbitrary nested objects — `GraphEdge`, `TensorPointerInfo`,
`NestedLoopIndices`, `GraphTimings`, sampling configs — none of which are
`msgspec.Struct`/JSON-safe today. Wiring msgspec in would mean either (a)
converting every `MessageBody` dataclass to a `msgspec.Struct` plus custom
encoders/decoders for the non-primitive fields, or (b) a hybrid
struct-envelope-around-pickled-body scheme. Either is a much larger, riskier
project than 100-500 LoC and outside this idea's scope. **This
implementation keeps the existing codec (ZMQ's `send_pyobj`/`recv_pyobj`,
i.e. pickle)** and attacks the other half of the win: collapsing **N
messages into 1 wire frame** per step per peer. The profiled cost (13% CPU)
is dominated by per-message overhead (syscall + pickle framing + Python
dispatch), not by pickle's per-byte throughput on these payloads, so frame
count is the lever that matches the profile. Swapping codecs is a clean
follow-up once frame count is fixed, noted below.

## Design

New wire message types in `mstar/utils/ipc_format.py`:
- `ConductorMessageType.PACKED` / `PackedConductorMessage(messages:
  list[ConductorMessage])` — worker → conductor direction.
- `WorkerMessageType.PACKED` / `PackedWorkerMessage(messages:
  list[WorkerMessage])` — conductor → worker direction.

**Worker send side** (`mstar/worker/worker.py`):
- `self._wgd_pack = os.environ.get("MSTAR_WGD_PACK", "0") == "1"`, cached in
  `__init__` and refreshed in `_refresh_dynamic_flags` (dynflag-toggleable).
- `_send_outputs` gained an optional `wgd_pack_buffer: list[ConductorMessage]
  | None` parameter; when supplied, the `WORKER_GRAPHS_DONE` message is
  appended to it instead of sent immediately (mirrors the existing
  `batch_collector` param for `MSTAR_BATCH_EMIT`).
- The step's call site (around `worker.py:3751`, the `for rid, routing in
  routing_per_request.items(): self._send_outputs(...)` loop) creates
  `wgd_pack_buffer = [] if self._wgd_pack else None` **once per step** —
  read once, at a natural iteration boundary, per the task's safety
  requirement — and flushes it right after the loop: a 1-message buffer is
  sent unpacked (no framing overhead for the common low-concurrency case), a
  ≥2-message buffer is sent as one `ConductorMessage(PACKED,
  PackedConductorMessage(messages=buffer))`.
- Receive side: `_process_message_list` gained a `PACKED` branch that calls
  `self._process_message_list(message.body.messages)` — i.e. it unpacks and
  redispatches through the exact same per-message handler, in order,
  including the existing out-of-order-request buffering
  (`_unprocessed_messages`) for any inner message whose rid isn't live yet.

**Conductor send side** (`mstar/conductor/conductor.py`):
- Same `self._wgd_pack` flag, read once in `__init__` (conductor has no
  dynflags-refresh mechanism today, unlike the worker).
- `self._pack_buffer_out: dict[str, list] | None`, set at the top of every
  `run()` main-loop iteration (`{}` if the flag is on, `None` otherwise) —
  again, read once per iteration boundary.
- New `_send_to_worker(worker_id, message)` helper, used by both
  `_send_partition_inputs` and `_send_producer_done`: buffers into
  `self._pack_buffer_out[worker_id]` when a buffer is open, else sends
  immediately (identical to before).
- New `_flush_pack_buffer(pack_buffer)`: one send per destination worker,
  same 1-message-unpacked / ≥2-message-PACKED rule as the worker side.
- The flush runs in a `finally` around the iteration's core work (message
  dispatch + `_process_done_forward` + `_process_request_done`), so a
  mid-iteration exception still flushes whatever was buffered up to that
  point — matching the legacy immediate-send path's failure behavior (drop
  only what wasn't reached yet, not what already "would have been sent").
- Receive side: the main loop's per-message `if/elif` chain was extracted
  into `_dispatch_message(message, done_partition_forwards)` (pure
  refactor, byte-identical control flow — `continue`→`return` is equivalent
  since it's now one call per message) with a new `PACKED` branch that
  recurses over `message.body.messages` in order.

**Wire-safety property** (the task's stated constraint): unpacking is
**unconditional** on both receivers — `_dispatch_message` and
`_process_message_list`'s `PACKED` branches never consult the local
`_wgd_pack`/`self._wgd_pack` flag. Only the *send* side gates on the flag.
Since a `PACKED` envelope is only ever emitted when the sender's flag is on,
and any receiver running this code (on or off) unconditionally knows how to
unpack one, a dynflags refresh lag between conductor and worker processes
(worker refreshes ~50 iters, conductor doesn't refresh at all today) can
never desync the protocol — the receiver doesn't care what its own flag
says.

## Files:lines

- `mstar/utils/ipc_format.py`: `WorkerMessageType.PACKED`,
  `PackedWorkerMessage`, `ConductorMessageType.PACKED`,
  `PackedConductorMessage`.
- `mstar/worker/worker.py`: flag in `__init__` (~L200) and
  `_refresh_dynamic_flags` (~L2241); `_send_outputs` signature +
  WGD-send block (~L1576, ~L1901); call site buffer create/flush
  (~L3748-3800); `_process_message_list` PACKED branch (~L1080).
- `mstar/conductor/conductor.py`: flag + `_pack_buffer_out` in `__init__`
  (~L232); `_send_to_worker` helper (~L1050); `_send_producer_done` call
  site (~L1096); `_dispatch_message` + `_flush_pack_buffer` (~L1121-1181);
  `run()` main loop restructure (~L1193-1236).

## Flag

`MSTAR_WGD_PACK` (default `"0"`, both processes must be started/refreshed
consistently for the win — but per above, an inconsistent state is
wire-safe, just leaves the win partially unrealized). Dynflag-toggleable on
the worker; boot-time-effective-immediately on the conductor (no dynflags
refresh loop there — would need one added to make it live-toggleable
mid-run, not attempted here since the conductor process has no existing
dynflags-refresh precedent to extend).

## Target metric

Per-step/per-iteration host latency on the control-plane hop → expect ITL
improvement at high concurrency (B16/B32) where many rids complete in the
same step/iteration, and TTFT improvement on i2t specifically (the "32
serialized bs=1 `prefill_vision` walks" amplifier in
`RESEARCH_MSTAR_TTFT_PATH.md` means B32 pays up to 32 WGD hops today; this
collapses the common case to 1). tok/s should move less than ITL/TTFT since
it's a host-latency cut, not a GPU-compute cut — expect it to show up mainly
under CPU-bound/high-concurrency conditions, in line with the
`project_mstar_decode_bottleneck` finding that B32 is CPU-floor + prefill-
serialization bound, not forward-pass bound.

## Risks

- **No win at low concurrency**: B1/B2 rarely has >1 rid completing per step
  and the buffer is usually length-1 (sent unpacked) — expect wash there,
  by design.
- **Conductor-side win is weaker than worker-side**: `_send_partition_inputs`
  fans out to *different* workers per partition more often than to the
  *same* worker repeatedly within one iteration (topology-dependent); the
  worker→conductor direction is the more reliably-hot path (every rid's WGD
  always targets the single "conductor" peer).
- **Sidecar path not covered**: `MSTAR_EMIT_SIDECAR`-scoped rids send their
  WGD via `emit_sidecar.py`'s own `_send_wgd` (a separate spawned process
  with its own strict ordering contract documented in `SIDECAR_DESIGN §6.3`)
  — deliberately left untouched to keep this change's blast radius
  contained; the identical buffer-and-flush-per-step technique applies
  there directly as a follow-up.
- **Pickle, not msgspec**: see codec check above — the real throughput
  ceiling of pickle-based batching is unmeasured; if per-step batching alone
  doesn't move ITL enough, the next lever is a msgspec/orjson envelope
  around a *subset* of hot fields (e.g. just the token ints + rid, skipping
  the rarely-nonempty `persist_signals`/`graph_timings`/`rx_info`/`tx_info`),
  not a full dataclass migration.
- **Larger single pickle blob**: packing trades "N small sends" for "1
  larger send" — for very large N (e.g. B32 all completing in one step),
  the packed pickle could momentarily hold more Python objects alive at
  once than the immediate-send path (which frees each message right after
  `send_pyobj`). Unmeasured; worth watching peak RSS under the B32 A/B.

## A/B recipe

1. Boot two servers, flags off vs `MSTAR_WGD_PACK=1`, same commit, same
   config, same GPU indices (per this workspace's convention: dedicated
   `CUDA_VISIBLE_DEVICES`, load-gated <25, paired cells, n>=96 for B32 per
   `LEARNINGS_TTFT.md` protocol lesson #15 — single loaded cells produced
   every false alarm this week).
2. i2t B32 food101 closed-loop, n>=96: compare TTFT p50, tok/s
   length-normalized, req/s. Expect the clearest signal here (32-walk
   amplifier).
3. s2t B32 n>=256 (wave-lottery protection per memory
   `project_mstar_dp_replicas`/`LEARNINGS_FIX20.md`): compare ITL, tok/s.
4. Identity check: B1x8 sequential, flags off vs on, byte-identical token
   output (packing must never change sampled tokens — pure transport
   change).
5. CPU-only correctness (already run this session, no GPU needed): see
   commit for the fake-socket test exercising `Conductor._dispatch_message`,
   `Conductor._send_to_worker`/`_flush_pack_buffer`, and
   `Worker._process_message_list` over a real ZMQ IPC round trip — in-order
   delivery, mixed pack/unpack (1-message buffers stay unwrapped), abort
   mid-batch (`ABORT_REQUEST` interleaved with `WORKER_GRAPHS_DONE` inside
   one `PACKED` envelope), out-of-order-request stashing preserved for inner
   messages, and receiver unpacking `PACKED` independent of its own flag
   state. All 14 checks passed; the test script was scratch (not committed).

## Validation performed (this session, no GPU)

- `python -m py_compile` + `compileall` on all three touched files: clean.
- `PYTHONPATH=<worktree> .venv/bin/python -c "import mstar"` (torch via
  `mstar-new/.venv`): clean, both new enums/dataclasses importable.
- CPU-only functional test (described above): 14/14 checks pass, exercising
  the actual shipped methods (not reimplementations) via lightweight fake
  `self` stand-ins bound with `Method.__get__`.

"""MSTAR_EMIT_SIDECAR — Stage 1 of docs/SIDECAR_DESIGN.md (design @ f0fd9e4).

One pure-CPU sidecar PROCESS per worker owns the client-bound emit path for
"sidecar-scoped" requests: emit message construction (full/slim items and the
slim-template protocol state ``_slim_emit_sent`` / ``_slim_emit_loop_layout``),
the api_server transport, the three WGD-feeding accumulators
(``pending_new_tokens`` / ``current_output_chunks`` / ``output_loop_indices``),
and WORKER_GRAPHS_DONE assembly + conductor transport. The worker main thread
feeds it one compact ``StepRecord`` per step: tuples of ints and interned
indices; a GraphEdge rides a record only at boundary rate (first inline
template per (rid, name), or a non-inline edge whose fresh tensor_info the
consumer must fetch via SHM).

Ownership contract (design §4.3, all-or-nothing per structure):

- The worker NEVER writes the three accumulators for a scoped rid — every
  write becomes a record field, on the steady, boundary, and non-inline
  paths alike (see ``Worker._send_outputs_sidecar``).
- Scoping is decided ONCE per rid at admission, from the full
  ``worker_graph_to_workers`` map (which covers every partition), so a later
  partition's add can never flip a rid between owners mid-flight — the
  split-brain trap called out in design §0.
- Tensor lifecycle (register_for_send / SHM-skip / producer-ref release) and
  route/store/check_stop stay on the worker (§3.2/§4.3); each record item's
  inline flag is derived from the worker's SHM-skip decision.

One-way data flow (§6.1): the sidecar produces nothing the worker, scheduler,
or tensor_manager ever reads. Transport (§4.1/§6.3): a single ZMQ PUSH/PULL
pair per worker — FIFO gives template-before-slim and token order per rid for
free. Never add a second worker→sidecar socket.

Ordering audit (§6.3 / open question 1): besides WORKER_GRAPHS_DONE, the
worker sends the conductor only SETUP_DONE (startup, before any request
exists) and — on sidecar failure — ABORT_REQUEST for stranded rids. Nothing
else implies partition completion, so moving WGD into the sidecar cannot
reorder it against other completion-implying worker→conductor traffic.

Flags: MSTAR_EMIT_SIDECAR (default "0") is read ONCE at worker init. The
sidecar is a spawned process, so this flag cannot follow MSTAR_DYNFLAGS flips
(process spawn requires a static flag; A/B via two-server alternation, not
dyn_ab). Scoped-rid construction is pinned to the winning emit stack
(MSTAR_BATCH_EMIT + MSTAR_SLIM_EMIT + MSTAR_SLIM_EMIT2 semantics); the worker
refuses to enable the sidecar unless those flags are on, so the flag-on
stream is byte-identical to the legacy stack's and the flag-off code paths
stay untouched. Because the scoped construction is static, the design's
dynflags flip rule (§6.3, "a flip forces full-template re-emission") has no
trigger in Stage 1: neither MSTAR_EMIT_SIDECAR nor the scoped slim semantics
can flip at runtime.

Failure policy (§7): the worker never blocks on the sidecar. The PUSH socket
has a bounded SNDHWM and sends NOBLOCK; an HWM trip or a dead process is a
permanent drain-and-disable — legacy path for new work, ABORT_REQUEST for
rids whose emit/WGD state is stranded in the sidecar. Per-step fallback is
forbidden: interleaving two producers would break the per-(rid, name) FIFO
the slim-template protocol needs.
"""

import logging
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass, field

import zmq

from mstar.api_server.request_types import (
    APIServerMessage,
    ResultTensors,
    ResultTensorsBatch,
    SlimResultTokens,
)
from mstar.communication.communicator import (
    CommProtocol,
    ZMQCommunicator,
    resolve_endpoint,
)
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    WorkerGraphsDone,
)

logger = logging.getLogger(__name__)

# Walk gate (design §4.1, the E4b lesson): only the text paths are ever
# sidecar-scoped. Talker/Code2Wav emit rides streaming edges, not this path,
# and must stay exactly flat. A rid is scoped only if EVERY walk its worker
# graphs can run on the owning worker is in this set.
SIDECAR_WALKS = frozenset({"thinker_decode", "prefill_text", "thinker_mixed"})

# Bounded send queue (design §7): ~512 steps ≈ 4.5 s of buffer at the 8.8 ms
# B32 step. The sidecar has 4-7x headroom per step, so the queue only grows
# when the sidecar has degraded — treat a trip as failure, not backpressure.
SIDECAR_SNDHWM = 512

# StepRecord item flags (bit field, plain int per design §4.2).
ITEM_INLINE = 1  # inline-qualifying: values carried, no SHM fetch downstream


@dataclass
class RidRegister:
    """Admission record: binds a rid string to its session-interned index
    (design §4.2 "rid-registration record on admission"). Sent once per
    scoped rid, before any step record can reference the index (FIFO)."""
    rid_idx: int
    rid: str


@dataclass
class RidRemove:
    """Teardown record: drop every per-rid structure the sidecar owns
    (accumulators, slim protocol state, cached edge templates)."""
    rid_idx: int


@dataclass
class StepRecord:
    """One postprocess step's client-bound work for every scoped rid.

    ``entries`` holds one tuple per rid, in batch iteration order (the order
    legacy ``_send_outputs`` ran in — per-destination message order depends
    on it):

        (rid_idx, new_tokens, items, boundary)

    - ``new_tokens``: ``[(name_idx, [int, ...]), ...] | None`` — the
      buffer_new_tokens write, verbatim (post name-dedup, insertion order).
    - ``items``: one tuple per emit_to_client edge, in edge order::

          (name_idx, flags, values, layout_idx, wg_fwd_pass_idx,
           loop_vals, edge)

      ``flags`` is an ITEM_* bit field; ``values`` is the prematerialized
      token list for inline items (the SAME list object as the new_tokens
      field where they overlap, so the record pickle memoizes it) or None;
      ``layout_idx``/``wg_fwd_pass_idx``/``loop_vals`` are the step's
      NestedLoopIndices decomposed to ints against an interned layout;
      ``edge`` is the GraphEdge when it must ride the record (see module
      docstring) or None on the steady path.
    - ``boundary``: None, or the worker-only WGD fields (design §4.2)::

          (completed_worker_graph_ids, is_first_tp_rank, persist_signals,
           per_label_seq_info, partition_name, partition_done,
           stream_tokens_consumed, graph_timings, rx_info, tx_info)

    ``new_names`` / ``new_layouts`` carry lazily-interned string tables:
    ``[(idx, name)]`` and ``[(idx, (loop_name_order, loop_key_names))]``.
    FIFO guarantees an index is defined before any record uses it.
    """
    step_id: int
    new_names: list = field(default_factory=list)
    new_layouts: list = field(default_factory=list)
    entries: list = field(default_factory=list)


class SidecarRecordBuilder:
    """Worker-side record assembly: interning tables + step assembly.

    Pure bookkeeping (no socket, no process) so the byte-identity harness
    can drive the exact production record path in-process;
    ``SidecarClient`` layers the spawned process and the bounded PUSH
    socket on top.
    """

    def __init__(self):
        self._rid_to_idx: dict[str, int] = {}
        self._next_rid_idx = 0
        self._name_to_idx: dict[str, int] = {}
        self._pending_names: list[tuple[int, str]] = []
        self._layout_to_idx: dict[tuple, int] = {}
        self._pending_layouts: list[tuple[int, tuple]] = []
        # (rid_idx, name_idx) pairs whose INLINE template edge has shipped.
        # Transport dedup only — the slim protocol state lives in the
        # sidecar. Keyed on inline ships alone so a preceding non-inline
        # step can never leave the sidecar holding a stale-tensor_info edge
        # for the template it must emit byte-identically.
        self._edge_shipped: set[tuple[int, int]] = set()
        self._step_id = 0

    def register_rid(self, rid: str) -> RidRegister:
        idx = self._next_rid_idx
        self._next_rid_idx += 1  # never reused: no ABA across remove/re-add
        self._rid_to_idx[rid] = idx
        return RidRegister(rid_idx=idx, rid=rid)

    def remove_rid(self, rid: str) -> RidRemove:
        idx = self._rid_to_idx.pop(rid)
        if self._edge_shipped:
            self._edge_shipped = {
                k for k in self._edge_shipped if k[0] != idx
            }
        return RidRemove(rid_idx=idx)

    def rid_idx(self, rid: str) -> int:
        return self._rid_to_idx[rid]

    def name_idx(self, name: str) -> int:
        idx = self._name_to_idx.get(name)
        if idx is None:
            idx = self._name_to_idx[name] = len(self._name_to_idx)
            self._pending_names.append((idx, name))
        return idx

    def layout_idx(self, nli: NestedLoopIndices) -> int:
        """Intern the step's loop layout (loop_name_order content + loop
        key ORDER — the exact pair MSTAR_SLIM_EMIT2 compares). Equal layouts
        get equal indices, so the sidecar's loop_key-validity check reduces
        to an int compare against the template's layout_idx."""
        key = (
            tuple(nli.loop_name_order),
            tuple(nli.loop_indices.keys()),
        )
        idx = self._layout_to_idx.get(key)
        if idx is None:
            idx = self._layout_to_idx[key] = len(self._layout_to_idx)
            self._pending_layouts.append((idx, key))
        return idx

    def ship_edge(self, rid_idx: int, name_idx: int, edge_inline: bool) -> bool:
        """Whether this item must carry its GraphEdge. Non-inline items
        always do (their fresh tensor_info is consumed via SHM every time);
        inline items ship it exactly once — the template occurrence."""
        if not edge_inline:
            return True
        key = (rid_idx, name_idx)
        if key in self._edge_shipped:
            return False
        self._edge_shipped.add(key)
        return True

    def build_step(self, entries: list) -> StepRecord:
        rec = StepRecord(
            step_id=self._step_id,
            new_names=self._pending_names,
            new_layouts=self._pending_layouts,
            entries=entries,
        )
        self._step_id += 1
        self._pending_names = []
        self._pending_layouts = []
        return rec


class SidecarClient(SidecarRecordBuilder):
    """Worker-side handle: spawns the sidecar process and owns the bounded,
    never-blocking PUSH socket. All failure modes collapse into
    ``healthy() == False`` / ``send() == False``; the worker reacts with the
    permanent fallback in ``Worker._disable_sidecar`` (design §7)."""

    def __init__(
        self,
        worker_id: str,
        socket_path_prefix: str = "/tmp/mstar",
        log_level: str = "INFO",
    ):
        super().__init__()
        self.worker_id = worker_id
        self.sidecar_id = f"{worker_id}_sidecar"
        self.failed = False
        self.hwm_trips = 0
        self.records_sent = 0

        # Spawn, not fork: the worker holds a CUDA context (design §7).
        # daemon=True is the backstop against orphaned sidecars; graceful
        # shutdown (SIGTERM → 5 s drain) goes through shutdown().
        ctx = mp.get_context("spawn")
        self.proc = ctx.Process(
            target=run_sidecar,
            kwargs=dict(
                worker_id=worker_id,
                sidecar_id=self.sidecar_id,
                socket_path_prefix=socket_path_prefix,
                log_level=log_level,
            ),
            daemon=True,
            name=self.sidecar_id,
        )
        self.proc.start()
        logger.info(
            "Worker %s: emit sidecar spawned (pid=%d)",
            worker_id, self.proc.pid,
        )

        # Single PUSH/PULL pair (design §4.1/§6.3) with a bounded send
        # queue; sends are NOBLOCK so the worker can never stall on the
        # sidecar. The endpoint scheme matches ZMQCommunicator's, so the
        # sidecar's plain communicator PULL binds the other end.
        transport = os.getenv("MSTAR_ZMQ_TRANSPORT", CommProtocol.IPC.value)
        self._socket = zmq.Context.instance().socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, SIDECAR_SNDHWM)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(resolve_endpoint(
            self.sidecar_id, CommProtocol(transport.upper()),
            socket_path_prefix,
        ))

    def send(self, rec) -> bool:
        """NOBLOCK send of one record. False = the sidecar is (now) failed;
        the caller must go through the permanent-fallback path. An HWM trip
        is failure by policy, not backpressure — mixing per-step fallback
        with sidecar sends would split the per-(rid, name) FIFO."""
        if self.failed:
            return False
        try:
            self._socket.send_pyobj(rec, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.hwm_trips += 1
            self.failed = True
            logger.critical(
                "Worker %s: emit sidecar send queue hit HWM (%d records "
                "buffered) — marking sidecar failed",
                self.worker_id, SIDECAR_SNDHWM,
            )
            return False
        except zmq.ZMQError:
            self.failed = True
            logger.critical(
                "Worker %s: emit sidecar socket error — marking sidecar "
                "failed", self.worker_id, exc_info=True,
            )
            return False
        self.records_sent += 1
        return True

    def healthy(self) -> bool:
        return not self.failed and self.proc.is_alive()

    def shutdown(self, timeout: float = 5.0) -> None:
        """SIGTERM the sidecar (it drains its queue with its own 5 s
        deadline) and wait at most ``timeout`` (design §7)."""
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=timeout)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=1.0)


class SidecarState:
    """The sidecar's pure logic: record stream in, api_server/conductor
    messages out. Separated from the process loop so the byte-identity
    harness can drive it in-process against a recording communicator.

    Construction mirrors ``Worker._send_outputs``'s emit block with the
    winning-stack flags pinned on (batch + slim + slim2) — the worker
    enforces that pairing at init, so the byte streams line up with the
    legacy baseline the benchmarks actually run.
    """

    def __init__(self, communicator, worker_id: str = "worker"):
        self.communicator = communicator
        self.worker_id = worker_id

        # Interning tables mirrored from the record stream.
        self.rids: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.layouts: dict[int, tuple] = {}

        # The three WGD-feeding accumulators — ownership moved WHOLESALE
        # from PerRequestInfo (design §4.3). Keyed by rid_idx.
        self.pending_new_tokens: dict[int, dict[str, list[int]]] = {}
        self.current_output_chunks: dict[int, list[str]] = {}
        self.output_loop_indices: dict[int, dict[str, NestedLoopIndices]] = {}

        # Slim-template protocol state — ownership moved from Worker
        # (_slim_emit_sent / _slim_emit_loop_layout). The layout compare
        # collapses to the template's layout_idx (equal content ⇔ equal
        # interned index).
        self.slim_sent: set[tuple[int, int]] = set()
        self.slim_layout: dict[tuple[int, int], int] = {}
        # Last shipped edge per (rid_idx, name_idx). Only consulted if a
        # template must be rebuilt without an edge on the record (cannot
        # happen on a healthy Stage-1 stream; belt for future flip paths).
        self.edge_templates: dict[tuple[int, int], object] = {}

        # MSTAR_EMIT_SEQNUMS: per-rid_idx emit sequence counter. For a
        # sidecar-scoped rid the sidecar process is the SOLE ordering authority
        # (all of that rid's client-bound emits ride the FIFO record stream and
        # are rebuilt here in build order), so numbering here matches the order
        # the consumer must deliver. Scoped rids are text-only (SIDECAR_WALKS is
        # all text; Talker/Code2Wav emit rides streaming edges, not this path),
        # so rid_idx alone identifies the (rid, "text") stream — the same
        # stream the consumer keys as (rid, "text"). Read once: the sidecar is a
        # spawned process and cannot follow MSTAR_DYNFLAGS (module docstring),
        # so this flag is effectively static per server boot. Cleared in
        # ``_remove``.
        self._emit_seqnums = os.environ.get("MSTAR_EMIT_SEQNUMS", "0") == "1"
        self._emit_seq: dict[int, int] = {}

        # Mechanism-alive counters (design §8 validation recipe 3). The
        # drain size per poll is the observable proxy for queue depth (ZMQ
        # doesn't expose it): >1 means the sidecar fell behind the producer
        # within a poll interval.
        self.steps = 0
        self.items_seen = 0
        self.wgd_sent = 0
        self.batch_msgs = 0
        self.drain_max = 0
        self.drain_sum = 0
        self.drain_polls = 0

    # ------------------------------------------------------------------
    # Record dispatch
    # ------------------------------------------------------------------

    def handle(self, rec) -> None:
        if type(rec) is StepRecord:
            self._handle_step(rec)
        elif type(rec) is RidRegister:
            self.rids[rec.rid_idx] = rec.rid
            self.pending_new_tokens[rec.rid_idx] = {}
            self.current_output_chunks[rec.rid_idx] = []
            self.output_loop_indices[rec.rid_idx] = {}
        elif type(rec) is RidRemove:
            self._remove(rec.rid_idx)
        else:
            raise TypeError(f"emit sidecar: unknown record type {type(rec)!r}")

    def _remove(self, rid_idx: int) -> None:
        self.rids.pop(rid_idx, None)
        self._emit_seq.pop(rid_idx, None)
        self.pending_new_tokens.pop(rid_idx, None)
        self.current_output_chunks.pop(rid_idx, None)
        self.output_loop_indices.pop(rid_idx, None)
        if self.slim_sent:
            self.slim_sent = {k for k in self.slim_sent if k[0] != rid_idx}
        if self.slim_layout:
            self.slim_layout = {
                k: v for k, v in self.slim_layout.items() if k[0] != rid_idx
            }
        if self.edge_templates:
            self.edge_templates = {
                k: v for k, v in self.edge_templates.items()
                if k[0] != rid_idx
            }

    def _rebuild_nli(
        self, layout_idx: int, wg_fwd: int, loop_vals: tuple
    ) -> NestedLoopIndices:
        """Reconstruct the step's NestedLoopIndices from its interned layout
        + ints. dict insertion order == the shipped key order, so the
        rebuilt object pickles byte-identically to the worker's original."""
        names, keys = self.layouts[layout_idx]
        return NestedLoopIndices(
            loop_name_order=list(names),
            loop_indices=dict(zip(keys, loop_vals, strict=True)),
            wg_fwd_pass_idx=wg_fwd,
        )

    def _handle_step(self, rec: StepRecord) -> None:
        for idx, name in rec.new_names:
            self.names[idx] = name
        for idx, layout in rec.new_layouts:
            self.layouts[idx] = layout

        collector: list = []
        for rid_idx, new_tokens, items, boundary in rec.entries:
            rid = self.rids[rid_idx]

            # buffer_new_tokens, verbatim (extend-in-order semantics).
            if new_tokens:
                pending = self.pending_new_tokens[rid_idx]
                for name_idx, toks in new_tokens:
                    name = self.names[name_idx]
                    if name not in pending:
                        pending[name] = []
                    pending[name].extend(toks)

            if items:
                chunks = self.current_output_chunks[rid_idx]
                out_loop = self.output_loop_indices[rid_idx]
                # Legacy shares ONE NestedLoopIndices object per (rid, step)
                # across that rid's items and output_loop_indices entries;
                # pickle memoization makes the sharing part of the message
                # BYTES. Rebuild once per distinct loop state per entry and
                # reuse the object so the bytes match.
                nli_memo: dict[tuple, NestedLoopIndices] = {}
                for (
                    name_idx, flags, values, layout_idx, wg_fwd, loop_vals,
                    edge,
                ) in items:
                    self.items_seen += 1
                    name = self.names[name_idx]
                    # MSTAR_EMIT_SEQNUMS: one seqnum per emitted edge, in the
                    # record's edge order (== the worker's build order), before
                    # the inline/non-inline split — mirrors the legacy
                    # _send_outputs stamping so both paths number identically.
                    emit_seq = None
                    if self._emit_seqnums:
                        emit_seq = self._emit_seq.get(rid_idx, 0)
                        self._emit_seq[rid_idx] = emit_seq + 1
                    # buffer_output_signals: one chunk name per emit edge,
                    # in edge order.
                    chunks.append(name)
                    mkey = (layout_idx, wg_fwd, loop_vals)
                    nli = nli_memo.get(mkey)
                    if nli is None:
                        nli = self._rebuild_nli(layout_idx, wg_fwd, loop_vals)
                        nli_memo[mkey] = nli
                    # register_output_loop_indices, verbatim.
                    out_loop[name] = nli

                    tkey = (rid_idx, name_idx)
                    if edge is not None:
                        self.edge_templates[tkey] = edge
                    if flags & ITEM_INLINE:
                        if tkey in self.slim_sent:
                            # Steady slim path (MSTAR_SLIM_EMIT/2): loop_key
                            # ints while the layout still matches the
                            # template step's, else the full object.
                            loop_key = None
                            if self.slim_layout.get(tkey) == layout_idx:
                                loop_key = (wg_fwd, *loop_vals)
                            collector.append(SlimResultTokens(
                                request_id=rid,
                                name=name,
                                values=values,
                                loop_indices=(
                                    None if loop_key is not None else nli
                                ),
                                loop_key=loop_key,
                                emit_seq=emit_seq,
                            ))
                        else:
                            # Template step: first full ResultTensors for
                            # this (rid, name) — the api_server caches it.
                            self.slim_sent.add(tkey)
                            self.slim_layout[tkey] = layout_idx
                            tmpl_edge = (
                                edge if edge is not None
                                else self.edge_templates.get(tkey)
                            )
                            collector.append(ResultTensors(
                                request_id=rid,
                                modality=tmpl_edge.output_modality,
                                graph_edge=tmpl_edge,
                                loop_indices=nli,
                                metadata={"inline_values": {name: values}},
                                emit_seq=emit_seq,
                            ))
                    else:
                        # Non-inline edge: full ResultTensors sent
                        # immediately, exactly as legacy's non-collector
                        # branch — the consumer fetches its tensors via SHM
                        # (the worker registered them for send).
                        self.communicator.send("api_server", APIServerMessage(
                            message_type="result_tensors",
                            body=ResultTensors(
                                request_id=rid,
                                modality=edge.output_modality,
                                graph_edge=edge,
                                loop_indices=nli,
                                metadata={},
                                emit_seq=emit_seq,
                            ),
                        ))

            if boundary is not None:
                # WGD rides the conductor stream at this rid's position in
                # the batch order, before the step's batch message — same
                # per-destination order as legacy.
                self._send_wgd(rid_idx, rid, boundary)

        if collector:
            self.communicator.send("api_server", APIServerMessage(
                message_type="result_tensors_batch",
                body=ResultTensorsBatch(items=collector),
            ))
            self.batch_msgs += 1

        self.steps += 1
        if self.steps % 2000 == 0:
            mean_drain = (
                self.drain_sum / self.drain_polls if self.drain_polls else 0.0
            )
            logger.info(
                "%s: steps=%d items=%d wgd=%d batch_msgs=%d rids_live=%d "
                "drain_max=%d drain_mean=%.2f",
                self.worker_id + "_sidecar", self.steps, self.items_seen,
                self.wgd_sent, self.batch_msgs, len(self.rids),
                self.drain_max, mean_drain,
            )

    def note_drain(self, n: int) -> None:
        """Record one poll's drained-record count (queue-depth proxy)."""
        if n:
            self.drain_max = max(self.drain_max, n)
            self.drain_sum += n
            self.drain_polls += 1

    def _send_wgd(self, rid_idx: int, rid: str, boundary: tuple) -> None:
        (
            worker_graph_ids, is_first_tp_rank, persist_signals,
            per_label_seq_info, partition_name, partition_done,
            stream_tokens_consumed, graph_timings, rx_info, tx_info,
        ) = boundary
        # Flush semantics identical to WorkerGraphsManager.flush_new_tokens /
        # flush_output_signals: hand off the dict, copy-then-clear the list,
        # pass the live output_loop_indices dict (legacy never clears it).
        new_tokens = self.pending_new_tokens[rid_idx]
        self.pending_new_tokens[rid_idx] = {}
        chunks = self.current_output_chunks[rid_idx]
        out_chunks = list(chunks)
        chunks.clear()
        self.communicator.send("conductor", ConductorMessage(
            message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
            body=WorkerGraphsDone(
                request_id=rid,
                worker_graph_ids=worker_graph_ids,
                is_first_tp_rank=is_first_tp_rank,
                persist_signals=persist_signals,
                new_tokens=new_tokens,
                output_signal_names=out_chunks,
                per_label_seq_info=per_label_seq_info,
                partition_name=partition_name,
                partition_done=partition_done,
                stream_tokens_consumed=stream_tokens_consumed,
                output_loop_indices=self.output_loop_indices[rid_idx],
                graph_timings=graph_timings,
                rx_info=rx_info,
                tx_info=tx_info,
            ),
        ))
        self.wgd_sent += 1


def run_sidecar(
    worker_id: str,
    sidecar_id: str,
    socket_path_prefix: str = "/tmp/mstar",
    log_level: str = "INFO",
) -> None:
    """Sidecar process target. Module-level for spawn picklability (same
    pattern as the conductor's _worker_process_target)."""
    # FIRST: make CUDA unreachable. The sidecar is a pure-CPU process with
    # its own GIL; torch.cuda must never initialize here (design §4.1/§7).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=f"%(asctime)s %(levelname)s [{sidecar_id}] %(name)s: %(message)s",
        force=True,
    )
    from mstar.utils.logging_config import quiet_noisy_loggers
    quiet_noisy_loggers()

    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    # SIGTERM (from SidecarClient.shutdown / worker teardown): drain the
    # queue with a 5 s deadline, then exit (design §7).
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    communicator = ZMQCommunicator(
        my_id=sidecar_id,
        push_ids=["api_server", "conductor"],
        ipc_socket_path_prefix=socket_path_prefix,
    )
    state = SidecarState(communicator, worker_id=worker_id)
    parent_pid = os.getppid()
    drain_deadline: float | None = None

    logger.info("%s ready (parent pid=%d)", sidecar_id, parent_pid)
    while True:
        messages = communicator.get_all_new_messages()
        state.note_drain(len(messages))
        for rec in messages:
            try:
                state.handle(rec)
            except Exception:
                # Loud fail-fast (design §7): a corrupted record stream must
                # not half-process silently. Exiting flips the worker to the
                # legacy path via its death watch.
                logger.critical(
                    "%s: record processing failed on %r — exiting so the "
                    "worker falls back to the legacy emit path",
                    sidecar_id, type(rec).__name__, exc_info=True,
                )
                raise
        if stop_requested:
            if drain_deadline is None:
                drain_deadline = time.monotonic() + 5.0
            # Drained (empty poll after the worker stopped feeding us) or
            # out of budget: exit.
            if not messages or time.monotonic() > drain_deadline:
                break
        elif not messages:
            # Parent-death watch: getppid() flips to the reaper when the
            # worker dies. Nothing left to drain reliably — exit.
            if os.getppid() != parent_pid:
                logger.warning(
                    "%s: worker (pid=%d) is gone; exiting", sidecar_id,
                    parent_pid,
                )
                break
            communicator.wait_for_work(50)
    logger.info(
        "%s exiting: steps=%d items=%d wgd=%d batch_msgs=%d",
        sidecar_id, state.steps, state.items_seen, state.wgd_sent,
        state.batch_msgs,
    )

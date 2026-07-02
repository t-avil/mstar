"""MSTAR_STEP_TXN: memoized decode-step transaction (default OFF).

Composes with W1 (``store_and_populate_graph_edges_fast``): W1 memoizes the
tensor-store / TensorPointerInfo derivation for one step; this module memoizes
the *rest* of ``_postprocess_batch``'s per-rid derivation — the node-complete
loop advance, ready-slot bookkeeping, output routing, and ref-count fanout —
for a uniform *continuing* ``thinker_decode`` step.

Why only that step is safe to memoize
--------------------------------------
The canonical decode loop (``thinker_decode_loop``) is a single-node ``Loop``
with **empty** ``outputs`` / ``accumulated_outputs``. On a *continuing* iter
(``curr_iter + 1 < max_iters`` and no finish signal) the slow chain

    mark_node_complete -> LoopStateRegistry.mark_entity_complete
        -> maybe_cache_output   (no-op: no declared loop outputs)
        -> complete_iter (continuing) -> _advance_one_iter
            -> curr_iter += 1
            -> inner_registry.reset_for_iter()   (swaps node ready slots)
            -> _uncache_outputs()                (no-op: cache empty)
        -> returns _ingested_external_inputs, filtered_signals == set()

produces a byte-identical routing shape every step, so the derivation can be
captured once and replayed. The moment ANY of those invariants can shift
(loop near max_iters, finish signal, declared loop outputs, spec-state change,
different tensor shapes) the transaction is invalid and the slow path runs.

The store seam stays W1's: ``fast_execute`` calls
``store_and_populate_graph_edges_fast`` unchanged and reuses its returned
``graph_node_info`` / ``per_request_uuids`` exactly as the slow block does.

This module deliberately reproduces, by hand, the mutations of the slow
path. Every reproduced site is annotated with the file:function it mirrors so
the two can be diffed. The guards in ``valid_for`` are conservative: on any
doubt the caller drops the txn and re-runs the slow block for that rid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.graph.graph_io import WorkerGraphIO
    from mstar.graph.base import LoopStateRegistry, WorkerGraphStateRegistry
    from mstar.worker.node_manager_utils import NodeOutputRouting


# Which ReadySignals slot the slow path's register_ingested_input writes for the
# re-injected loop-back input, given the node's speculative state at capture.
# Determined at capture time and asserted unchanged by valid_for.
@dataclass
class _ReadySlotPlan:
    """How to reproduce the loop-back re-ingestion for the advanced iteration.

    On a continuing thinker_decode step the node's ``text_inputs`` output is a
    loop-back edge (``next_node == node.name``). ``process_node_outputs`` routes
    it locally via ``process_new_inputs -> WorkerGraphIO.ingest_input ->
    GraphNode.ingest_input`` (graph_io.py:59, base.py:276). Because
    ``mark_node_complete`` already advanced the iter and swapped the node's
    ready slots (``reset_for_outer_iter``, base.py:333), the freshly-emptied
    ``ready_signals`` slot receives the edge via ``ReadySignals.update``
    (base.py:163), then ``register_ingested_input`` fires:

      GraphNode.ingest_input -> node._managing_registry (LoopStateRegistry)
        .register_ingested_input (base.py:700)
          -> Loop.ingest_external_input (base.py:496): for a pure loop-back
             input the ``(name, next_node) in _external_inputs`` test is False
             (loop-back inputs are not external inputs), so it does NOT persist
             the edge; it forwards to
          -> outer WorkerGraphStateRegistry.register_ingested_input (base.py:721)
             which, gated on ``not node._speculatively_scheduled``, may add the
             node to the outer ``ready_names`` / ``ready_for_streaming`` sets
             based on the node's now-updated ready_signals.

    So the reproduced effect touches BOTH the node's live ready_signals slot AND
    the outer registry's ready sets, exactly as base.py:721-742 dictates.
    """
    # names of the loop-back edges routed locally back into the node, in routing
    # order (sourced from routing.routed_to_this_worker_graph at capture)
    loopback_input_names: tuple[str, ...]
    # captured node._speculatively_scheduled at capture time; the outer-registry
    # ready-set add is gated on ``not this``
    speculatively_scheduled: bool


@dataclass
class StepTransaction:
    """Memoized derivation of one uniform continuing ``thinker_decode`` step
    for a single ``(rid, node_name, graph_walk)``.

    Holds *references* to identity-stable live objects (the same node / loop /
    registry / edge objects are reused every step), plus a captured routing
    template and the invariants ``valid_for`` re-checks before every replay.
    Never caches tensor payloads or uuids — those come fresh from W1 each step.
    """

    # ---- identity-stable live refs (reused every step) -------------------
    node: "GraphNode"
    loop: "Loop"
    inner_registry: "LoopStateRegistry"
    wg_state_registry: "WorkerGraphStateRegistry"
    wgio: "WorkerGraphIO"
    graph_walk: str

    # the routing template returned by the slow process_node_outputs on capture;
    # its GraphEdge lists are re-pointed (tensor_info) each replay, never rebuilt
    routing_template: "NodeOutputRouting"

    # per-name fanout counts for the ref-count delta step (mirrors
    # set_output_ref_counts, tensors.py ~758): how many routed edges consume
    # each captured uuid-name. Keyed by output edge name.
    fanout_counts: dict[str, int]

    # how to reproduce the ready-slot bookkeeping after reset_for_iter
    ready_slot_plan: _ReadySlotPlan

    # ---- invariants re-checked by valid_for ------------------------------
    expected_output_names: frozenset
    max_iters: int
    captured_spec_state: bool
    # fingerprint = (id(node), id(loop), id(wgio), sorted output names)
    fingerprint: tuple

    # ---- shadow-verify bookkeeping (MSTAR_STEP_TXN_VERIFY) ----------------
    # the routed edge lists (by identity) the fast path would treat as consumers
    routed_edge_ids: tuple = field(default_factory=tuple)


def build_fingerprint(node, loop, wgio, output_names) -> tuple:
    return (id(node), id(loop), id(wgio), tuple(sorted(output_names)))


def is_uniform_continuing(loop, graph_walk: str, node) -> bool:
    """Gate: the same restriction the existing prem-ints code uses at
    worker.py ~2252 (``graph_walk == "thinker_decode"``), tightened to the
    structural conditions that make the step's derivation memoizable.

    Requires: the thinker_decode walk; the node lives in a Loop with no
    declared/accumulated loop outputs (so maybe_cache_output/_uncache_outputs
    are no-ops); and the loop is strictly continuing after this step
    (curr_iter + 1 < max_iters, no finish signal). Near the boundary we defer
    to the slow path so the terminal step's different routing is never
    memoized.
    """
    if graph_walk != "thinker_decode":
        return False
    if loop is None:
        return False
    if loop._finish_signal:
        return False
    # empty declared loop outputs → maybe_cache_output / _uncache_outputs no-op
    if loop.outputs or loop.accumulated_outputs:
        return False
    # After this step the loop advances one iter. Require it to remain strictly
    # continuing (mirror the continuing branch condition in Loop.complete_iter:
    # continuing iff NOT (max_iters == curr_iter + 1 or _finish_signal)). We add
    # a one-step margin so the NEXT step (the replay) is itself continuing.
    if loop.max_iters <= loop.curr_iter + 2:
        return False
    return True


class StepTransactionRegistry:
    """Keyed by (rid, node_name, graph_walk). Lives on the Worker.

    A transaction is a pure optimization: dropping one only ever forces a slow
    rebuild, never a correctness hazard, so every invalidation path is
    conservative (drops broadly).
    """

    def __init__(self) -> None:
        self._txns: dict[tuple, StepTransaction] = {}

    def get(self, rid: str, node_name: str, graph_walk: str) -> Optional[StepTransaction]:
        return self._txns.get((rid, node_name, graph_walk))

    def put(self, rid: str, node_name: str, graph_walk: str, txn: StepTransaction) -> None:
        self._txns[(rid, node_name, graph_walk)] = txn

    def invalidate(
        self, rid: str, node_name: str | None = None, graph_walk: str | None = None
    ) -> None:
        """Drop transactions for an rid. With ``node_name``/``graph_walk``
        omitted, every txn for the rid is dropped (mirrors
        ``TensorCommunicationManager.invalidate_populate_plan``'s conservative
        contract, tensors.py:577)."""
        if node_name is None:
            for key in [k for k in self._txns if k[0] == rid]:
                self._txns.pop(key, None)
        else:
            self._txns.pop((rid, node_name, graph_walk), None)

    def clear(self) -> None:
        self._txns.clear()


# ---------------------------------------------------------------------------
# Capture and replay
# ---------------------------------------------------------------------------

def _routed_edges_of(routing: "NodeOutputRouting") -> list["GraphEdge"]:
    """The exact flat routed-edge list the slow path feeds to
    ``set_output_ref_counts`` (worker.py ~2233-2240). Kept in one place so
    capture and replay agree by construction."""
    return (
        routing.routed_to_this_worker_graph
        + routing.persist
        + routing.emit_to_client
        + routing.streaming_local
        + sum(routing.to_workers.values(), start=[])
        + sum(routing.streaming_to_workers.values(), start=[])
    )


def capture(
    *,
    rid: str,
    node: "GraphNode",
    loop: "Loop",
    wgio: "WorkerGraphIO",
    graph_walk: str,
    routing: "NodeOutputRouting",
) -> StepTransaction:
    """Build a StepTransaction from the values the slow block just computed
    for this rid. Called AFTER the slow block ran (so ``routing`` is the real,
    correct output to memoize).

    ``node``/``loop``/``wgio`` are the identity-stable live objects; ``routing``
    is the NodeOutputRouting the slow ``process_node_outputs`` returned. The
    output-name set comes from ``node.outputs`` (stable); no payloads/uuids are
    ever cached — those come fresh from W1 each replay.
    """
    inner_registry = node._managing_registry
    wg_state_registry = wgio.wg_state_registry

    output_names = frozenset(e.name for e in node.outputs)

    # fanout_counts[name] = number of routed edges carrying that name. Each such
    # edge's tensor_info IS the shared per-name list, so in set_output_ref_counts
    # every uuid of the name is counted once per routed edge of that name. This
    # reproduces actual_counts without needing the (per-step) uuids.
    routed_edges = _routed_edges_of(routing)
    fanout_counts: dict[str, int] = {}
    for edge in routed_edges:
        fanout_counts[edge.name] = fanout_counts.get(edge.name, 0) + 1

    # Loop-back inputs re-ingested into the node for the advanced iteration are
    # exactly the local-routed edges whose next_node is the node itself
    # (process_node_outputs routes them via process_new_inputs). Capture their
    # names in routing order.
    loopback_input_names = tuple(
        e.name
        for e in routing.routed_to_this_worker_graph
        if e.next_node == node.name and e.name in node.input_names
    )
    ready_slot_plan = _ReadySlotPlan(
        loopback_input_names=loopback_input_names,
        speculatively_scheduled=node._speculatively_scheduled,
    )

    return StepTransaction(
        node=node,
        loop=loop,
        inner_registry=inner_registry,
        wg_state_registry=wg_state_registry,
        wgio=wgio,
        graph_walk=graph_walk,
        routing_template=routing,
        fanout_counts=fanout_counts,
        ready_slot_plan=ready_slot_plan,
        expected_output_names=output_names,
        max_iters=loop.max_iters,
        captured_spec_state=node._speculatively_scheduled,
        fingerprint=build_fingerprint(node, loop, wgio, output_names),
        routed_edge_ids=tuple(id(e) for e in routed_edges),
    )


def valid_for(
    txn: StepTransaction,
    *,
    node: "GraphNode",
    graph_walk: str,
    req_output_tensors: dict,
) -> bool:
    """Guards, all conservative — any failure means run the slow block.

    * fingerprint ids match (same node/loop/wgio/output-name objects),
    * graph_walk matches,
    * spec-state unchanged since capture,
    * loop still strictly continuing after this step (curr_iter + 1 < max_iters,
      no finish signal) — near-boundary defers to slow so the terminal step's
      different routing is never replayed,
    * live output-tensor name set matches capture (shape/dtype re-checked inside
      W1's store_and_populate_graph_edges_fast, so we only guard the name set
      here; a mismatch there falls back on its own).
    """
    loop = txn.loop
    if graph_walk != txn.graph_walk:
        return False
    if node is not txn.node:
        return False
    if node._speculatively_scheduled != txn.captured_spec_state:
        return False
    if loop._finish_signal:
        return False
    # strictly continuing after the advance (mirror is_uniform_continuing margin)
    if loop.max_iters <= loop.curr_iter + 2:
        return False
    if loop.outputs or loop.accumulated_outputs:
        return False
    if txn.fingerprint != build_fingerprint(
        node, loop, txn.wgio, txn.expected_output_names
    ):
        return False
    if frozenset(req_output_tensors.keys()) - txn.expected_output_names:
        # extra output names not seen at capture → structural change
        return False
    return True


def fast_execute(
    txn: StepTransaction,
    *,
    rid: str,
    req_output_tensors: dict,
    tensor_manager,
) -> tuple["NodeOutputRouting", set]:
    """Replay the memoized step. Returns ``(routing, per_request_uuids)`` for the
    rid, matching what the slow block would have placed into
    ``routing_per_request[rid]`` / ``per_request_uuids[rid]``.

    Order is load-bearing and mirrors _postprocess_batch's per-rid slow block;
    each step is annotated with the slow site it reproduces.

    Failure atomicity: every lookup that can fail is resolved BEFORE the first
    state mutation. Once mutation begins (step 3 advances curr_iter) nothing
    below can raise, so the caller's slow-block fallback never runs against a
    half-advanced step. If the pre-mutation resolution raises, no state changed.
    """
    node = txn.node
    loop = txn.loop
    inner_registry = txn.inner_registry
    wg_state_registry = txn.wg_state_registry

    # (0) PRE-MUTATION resolution: look up the loop-back edge objects now, while
    # a miss can still raise harmlessly (nothing mutated yet).
    loopback_edges: list["GraphEdge"] = []
    for name in txn.ready_slot_plan.loopback_input_names:
        lb = _find_loopback_edge(txn.routing_template, node.name, name)
        if lb is None:
            raise RuntimeError(
                f"step_txn: loop-back edge {name!r} not found for rid {rid}"
            )
        loopback_edges.append((name, lb))

    # (1) reset stale outputs. Mirrors node.reset_outputs() (base.py:345) which
    # the slow block calls at worker.py:2192, restricted to this node's edges.
    for e in node.outputs:
        e.tensor_info.clear()

    # (2) store via W1 (unchanged seam). Same call the slow fast-postproc block
    # makes at worker.py:2198. Sets edge.tensor_info per name + safety-hold refs.
    gni = tensor_manager.store_and_populate_graph_edges_fast(
        request_id=rid,
        tensors=req_output_tensors,
        graph_edges=node.outputs,
        node_name=node.name,
        graph_walk=txn.graph_walk,
    )
    last_uuids = {info.uuid for infos in gni.values() for info in infos}

    # (3) advance the iteration. Mirrors Loop._advance_one_iter (base.py:491),
    # reached via mark_node_complete on a continuing step. maybe_cache_output /
    # _uncache_outputs are no-ops here (empty loop outputs — guarded), and
    # mark_entity_complete's counter mutations are subsumed by reset_for_iter
    # (which sets is_done=False, _num_completed_entities=0). We call the same
    # LoopStateRegistry.reset_for_iter the slow path calls.
    loop.curr_iter += 1
    inner_registry.reset_for_iter()

    # (4) LIVE-READ the node's current ready slot. reset_for_iter -> (per entity)
    # reset_for_outer_iter REBOUND node.ready_signals <-> ready_next_iter
    # (base.py:336); the object we want is whatever node.ready_signals points at
    # NOW, never a cached reference. Re-ingest the loop-back input(s) into that
    # slot, reproducing GraphNode.ingest_input -> ReadySignals.update
    # (base.py:163) followed by register_ingested_input's inner->outer chain
    # (base.py:700 -> 496 -> 721).
    ready = node.ready_signals  # live re-read AFTER the swap in step (3)
    for name, lb in loopback_edges:
        # the loop-back edge object is the routing template's local-routed edge
        # (pre-resolved in step 0); its tensor_info is re-pointed in step (5), so
        # storing it here is consistent with the slow path (which stores a fresh
        # per-step clone).
        # Mirror GraphNode.ingest_input's slot selection (base.py:286-296): the
        # freshly-swapped ready_signals slot does NOT yet have this name, so it
        # takes the ready_signals branch (never the ready_next_iter buffer).
        if name not in ready.ready_names:
            ready.ready_inputs[name] = lb
            ready.ready_names.add(name)
    # ReadySignals.update recomputes these on each add (base.py:171-178). Compute
    # once over the final ready_names — identical result for the single-input
    # decode node, and correct for any additional loop-back inputs.
    ready.is_ready = node.input_names.issubset(ready.ready_names)
    ready.is_ready_for_streaming = (
        ready.is_ready
        or ready.is_ready_for_streaming
        or node.input_names.issuperset(
            ready.ready_names | ready.streaming_inputs
        )
    )
    # Reproduce WorkerGraphStateRegistry.register_ingested_input's outer effect
    # (base.py:721-742) EXACTLY, reading the node's now-final live slot state.
    # The slow path calls this once per ingested edge; since readiness is
    # monotonic as inputs are added, evaluating once over the final state yields
    # the same set membership. Gated on ``not _speculatively_scheduled`` as the
    # slow path. Both current-iter (ready_signals) and next-iter branches are
    # reproduced so any additional loop-back inputs are handled identically.
    spec = txn.ready_slot_plan.speculatively_scheduled
    if node.ready_signals.is_ready:
        if not spec:
            wg_state_registry.ready_names.add(node.name)
        wg_state_registry.ready_for_streaming.discard(node.name)
    elif node.ready_signals.is_ready_for_streaming:
        if not spec:
            wg_state_registry.ready_for_streaming.add(node.name)

    if node.ready_next_iter.is_ready:
        if not spec:
            wg_state_registry.ready_next_iter.add(node.name)
        wg_state_registry.ready_streaming_next_iter.discard(node.name)
    elif node.ready_next_iter.is_ready_for_streaming:
        if not spec:
            wg_state_registry.ready_streaming_next_iter.add(node.name)

    # (5) re-point every routing-template edge's tensor_info at this step's
    # per-name list (mirrors the slow path where process_node_outputs returns
    # edges whose tensor_info was set by the store). Each name shares one list.
    for edge in _iter_routing_edges(txn.routing_template):
        if edge.name in gni:
            edge.tensor_info = gni[edge.name]

    # (6) ref-count deltas. Mirrors set_output_ref_counts (tensors.py:758): the
    # safety hold gave each uuid ref=1; adjust to the real fanout. delta =
    # fanout_count - 1 per name; >0 increment, <0 dereference. All uuids of a
    # name share the same fanout, so apply per uuid of the name.
    for name, infos in gni.items():
        count = txn.fanout_counts.get(name, 0)
        delta = count - 1
        for info in infos:
            if delta > 0:
                tensor_manager.increment_ref(rid, info.uuid, n=delta)
            elif delta < 0:
                tensor_manager.dereference(rid, info.uuid, n=-delta)

    return txn.routing_template, last_uuids


def _find_loopback_edge(
    routing: "NodeOutputRouting", node_name: str, name: str
) -> Optional["GraphEdge"]:
    for e in routing.routed_to_this_worker_graph:
        if e.next_node == node_name and e.name == name:
            return e
    return None


def _iter_routing_edges(routing: "NodeOutputRouting"):
    yield from routing.routed_to_this_worker_graph
    yield from routing.persist
    yield from routing.emit_to_client
    yield from routing.new_token_outputs
    yield from routing.streaming_local
    for edges in routing.to_workers.values():
        yield from edges
    for edges in routing.streaming_to_workers.values():
        yield from edges

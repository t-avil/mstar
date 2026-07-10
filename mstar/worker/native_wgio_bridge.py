"""Bridge between the live Python ``WorkerGraphIO`` state machine and the
validated C++ pybind11 port (``native_wgio.WGIO``).

Gated entirely by ``MSTAR_NATIVE_WGIO`` (default ``0`` = off). When off, this
module is never imported and there is zero overhead on the worker path — the
import is lazy, guarded by the flag read once in ``node_manager_utils``.

Modes (env ``MSTAR_NATIVE_WGIO``):
  * ``0`` / unset          : off. Bridge never constructed.
  * ``shadow``             : run BOTH the Python ``WorkerGraphIO`` and the C++
                             port on every mutating/reading call, assert the
                             projected state matches (ready_node_names,
                             ready_for_streaming, is_done, num_times_run, the
                             mark_node_complete (name,next) edge lists +
                             filtered set, and the process_new_inputs not-
                             ingested list), count matches/mismatches, and
                             RETURN THE PYTHON RESULT. Zero behavior change —
                             pure validation. Works for ALL worker graphs
                             (looped decode included).
  * ``1`` / ``native``     : use the C++ result. The C++ owns readiness /
                             iteration; Python owns the tensor payload; the
                             bridge stitches them at the handle boundary:
                               - loopless walks (prefill / encoder) run fully
                                 native — mark_node_complete reconstructs the
                                 completing node's live ``node.outputs`` edges;
                               - the hot ``thinker_decode`` self-loop (and any
                                 loop whose section has NO external inputs and
                                 NO terminal/accumulated Loop.outputs) also runs
                                 native: every edge the C++ emits from
                                 mark_node_complete is a subset of the
                                 completing node's own static ``node.outputs``
                                 (the loop re-injects nothing and caches no
                                 terminal tensor_info), so the SAME loopless
                                 reconstruction is byte-identical. The bridge
                                 additionally (a) populates the live
                                 ``ready_signals`` / ``ready_next_iter`` payload
                                 so the worker can assemble a node's INPUT
                                 tensors, driving the ReadySignals clear/swap
                                 lifecycle from C++-emitted slot events, and
                                 (b) mirrors the C++ loop ``curr_iter`` /
                                 ``num_times_run`` onto the live Python objects
                                 for the loop-index consumers.
                             A loop that DOES re-inject external inputs or cache
                             terminal Loop.outputs still needs dynamic payload
                             reconstruction the int-keyed C++ cannot hold, so it
                             FALLS BACK to Python (logged once). Shadow validates
                             every worker graph regardless.

Safety contract (see task CRITICAL): default-off is a no-op; the flag is read
once; any construction failure or unrepresentable graph falls back to Python
and logs — it never crashes the worker.

``MSTAR_NATIVE_WGIO_STRICT=1`` turns shadow mismatches into hard
``AssertionError`` (used by the offline unit test); default is loud-log +
count so a live decode server is never taken down by a shadow divergence.

``MSTAR_NATIVE_WGIO_SO_DIR`` overrides where ``native_wgio`` (the .so) is
imported from (default the gilext_poc build dir).
"""
import logging
import os as _os
import sys as _sys

from mstar.graph.base import (
    GraphNode,
    Loop,
    NodeCompletionOutput,
    WorkerGraphStateRegistry,
)
from mstar.graph.graph_io import WorkerGraphIO

logger = logging.getLogger(__name__)

_DEFAULT_SO_DIR = "/m-coriander/coriander/tim/gilext_poc"

# Sentinels for edge->int translation. Both are guaranteed absent from the C++
# id spaces: real node ids are 0..N-1 (a <0 or >=N id => "not in this graph"),
# real string ids are >=0 (a negative id => name never seen at desc-build time,
# so no node's name_to_bit contains it => not ingested).
_NODE_MISS = -1
_NAME_MISS = -(10 ** 9)

_STRICT = _os.environ.get("MSTAR_NATIVE_WGIO_STRICT", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# How often (in matched compare calls) to emit a rolling shadow-match summary.
_LOG_EVERY = int(_os.environ.get("MSTAR_NATIVE_WGIO_LOG_EVERY", "2000") or "2000")

# Process-global shadow counters, aggregated across all worker graphs, so the
# operator can grep one line for "shadow" and see the running match count.
_SHADOW_MATCHES = 0
_SHADOW_MISMATCHES = 0


def native_wgio_shadow_stats() -> tuple[int, int]:
    """Return (matches, mismatches) accumulated across all bridges this process."""
    return _SHADOW_MATCHES, _SHADOW_MISMATCHES


def _import_native():
    so_dir = _os.environ.get("MSTAR_NATIVE_WGIO_SO_DIR", _DEFAULT_SO_DIR)
    if so_dir and so_dir not in _sys.path:
        _sys.path.insert(0, so_dir)
    import native_wgio  # noqa: E402  (lazy, flag-gated)
    return native_wgio


# --------------------------------------------------------------------------
# Introspection: real WorkerGraphIO / registries -> integer POD description.
# Ported verbatim (semantics) from gilext_poc/test_wgio.py:build_desc — the
# same construction that the 4000-trial differential test validated.
# --------------------------------------------------------------------------
def build_desc(wgio: WorkerGraphIO):
    string_ids: dict[str, int] = {}

    def sid(s):
        if s not in string_ids:
            string_ids[s] = len(string_ids)
        return string_ids[s]

    node_list = list(wgio.nodes.keys())              # graphid order
    node_id = {nm: i for i, nm in enumerate(node_list)}
    N = len(node_list)
    loop_list = list(wgio.loops.keys())
    loop_id = {nm: N + i for i, nm in enumerate(loop_list)}   # entity id space

    def entity_id(name):
        return node_id[name] if name in node_id else loop_id[name]

    # registry object -> reg id (worker registry forced to id 0)
    reg_id: dict[int, int] = {}

    def rid_of(reg_obj):
        key = id(reg_obj)
        if key not in reg_id:
            reg_id[key] = len(reg_id)
        return reg_id[key]

    rid_of(wgio.wg_state_registry)
    assert reg_id[id(wgio.wg_state_registry)] == 0
    for lname in loop_list:
        rid_of(wgio.loops[lname].inner_registry)

    node_string_ids = [sid(nm) for nm in node_list]
    node_inputs, node_streaming, node_mreg, node_outputs = [], [], [], []
    for nm in node_list:
        node = wgio.nodes[nm]
        node_inputs.append([sid(x) for x in node.input_names])
        node_streaming.append([sid(x) for x in node._streaming_inputs])
        node_mreg.append(reg_id[id(node._managing_registry)])
        node_outputs.append([[sid(e.name), sid(e.next_node)] for e in node.outputs])

    loops_desc = []
    for lname in loop_list:
        L = wgio.loops[lname]
        loops_desc.append({
            "inner_reg": reg_id[id(L.inner_registry)],
            "managing_reg": reg_id[id(L._managing_registry)],
            "entity_id": loop_id[lname],
            "max_iters": L.max_iters,
            "outputs": [[sid(e.name), sid(e.next_node)] for e in L.outputs],
            "accumulated_outputs":
                [[sid(e.name), sid(e.next_node)] for e in L.accumulated_outputs],
            "loop_back_inputs": [[sid(n), sid(d)] for (n, d) in L._loop_back_inputs],
            "external_inputs":
                [[sid(n), node_id[d]] for (n, d) in L._external_inputs if d in node_id],
        })

    id_to_reg = {v: k for k, v in reg_id.items()}
    obj_by_id = {id(wgio.wg_state_registry): wgio.wg_state_registry}
    for lname in loop_list:
        obj_by_id[id(wgio.loops[lname].inner_registry)] = wgio.loops[lname].inner_registry

    regs = []
    for i in range(len(reg_id)):
        reg_obj = obj_by_id[id_to_reg[i]]
        managed = [entity_id(nm) for nm in reg_obj.managed_entities.keys()]
        is_worker = isinstance(reg_obj, WorkerGraphStateRegistry)
        regs.append({
            "kind": 0 if is_worker else 1,
            "managed": managed,
            "loop_index": (-1 if is_worker else loop_list.index(reg_obj.loop.name)),
        })

    only_streaming = [node_id[nm] for nm in wgio.wg_state_registry.only_streaming_inputs]

    desc = {
        "num_nodes": N,
        "node_string_ids": node_string_ids,
        "node_inputs": node_inputs,
        "node_streaming": node_streaming,
        "node_managing_reg": node_mreg,
        "node_outputs": node_outputs,
        "loops": loops_desc,
        "regs": regs,
        "only_streaming": only_streaming,
    }
    meta = {
        "node_id": node_id,
        "node_list": node_list,
        "loop_list": loop_list,
        "sid": string_ids,
        "inv_sid": {v: k for k, v in string_ids.items()},
        "num_loops": len(loop_list),
    }
    return desc, meta


class NativeWGIOBridge:
    """Per-worker-graph bridge. One C++ ``WGIO`` object holds the disjoint
    per-request state for every request on this worker graph (mirrors the
    graph structure, which is identical across requests since each per-request
    ``WorkerGraphIO`` is a deepcopy of the same section)."""

    def __init__(self, section, worker_graph_id, mode: str):
        assert mode in ("shadow", "native")
        self.mode = mode
        self.worker_graph_id = worker_graph_id

        # Build the integer description from a throwaway template WorkerGraphIO
        # (structure only — never driven). Raises on an unrepresentable graph;
        # the caller catches and falls back to pure Python.
        from copy import deepcopy
        template = WorkerGraphIO(deepcopy(section), wg_id=worker_graph_id)
        template.register_communication_info(None, "__native_template__")
        self._desc, self._meta = build_desc(template)

        self.node_id = self._meta["node_id"]
        self.node_list = self._meta["node_list"]
        self.inv_sid = self._meta["inv_sid"]
        self.sid = self._meta["sid"]
        self.loop_list = self._meta["loop_list"]
        # A worker graph is native-reconstructable iff every Loop it contains
        # re-injects NO external inputs AND caches NO terminal / accumulated
        # Loop.outputs. For such loops (the ``thinker_decode`` self-loop is the
        # canonical case) every edge the C++ emits from mark_node_complete is a
        # subset of the completing node's own static ``node.outputs`` — the
        # exact loopless reconstruction — and the loop's re-injection list
        # (``ing_inputs``) is always empty, so there is no runtime GraphEdge
        # payload the int-keyed C++ needs to hold. A loop with external inputs
        # or terminal outputs would need those live edges reconstructed and is
        # left to Python (shadow still validates it). Loopless => trivially true.
        self.native_capable = all(
            not L["external_inputs"]
            and not L["outputs"]
            and not L["accumulated_outputs"]
            for L in self._desc["loops"]
        )

        native = _import_native()
        self.cpp = native.WGIO(self._desc)

        self._rid: dict[str, int] = {}          # request_id -> int
        self._wgio: dict[str, WorkerGraphIO] = {}  # request_id -> live py wgio
        self._next_rid = 0
        # Hot-path spec-flag forwarding state, per request_id:
        #   _spec_pairs[request_id]  = [(node_id, node_obj), ...] for every node
        #     present in the C++ desc, PRECOMPUTED at add_request so the hot
        #     _sync_spec_flags loop needs neither ``wgio.nodes.items()`` nor a
        #     name->id dict lookup nor the _NODE_MISS filter per call.
        #   _spec_cache[request_id] = [last bool pushed to C++, ...] parallel to
        #     _spec_pairs, so change-detection is a list index (no (rid,node_id)
        #     tuple build + dict hash). Only a CHANGED flag crosses pybind.
        self._spec_pairs: dict[str, list] = {}
        self._spec_cache: dict[str, list] = {}

        self._matches = 0
        self._mismatches = 0
        logger.info(
            "MSTAR_NATIVE_WGIO=%s: bridge active for wg=%s (nodes=%d loops=%d "
            "native_capable=%s)",
            mode, worker_graph_id, len(self.node_list),
            self._meta["num_loops"], self.native_capable,
        )

    # ------------------------------------------------------------------
    # translation helpers
    # ------------------------------------------------------------------
    def _edges_to_ints(self, edges):
        next_ids = [self.node_id.get(e.next_node, _NODE_MISS) for e in edges]
        name_ids = [self.sid.get(e.name, _NAME_MISS) for e in edges]
        return next_ids, name_ids

    # ------------------------------------------------------------------
    # native-mode payload stitching (C++ owns readiness; Python owns payload)
    # ------------------------------------------------------------------
    def _ingest_payload(self, node, edge, can_buffer):
        """Place ``edge`` (carrying tensor_info) into the live node's payload
        slot the C++ just chose. Mirrors ``GraphNode.ingest_input`` MINUS the
        registry chain-up (the readiness bookkeeping the C++ owns): the current
        vs next-iter slot decision is the SAME predicate the C++ ``ingest_node``
        used (current slot free? else buffer to next-iter), evaluated on the
        Python ``ReadySignals.ready_names`` the bridge keeps in lockstep. The
        C++ has already claimed this edge (it is not in the not-ingested set),
        so exactly one of the two branches fires; ``ReadySignals.update`` fills
        ``ready_inputs`` so the worker's ``_build_node_batch`` can gather the
        input tensors, and recomputes the (locally-ignored) ready flags."""
        if edge.name not in node.input_names:
            return
        if edge.name not in node.ready_signals.ready_names:
            node.ready_signals.update(edge)
        elif can_buffer and edge.name not in node.ready_next_iter.ready_names:
            node.ready_next_iter.update(edge)
        # else: both slots already hold this input — C++ would not have claimed
        # it either (unreachable while state is in lockstep); leave payload be.

    def _apply_events(self, wgio, events):
        """Drive the ReadySignals clear/swap lifecycle the C++ state machine
        just performed onto the live Python nodes, so the payload dicts (and
        their tensor_manager dereference on clear) stay byte-identical to the
        pure-Python registry path. Event kinds mirror node_manager_utils:
          0 CLEAR_CUR : top-level node completed  -> ready_signals.clear()
          1 CLEAR_ALL : registry clear (loop done / wgio clear) -> node.clear()
          2 SWAP      : loop advanced -> reset_for_outer_iter (clear cur, swap)
        """
        for node_id, kind in events:
            node = wgio.nodes[self.node_list[node_id]]
            if kind == 0:
                node.ready_signals.clear()
            elif kind == 1:
                node.clear()
            else:
                node.reset_for_outer_iter()

    def _mirror_iter_state(self, wgio, rid):
        """The C++ owns iteration, so the live Python objects (never advanced
        natively) are stale for the loop-index consumers
        (``get_loop_indices`` / ``get_nested_loop_idxs`` read
        ``loop.curr_iter`` and ``wgio.num_times_run`` directly). Mirror them
        from the authoritative C++ after every completion / reset."""
        wgio.num_times_run = self.cpp.num_times_run(rid)
        for i, lname in enumerate(self.loop_list):
            loop = wgio.loops.get(lname)
            if loop is not None:
                loop.curr_iter = self.cpp.loop_curr_iter(rid, i)

    # ------------------------------------------------------------------
    # out-of-band _speculatively_scheduled forwarding
    # ------------------------------------------------------------------
    def _sync_spec_flags(self, request_id):
        """Forward the worker's out-of-band ``node._speculatively_scheduled``
        writes to the C++ port so its ready-queue projection stays identical.

        During a decode speculation chain the worker flips this flag DIRECTLY on
        the Python node object (``worker.py``), outside any ingest call, so a
        node already executing as a spec batch is not double-queued.
        ``WorkerGraphStateRegistry.register_ingested_input`` gates the ready-queue
        ADD on that flag; the C++ ``worker_register`` mirrors the gate but only
        via ``spec_scheduled``, which nothing sets unless we push it here. Called
        at the top of every op that reads or mutates ready state so the C++ sees
        the same spec-scheduling the Python nodes have at that instant.

        Cheap on the hot path: the (node_id, node) pairs and the last-pushed
        flags are precomputed at add_request, so this is a bare list walk —
        one attribute read + one list compare per node, and a pybind call ONLY
        for a node whose flag actually changed. Nodes absent from the C++ desc
        were filtered out of ``_spec_pairs`` at add_request, so they cost
        nothing here.
        """
        pairs = self._spec_pairs.get(request_id)
        if not pairs:
            return
        rid = self._rid[request_id]
        cache = self._spec_cache[request_id]
        cpp_set = self.cpp.set_speculatively_scheduled
        for idx, (node_id, node) in enumerate(pairs):
            flag = node._speculatively_scheduled   # dataclass field, always present
            if cache[idx] != flag:
                cpp_set(rid, node_id, flag)
                cache[idx] = flag

    # ------------------------------------------------------------------
    # shadow comparison
    # ------------------------------------------------------------------
    def _record(self, ok: bool, ctx: str, detail: str = ""):
        global _SHADOW_MATCHES, _SHADOW_MISMATCHES
        if ok:
            self._matches += 1
            _SHADOW_MATCHES += 1
            if _LOG_EVERY and (_SHADOW_MATCHES % _LOG_EVERY == 0):
                logger.info(
                    "MSTAR_NATIVE_WGIO shadow: matches=%d mismatches=%d "
                    "(wg=%s local matches=%d)",
                    _SHADOW_MATCHES, _SHADOW_MISMATCHES,
                    self.worker_graph_id, self._matches,
                )
        else:
            self._mismatches += 1
            _SHADOW_MISMATCHES += 1
            msg = ("MSTAR_NATIVE_WGIO shadow MISMATCH wg=%s ctx=%s: %s"
                   % (self.worker_graph_id, ctx, detail))
            if _STRICT:
                raise AssertionError(msg)
            logger.error(msg)

    def _compare_state(self, request_id, ctx):
        wgio = self._wgio[request_id]
        rid = self._rid[request_id]
        py_ready = set(wgio.ready_node_names)
        cpp_ready = {self.node_list[i] for i in self.cpp.ready_node_names(rid)}
        py_stream = set(wgio.ready_for_streaming)
        cpp_stream = {self.node_list[i] for i in self.cpp.ready_for_streaming(rid)}
        py_done = wgio.wg_state_registry.is_done
        cpp_done = self.cpp.is_done(rid)
        py_ntr = wgio.num_times_run
        cpp_ntr = self.cpp.num_times_run(rid)
        ok = (py_ready == cpp_ready and py_stream == cpp_stream
              and py_done == cpp_done and py_ntr == cpp_ntr)
        if not ok:
            self._record(False, ctx, (
                f"ready {py_ready} != {cpp_ready}; stream {py_stream} != "
                f"{cpp_stream}; done {py_done}!={cpp_done}; ntr {py_ntr}!={cpp_ntr}"
            ))
        else:
            self._record(True, ctx)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def add_request(self, request_id: str, wgio: WorkerGraphIO):
        rid = self._next_rid
        self._next_rid += 1
        self._rid[request_id] = rid
        self._wgio[request_id] = wgio
        # Precompute the hot-path spec-flag walk: (node_id, node) for every node
        # in the C++ desc, plus a parallel all-False cache matching the C++
        # per-request initial state (init_request zeroes spec_scheduled).
        node_id_map = self.node_id
        pairs = [(nid, node) for name, node in wgio.nodes.items()
                 if (nid := node_id_map.get(name, _NODE_MISS)) != _NODE_MISS]
        self._spec_pairs[request_id] = pairs
        self._spec_cache[request_id] = [False] * len(pairs)
        self.cpp.add_request(rid)
        if self.mode == "shadow":
            self._compare_state(request_id, "add_request")

    def remove_request(self, request_id: str):
        rid = self._rid.pop(request_id, None)
        self._wgio.pop(request_id, None)
        self._spec_pairs.pop(request_id, None)
        self._spec_cache.pop(request_id, None)
        if rid is not None:
            self.cpp.remove_request(rid)

    # ------------------------------------------------------------------
    # mutating ops — signature: (..., py_fn) where py_fn() runs the original
    # Python WorkerGraphQueues body and returns its result.
    # ------------------------------------------------------------------
    def process_new_inputs(self, request_id, inputs, can_buffer, py_fn):
        self._sync_spec_flags(request_id)
        rid = self._rid[request_id]
        next_ids, name_ids = self._edges_to_ints(inputs)
        if self.mode == "native" and self.native_capable:
            miss_idx = self.cpp.process_new_inputs(rid, next_ids, name_ids, can_buffer)
            # C++ decided readiness; now stitch the tensor payload into the live
            # nodes for every edge it claimed (miss_idx = the ones it did not).
            miss = set(miss_idx)
            wgio = self._wgio[request_id]
            for i, e in enumerate(inputs):
                if i in miss:
                    continue
                node = wgio.nodes.get(e.next_node)
                if node is not None:
                    self._ingest_payload(node, e, can_buffer)
            return [inputs[i] for i in miss_idx]
        # shadow: run both, compare not-ingested + state, return Python result
        py_not = py_fn()
        miss_idx = self.cpp.process_new_inputs(rid, next_ids, name_ids, can_buffer)
        py_pairs = [(e.name, e.next_node) for e in py_not]
        cpp_pairs = [(inputs[i].name, inputs[i].next_node) for i in miss_idx]
        if py_pairs != cpp_pairs:
            self._record(False, "process_new_inputs",
                         f"not_ingested {py_pairs} != {cpp_pairs}")
        else:
            self._compare_state(request_id, "process_new_inputs")
        return py_not

    def process_new_streaming_inputs(self, request_id, inputs, can_buffer, py_fn):
        self._sync_spec_flags(request_id)
        rid = self._rid[request_id]
        next_ids, name_ids = self._edges_to_ints(inputs)
        if self.mode == "native" and self.native_capable:
            miss_idx = self.cpp.process_new_streaming_inputs(
                rid, next_ids, name_ids, can_buffer)
            miss = set(miss_idx)
            wgio = self._wgio[request_id]
            for i, e in enumerate(inputs):
                if i in miss:
                    continue
                node = wgio.nodes.get(e.next_node)
                if node is not None:
                    self._ingest_payload(node, e, can_buffer)
            return [inputs[i] for i in miss_idx]
        py_not = py_fn()
        miss_idx = self.cpp.process_new_streaming_inputs(
            rid, next_ids, name_ids, can_buffer)
        py_pairs = [(e.name, e.next_node) for e in py_not]
        cpp_pairs = [(inputs[i].name, inputs[i].next_node) for i in miss_idx]
        if py_pairs != cpp_pairs:
            self._record(False, "process_new_streaming_inputs",
                         f"not_ingested {py_pairs} != {cpp_pairs}")
        else:
            self._compare_state(request_id, "process_new_streaming_inputs")
        return py_not

    def mark_node_complete(self, request_id, node_name, py_fn):
        rid = self._rid[request_id]
        nid = self.node_id[node_name]
        wgio = self._wgio[request_id]
        # The _speculatively_scheduled gate is read ONLY on the ingest path
        # (worker_register), so completion never depends on it — skip the sync on
        # the native hot path. Shadow keeps it (its per-op _compare_state must see
        # the same flags Python does).
        if self.mode == "native" and self.native_capable:
            # ONE fused crossing returns filtered-set + slot events + advanced
            # num_times_run + per-loop curr_iter, replacing the previous four
            # crossings (mark_node_complete + take_last_events + num_times_run +
            # loop_curr_iter-per-loop). Byte-identical: the C++ does the same
            # state mutation, and the values are exactly what the getters
            # returned. The output-edge list the old C++ built here was always
            # discarded on this path — native reconstructs edges from the live
            # node.outputs using only ``filt`` below.
            filt_c, events, ntr, iters = self.cpp.mark_node_complete_native(rid, nid)
            filt = {(self.inv_sid[a], self.inv_sid[b]) for (a, b) in filt_c}
            # For a native-capable graph (loopless OR a loop with no external
            # inputs / no terminal outputs) every edge the C++ emits is a subset
            # of the completing node's own static ``node.outputs`` GraphEdge
            # objects: the loop re-injects nothing and caches no terminal
            # tensor_info, so the ONLY runtime payload is the tensor_info the
            # worker wrote onto ``node.outputs`` before completing. Reconstruct
            # from those real objects (identity-stable, tensor_info live),
            # preserving node.outputs order (== C++ emission order).
            node = wgio.nodes[node_name]
            out_edges = [e for e in node.outputs
                         if (e.name, e.next_node) not in filt]
            # Drive the Python ReadySignals clear/swap lifecycle the C++ just
            # performed (frees consumed input payload; promotes the buffered
            # loop-back for the next iter), then mirror the advanced iteration
            # counters onto the live Python objects — both from the values the
            # fused call already returned (no extra crossings).
            self._apply_events(wgio, events)
            wgio.num_times_run = ntr
            loops = wgio.loops
            for i, lname in enumerate(self.loop_list):
                loop = loops.get(lname)
                if loop is not None:
                    loop.curr_iter = iters[i]
            return NodeCompletionOutput(output_edges=out_edges, filtered_signals=filt)
        # shadow
        self._sync_spec_flags(request_id)
        completion = py_fn()
        edges_c, filt_c = self.cpp.mark_node_complete(rid, nid)
        py_out = [(e.name, e.next_node) for e in completion.output_edges]
        cpp_out = [(self.inv_sid[a], self.inv_sid[b]) for (a, b) in edges_c]
        py_filt = set(completion.filtered_signals)
        cpp_filt = {(self.inv_sid[a], self.inv_sid[b]) for (a, b) in filt_c}
        if py_out != cpp_out or py_filt != cpp_filt:
            self._record(False, "mark_node_complete",
                         f"node={node_name} edges {py_out} != {cpp_out}; "
                         f"filt {py_filt} != {cpp_filt}")
        else:
            self._compare_state(request_id, "mark_node_complete")
            # Extra cross-check (native-capable graphs only): the edges the
            # NATIVE path would reconstruct from the live node.outputs must be
            # the SAME objects (identity) and carry the SAME tensor_info as the
            # Python completion produced — proving the native reconstruction is
            # payload-identical without taking over the state machine.
            if self.native_capable:
                node = wgio.nodes[node_name]
                recon = [e for e in node.outputs
                         if (e.name, e.next_node) not in cpp_filt]
                py_edges = completion.output_edges
                ok = len(recon) == len(py_edges) and all(
                    r is p and r.tensor_info is p.tensor_info
                    for r, p in zip(recon, py_edges)
                )
                if not ok:
                    self._record(False, "mark_node_complete.payload",
                                 f"node={node_name} native recon "
                                 f"{[(e.name, e.next_node) for e in recon]} != "
                                 f"py {py_out}")
        return completion

    def pop_ready_nodes(self, request_id, node_names, py_fn):
        rid = self._rid[request_id]
        # pop only erases from ready_names; it never ingests, so the spec gate is
        # irrelevant — skip the sync on the native hot path (shadow keeps it).
        if self.mode == "native" and self.native_capable:
            self.cpp.pop_ready_nodes(rid, [self.node_id[n] for n in node_names])
            wgio = self._wgio[request_id]
            return [wgio.nodes[n] for n in node_names if n in wgio.nodes]
        self._sync_spec_flags(request_id)
        nodes = py_fn()
        self.cpp.pop_ready_nodes(rid, [self.node_id[n] for n in node_names
                                       if n in self.node_id])
        self._compare_state(request_id, "pop_ready_nodes")
        return nodes

    def push_back_node(self, request_id, node, py_fn):
        rid = self._rid[request_id]
        if self.mode == "native" and self.native_capable:
            self.cpp.push_back_node(rid, self.node_id[node.name])
            return None
        r = py_fn()
        self.cpp.push_back_node(rid, self.node_id[node.name])
        self._compare_state(request_id, "push_back_node")
        return r

    def reset(self, request_id, py_fn):
        rid = self._rid[request_id]
        if self.mode == "native" and self.native_capable:
            self.cpp.clear(rid)
            # free any lingering input payload (dereference) + reset the mirrored
            # iteration counters, exactly as the Python wgio.clear() would.
            wgio = self._wgio[request_id]
            self._apply_events(wgio, self.cpp.take_last_events())
            self._mirror_iter_state(wgio, rid)
            return None
        r = py_fn()
        self.cpp.clear(rid)
        self._compare_state(request_id, "reset")
        return r

    def stop_loops(self, request_id, loop_names, py_fn):
        rid = self._rid[request_id]
        # native_capable may now include loops (thinker_decode): register the
        # finish signal in the C++ (authoritative for iteration) and return the
        # loop-back (name, dest) set straight off the live Python Loop objects
        # (static structure — identical to what _py_stop_loops returns).
        if self.mode == "native" and self.native_capable:
            wgio = self._wgio[request_id]
            signals = set()
            for name in loop_names:
                if name in wgio.loops:
                    self.cpp.register_loop_finish_signal(
                        rid, self.loop_list.index(name))
                    signals |= wgio.loops[name]._loop_back_inputs
            return signals
        self._sync_spec_flags(request_id)
        r = py_fn()
        for name in loop_names:
            if name in self.loop_list:
                self.cpp.register_loop_finish_signal(rid, self.loop_list.index(name))
        self._compare_state(request_id, "stop_loops")
        return r

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def is_done(self, request_id, py_fn):
        rid = self._rid[request_id]
        # is_done reads the completion counter, not ready state — the spec gate
        # never affects it, so skip the sync on the native hot path.
        if self.mode == "native" and self.native_capable:
            return self.cpp.is_done(rid)
        self._sync_spec_flags(request_id)
        py = py_fn()
        cpp = self.cpp.is_done(rid)
        if py != cpp:
            self._record(False, "is_done", f"{py} != {cpp}")
        else:
            self._record(True, "is_done")
        return py

    def get_ready_node_names(self, py_fn):
        # no request_id arg — this op projects ready state for ALL live rids, so
        # forward every request's spec flags before reading.
        for request_id in list(self._rid):
            self._sync_spec_flags(request_id)
        if self.mode == "native" and self.native_capable:
            return {
                request_id: {self.node_list[i]
                             for i in self.cpp.ready_node_names(rid)}
                for request_id, rid in self._rid.items()
            }
        py = py_fn()
        for request_id, rid in self._rid.items():
            cpp_ready = {self.node_list[i] for i in self.cpp.ready_node_names(rid)}
            py_ready = set(py.get(request_id, set()))
            if py_ready != cpp_ready:
                self._record(False, "get_ready_node_names",
                             f"rid={request_id} {py_ready} != {cpp_ready}")
            else:
                self._record(True, "get_ready_node_names")
        return py

    def get_ready_for_streaming(self, request_id, py_fn):
        self._sync_spec_flags(request_id)
        rid = self._rid[request_id]
        if self.mode == "native" and self.native_capable:
            return {self.node_list[i] for i in self.cpp.ready_for_streaming(rid)}
        py = py_fn()
        cpp_stream = {self.node_list[i] for i in self.cpp.ready_for_streaming(rid)}
        if set(py) != cpp_stream:
            self._record(False, "get_ready_for_streaming",
                         f"rid={request_id} {set(py)} != {cpp_stream}")
        else:
            self._record(True, "get_ready_for_streaming")
        return py

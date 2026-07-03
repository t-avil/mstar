"""CPU unit tests for the MSTAR_ASYNC_SCHED (V1 async scheduling) worker
state machine.

These exercise the pure-Python bookkeeping the flag adds — deferred-postprocess
roll-forward/flush, treating a deferred step's rids as in-flight for remove
safety, and late-stop-trim cleanup on removal — without constructing a real
engine or touching a GPU. The Worker is built with ``object.__new__`` and only
the attributes the methods under test read are populated, so the tests stay at
the logic level (the end-to-end token-identity check is the GPU smoke in
SMOKE.md).
"""
from types import SimpleNamespace

from mstar.worker.worker import Worker


def _batch(*rids):
    """A stand-in PendingBatch: only ``.batch.node_objects`` is read by the
    methods under test."""
    return SimpleNamespace(
        batch=SimpleNamespace(node_objects={r: object() for r in rids})
    )


def _bare_worker():
    w = object.__new__(Worker)
    w._async_sched = True
    w._deferred_pp = None
    w._async_trim = set()
    w._walk_stats = None
    w._pending_removes = set()
    w._side_in_flight_rids = set()
    w._slim_emit = False
    w._slim_emit_sent = set()
    w._slim_emit_loop_layout = {}
    return w


def test_run_deferred_is_noop_when_nothing_owed():
    w = _bare_worker()
    calls = []
    w._postprocess_batch = lambda p, o: calls.append((p, o))
    w._run_deferred_postprocess()
    assert calls == []
    assert w._deferred_pp is None


def test_run_deferred_runs_and_clears_the_owed_step():
    w = _bare_worker()
    calls = []
    w._postprocess_batch = lambda p, o: calls.append((p, o))
    pb, out = _batch("r1", "r2"), object()
    w._deferred_pp = (pb, out)
    w._run_deferred_postprocess()
    assert calls == [(pb, out)]
    # Slot cleared BEFORE the postprocess-driven reschedule can observe it.
    assert w._deferred_pp is None


def test_deferred_rids_are_held_from_removal():
    w = _bare_worker()
    removed = []
    w._remove_request = lambda body: removed.append(body.request_id)
    # r_defer is owed a postprocess; r_free is not referenced anywhere.
    w._deferred_pp = (_batch("r_defer"), object())
    w._pending_removes = {"r_defer", "r_free"}
    w._apply_pending_removes_safe_to_drop(in_flight_rids=set())
    assert removed == ["r_free"]
    # r_defer stays queued until its deferral is flushed.
    assert w._pending_removes == {"r_defer"}


def test_deferred_rid_released_once_flushed():
    w = _bare_worker()
    removed = []
    w._remove_request = lambda body: removed.append(body.request_id)
    w._deferred_pp = None  # already flushed
    w._pending_removes = {"r_defer"}
    w._apply_pending_removes_safe_to_drop(in_flight_rids=set())
    assert removed == ["r_defer"]
    assert w._pending_removes == set()


def test_remove_request_clears_async_trim():
    from mstar.worker.worker import RemoveRequest, MessageSource

    w = _bare_worker()
    w.is_tp_follower = False
    w._in_flight_rids = set()
    w._async_trim = {"r_stopped", "r_other"}
    w._sidecar_rids = set()
    w._sidecar_condemned = set()
    w._sidecar_client = None
    w.streaming_buffers = {}
    w._last_active = {}
    _noop = lambda *a, **k: None
    w.worker_graphs_manager = SimpleNamespace(
        per_request_info={}, remove_request=_noop
    )
    w.engine_manager = SimpleNamespace(
        remove_request=_noop, lru_tracked_nodes=lambda: []
    )
    w.tensor_manager = SimpleNamespace(cleanup_request=_noop)
    w.profile_info = SimpleNamespace(pop_request=_noop)

    w._remove_request(
        RemoveRequest(request_id="r_stopped", source=MessageSource.SELF)
    )
    # The removed rid can no longer have an overrun step in flight — its trim
    # entry is dropped; the unrelated one is left alone.
    assert w._async_trim == {"r_other"}


def test_remove_request_defers_a_rid_whose_postprocess_is_still_deferred():
    # Regression: a REMOVE_REQUEST arriving via _process_messages for a rid
    # that is still in _deferred_pp must be QUEUED, not applied — applying it
    # empties per_request_info before the owed deferred postprocess reads it
    # (KeyError in get_fwd_info). Caught on the first GPU smoke.
    from mstar.worker.worker import RemoveRequest, MessageSource

    w = _bare_worker()
    w.is_tp_follower = False
    w._in_flight_rids = set()  # NOT in the current spec batch
    w._deferred_pp = (_batch("r_defer"), object())
    torn_down = []
    w.engine_manager = SimpleNamespace(
        remove_request=lambda r: torn_down.append(r), lru_tracked_nodes=lambda: []
    )

    w._remove_request(
        RemoveRequest(request_id="r_defer", source=MessageSource.SELF)
    )
    # Queued, not torn down.
    assert "r_defer" in w._pending_removes
    assert torn_down == []


def test_in_flight_and_side_rids_still_held():
    w = _bare_worker()
    removed = []
    w._remove_request = lambda body: removed.append(body.request_id)
    w._side_in_flight_rids = {"r_side"}
    w._pending_removes = {"r_inflight", "r_side", "r_free"}
    w._apply_pending_removes_safe_to_drop(in_flight_rids={"r_inflight"})
    assert removed == ["r_free"]
    assert w._pending_removes == {"r_inflight", "r_side"}

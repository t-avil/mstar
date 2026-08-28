"""A data-worker failure reaches the client instead of its request timeout.

Only `_process_input` used to route to `_fail_request`; a raise anywhere else
in the loop hit the catch-all in `run`, which logs and moves on. The client
then waited out `timeout_seconds` and got a generic 500 that named neither the
failure nor its status.
"""
from types import SimpleNamespace

import pytest

from mstar.api_server.data_worker import PreprocessWorkerThread


class _Queue:
    def __init__(self, items=()):
        self._items = list(items)

    def empty(self):
        return not self._items

    def get(self):
        return self._items.pop(0)

    def put(self, item):
        self._items.append(item)


@pytest.fixture
def worker():
    w = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    w.out_queue = _Queue()
    w.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"req-1": ["edge"]},
        cleanup_request=lambda rid: None,
    )
    w.request_model_kwargs = {}
    w.tensor_uuid_to_metadata_per_request = {}
    return w


def _chunks(worker):
    return [c for c in worker.out_queue._items if c.modality == "error"]


def test_a_failure_reading_outputs_fails_that_request(worker, monkeypatch):
    monkeypatch.setattr(
        worker, "_read_ready_graph_edges",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("transport gone")),
    )
    assert worker._process_read_tensors() is True
    errors = _chunks(worker)
    assert len(errors) == 1
    assert errors[0].request_id == "req-1"
    assert b"transport gone" in errors[0].data
    assert errors[0].metadata["status"] == 500


def test_a_bad_value_is_reported_as_a_client_error(worker, monkeypatch):
    """`_fail_request` already maps ValueError to 400; keep that reachable."""
    monkeypatch.setattr(
        worker, "_read_ready_graph_edges",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad shape")),
    )
    worker._process_read_tensors()
    assert _chunks(worker)[0].metadata["status"] == 400


def test_one_failed_request_does_not_strand_the_others(worker, monkeypatch):
    worker.tensor_manager.get_ready_tensors = lambda: {"req-1": ["a"], "req-2": ["b"]}
    done = []

    def flaky(request_id, graph_edges):
        if request_id == "req-1":
            raise RuntimeError("boom")
        done.append(request_id)

    monkeypatch.setattr(worker, "_read_ready_graph_edges", flaky)
    worker._process_read_tensors()
    assert done == ["req-2"]
    assert [c.request_id for c in _chunks(worker)] == ["req-1"]


def test_a_worker_message_that_names_a_request_fails_that_request(worker, monkeypatch):
    message = SimpleNamespace(body=SimpleNamespace(request_id="req-9"))
    worker.communicator = SimpleNamespace(get_all_new_messages=lambda: [message])
    monkeypatch.setattr(
        worker, "_handle_worker_message",
        lambda m: (_ for _ in ()).throw(RuntimeError("deref failed")),
    )
    assert worker._process_messages() is True
    assert [c.request_id for c in _chunks(worker)] == ["req-9"]


def test_an_unattributable_message_is_logged_not_dropped_silently(worker, monkeypatch, caplog):
    message = SimpleNamespace(body=SimpleNamespace())
    worker.communicator = SimpleNamespace(get_all_new_messages=lambda: [message])
    monkeypatch.setattr(
        worker, "_handle_worker_message",
        lambda m: (_ for _ in ()).throw(RuntimeError("no rid here")),
    )
    worker._process_messages()
    assert _chunks(worker) == []
    assert "unattributable" in caplog.text.lower()

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the device-backed Talker text FIFO.

Covers the standalone :class:`PendingTextTensorQueue` (append / popleft / cursor
wrap / equivalence to a reference deque) and the :class:`StreamBuffer`
integration gated by ``MSTAR_TALKER_PENDING_QUEUE`` (off / on / parity must all
produce identical popped values).

CPU-safe: the FIFO is pure data-structure logic.  Any device-specific behavior
is guarded by ``skipif`` on CUDA availability.
"""

from __future__ import annotations

from collections import deque

import pytest
import torch

from mstar.streaming.chunk_policy import (
    FixedChunkPolicy,
    LeftContextChunkPolicy,
    SlidingWindowChunkPolicy,
)
from mstar.streaming.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
    pending_queue_mode,
)
from mstar.streaming.stream_buffer import StreamBuffer

HIDDEN = 8


def _row(val: float, hidden: int = HIDDEN) -> torch.Tensor:
    return torch.full((1, hidden), val, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PendingTextTensorQueue: core FIFO semantics
# ---------------------------------------------------------------------------


def test_append_popleft_order_matches_deque():
    pq = PendingTextTensorQueue()
    ref: deque = deque()
    for i in range(5):
        t = torch.arange(HIDDEN, dtype=torch.float32) + i
        pq.append(t)
        ref.append(t)
    assert len(pq) == len(ref) == 5
    while ref:
        got = pq.popleft()
        exp = ref.popleft()
        assert torch.equal(got, exp)
    assert len(pq) == 0
    assert not pq


def test_cursor_advances_without_copy():
    # Append a batch as one tensor; popleft must just walk the cursor and
    # return rows that are views into the SAME backing storage.
    pq = PendingTextTensorQueue.from_tensor(torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN))
    backing = pq.rows
    assert backing is not None
    base_ptr = backing.data_ptr()
    pq.popleft()
    assert pq.cursor == 1
    # Still the same backing tensor (no realloc on popleft).
    assert pq.rows is backing
    assert pq.rows.data_ptr() == base_ptr


def test_drains_to_empty_and_resets():
    pq = PendingTextTensorQueue()
    pq.append(_row(1.0))
    pq.popleft()
    # Fully drained -> backing dropped, cursor reset to take the fast path.
    assert pq.rows is None
    assert pq.cursor == 0
    assert len(pq) == 0


def test_append_after_partial_drain_compacts():
    pq = PendingTextTensorQueue()
    pq.append(_row(1.0))
    pq.append(_row(2.0))
    first = pq.popleft()  # advance cursor to 1, one row remaining
    assert torch.equal(first, _row(1.0).reshape(-1))
    pq.append(_row(3.0))  # compaction: remaining row + new row
    assert len(pq) == 2
    assert torch.equal(pq.popleft(), _row(2.0).reshape(-1))
    assert torch.equal(pq.popleft(), _row(3.0).reshape(-1))


def test_slice_api_preserves_leading_dim():
    pq = PendingTextTensorQueue()
    pq.append(_row(7.0))
    sl = pq.pop_slice(1)
    assert sl.shape == (1, HIDDEN)
    assert torch.equal(sl, _row(7.0))


def test_interleaved_append_popleft_vs_deque():
    pq = PendingTextTensorQueue()
    ref: deque = deque()
    rng = [3, -2, 1, 4, -3, -3, 2, -1, -1]  # +append n rows / -pop |n| rows
    counter = 0.0
    for op in rng:
        if op > 0:
            for _ in range(op):
                t = _row(counter)
                pq.append(t)
                ref.append(t)
                counter += 1.0
        else:
            for _ in range(-op):
                if not ref:
                    break
                got = pq.popleft()
                exp = ref.popleft()
                assert torch.equal(got, exp.reshape(-1))
    assert len(pq) == len(ref)


def test_coerce_variants():
    assert len(coerce_pending_text_queue(None)) == 0
    t = torch.arange(2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)
    assert len(coerce_pending_text_queue(t)) == 2
    assert len(coerce_pending_text_queue([_row(1.0), _row(2.0)])) == 2
    src = PendingTextTensorQueue.from_tensor(t)
    cp = coerce_pending_text_queue(src)
    assert len(cp) == 2 and cp is not src
    with pytest.raises(TypeError):
        coerce_pending_text_queue(object())


def test_empty_and_bad_rows():
    pq = PendingTextTensorQueue()
    pq.append(torch.empty(0, HIDDEN))  # no-op
    assert len(pq) == 0
    with pytest.raises(ValueError):
        pq.append(torch.empty(1, 0))
    with pytest.raises(ValueError):
        pq.append(torch.zeros(2, 2, 2))


# ---------------------------------------------------------------------------
# Env mode parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "off"), ("0", "off"), ("", "off"),
        ("1", "on"), ("on", "on"), ("TRUE", "on"),
        ("2", "parity"), ("parity", "parity"), ("shadow", "parity"),
    ],
)
def test_pending_queue_mode(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MSTAR_TALKER_PENDING_QUEUE", raising=False)
    else:
        monkeypatch.setenv("MSTAR_TALKER_PENDING_QUEUE", value)
    assert pending_queue_mode() == expected


# ---------------------------------------------------------------------------
# StreamBuffer integration: off / on / parity must agree
# ---------------------------------------------------------------------------


def _make_buffer(policy) -> StreamBuffer:
    return StreamBuffer(
        request_id="r0", edge_name="thinker_states",
        from_partition="thinker", policy=policy,
    )


def _drive(buf: StreamBuffer, rows: list[torch.Tensor]):
    """Feed rows one-by-one (in order) and pop every ready chunk."""
    popped = []
    for i, row in enumerate(rows):
        tid = f"t{i}"
        buf.pre_read_register(tid)
        buf.put(tid, row)
        while buf.has_chunk_ready():
            popped.append(buf.pop_chunk())
    buf.signal_done()
    while buf.has_chunk_ready():
        popped.append(buf.pop_chunk())
    return popped


def test_stream_buffer_off_on_parity_identical(monkeypatch):
    rows = [_row(float(i)) for i in range(6)]

    def run(mode: str):
        monkeypatch.setenv("MSTAR_TALKER_PENDING_QUEUE", mode)
        buf = _make_buffer(FixedChunkPolicy(chunk_size=1, continue_after_done=True))
        return _drive(buf, rows)

    base = run("0")
    base_tensors = [c.data["data"] for c in base if c.data["data"] is not None]
    assert len(base_tensors) == len(rows)
    for got, exp in zip(base_tensors, rows):
        assert torch.equal(got, exp)

    for mode in ("1", "parity"):
        out = run(mode)
        tensors = [c.data["data"] for c in out if c.data["data"] is not None]
        assert len(tensors) == len(base_tensors)
        for got, exp in zip(tensors, rows):
            assert got.shape == exp.shape
            assert torch.equal(got, exp)
        # is_final / start_offset bookkeeping must also match the list path.
        assert [c.start_offset for c in out] == [c.start_offset for c in base]
        assert [c.is_final for c in out] == [c.is_final for c in base]


def test_stream_buffer_continue_after_done_empty_chunks(monkeypatch):
    # continue_after_producer_done -> after draining, has_chunk_ready stays True
    # and pop yields empty ({"data": None}) chunks, never marked final.
    for mode in ("0", "1", "parity"):
        monkeypatch.setenv("MSTAR_TALKER_PENDING_QUEUE", mode)
        buf = _make_buffer(FixedChunkPolicy(chunk_size=1, continue_after_done=True))
        buf.pre_read_register("a")
        buf.put("a", _row(1.0))
        buf.signal_done()
        assert buf.has_chunk_ready()
        c0 = buf.pop_chunk()
        assert torch.equal(c0.data["data"], _row(1.0))
        assert c0.is_final is False
        # Drained but continue_after_done -> empty chunk available.
        assert buf.has_chunk_ready()
        c1 = buf.pop_chunk()
        assert c1.data["data"] is None
        assert c1.is_final is False


@pytest.mark.parametrize("policy", [
    SlidingWindowChunkPolicy(window=4, stride=2),
    LeftContextChunkPolicy(chunk=3, left_context=1),
    FixedChunkPolicy(chunk_size=25),
])
def test_non_single_row_policies_not_eligible(monkeypatch, policy):
    # Multi-item / overlapping policies keep the list path even when enabled.
    monkeypatch.setenv("MSTAR_TALKER_PENDING_QUEUE", "1")
    buf = _make_buffer(policy)
    assert buf._use_pending_queue is False
    assert policy.is_single_row_fifo() is False


def test_single_row_policy_eligible(monkeypatch):
    monkeypatch.setenv("MSTAR_TALKER_PENDING_QUEUE", "1")
    buf = _make_buffer(FixedChunkPolicy(chunk_size=1))
    assert buf._use_pending_queue is True
    assert FixedChunkPolicy(chunk_size=1).is_single_row_fifo() is True


# ---------------------------------------------------------------------------
# Device-specific: backing tensor lives on the requested device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_zero_copy_popleft():
    rows = torch.arange(4 * HIDDEN, dtype=torch.float32, device="cuda").reshape(4, HIDDEN)
    pq = PendingTextTensorQueue.from_tensor(rows)
    assert pq.rows.is_cuda
    ptr = pq.rows.data_ptr()
    out = pq.pop_slice(1)
    assert out.is_cuda
    assert out.data_ptr() == ptr  # view, no device copy

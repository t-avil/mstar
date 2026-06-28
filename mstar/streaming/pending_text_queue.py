# SPDX-License-Identifier: Apache-2.0
"""Device-backed FIFO for the Qwen3-Omni Talker text/hidden feed.

Port of SGLang-Omni's ``PendingTextTensorQueue``
(``sglang_omni/models/qwen3_omni/pending_text_queue.py``).

Motivation
----------
The Talker consumes exactly one future-text row (a projected Thinker hidden
state, shape ``[1, thinker_hidden]``) per decode step.  M*'s streaming layer
(:class:`mstar.streaming.stream_buffer.StreamBuffer`) buffers those rows in a
Python ``list`` and advances by re-slicing the list every pop
(``self._buffer = self._buffer[stride:]``), which churns Python list objects
once per decode step on the Talker's hottest loop.

This FIFO keeps the buffered rows as a *single* device tensor plus an integer
cursor: ``popleft`` advances the cursor (no copy, no list realloc) and
``append`` concatenates onto the device tensor.  In the steady B=1 case where
the consumer keeps up with the producer, the queue drains to empty between
steps, so ``append`` hits the fast ``len == 0`` path (a reference assignment,
no concat) and ``popleft`` is a cursor bump — eliminating the per-step Python
object churn entirely.

This module is data-structure-only and CPU-safe; the optimization is gated at
the call site (:class:`StreamBuffer`) behind ``MSTAR_TALKER_PENDING_QUEUE``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------

# Mode values:
#   off    (default) : current list-backed behavior, byte-identical.
#   on / 1 / true    : device-backed single-tensor FIFO for eligible edges.
#   parity / 2       : run BOTH the list path and the FIFO in lockstep and
#                      assert popped values are identical (runtime parity gate,
#                      returns the list result so behavior matches `off`).
_ON_VALUES = ("1", "on", "true", "yes")
_PARITY_VALUES = ("2", "parity", "shadow")


def pending_queue_mode() -> str:
    """Return one of ``"off"``, ``"on"``, ``"parity"`` from the environment."""
    raw = os.environ.get("MSTAR_TALKER_PENDING_QUEUE", "0").strip().lower()
    if raw in _PARITY_VALUES:
        return "parity"
    if raw in _ON_VALUES:
        return "on"
    return "off"


def _as_rows(tensor: torch.Tensor) -> torch.Tensor | None:
    """Normalize an appended item to a 2D ``[rows, hidden]`` tensor.

    A 1D ``[hidden]`` tensor becomes ``[1, hidden]``.  An empty tensor maps to
    ``None`` (nothing to append).  Anything else is rejected.
    """
    try:
        tensor = tensor.detach()
    except AttributeError as exc:
        raise TypeError("pending text rows must be tensors") from exc
    if tensor.dim() == 1:
        if tensor.shape[0] == 0:
            return None
        return tensor.reshape(1, -1)
    if tensor.dim() == 2:
        if tensor.shape[0] == 0:
            return None
        if tensor.shape[1] == 0:
            raise ValueError("pending text rows must have a non-empty hidden dimension")
        return tensor
    raise ValueError("pending text rows must be a 1D row tensor or a 2D row batch")


@dataclass
class PendingTextTensorQueue:
    """FIFO queue backed by one tensor plus a cursor.

    The Talker consumes one future text row per decode step.  Keeping those
    rows as a single device tensor avoids row-wise copies and Python list/deque
    object churn while preserving the small queue API used by the runner.

    Two complementary read APIs are exposed:

    * SGLang-compatible row API — :meth:`popleft` / :meth:`__getitem__` /
      :meth:`__iter__` return a *1D* ``[hidden]`` row (squeezed).
    * Slice API — :meth:`front_slice` / :meth:`pop_slice` return a *2D*
      ``[n, hidden]`` view, preserving the leading dim.  :class:`StreamBuffer`
      uses this so a single-row pop yields a ``[1, hidden]`` tensor identical to
      the list path's ``items[0]``.
    """

    rows: torch.Tensor | None = None
    cursor: int = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "PendingTextTensorQueue":
        queue = cls()
        queue.append_rows(tensor)
        return queue

    def copy(self) -> "PendingTextTensorQueue":
        return type(self)(rows=self.rows, cursor=self.cursor)

    # -- size / iteration --------------------------------------------------

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        if self.rows is None:
            return 0
        return max(0, int(self.rows.shape[0]) - self.cursor)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.rows is None:
            return
        for idx in range(self.cursor, int(self.rows.shape[0])):
            yield self.rows[idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not isinstance(idx, int):
            raise TypeError("PendingTextTensorQueue indices must be integers")
        if idx < 0:
            idx += len(self)
        absolute_idx = self.cursor + idx
        if self.rows is None or idx < 0 or absolute_idx >= int(self.rows.shape[0]):
            raise IndexError(idx)
        return self.rows[absolute_idx]

    # -- SGLang-compatible row pop (1D) ------------------------------------

    def popleft(self) -> torch.Tensor:
        row = self[0]
        self._advance(1)
        return row

    # -- slice API (2D, dim-preserving) ------------------------------------

    def front_slice(self, n: int) -> torch.Tensor:
        """Return the next ``n`` rows as a ``[n, hidden]`` view (no advance)."""
        if n <= 0:
            raise ValueError("front_slice size must be positive")
        if len(self) < n:
            raise IndexError(n)
        start = self.cursor
        return self.rows[start : start + n]

    def pop_slice(self, n: int) -> torch.Tensor:
        """Return the next ``n`` rows as ``[n, hidden]`` and advance the cursor."""
        view = self.front_slice(n)
        self._advance(n)
        return view

    def _advance(self, n: int) -> None:
        self.cursor += n
        # Drop the backing tensor the moment the queue is fully drained so a
        # subsequent append re-takes the zero-copy fast path and stale device
        # memory is released.
        if self.rows is not None and self.cursor >= int(self.rows.shape[0]):
            self.rows = None
            self.cursor = 0

    # -- append ------------------------------------------------------------

    def append(self, row: torch.Tensor) -> None:
        self.append_rows(row)

    def append_rows(self, rows: torch.Tensor) -> None:
        rows = _as_rows(rows)
        if rows is None:
            return
        if self.rows is None or len(self) == 0:
            # Fast path: empty queue -> adopt the incoming tensor by reference.
            self.rows = rows
            self.cursor = 0
            return

        remaining = self.rows[self.cursor :]
        rows = rows.to(device=remaining.device, dtype=remaining.dtype)
        self.rows = torch.cat([remaining, rows], dim=0)
        self.cursor = 0


def coerce_pending_text_queue(value: object) -> PendingTextTensorQueue:
    if value is None:
        return PendingTextTensorQueue()
    if isinstance(value, PendingTextTensorQueue):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return PendingTextTensorQueue.from_tensor(value)
    if isinstance(value, Iterable):
        queue = PendingTextTensorQueue()
        for row in value:
            queue.append(row)
        return queue
    raise TypeError(
        "pending text queue must be None, a tensor, a PendingTextTensorQueue, "
        "or an iterable of tensors"
    )

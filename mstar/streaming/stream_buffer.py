from collections import deque
from dataclasses import dataclass, field

import torch

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy
from mstar.streaming.pending_text_queue import (
    PendingTextTensorQueue,
    pending_queue_mode,
)


@dataclass
class StreamChunk:
    """A chunk of data popped from a StreamBuffer."""
    data: dict[str, torch.Tensor | None]
    chunk_index: int
    start_offset: int = 0  # global position of the first item in this chunk
    is_final: bool = False


@dataclass
class StreamBuffer:
    """Per-request, per-edge buffer on the CONSUMER worker.

    Tensors arrive one-by-one via normal RDMA routing.
    The buffer accumulates them and applies a ChunkPolicy to decide
    when the consuming node has enough data to proceed.

    For sliding-window policies the buffer keeps old items so that
    pop_chunk can return the full window while only advancing by stride.
    """
    request_id: str
    edge_name: str
    from_partition: str
    policy: ChunkPolicy

    _waiting_graph_edges: deque = field(default_factory=deque)

    _buffer: list = field(default_factory=list)
    _tensor_ids_in_order: deque = field(default_factory=deque)
    _id_to_tensor: dict = field(default_factory=dict)
    _consumed: int = 0
    _chunks_popped: int = 0
    producer_done: bool = False

    _num_tensors_registered = 0
    _num_buffer_writes = 0

    # Device-backed single-tensor FIFO for the Talker text/hidden feed.
    # See mstar/streaming/pending_text_queue.py and DESIGN_pending_queue.md.
    #   _pending_mode in {"off", "on", "parity"} (from MSTAR_TALKER_PENDING_QUEUE)
    #   _use_pending_queue: True only when enabled AND the policy is a
    #       single-row FIFO (window == stride == 1, e.g. thinker_states).
    #   _parity: shadow mode -- maintain both paths and assert equality.
    _pending_mode: str = "off"
    _use_pending_queue: bool = False
    _parity: bool = False
    _pending_queue: PendingTextTensorQueue | None = None

    def __post_init__(self) -> None:
        self._pending_mode = pending_queue_mode()
        if self._pending_mode != "off" and self.policy.is_single_row_fifo():
            self._use_pending_queue = True
            self._parity = self._pending_mode == "parity"
            self._pending_queue = PendingTextTensorQueue()

    def _buffer_len(self) -> int:
        """Number of ready (HOL-ordered) items available to pop."""
        if self._use_pending_queue and not self._parity:
            return len(self._pending_queue)
        # `off` and `parity` both use the list as the length source of truth;
        # in `parity` the pending queue is a shadow validated on every pop.
        return len(self._buffer)

    def pre_read_register(self, tensor_id: str):
        self._num_tensors_registered += 1
        self._tensor_ids_in_order.append(tensor_id)

    def put(self, tensor_id: str, item: torch.Tensor) -> None:
        """Called when a tensor arrives via normal RDMA routing."""
        self._id_to_tensor[tensor_id] = item

    def _update_buffer(self):
        while len(self._tensor_ids_in_order) > 0:
            tensor_id = self._tensor_ids_in_order[0]
            if tensor_id not in self._id_to_tensor:
                return
            self._tensor_ids_in_order.popleft()
            item = self._id_to_tensor[tensor_id]
            if self._use_pending_queue:
                # HOL ordering is already enforced by this loop, so appending
                # in arrival order is safe; the device FIFO replaces the list's
                # per-step object churn.
                self._pending_queue.append(item)
                if self._parity:
                    self._buffer.append(item)
            else:
                self._buffer.append(item)
            self._num_buffer_writes += 1
            del self._id_to_tensor[tensor_id]

    def signal_done(self) -> None:
        """Producer signals no more items will arrive."""
        self.producer_done = True

    def _producer_done_and_all_read(self) -> bool:
        return self.producer_done and self._num_buffer_writes >= self._num_tensors_registered

    def pop_waiting_edge(self) -> GraphEdge | None:
        if len(self._waiting_graph_edges) > 0:
            return self._waiting_graph_edges.popleft()

    def has_chunk_ready(self) -> bool:
        self._update_buffer()
        buf_len = self._buffer_len()
        if self._producer_done_and_all_read() and buf_len > 0:
            return True
        # When continue_after_producer_done is set, keep producing empty
        # chunks after the producer finishes and all items are consumed.
        # This allows the consumer to keep running (e.g., Talker continues
        # generating codec tokens after the Thinker hits text EOS).
        if (self._producer_done_and_all_read()
                and buf_len == 0
                and self.policy.continue_after_producer_done()):
            return True
        return self.policy.is_ready(buf_len)

    def pop_chunk(self) -> StreamChunk:
        """Pop the next chunk. Only call when has_chunk_ready() is True.

        For sliding-window: returns `window_size` items, advances by
        `stride` items, discards items that have fallen out of the window.
        start_offset is the global position of the first item in the chunk.
        """
        self._update_buffer()

        # Fast path: device-backed single-tensor FIFO (one item per chunk).
        if self._use_pending_queue and not self._parity:
            return self._pop_chunk_pending()

        buf_len = len(self._buffer)
        window = self.policy.window_size()
        offset = self._consumed  # global position of buffer[0]

        if self._producer_done_and_all_read() and not self.policy.is_ready(buf_len):
            # Flush remainder — return whatever is left (may be empty)
            items = list(self._buffer)
            self._buffer.clear()
            self._consumed += len(items)
            stride = len(items)
        else:
            stride = self.policy.next_chunk_size(buf_len)
            # Return the first `window` items (overlapping sliding window)
            items = self._buffer[:window]
            # Advance by stride — discard items that fell out of the window
            self._buffer = self._buffer[stride:]
            self._consumed += stride
        self.policy.register_chunk(stride)

        is_final = self._producer_done_and_all_read() and len(self._buffer) == 0
        # When continue_after_producer_done, never mark as final — the
        # consumer decides when it's done via its own model logic.
        if self.policy.continue_after_producer_done():
            is_final = False

        data = self._collate(items)
        if self._parity:
            # Shadow-pop the device FIFO and assert byte-identical values, then
            # return the list result so behavior matches `off` exactly.
            self._assert_pending_parity(data, stride)

        chunk = StreamChunk(
            data=data,
            chunk_index=self._chunks_popped,
            start_offset=offset,
            is_final=is_final,
        )
        self._chunks_popped += 1
        return chunk

    def _pop_chunk_pending(self) -> StreamChunk:
        """``pop_chunk`` for the device-backed single-row FIFO (window==stride==1).

        Mirrors the list path for a one-item-per-chunk policy:
          * empty flush after producer-done -> ``{"data": None}``
          * otherwise pop exactly one ``[1, hidden]`` row (== list ``items[0]``).
        """
        buf_len = len(self._pending_queue)
        offset = self._consumed

        if self._producer_done_and_all_read() and not self.policy.is_ready(buf_len):
            # buf_len == 0 here (chunk_size == 1): nothing left to flush.
            data: dict[str, torch.Tensor | None] = {"data": None}
            stride = 0
        else:
            row = self._pending_queue.pop_slice(1)  # [1, hidden]
            self._consumed += 1
            stride = 1
            data = {"data": row}
        self.policy.register_chunk(stride)

        is_final = self._producer_done_and_all_read() and len(self._pending_queue) == 0
        if self.policy.continue_after_producer_done():
            is_final = False

        chunk = StreamChunk(
            data=data,
            chunk_index=self._chunks_popped,
            start_offset=offset,
            is_final=is_final,
        )
        self._chunks_popped += 1
        return chunk

    def _assert_pending_parity(
        self, list_data: dict[str, torch.Tensor | None], stride: int
    ) -> None:
        """Parity gate: pop the shadow FIFO and assert it matches the list path."""
        ref = list_data.get("data")
        if stride == 0:
            # Empty flush: the shadow FIFO must also be empty.
            if len(self._pending_queue) != 0:
                raise AssertionError(
                    "PendingTextTensorQueue parity: expected empty FIFO on flush, "
                    f"got len={len(self._pending_queue)}"
                )
            return
        got = self._pending_queue.pop_slice(stride)
        if ref is None:
            raise AssertionError(
                "PendingTextTensorQueue parity: list path produced no tensor"
            )
        if got.shape != ref.shape or not torch.equal(got, ref.to(got.device)):
            raise AssertionError(
                "PendingTextTensorQueue parity mismatch: "
                f"fifo.shape={tuple(got.shape)} list.shape={tuple(ref.shape)}"
            )

    def store_uningested_edge(self, edge: GraphEdge):
        self._waiting_graph_edges.append(edge)

    def _collate(self, items: list) -> dict[str, torch.Tensor | None]:
        if not items:
            return {"data": None}
        if isinstance(items[0], torch.Tensor):
            if len(items) == 1:
                return {"data": items[0]}
            return {"data": torch.stack(items)}
        return {"data": items}

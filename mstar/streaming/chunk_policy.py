from abc import ABC, abstractmethod


class ChunkPolicy(ABC):
    """Determines when a StreamBuffer has enough items for the consumer node."""
    def __init__(self):
        self.first_chunk_read = False
        self.items_consumed = 0

    def register_chunk(self, chunk_size: int):
        self.first_chunk_read = True
        self.items_consumed += chunk_size

    @abstractmethod
    def is_ready(self, buffer_len: int) -> bool:
        """Return True if the buffer has enough items for a chunk."""
        ...

    @abstractmethod
    def next_chunk_size(self, buffer_len: int) -> int:
        """Return the number of items to consume for the next chunk.

        Only called when is_ready() returns True.
        For sliding-window policies this is the stride, not the window.
        """
        ...

    @abstractmethod
    def window_size(self) -> int:
        """Return the full window of items to include in the chunk.

        For non-overlapping policies, equals next_chunk_size.
        For sliding-window policies, this is larger than the stride —
        the buffer retains older items so the chunk contains the full window.
        """
        ...

    def continue_after_producer_done(self) -> bool:
        """Whether the buffer should keep producing (empty) chunks after the
        producer signals done and all buffered items have been consumed.

        Default ``False``: partition-done is propagated to the conductor after
        the last item is flushed.

        Set to ``True`` for connections where the consumer must keep running
        after the producer finishes (e.g., Thinker→Talker: the Talker
        continues generating codec tokens after the Thinker hits text EOS).
        In this case the buffer produces empty chunks (``_collate([])`` →
        ``{"data": None}``), and the consumer's partition-done is determined
        by its own model logic, not by the StreamBuffer.
        """
        return False

    def observe_batch_size(self, batch_size: int) -> None:
        """Hint: how many requests are concurrently feeding this consumer node.

        Called by the worker poll loop before reading the buffer. The default
        is a no-op -- fixed policies ignore it and behave byte-identically
        regardless of co-processing concurrency. Batch-adaptive policies
        override this to size their chunks to the current batch (small chunk
        for low B=1 latency, large chunk for high throughput once several
        requests batch together).
        """
        return None


class SlidingWindowChunkPolicy(ChunkPolicy):
    """Fixed-size sliding window that advances by a stride.

    Each pop_chunk returns `window` items and advances the consumed
    pointer by `stride`. Old items before the window are discarded.

    Example (Orpheus SNAC): window=28 tokens (4 frames), stride=7 (1 frame).
    """

    def __init__(self, window: int, stride: int):
        super().__init__()
        self._window = window
        self._stride = stride

    def is_ready(self, buffer_len: int) -> bool:
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._stride

    def window_size(self) -> int:
        return self._window


class LeftContextChunkPolicy(ChunkPolicy):
    """Chunk policy for streaming vocoders with left-context overlap.

    Matches HuggingFace's ``Qwen3OmniMoeCode2Wav.chunked_decode`` pattern:

        Iter 0: codes[0 : chunk]                → emit all (no context)
        Iter 1: codes[chunk-ctx : 2*chunk]       → trim first ctx, emit rest
        Iter 2: codes[2*chunk-ctx : 3*chunk]     → trim first ctx, emit rest

    The first pop returns ``chunk`` items (no context).  Subsequent pops
    return ``chunk + left_context`` items, where the leading ``left_context``
    items OVERLAP with the tail of the previous chunk.  This overlap allows
    the causal ConvNet vocoder to "warm up" its internal state on frames
    it has already processed, ensuring a smooth transition at chunk
    boundaries.

    The key invariant: the first pop advances by ``chunk - left_context``
    (not ``chunk``), so the last ``left_context`` items of the first chunk
    remain in the buffer as overlap for the second pop.  All subsequent
    pops advance by ``chunk``.
    """

    def __init__(self, chunk: int, left_context: int):
        super().__init__()
        self._chunk = chunk
        self._left_context = left_context
        self._window = chunk + left_context

    def is_ready(self, buffer_len: int) -> bool:
        if not self.first_chunk_read:
            return buffer_len >= self._chunk
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        # First pop: advance by (chunk - left_context) so the tail of the
        # first chunk stays in the buffer as overlap for the next pop.
        if not self.first_chunk_read:
            return self._chunk - self._left_context
        return self._chunk

    def window_size(self) -> int:
        if not self.first_chunk_read:
            return self._chunk
        return self._window


class BatchAdaptiveLeftContextChunkPolicy(LeftContextChunkPolicy):
    """Left-context chunk policy whose chunk size adapts to the vocoder batch.

    Same overlap mechanics as :class:`LeftContextChunkPolicy` (the leading
    ``left_context`` frames of every non-first chunk overlap the previous
    chunk's tail to warm up the causal ConvNet vocoder), but the number of
    *fresh* frames per chunk switches between a small "latency" value and a
    large "throughput" value based on how many requests are currently
    co-vocoding:

        batch_size <  threshold  -> ``latency_chunk``   (low B=1 latency)
        batch_size >= threshold  -> ``large_chunk``     (high GPU utilization)

    M* already batches the Code2Wav vocoder across requests, so once several
    requests stream codec tokens at once a larger chunk packs more frames into
    each batched forward and amortizes launch/overhead -- compounding with the
    cross-request batching for throughput on the audio-output paths.

    Safety: ``left_context`` is held FIXED while the chunk varies. This keeps
    the overlap accounting seamless (every pop leaves exactly ``left_context``
    frames in the buffer as warm-up for the next pop, independent of chunk
    size) AND guarantees the first-pop stride ``chunk - left_context`` is
    never negative, because every selectable chunk is clamped up to
    ``left_context`` via ``max(chunk, left_context)``. (A chunk below the
    left context produced a negative pop stride and corrupted audio; see the
    project FINDINGS. The clamp makes that unrepresentable.)

    The chosen chunk is *latched* the moment :meth:`is_ready` first returns
    True, so the immediately-following ``window_size`` / ``next_chunk_size`` /
    ``register_chunk`` calls for that one pop are mutually consistent even if
    the observed batch size changes mid-poll. The latch is released after the
    chunk is registered so the next chunk re-evaluates the batch size.
    """

    def __init__(
        self,
        latency_chunk: int,
        large_chunk: int,
        left_context: int,
        threshold: int,
    ):
        # Initialize the base with the latency chunk so all base fields exist;
        # the adaptive overrides below pick the effective chunk per pop.
        super().__init__(chunk=max(latency_chunk, left_context),
                         left_context=left_context)
        self._latency_chunk = max(latency_chunk, left_context)
        self._large_chunk = max(large_chunk, left_context)
        self._threshold = max(threshold, 1)
        self._batch_size = 1
        self._pending_chunk: int | None = None

    def observe_batch_size(self, batch_size: int) -> None:
        self._batch_size = max(int(batch_size), 1)

    def _choose_chunk(self) -> int:
        # Already clamped >= left_context in __init__, so the first-pop stride
        # (chunk - left_context) is guaranteed non-negative.
        if self._batch_size >= self._threshold:
            return self._large_chunk
        return self._latency_chunk

    def _effective_chunk(self) -> int:
        # Use the latched choice when one is held for the in-progress pop.
        if self._pending_chunk is not None:
            return self._pending_chunk
        return self._choose_chunk()

    def is_ready(self, buffer_len: int) -> bool:
        chunk = self._effective_chunk()
        if not self.first_chunk_read:
            ready = buffer_len >= chunk
        else:
            ready = buffer_len >= (chunk + self._left_context)
        if ready:
            self._pending_chunk = chunk  # latch for the upcoming pop
        return ready

    def next_chunk_size(self, buffer_len: int) -> int:
        chunk = self._effective_chunk()
        if not self.first_chunk_read:
            return chunk - self._left_context
        return chunk

    def window_size(self) -> int:
        chunk = self._effective_chunk()
        if not self.first_chunk_read:
            return chunk
        return chunk + self._left_context

    def register_chunk(self, chunk_size: int):
        super().register_chunk(chunk_size)
        self._pending_chunk = None  # release latch; re-evaluate next pop


class FixedChunkPolicy(ChunkPolicy):
    """Release non-overlapping chunks of fixed size.

    Each pop_chunk returns exactly `chunk_size` items and advances by
    `chunk_size`. No overlap, no sliding window.

    Args:
        chunk_size: number of items per chunk.
        continue_after_done: if True, keep producing empty chunks after
            the producer finishes and all buffered items are consumed.
    """

    def __init__(self, chunk_size: int, continue_after_done: bool = False):
        super().__init__()
        self._chunk_size = chunk_size
        self._continue_after_done = continue_after_done

    def is_ready(self, buffer_len) -> bool:
        return buffer_len >= self._chunk_size

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._chunk_size

    def window_size(self) -> int:
        return self._chunk_size

    def continue_after_producer_done(self) -> bool:
        return self._continue_after_done

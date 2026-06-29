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


class AdaptiveChunkPolicy(ChunkPolicy):
    """Left-context chunk policy whose chunk size adapts to batch level.

    Behaves identically to ``LeftContextChunkPolicy`` but dynamically
    scales the chunk size based on how many requests the worker is
    currently serving.  At low batch (B=1-3) the chunk stays at
    ``min_chunk`` to preserve pipeline overlap and low latency.  At
    higher batch the chunk grows (fewer, larger vocoder launches →
    better GPU utilization) up to ``max_chunk``.

    Scaling rule::

        effective_chunk = min(min_chunk * max(1, batch_level // 2), max_chunk)

        B=1-3  → chunk = min_chunk          (25)
        B=4-5  → chunk = min_chunk * 2      (50)
        B=6-7  → chunk = min_chunk * 3      (75)
        B≥8    → chunk = max_chunk          (100)

    ``left_context`` stays fixed: the vocoder's receptive field doesn't
    change with batch size.

    Thread safety: ``set_batch_level`` is called from the worker's poll
    loop on the same thread that calls ``is_ready``/``next_chunk_size``,
    so no lock is needed.
    """

    def __init__(
        self,
        min_chunk: int = 25,
        max_chunk: int = 100,
        left_context: int = 25,
    ):
        super().__init__()
        self._min_chunk = min_chunk
        self._max_chunk = max_chunk
        self._left_context = left_context
        # Start at the minimum (conservative for B=1)
        self._chunk = min_chunk
        self._window = min_chunk + left_context

    # -- batch-level knob ---------------------------------------------------

    def set_batch_level(self, n: int) -> None:
        """Update the effective chunk size based on current batch count *n*."""
        effective = min(self._min_chunk * max(1, n // 2), self._max_chunk)
        self._chunk = effective
        self._window = effective + self._left_context

    # -- ChunkPolicy interface (mirrors LeftContextChunkPolicy) -------------

    def is_ready(self, buffer_len: int) -> bool:
        if not self.first_chunk_read:
            return buffer_len >= self._chunk
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        if not self.first_chunk_read:
            return self._chunk - self._left_context
        return self._chunk

    def window_size(self) -> int:
        if not self.first_chunk_read:
            return self._chunk
        return self._window

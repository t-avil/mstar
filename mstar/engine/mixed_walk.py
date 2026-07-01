"""Varlen tensor layout for the mixed prefill+decode ("piggyback") step.

This is the forward-build core for MSTAR_MIXED_WALK (continuous batching). It is
deliberately a PURE, device-agnostic builder: given the per-request decode/
prefill bookkeeping the scheduler + KV manager already hold, it computes the
flat varlen layout a single mixed forward needs — token order, ``qo_indptr``,
per-request KV ``seq_lens``, M-RoPE positions, and the KV-append offsets. It
allocates only small CPU/GPU index tensors and does no model work, so it is
fully unit-testable without a GPU.

The actual graph capture/replay that consumes this layout lives (stubbed) in
``CudaGraphRunner.run_mixed``. See DESIGN_mixed_walk.md for the end-to-end
contract.

Layout convention (matches vLLM v1's SchedulerOutput -> single varlen forward):
running decodes come FIRST (one query token each), then each prefill request's
chunk in order. So for D decodes and prefill chunks [P0, P1, ...]:

    flat tokens:  [d0 d1 ... d(D-1) | p0_0 ... p0_(P0-1) | p1_0 ... ]
    qo_indptr:    [0, 1, 2, ..., D, D+P0, D+P0+P1, ...]

``qo_indptr`` is the per-request *query* offset array FlashInfer's varlen
prefill wrapper expects; the matching ``kv_indptr`` (paged) is built by the
cache manager from ``kv_seq_lens`` at plan time and is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MixedWalkLayout:
    """Flat varlen layout for one mixed prefill+decode forward.

    All index tensors are 1-D ``int32`` on ``device`` unless noted.
    """
    num_decode: int
    num_prefill_reqs: int
    num_tokens: int                 # == num_decode + sum(prefill_lengths)
    batch_size: int                 # == num_decode + num_prefill_reqs

    # FlashInfer varlen query offsets, shape [batch_size + 1].
    qo_indptr: torch.Tensor
    # Per-request total KV length AFTER this step (for kv_indptr planning).
    # Order: decode requests, then prefill requests. shape [batch_size].
    kv_seq_lens: torch.Tensor
    # Per-request query length this step (decode=1, prefill=chunk len).
    qo_seq_lens: torch.Tensor

    # M-RoPE positions, shape [3, num_tokens] (temporal, height, width rows).
    # For text the three rows are identical; audio/vision callers can override
    # per request via ``prefill_mrope_fn``.
    mrope_positions: torch.Tensor
    # Flat 1-D positions [num_tokens] — convenience for non-MRoPE reference /
    # tests; equals mrope_positions[0].
    positions: torch.Tensor

    # Flat-tensor slice [start, end) for each request, order decode then
    # prefill. Lets the replay copy each request's input embeddings into the
    # right rows and scatter sampled logits back. shape [batch_size, 2].
    request_token_spans: torch.Tensor


def build_mixed_varlen_layout(
    *,
    decode_kv_lens: list[int],
    decode_positions: list[int],
    prefill_lengths: list[int],
    prefill_kv_starts: list[int] | None = None,
    prefill_pos_starts: list[int] | None = None,
    prefill_chunk_cap: int | None = None,
    prefill_mrope_fn=None,
    device: torch.device | str = "cpu",
) -> MixedWalkLayout:
    """Build the flat varlen layout for a mixed prefill+decode step.

    Args:
        decode_kv_lens: existing KV length per decode request (tokens already in
            cache BEFORE this step). The decode appends 1 token, so its KV
            length after the step is ``decode_kv_lens[i] + 1``.
        decode_positions: position id of the single new token for each decode
            request (typically ``decode_kv_lens[i]`` for plain text, but passed
            explicitly because M-RoPE position ids can diverge from raw token
            counts after multimodal spans).
        prefill_lengths: chunk token count for each piggybacking prefill
            request (already the count to process THIS step). Each is capped at
            ``prefill_chunk_cap`` if given (defensive — the scheduler already
            caps, this guarantees finite capture shapes).
        prefill_kv_starts: KV length already present for each prefill request
            (0 for a fresh request; >0 if a previous chunk was processed).
            Defaults to all-zeros.
        prefill_pos_starts: starting position id for each prefill chunk.
            Defaults to ``prefill_kv_starts`` (text default).
        prefill_chunk_cap: optional hard cap applied to each prefill length.
        prefill_mrope_fn: optional ``fn(req_idx, pos_start, length) ->
            torch.Tensor[3, length]`` producing the 3-row M-RoPE positions for a
            prefill request (audio/vision). Defaults to text (all rows =
            ``arange(pos_start, pos_start + length)``).
        device: device for the returned index tensors.

    Returns:
        A ``MixedWalkLayout``.
    """
    num_decode = len(decode_kv_lens)
    if len(decode_positions) != num_decode:
        raise ValueError(
            f"decode_positions ({len(decode_positions)}) must match "
            f"decode_kv_lens ({num_decode})"
        )
    num_prefill = len(prefill_lengths)
    if prefill_kv_starts is None:
        prefill_kv_starts = [0] * num_prefill
    if prefill_pos_starts is None:
        prefill_pos_starts = list(prefill_kv_starts)
    if not (len(prefill_kv_starts) == len(prefill_pos_starts) == num_prefill):
        raise ValueError("prefill_* lists must all have the same length")

    capped_lengths = []
    for p in prefill_lengths:
        if p <= 0:
            raise ValueError(f"prefill chunk length must be positive, got {p}")
        if prefill_chunk_cap is not None and p > prefill_chunk_cap:
            p = prefill_chunk_cap
        capped_lengths.append(p)

    batch_size = num_decode + num_prefill
    num_tokens = num_decode + sum(capped_lengths)

    # --- qo_seq_lens / qo_indptr (query offsets) ---
    qo_seq_lens = [1] * num_decode + list(capped_lengths)
    qo_indptr = [0]
    for q in qo_seq_lens:
        qo_indptr.append(qo_indptr[-1] + q)

    # --- kv_seq_lens (post-step total KV per request) ---
    kv_seq_lens = [kv + 1 for kv in decode_kv_lens] + [
        prefill_kv_starts[i] + capped_lengths[i] for i in range(num_prefill)
    ]

    # --- positions / mrope_positions ---
    mrope = torch.zeros((3, num_tokens), dtype=torch.long, device=device)
    # decode rows: each decode token's 3 mrope rows all equal its scalar pos
    for i, pos in enumerate(decode_positions):
        mrope[:, i] = pos
    # prefill rows
    cursor = num_decode
    request_token_spans = [(i, i + 1) for i in range(num_decode)]
    for r in range(num_prefill):
        length = capped_lengths[r]
        pos_start = prefill_pos_starts[r]
        if prefill_mrope_fn is not None:
            block = prefill_mrope_fn(r, pos_start, length)
            block = torch.as_tensor(block, dtype=torch.long, device=device)
            if tuple(block.shape) != (3, length):
                raise ValueError(
                    f"prefill_mrope_fn returned shape {tuple(block.shape)}, "
                    f"expected (3, {length})"
                )
            mrope[:, cursor:cursor + length] = block
        else:
            rng = torch.arange(
                pos_start, pos_start + length, dtype=torch.long, device=device
            )
            mrope[0, cursor:cursor + length] = rng
            mrope[1, cursor:cursor + length] = rng
            mrope[2, cursor:cursor + length] = rng
        request_token_spans.append((cursor, cursor + length))
        cursor += length

    assert cursor == num_tokens, (cursor, num_tokens)

    int32 = torch.int32
    return MixedWalkLayout(
        num_decode=num_decode,
        num_prefill_reqs=num_prefill,
        num_tokens=num_tokens,
        batch_size=batch_size,
        qo_indptr=torch.tensor(qo_indptr, dtype=int32, device=device),
        kv_seq_lens=torch.tensor(kv_seq_lens, dtype=int32, device=device),
        qo_seq_lens=torch.tensor(qo_seq_lens, dtype=int32, device=device),
        mrope_positions=mrope,
        positions=mrope[0].clone(),
        request_token_spans=torch.tensor(
            request_token_spans, dtype=int32, device=device
        ),
    )


# Fixed prefill-chunk capture buckets. The mixed CudaGraphKey draws
# ``num_prefill_tokens`` from this set (padding up to the next bucket) so the
# captured-graph count stays finite — without bucketing, every distinct prompt
# length would force a new capture and blow up warmup time + memory. See the
# "capture-shape explosion risk + mitigation" section of DESIGN_mixed_walk.md.
DEFAULT_MIXED_PREFILL_BUCKETS: tuple[int, ...] = (64, 128, 256, 512)


def pad_prefill_tokens_to_bucket(
    num_prefill_tokens: int,
    buckets: tuple[int, ...] = DEFAULT_MIXED_PREFILL_BUCKETS,
) -> int | None:
    """Return the smallest bucket >= ``num_prefill_tokens`` (or None if it
    exceeds the largest bucket — caller falls back to a separate prefill step).
    """
    for b in buckets:
        if num_prefill_tokens <= b:
            return b
    return None

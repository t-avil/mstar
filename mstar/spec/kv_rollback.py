"""Paged-KV bookkeeping for a lossless speculative step (pure logic, no GPU).

A spec step appends K *drafted* tokens' KV to a request's paged cache, runs one
verify forward over those K rows, accepts a prefix of length ``a`` (0..K), and
emits the bonus token. The cache must end in EXACTLY the state a non-spec run of
``a+1`` real tokens would have produced — otherwise decode silently corrupts.

The delicate part is the *rollback*: we speculatively advanced seq_len by K to
place the drafted KV, but only ``a`` of those drafts were correct. The bonus
token (the verify model's own token at position ``a``) is real and must occupy
slot ``a``. So the committed advance is ``a + 1``, and any pages that became
allocated purely to hold rejected drafts (positions > a within this step) must be
released back to the free pool. This module is that bookkeeping, isolated so it is
unit-testable without a model boot.
"""
from __future__ import annotations


def slots_needed(seq_len: int, page_size: int) -> int:
    """Number of pages required to hold ``seq_len`` tokens (ceil-div)."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be >=0, got {seq_len}")
    return (seq_len + page_size - 1) // page_size


def plan_spec_append(seq_len: int, k: int, block_table: list[int],
                     free_pages: list[int], page_size: int) -> dict:
    """Reserve pages so ``k`` drafted tokens can be written after ``seq_len``.

    Returns the *tentative* block_table and remaining free list assuming all K
    drafts land. Pure planning: no writes. ``block_table`` is the request's
    current page ids (len == slots_needed(seq_len)).
    """
    if len(block_table) != slots_needed(seq_len, page_size):
        raise ValueError("block_table length inconsistent with seq_len")
    if k < 0:
        raise ValueError("k must be >=0")
    need = slots_needed(seq_len + k, page_size)
    bt = list(block_table)
    free = list(free_pages)
    while len(bt) < need:
        if not free:
            raise RuntimeError("out of KV pages planning spec append")
        bt.append(free.pop(0))
    return {"block_table": bt, "free_pages": free, "tentative_seq_len": seq_len + k}


def commit_after_verify(seq_len: int, k: int, accepted: int,
                        tentative_block_table: list[int], free_pages: list[int],
                        page_size: int) -> dict:
    """Finalize the cache after verify accepted ``accepted`` of ``k`` drafts.

    Committed advance = accepted + 1 (the bonus token is always real). Pages that
    were reserved only to hold rejected drafts (beyond the committed length) are
    returned to the FRONT of the free pool (LIFO — they were just taken). The
    result is bit-identical to having run ``accepted + 1`` non-spec steps.

    Invariant checks (raise, never silently mis-account):
      * 0 <= accepted <= k
      * committed advance never exceeds the tentative advance (a+1 <= k ... except
        the a==k full-accept case where bonus rides one past: a+1 == k+1).
    """
    if not (0 <= accepted <= k):
        raise ValueError(f"accepted {accepted} out of range [0,{k}]")
    committed_len = seq_len + accepted + 1          # +1 bonus token, always real
    # full-accept: bonus sits one past the K drafts => may need one more page than
    # the tentative plan reserved. Caller must have reserved k+1 in that case; guard it.
    need = slots_needed(committed_len, page_size)
    bt = list(tentative_block_table)
    free = list(free_pages)
    if need > len(bt):
        raise RuntimeError(
            f"full-accept bonus needs page {need} but only {len(bt)} reserved; "
            "reserve k+1 pages when drafting")
    # release pages beyond the committed length back to the free pool (LIFO front)
    released = bt[need:]
    bt = bt[:need]
    free = list(reversed(released)) + free
    return {"block_table": bt, "free_pages": free, "committed_seq_len": committed_len,
            "released_pages": released}

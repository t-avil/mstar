"""Standalone n-gram / prompt-lookup drafter for lossless speculative decoding.

Pure logic (no model / no GPU) so it is unit-testable without a boot. Mirrors
vLLM's ngram_proposer: given a request's own token history, propose the next few
tokens by finding the most recent prior occurrence of the current suffix n-gram
and copying what followed it. The proposal is only a *speculation* — the target
model verifies it, so a wrong draft costs nothing but a rejected step (lossless).
"""
from __future__ import annotations


def propose_draft(
    token_ids: list[int],
    min_n: int = 2,
    max_n: int = 4,
    k: int = 4,
) -> list[int]:
    """Propose up to ``k`` draft tokens by prompt-lookup.

    Try the LONGEST suffix first (``max_n`` down to ``min_n``): take the last
    ``n`` tokens as the query, scan the history for the most RECENT earlier
    occurrence of that exact n-gram, and return the up-to-``k`` tokens that
    followed it. Longer matches are more specific, so they win. Returns [] when
    no suffix of length in [min_n, max_n] recurs (no speculation this step).

    Invariants:
      * never proposes from the current suffix itself (search excludes the
        trailing occurrence);
      * returns at most ``k`` tokens, at most what actually followed the match;
      * deterministic; O(len * (max_n - min_n)) worst case (fine for serving K).
    """
    L = len(token_ids)
    if L < min_n + 1 or k <= 0:
        return []
    hi = min(max_n, L - 1)               # can't match a suffix longer than history-1
    for n in range(hi, min_n - 1, -1):
        suffix = token_ids[L - n:]
        # search backward for the most recent earlier occurrence of `suffix`
        # (start positions in [0, L-n-1]; exclude the trailing one at L-n)
        for start in range(L - n - 1, -1, -1):
            if token_ids[start:start + n] == suffix:
                draft = token_ids[start + n: start + n + k]
                if draft:
                    return draft
                break  # matched but nothing followed within this window; try shorter n
    return []


def accept_prefix(draft: list[int], target: list[int]) -> int:
    """Greedy accept: length of the longest prefix where target argmax == draft.

    ``target[i]`` is the target model's greedy token at draft position ``i``
    (i.e. what it would emit given the accepted context so far). The accepted
    count is the run of leading matches; the caller then also appends the FIRST
    non-matching target token (the guaranteed "bonus" token), so a step always
    makes >=1 token of real progress. Token-exact by construction => LOSSLESS.
    """
    n = 0
    for d, t in zip(draft, target):
        if d != t:
            break
        n += 1
    return n

"""One-request spec step orchestration (pure logic, no GPU).

Collapses draft -> reserve -> (worker runs verify forward) -> accept -> commit
into the TWO calls the worker makes per speculating request, so the live decode
loop only touches:

    plan = begin_spec_step(history, seq_len, block_table, free_pages, page_size, K, min_n)
    if plan.draft:                       # worker appends plan.draft as extra query rows,
        ...                              # runs the (1+len(draft))-query verify forward,
        out = finish_spec_step(plan, verify_greedy_tokens)   # gets model argmax per row
        # emit out.emitted_tokens; set seq_len/block_table/free_pages to out.*

Everything numeric (accept length, KV rollback, bonus) lives here and is unit-
tested without a boot. LOSSLESS by construction: emitted tokens are exactly the
verify model's own greedy tokens for the committed positions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ngram_drafter import propose_draft, accept_prefix
from kv_rollback import plan_spec_append, commit_after_verify


@dataclass
class SpecPlan:
    draft: list[int]                 # drafted tokens (may be []); q_len = 1 + len(draft)
    seq_len: int                     # request seq_len BEFORE this step
    block_table: list[int]           # pages reserved for up to len(draft)+1 headroom
    free_pages: list[int]            # remaining free pool after reservation
    page_size: int


@dataclass
class SpecResult:
    emitted_tokens: list[int]        # accepted drafts + bonus (always >=1 token)
    accepted: int                    # how many drafts verified correct
    committed_seq_len: int
    block_table: list[int]
    free_pages: list[int]


def begin_spec_step(history: list[int], seq_len: int, block_table: list[int],
                    free_pages: list[int], page_size: int,
                    k: int = 4, min_n: int = 2, max_n: int = 4) -> SpecPlan:
    """Draft up to K tokens and reserve KV headroom for K drafts + 1 bonus.

    Returns a SpecPlan whose ``draft`` may be empty (no recurrence found) — in
    that case the worker runs an ordinary q_len==1 decode (byte-identical path).
    We reserve len(draft)+1 slots so a FULL accept's bonus token has a home (the
    k+1 headroom trap that kv_rollback tests pin).
    """
    draft = propose_draft(history, min_n=min_n, max_n=max_n, k=k)
    reserve = len(draft) + 1                      # +1 for the guaranteed bonus token
    p = plan_spec_append(seq_len, reserve, block_table, free_pages, page_size)
    return SpecPlan(draft=draft, seq_len=seq_len, block_table=p["block_table"],
                    free_pages=p["free_pages"], page_size=page_size)


def finish_spec_step(plan: SpecPlan, verify_greedy: list[int]) -> SpecResult:
    """Given the verify model's greedy token at each of the 1+len(draft) rows,
    accept the matching prefix, emit accepted drafts + the bonus, and commit the
    paged cache to the byte-identical non-spec state.

    ``verify_greedy[i]`` = model argmax at query row i. Row 0 is the model's token
    for the *current* position (the ordinary next token). Rows 1..len(draft) are
    the model's tokens assuming draft[0..i-1] were correct. accept_prefix compares
    the DRAFT (plan.draft) against the model's tokens at the rows that PREDICT
    them: draft[j] is correct iff verify_greedy[j] == draft[j]. The bonus is the
    model's token at the first unaccepted row (always real, always emitted).
    """
    draft = plan.draft
    reserve = len(draft) + 1
    if len(verify_greedy) != reserve:
        raise ValueError(
            f"verify_greedy has {len(verify_greedy)} rows, expected {reserve} "
            f"(1 + {len(draft)} drafts)")
    # accepted = longest prefix where the model, at row j, would itself emit draft[j].
    accepted = accept_prefix(draft, verify_greedy)
    # emitted = the accepted drafts + the bonus (model's own token at row `accepted`).
    emitted = list(draft[:accepted]) + [verify_greedy[accepted]]
    c = commit_after_verify(plan.seq_len, reserve, accepted,
                            plan.block_table, plan.free_pages, plan.page_size)
    # committed advance from commit_after_verify is accepted+1 == len(emitted). check.
    assert c["committed_seq_len"] == plan.seq_len + len(emitted), "advance/emit mismatch"
    return SpecResult(emitted_tokens=emitted, accepted=accepted,
                      committed_seq_len=c["committed_seq_len"],
                      block_table=c["block_table"], free_pages=c["free_pages"])

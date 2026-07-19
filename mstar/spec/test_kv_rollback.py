"""No-boot tests for paged-KV spec-step bookkeeping (plan -> verify -> commit).

The property that matters: after a spec step that accepted `a` of `k` drafts, the
cache state (seq_len, block_table, free_pages) is EXACTLY what `a+1` ordinary
decode steps would have produced, and every page reserved for a rejected draft is
returned to the free pool. Run: `python test_kv_rollback.py` from this dir.
"""
from kv_rollback import slots_needed, plan_spec_append, commit_after_verify

PS = 4  # small page size to force boundary crossings


def _ok(name):
    print(f"  ok  {name}")


def reference_pages(seq_len, advance, block_table, free_pages, page_size):
    """Ground truth: run `advance` plain steps, taking pages from free front."""
    bt, free = list(block_table), list(free_pages)
    for _ in range(advance):
        seq_len += 1
        if slots_needed(seq_len, page_size) > len(bt):
            bt.append(free.pop(0))
    return seq_len, bt, free


def test_no_draft_is_plain_step():
    # k=0 => pure fallback: commit advances exactly 1 (the bonus == the real token)
    seq, bt, free = 5, [10, 11], list(range(20, 30))
    p = plan_spec_append(seq, 0, bt, free, PS)
    c = commit_after_verify(seq, 0, 0, p["block_table"], p["free_pages"], PS)
    rseq, rbt, rfree = reference_pages(seq, 1, bt, free, PS)
    assert c["committed_seq_len"] == rseq == 6
    assert c["block_table"] == rbt and c["free_pages"] == rfree
    _ok("k=0 reduces to one plain step")


def test_partial_accept_releases_rejected_pages():
    # seq_len 7 (2 pages: [0..3],[4..7]-> uses slot 7? 7//4=1 -> pages ceil(8/4)=2). draft 4.
    seq, bt, free = 7, [100, 101], [200, 201, 202, 203]
    p = plan_spec_append(seq, 4, bt, free, PS)   # need ceil(11/4)=3 pages -> +1 page (200)
    assert p["block_table"] == [100, 101, 200]
    # accept 1 of 4 => committed len 7+1+1=9 -> ceil(9/4)=3 pages. no release (bonus needs page 200)
    c = commit_after_verify(seq, 4, 1, p["block_table"], p["free_pages"], PS)
    rseq, rbt, rfree = reference_pages(seq, 2, bt, free, PS)
    assert c["committed_seq_len"] == rseq == 9
    assert c["block_table"] == rbt == [100, 101, 200]
    assert c["free_pages"] == rfree == [201, 202, 203]
    _ok("partial accept matches plain a+1 steps (page kept for bonus)")


def test_accept_zero_releases_the_speculative_page():
    seq, bt, free = 7, [100, 101], [200, 201, 202]
    p = plan_spec_append(seq, 4, bt, free, PS)   # grabs 200
    # accept 0 => committed len 8 -> ceil(8/4)=2 pages. page 200 must go BACK.
    c = commit_after_verify(seq, 4, 0, p["block_table"], p["free_pages"], PS)
    rseq, rbt, rfree = reference_pages(seq, 1, bt, free, PS)
    assert c["committed_seq_len"] == rseq == 8
    assert c["block_table"] == rbt == [100, 101]
    assert c["released_pages"] == [200]
    assert c["free_pages"] == rfree == [200, 201, 202]   # 200 returned to front
    _ok("accept 0 returns the speculative page to free pool")


def test_full_accept_needs_k_plus_one_reservation():
    # full accept: bonus rides one past K drafts. must reserve k+1 to be safe.
    seq, bt, free = 7, [100, 101], [200, 201, 202, 203]
    p = plan_spec_append(seq, 5, bt, free, PS)   # k+1=... reserve for seq+5=12 -> ceil=3 pages
    # accept 5 of 5 => committed 7+5+1=13 -> ceil(13/4)=4 pages. plan only reserved 3 -> must raise.
    raised = False
    try:
        commit_after_verify(seq, 5, 5, p["block_table"], p["free_pages"], PS)
    except RuntimeError:
        raised = True
    assert raised, "full-accept beyond reservation must raise, not corrupt"
    # correct: reserve k+1 tokens of headroom
    p2 = plan_spec_append(seq, 6, bt, free, PS)  # seq+6=13 -> 4 pages reserved
    c = commit_after_verify(seq, 5, 5, p2["block_table"], p2["free_pages"], PS)
    rseq, rbt, rfree = reference_pages(seq, 6, bt, free, PS)
    assert c["committed_seq_len"] == rseq == 13
    assert c["block_table"] == rbt and c["free_pages"] == rfree
    _ok("full accept needs k+1 headroom; guarded and correct")


def test_page_boundary_release_multi():
    # draft spanning two fresh pages, accept 0 => both fresh pages released
    seq, bt, free = 8, [100, 101], [200, 201, 202, 203]  # seq 8 -> exactly 2 pages
    p = plan_spec_append(seq, 6, bt, free, PS)   # seq+6=14 -> ceil=4 -> grabs 200,201
    assert p["block_table"] == [100, 101, 200, 201]
    c = commit_after_verify(seq, 6, 0, p["block_table"], p["free_pages"], PS)
    assert c["committed_seq_len"] == 9
    assert c["block_table"] == [100, 101, 200]     # keeps one for the bonus (slot 8)
    assert c["released_pages"] == [201]
    assert c["free_pages"][0] == 201               # returned to front
    _ok("multi-page draft releases only truly-unused pages")


def test_out_of_pages_raises():
    raised = False
    try:
        plan_spec_append(4, 8, [10], [], PS)       # need pages, none free
    except RuntimeError:
        raised = True
    assert raised
    _ok("out-of-pages raises during planning (no silent overwrite)")


def test_invariants_reject_bad_accept():
    seq, bt, free = 7, [100, 101], [200]
    p = plan_spec_append(seq, 4, bt, free, PS)
    for bad in (-1, 5):
        raised = False
        try:
            commit_after_verify(seq, 4, bad, p["block_table"], p["free_pages"], PS)
        except ValueError:
            raised = True
        assert raised, f"accepted={bad} must be rejected"
    _ok("accepted outside [0,k] rejected")


if __name__ == "__main__":
    print("kv_rollback bookkeeping (no boot):")
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1
    print(f"ALL {n} PASS")

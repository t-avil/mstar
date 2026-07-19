"""Unit tests for the n-gram drafter — pure logic, no model/boot."""
from ngram_drafter import propose_draft, accept_prefix

def _eq(name, got, want):
    assert got == want, f"FAIL {name}: got {got} want {want}"
    print(f"  ok {name}: {got}")

def test_draft():
    # exact recurring 4-gram: "the cat sat on ..." repeats -> propose what followed
    seq = [5,1,2,3,4,9,9,1,2,3,4]        # suffix [1,2,3,4] recurs at idx1; followed by 9
    _eq("recur-basic", propose_draft(seq, min_n=2, max_n=4, k=3), [9,9,1])
    # no recurrence -> empty
    _eq("no-match", propose_draft([1,2,3,4,5], min_n=2, max_n=4, k=3), [])
    # most-RECENT match wins (two occurrences of [7,8]; take the later one's follow)
    _eq("most-recent", propose_draft([7,8,100, 7,8,200, 7,8], min_n=2, max_n=2, k=1), [200])
    # longer n-gram preferred over shorter
    #   suffix4 [2,3,4,5] recurs (->6); suffix2 [4,5] also recurs. longer wins.
    _eq("longer-wins", propose_draft([2,3,4,5,6, 40,50, 2,3,4,5], min_n=2, max_n=4, k=1), [6])
    # k truncation: only 2 tokens followed the match though k=5
    _eq("k-truncate", propose_draft([1,2,3, 9,9, 1,2,3], min_n=2, max_n=3, k=5), [9,9,1,2,3])
    # too-short history
    _eq("too-short", propose_draft([1], min_n=2, max_n=4, k=3), [])

def test_accept():
    _eq("accept-all", accept_prefix([1,2,3],[1,2,3]), 3)
    _eq("accept-partial", accept_prefix([1,2,9],[1,2,3]), 2)
    _eq("accept-none", accept_prefix([9,2,3],[1,2,3]), 0)
    _eq("accept-empty", accept_prefix([],[1,2,3]), 0)
    _eq("accept-shorter-target", accept_prefix([1,2,3],[1,2]), 2)

if __name__ == "__main__":
    test_draft(); test_accept(); print("ALL NGRAM UNIT TESTS PASSED")

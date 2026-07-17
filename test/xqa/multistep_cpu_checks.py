"""CPU/meta-level checks for the in-graph multistep xqa decode primitives.

Runs with NO GPU (``CUDA_VISIBLE_DEVICES=""``). Exercises the pure index math
and host stop-handling that the K-step capture body depends on, against
independent brute-force references. These are the pieces most likely to be
wrong (page-roll across boundaries, EOS trim), and the ones that do NOT need the
model to validate.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
      /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/multistep_cpu_checks.py
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from mstar.utils.flashinfer_utils import (  # noqa: E402
    ingraph_greedy_token,
    multistep_pages_needed,
    multistep_write_locations,
    trim_after_eos,
    xqa_multistep_k,
)

PAGE = 128
FAILS = 0


def check(name, cond, detail=""):
    global FAILS
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS += 1
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")


def brute_write_loc(L, block_tables, page_size):
    """Independent reference for multistep_write_locations: token idx = L-1."""
    locs = []
    for r, Lr in enumerate(L.tolist()):
        idx = Lr - 1
        col = idx // page_size
        off = idx % page_size
        locs.append([int(block_tables[r, col]), off])
    return torch.tensor(locs, dtype=torch.int64)


def test_flag():
    print("flag xqa_multistep_k()")
    # xqa_multistep_k requires MSTAR_XQA_DECODE=1 (else 0 regardless of K).
    for xqa, k, expect in [
        ("0", "4", 0), ("1", "0", 0), ("1", "1", 0),
        ("1", "2", 2), ("1", "8", 8), ("1", "abc", 0), ("1", "-3", 0),
    ]:
        os.environ["MSTAR_XQA_DECODE"] = xqa
        os.environ["MSTAR_XQA_MULTISTEP"] = k
        got = xqa_multistep_k()
        check(f"XQA={xqa} K={k!r} -> {got}", got == expect, f"expected {expect}")
    os.environ.pop("MSTAR_XQA_DECODE", None)
    os.environ.pop("MSTAR_XQA_MULTISTEP", None)


def test_write_locations_roll():
    print("multistep_write_locations() page-roll vs brute force")
    # 3 requests at different lengths; block tables with NON-trivial physical
    # page ids so a wrong column is caught. Width 4 pages = 512 tokens.
    block_tables = torch.tensor([
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ], dtype=torch.int32)
    # Start lengths chosen to straddle a page boundary within K steps:
    #  r0 at 127 -> next write at idx126(page10) then crosses to idx127..(page10) then 128(page11)
    #  r1 at 128 (exactly full first page) ; r2 at 250
    L0 = torch.tensor([127, 128, 250], dtype=torch.int64)
    K = 4
    L = L0.clone()
    ok_all = True
    trace = []
    for step in range(K):
        got = multistep_write_locations(L, block_tables, PAGE)
        ref = brute_write_loc(L, block_tables, PAGE)
        if not torch.equal(got, ref):
            ok_all = False
        trace.append((L.tolist(), got.tolist()))
        L = L + 1  # advance_step_ingraph does seq_lens += 1
    check("all K steps match brute force", ok_all)
    # Spot-check the boundary: r0 length 128 -> idx127 -> col0 page10 off127;
    #                          r0 length 129 -> idx128 -> col1 page11 off0.
    g128 = multistep_write_locations(torch.tensor([128]), torch.tensor([[10, 11, 12, 13]], dtype=torch.int32), PAGE)
    g129 = multistep_write_locations(torch.tensor([129]), torch.tensor([[10, 11, 12, 13]], dtype=torch.int32), PAGE)
    check("idx127 -> (page10, off127)", g128.tolist() == [[10, 127]], str(g128.tolist()))
    check("idx128 rolls -> (page11, off0)", g129.tolist() == [[11, 0]], str(g129.tolist()))
    for L_, loc in trace:
        print(f"      L={L_} loc={loc}")


def test_pages_needed():
    print("multistep_pages_needed()")
    # L0=127, K=4 -> token indices written: 126,127,128,129 -> max col = 129//128 = 1 -> 2 pages.
    check("L0=127 K=4 -> 2 pages", multistep_pages_needed(torch.tensor([127]), 4, PAGE) == 2)
    # L0=200, K=4 -> indices 199..202 -> max col 1 -> 2 pages.
    check("L0=200 K=4 -> 2 pages", multistep_pages_needed(torch.tensor([200]), 4, PAGE) == 2)
    # Batch: max drives it. L0 in {50, 500}, K=8 -> max idx 505 -> col 3 -> 4 pages.
    check("L0=[50,500] K=8 -> 4 pages", multistep_pages_needed(torch.tensor([50, 500]), 8, PAGE) == 4)
    # Exact multiple boundary: L0=128 K=2 -> indices 127,128 -> max col 1 -> 2 pages.
    check("L0=128 K=2 -> 2 pages", multistep_pages_needed(torch.tensor([128]), 2, PAGE) == 2)


def test_greedy():
    print("ingraph_greedy_token() first-index tie-break")
    logits = torch.tensor([
        [1.0, 3.0, 3.0, 2.0],   # tie at 1,2 -> first index 1
        [5.0, 1.0, 0.0, 0.0],   # 0
        [0.0, 0.0, 0.0, 9.0],   # 3
    ])
    got = ingraph_greedy_token(logits)
    check("argmax first-index", got.tolist() == [1, 0, 3], str(got.tolist()))


def test_trim():
    print("trim_after_eos()")
    EOS = 2
    tokens = torch.tensor([
        [5, 6, 2, 7],    # eos at pos2 -> keep [5,6,2]
        [1, 1, 1, 1],    # no eos -> keep all
        [2, 9, 9, 9],    # eos immediately -> keep [2]
    ])
    out = trim_after_eos(tokens, EOS)
    check("row0 trims after eos (incl eos)", out[0] == [5, 6, 2], str(out[0]))
    check("row1 no eos keeps all", out[1] == [1, 1, 1, 1], str(out[1]))
    check("row2 eos-first keeps [2]", out[2] == [2], str(out[2]))
    # include_eos=False drops the eos token
    out2 = trim_after_eos(tokens, EOS, include_eos=False)
    check("include_eos=False drops eos", out2[0] == [5, 6] and out2[2] == [], str((out2[0], out2[2])))
    # already-finished request emits nothing
    out3 = trim_after_eos(tokens, EOS, already_finished=[False, True, False])
    check("already_finished emits []", out3[1] == [], str(out3[1]))
    # multiple eos ids
    out4 = trim_after_eos(torch.tensor([[5, 8, 6, 7]]), [2, 8])
    check("multi eos-id set", out4[0] == [5, 8], str(out4[0]))


def main():
    print("=== XQA multistep CPU/meta checks (no GPU) ===")
    test_flag()
    test_write_locations_roll()
    test_pages_needed()
    test_greedy()
    test_trim()
    print()
    if FAILS:
        print(f"RESULT: {FAILS} CHECK(S) FAILED")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

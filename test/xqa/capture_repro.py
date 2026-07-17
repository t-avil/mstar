"""Minimal standalone CUDA-graph capture repro for the in-graph multistep xqa
decode advance machinery. NO server boot, runs in seconds on one free GPU.

Builds a tiny FlashInferXqaDecodeWrapper(use_cuda_graph=True), enables multistep
K=2, fills synthetic seq_lens/block_tables, and tries to capture a K-loop that
calls advance_step_ingraph() (+ a dummy attention). Reproduces / then verifies
the fix for cudaErrorInvalidValue.

    FLASHINFER_WORKSPACE_BASE=... TRITON_CACHE_DIR=... CUDA_VISIBLE_DEVICES=3 \
      PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
      /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/capture_repro.py
"""
import os
import sys

import torch

from mstar.utils.flashinfer_utils import (
    FlashInferXqaDecodeWrapper,
    multistep_write_locations,
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
    locs = []
    for r, Lr in enumerate(L.tolist()):
        idx = Lr - 1
        col = idx // page_size
        off = idx % page_size
        locs.append([int(block_tables[r, col]), off])
    return torch.tensor(locs, dtype=torch.int64, device=block_tables.device)


def build_wrapper(bs, max_pages_per_seq, device):
    ws = torch.zeros(16, dtype=torch.uint8, device=device)  # unused placeholder
    w = FlashInferXqaDecodeWrapper(
        workspace_buffer=ws,
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=128,
        page_size=PAGE,
        batch_size=bs,
        max_num_pages=1024,
        max_pages_per_seq=max_pages_per_seq,
        device=device,
        use_cuda_graph=True,
    )
    return w


def seed_state(w, bs, L0, block_tables):
    """Directly seed the static graph buffers (bypass plan(), which needs the
    real KV metadata). Mirrors what plan() leaves behind."""
    w._n_req = bs
    w._max_seq_len = w.max_pages_per_seq * PAGE
    w.dtype = torch.bfloat16
    w._seq_lens_buf[:bs].copy_(L0.to(torch.uint32))
    w._ms_len[:bs].copy_(L0.to(torch.int64))
    w._block_tables_buf[:bs, : block_tables.shape[1]].copy_(block_tables)
    w._seq_lens = w._seq_lens_buf[:bs]
    w._block_tables = w._block_tables_buf[:bs]
    # initial kv write locations for L0
    locs0 = brute_write_loc(L0, block_tables, PAGE)
    w.kv_cache_locations[:bs].copy_(locs0)


def main():
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    K = 2
    bs = 3
    max_pages = 4

    block_tables = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32, device=dev,
    )
    # r0 straddles a page boundary within K steps (127 -> writes idx126, idx127)
    L0 = torch.tensor([127, 128, 250], dtype=torch.int64, device=dev)

    # ---- Reference (eager, no capture): advance K steps, record locations ----
    w_ref = build_wrapper(bs, max_pages, dev)
    w_ref.enable_multistep(K)
    seed_state(w_ref, bs, L0, block_tables)
    eager_locs = []
    # step 0 uses seeded L0 location; then advance.
    eager_locs.append(w_ref.kv_cache_locations[:bs].clone())
    for _ in range(K - 1):
        w_ref.advance_step_ingraph()
        eager_locs.append(w_ref.kv_cache_locations[:bs].clone())
    # brute-force reference
    ref_locs = []
    Lc = L0.clone()
    for s in range(K):
        ref_locs.append(brute_write_loc(Lc, block_tables, PAGE))
        Lc = Lc + 1
    for s in range(K):
        check(f"eager step{s} loc == brute", torch.equal(eager_locs[s], ref_locs[s]),
              f"{eager_locs[s].tolist()} vs {ref_locs[s].tolist()}")

    # ---- Capture: a K-loop calling advance_step_ingraph + dummy attention ----
    w = build_wrapper(bs, max_pages, dev)
    w.enable_multistep(K)
    seed_state(w, bs, L0, block_tables)

    # dummy "attention": a matmul that gives the graph real work
    q = torch.randn(bs, 8, 128, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(1024, 2, PAGE, 1, 128, device=dev, dtype=torch.bfloat16)
    kv_tok = torch.randn(bs, 1, 128, device=dev, dtype=torch.bfloat16)  # [n, kv_heads, hd]
    out_buf = torch.zeros(bs, 8, 128, device=dev, dtype=torch.bfloat16)

    def dummy_forward(step):
        # write K,V at the current (advancing) write locations, like set_kv_cache
        w.set_kv_cache(kv, kv_tok, kv_tok)
        out_buf.copy_(torch.matmul(q, q.transpose(1, 2)).softmax(-1) @ q)
        w.record_token_ingraph(step, torch.zeros(bs, dtype=torch.int64, device=dev))

    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        seed_state(w, bs, L0, block_tables)
        for step in range(K):
            dummy_forward(step)
            if step < K - 1:
                w.advance_step_ingraph()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # reset state to L0 for the captured run
    seed_state(w, bs, L0, block_tables)

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for step in range(K):
                dummy_forward(step)
                if step < K - 1:
                    w.advance_step_ingraph()
        print("  [PASS] capture succeeded")
    except Exception as e:
        print(f"  [FAIL] capture raised: {type(e).__name__}: {e}")
        torch.cuda.synchronize()
        global FAILS
        FAILS += 1
        return

    # Replay: state must be re-seeded to L0 before each replay (graph re-runs the
    # same in-place advances). Capture recorded final locations into the static
    # buffer; verify a replay advances correctly and is deterministic.
    def run_and_grab():
        seed_state(w, bs, L0, block_tables)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        # after K-1 advances, seq_lens should be L0 + (K-1)
        return w._seq_lens_buf[:bs].clone(), w.kv_cache_locations[:bs].clone()

    sl1, loc1 = run_and_grab()
    expected_sl = (L0 + (K - 1)).to(torch.uint32)
    check("replay seq_lens advanced", torch.equal(sl1, expected_sl),
          f"{sl1.tolist()} vs {expected_sl.tolist()}")
    # final kv_cache_locations should equal brute at L0 + (K-1)
    expected_final = brute_write_loc((L0 + (K - 1)), block_tables, PAGE)
    check("replay final loc == brute", torch.equal(loc1, expected_final),
          f"{loc1.tolist()} vs {expected_final.tolist()}")

    sl2, loc2 = run_and_grab()
    check("replay deterministic (seq_lens)", torch.equal(sl1, sl2))
    check("replay deterministic (locations)", torch.equal(loc1, loc2))


if __name__ == "__main__":
    main()
    print()
    if FAILS:
        print(f"RESULT: {FAILS} CHECK(S) FAILED")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")

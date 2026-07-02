#!/usr/bin/env python3
"""In-graph microbench: sizes the POD/split-attention lever for thinker_mixed.

Question: how much of a mixed step's attention cost comes from running the
31 decode rows (qo_len=1, long KV) through the PREFILL kernel instead of the
decode kernel? If the penalty is small, the split-attention build is not
worth it; if it's ~5-10ms/step, it is THE i2t B32 lever.

Method (in-graph law — never time these out of graph):
  A) BatchPrefillWithPagedKVCacheWrapper planned for [1]*31 + [C] rows
     (the packed mixed shape today).
  B) BatchDecodeWithPagedKVCacheWrapper planned for [1]*31 rows
     + BatchPrefillWithPagedKVCacheWrapper planned for [C] alone
     (what the split would run).
  Each: capture a graph replaying the wrapper.run(s) x LAYERS times to
  amortize per-replay overhead like the real 48-layer forward, then time
  graph.replay() over many iters.

Shapes: Qwen3-Omni thinker GQA 28 qo heads / 4 kv heads, head_dim 128,
page 128; KV lens drawn to mimic i2t mid-generation (prompt ~600-1200 +
~90 generated). C = 256.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/m-coriander/coriander/tim/mstar-p2")

import flashinfer  # noqa: E402

DEV = torch.device("cuda:0")
NUM_QO, NUM_KV, HDIM, PAGE = 28, 4, 128, 128
N_DEC, C = 31, 256
LAYERS = 48          # replays per graph = layers per forward
ITERS = 50
KV_LENS = [600 + (i * 37) % 700 + 90 for i in range(N_DEC)]  # ~690-1390
CHUNK_KV = 256       # chunk row: fresh prefill, kv == qo == C

torch.manual_seed(0)


def build_paged(kv_lens):
    """Allocate distinct pages per row; return indptr/indices/last_page_len."""
    indptr = [0]
    indices = []
    lastlen = []
    next_page = 1
    for L in kv_lens:
        np_ = (L + PAGE - 1) // PAGE
        indices.extend(range(next_page, next_page + np_))
        next_page += np_
        indptr.append(len(indices))
        lastlen.append(L % PAGE or PAGE)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        torch.tensor(indices, dtype=torch.int32, device=DEV),
        torch.tensor(lastlen, dtype=torch.int32, device=DEV),
        next_page,
    )


def time_graph(fn, tag):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(LAYERS):
            fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"{tag}: {ms:.3f} ms per {LAYERS}-layer replay "
          f"({ms/LAYERS*1000:.1f} µs/layer)")
    return ms


def main():
    total_pages = 4096
    kv = torch.randn(total_pages, 2, PAGE, NUM_KV, HDIM,
                     dtype=torch.bfloat16, device=DEV)

    # ---- A: single prefill wrapper over [1]*31 + [C] (today's mixed) ----
    ws_a = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    ip, idx, lpl, _ = build_paged(KV_LENS + [CHUNK_KV])
    qo = torch.tensor([0] + list(range(1, N_DEC + 1)) + [N_DEC + C],
                      dtype=torch.int32, device=DEV)
    wa = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws_a, "NHD")
    wa.plan(qo_indptr=qo, paged_kv_indptr=ip, paged_kv_indices=idx,
            paged_kv_last_page_len=lpl, num_qo_heads=NUM_QO,
            num_kv_heads=NUM_KV, head_dim_qk=HDIM, page_size=PAGE,
            causal=True, q_data_type=torch.bfloat16)
    q_all = torch.randn(N_DEC + C, NUM_QO, HDIM, dtype=torch.bfloat16, device=DEV)
    a = time_graph(lambda: wa.run(q_all, kv), "A  prefill-kernel[31x1 + 256]")

    # ---- B1: decode wrapper over [1]*31 ----
    ws_b1 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    ip_d, idx_d, lpl_d, _ = build_paged(KV_LENS)
    wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws_b1, "NHD", use_tensor_cores=True)
    wd.plan(indptr=ip_d, indices=idx_d, last_page_len=lpl_d,
            num_qo_heads=NUM_QO, num_kv_heads=NUM_KV, head_dim=HDIM,
            page_size=PAGE, q_data_type=torch.bfloat16)
    q_dec = torch.randn(N_DEC, NUM_QO, HDIM, dtype=torch.bfloat16, device=DEV)
    b1 = time_graph(lambda: wd.run(q_dec, kv), "B1 decode-kernel[31x1]")

    # ---- B2: prefill wrapper over [C] alone ----
    ws_b2 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    ip_c, idx_c, lpl_c, _ = build_paged([CHUNK_KV])
    qo_c = torch.tensor([0, C], dtype=torch.int32, device=DEV)
    wc = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws_b2, "NHD")
    wc.plan(qo_indptr=qo_c, paged_kv_indptr=ip_c, paged_kv_indices=idx_c,
            paged_kv_last_page_len=lpl_c, num_qo_heads=NUM_QO,
            num_kv_heads=NUM_KV, head_dim_qk=HDIM, page_size=PAGE,
            causal=True, q_data_type=torch.bfloat16)
    q_ch = torch.randn(C, NUM_QO, HDIM, dtype=torch.bfloat16, device=DEV)
    b2 = time_graph(lambda: wc.run(q_ch, kv), "B2 prefill-kernel[256 chunk]")

    # ---- B3: both back-to-back in one graph (the actual split step) ----
    b3 = time_graph(lambda: (wd.run(q_dec, kv), wc.run(q_ch, kv)),
                    "B3 split decode[31]+prefill[256]")

    print(f"\nA (today)  = {a:.3f} ms/forward-of-attn")
    print(f"B3 (split) = {b3:.3f} ms  -> saving {a-b3:.3f} ms per mixed step")
    print(f"(components: decode {b1:.3f} + chunk {b2:.3f})")
    print("At ~400 folds/cell, cell win ≈ %.1f s" % ((a - b3) * 400 / 1000))


if __name__ == "__main__":
    main()

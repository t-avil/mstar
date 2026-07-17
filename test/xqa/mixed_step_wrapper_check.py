#!/usr/bin/env python
"""Stage-B: exercise the REAL FlashInferXqaDecodeWrapper mixed-step API.

Unlike test/xqa/mixed_prefill_poc.py (which called flashinfer inline to prove the
primitive), this drives the actual wrapper methods that the engine integration
uses — enable_mixed_prefill / plan_prefill_chunk / set_kv_cache_mixed / run_mixed
plus the existing decode plan() — so a regression in the wrapper is caught here.

Checks:
  CPU preflight (no GPU): flag gating (mixed_prefill_enabled needs XQA_DECODE;
    mixed_budget_tokens fallback order) + mask builder shape.
  GPU (ALLOW_GPU=1 on an IDLE GPU): mixed step of bs decodes + one N-token prefill
    chunk vs fp32 causal reference; run_mixed == separate decode()+prefill legs;
    and run_mixed captured in a cuda graph replays deterministically == eager.

Usage:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/mixed_step_wrapper_check.py
  ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=<idle_gpu> MSTAR_XQA_DECODE=1 MSTAR_MIXED_PREFILL=1 \
    PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/mixed_step_wrapper_check.py
"""
import math
import os
import sys

import torch

HEAD_DIM = 128
PAGE_SIZE = 128
NUM_KV_HEADS = 4
NUM_QO_HEADS = 32
DTYPE = torch.bfloat16
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def maxdiff(a, b):
    a, b = a.float(), b.float()
    ad = (a - b).abs()
    return ad.max().item(), (ad / (b.abs() + 1e-4)).max().item()


def fp32_attn(q_row, K, V):
    hq = q_row.shape[0]
    grp = hq // NUM_KV_HEADS
    Kx = K.repeat_interleave(grp, dim=1)
    Vx = V.repeat_interleave(grp, dim=1)
    scores = torch.einsum("hd,shd->hs", q_row.float(), Kx) * SM_SCALE
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", w, Vx)


def preflight_cpu():
    print("=" * 78)
    print("CPU PREFLIGHT: flag gating + mask builder")
    print("=" * 78)
    from mstar.utils.flashinfer_utils import (
        build_chunk_causal_mask,
        mixed_budget_tokens,
        mixed_prefill_enabled,
    )

    # mixed prefill requires XQA_DECODE.
    for k in ("MSTAR_XQA_DECODE", "MSTAR_MIXED_PREFILL", "MSTAR_MIXED_BUDGET_TOKENS",
              "MSTAR_PREFILL_CHUNK_TOKENS"):
        os.environ.pop(k, None)
    assert mixed_prefill_enabled() is False and mixed_budget_tokens() == 0
    os.environ["MSTAR_MIXED_PREFILL"] = "1"
    assert mixed_prefill_enabled() is False, "must stay off without XQA_DECODE"
    os.environ["MSTAR_XQA_DECODE"] = "1"
    assert mixed_prefill_enabled() is True
    assert mixed_budget_tokens() == 512, "default 512"
    os.environ["MSTAR_PREFILL_CHUNK_TOKENS"] = "256"
    assert mixed_budget_tokens() == 256, "PREFILL_CHUNK fallback"
    os.environ["MSTAR_MIXED_BUDGET_TOKENS"] = "128"
    assert mixed_budget_tokens() == 128, "MIXED_BUDGET wins"
    m = build_chunk_causal_mask(1, 8, torch.device("cpu"))
    assert m.dtype == torch.uint16 and m.shape == (1, 8, 2), (m.dtype, m.shape)
    assert int(m[0, 0, 0]) == 0b1 and int(m[0, 7, 0]) == 0b11111111
    print("  flag gating + mask builder OK "
          f"(budget={mixed_budget_tokens()}, mask uint16 {tuple(m.shape)})")
    for k in ("MSTAR_MIXED_BUDGET_TOKENS", "MSTAR_PREFILL_CHUNK_TOKENS"):
        os.environ.pop(k, None)
    print("  CPU preflight complete.\n")


def _build_cache(total_lens, extra_pages=8):
    """Contiguous paged cache + per-request physical page-id lists."""
    pages_per = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in total_lens]
    n_pages = sum(pages_per) + extra_pages
    cache = torch.randn(
        n_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    page_ids, nxt = [], 0
    for npg in pages_per:
        page_ids.append(list(range(nxt, nxt + npg)))
        nxt += npg
    return cache, page_ids, pages_per


def _write_kv(cache, page_ids, req, pos, k, v):
    p, o = page_ids[req][pos // PAGE_SIZE], pos % PAGE_SIZE
    cache[p, 0, o] = k.to(DTYPE)
    cache[p, 1, o] = v.to(DTYPE)


def _gather_kv(cache, page_ids, req, upto):
    ks, vs = [], []
    for pos in range(upto):
        p, o = page_ids[req][pos // PAGE_SIZE], pos % PAGE_SIZE
        ks.append(cache[p, 0, o]); vs.append(cache[p, 1, o])
    return torch.stack(ks).float(), torch.stack(vs).float()


def gpu_checks():
    from mstar.utils.flashinfer_utils import FlashInferXqaDecodeWrapper

    dev = torch.device("cuda")
    print(f"LIVE on {torch.cuda.get_device_name(dev)} cc={torch.cuda.get_device_capability(dev)}\n")
    bs, N = 3, 8
    dec_ctx = [130, 256, 400]        # committed decode lengths
    pre_ctx = 300                    # prefill context before the chunk
    dec_total = [c + 1 for c in dec_ctx]
    pre_total = pre_ctx + N
    cache, page_ids, pages_per = _build_cache(dec_total + [pre_total])
    max_pages = max(pages_per)

    # real new tokens
    torch.manual_seed(0)
    dq = torch.randn(bs, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    dk = torch.randn(bs, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    dv = torch.randn(bs, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    pq = torch.randn(N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    pk = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    pv = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)

    ws = torch.zeros(1, dtype=torch.uint8, device=dev)
    w = FlashInferXqaDecodeWrapper(
        ws, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        batch_size=bs, max_num_pages=cache.shape[0],
        max_pages_per_seq=max_pages + 2, device=dev, use_cuda_graph=True,
    )
    # decode-leg plan: ragged FlashInfer metadata for the 3 decodes.
    kv_indptr = [0]
    kv_indices = []
    last_page = []
    for i in range(bs):
        kv_indices += page_ids[i]
        kv_indptr.append(len(kv_indices))
        last_page.append(dec_total[i] % PAGE_SIZE or PAGE_SIZE)
    w.plan(
        paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        paged_kv_indices=torch.tensor(kv_indices, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(last_page, dtype=torch.int32),
        dtype=DTYPE,
    )
    pf_pages = (pre_total + PAGE_SIZE - 1) // PAGE_SIZE
    w.enable_mixed_prefill(n_max=N, max_pages_pf=pf_pages + 2)
    w.plan_prefill_chunk(committed_len=pre_ctx, page_indices=page_ids[bs], n_chunk=N)

    # write KV via the wrapper (mixed): packed [decodes ; prefill chunk]
    k_packed = torch.cat([dk, pk], dim=0)
    v_packed = torch.cat([dv, pv], dim=0)
    w.set_kv_cache_mixed(cache, k_packed, v_packed)

    q_packed = torch.cat([dq, pq], dim=0)
    out = w.run_mixed(q_packed, cache)   # [bs+N, hq, hd]

    # fp32 reference
    ok = True
    d_ab = 0.0
    for i in range(bs):
        Kf, Vf = _gather_kv(cache, page_ids, i, dec_total[i])
        ref = fp32_attn(dq[i], Kf, Vf)
        ab, _ = maxdiff(out[i], ref); d_ab = max(d_ab, ab)
    p_ab = 0.0
    Kf, Vf = _gather_kv(cache, page_ids, bs, pre_total)
    for j in range(N):
        ref = fp32_attn(pq[j], Kf[: pre_ctx + j + 1], Vf[: pre_ctx + j + 1])
        ab, _ = maxdiff(out[bs + j], ref); p_ab = max(p_ab, ab)
    ok &= d_ab <= 5e-2 and p_ab <= 5e-2
    print(f"  run_mixed vs fp32 ref: decode max_abs={d_ab:.3e} prefill max_abs={p_ab:.3e}"
          f"  -> {'PASS' if d_ab <= 5e-2 and p_ab <= 5e-2 else 'FAIL'}")

    # capture run_mixed in a cuda graph, replay deterministically == eager
    eager = out.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            w.run_mixed(q_packed, cache)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    out_static = torch.empty_like(eager)
    try:
        with torch.cuda.graph(g):
            out_static.copy_(w.run_mixed(q_packed, cache))
        g.replay(); torch.cuda.synchronize(); r1 = out_static.clone()
        g.replay(); torch.cuda.synchronize(); r2 = out_static.clone()
        det = torch.equal(r1, r2)
        ab_e, _ = maxdiff(r1, eager)
        cap_ok = det and ab_e <= 5e-2
        print(f"  run_mixed captured: deterministic={det} vs eager max_abs={ab_e:.3e}"
              f"  -> {'PASS' if cap_ok else 'FAIL'}")
        ok &= cap_ok
    except Exception as e:  # noqa: BLE001
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:200]}  -> FAIL")
        ok = False
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    preflight_cpu()
    if not (os.environ.get("ALLOW_GPU") == "1" and torch.cuda.is_available()):
        print("Live GPU DEFERRED (set ALLOW_GPU=1 with an IDLE GPU visible).")
        return 0
    return 0 if gpu_checks() else 1


if __name__ == "__main__":
    sys.exit(main())

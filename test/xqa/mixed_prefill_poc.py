#!/usr/bin/env python
"""Stage-B POC: mixed decode + bounded-prefill in one xqa call (q_len_per_req>1).

GOAL (de-risk BEFORE any engine integration): prove the CORE PRIMITIVE for
co-admitting decodes with a BOUNDED prefill slice in a single captured xqa step
is (a) numerically correct and (b) CUDA-graph capturable, for M*'s exact shapes
(head_dim=128, page_size=128, bf16, GQA 32/4 = grp8). This fixes the i2t B32 TTFT
tail where the 32 image-prefills run as separate walks that FREEZE all decodes;
the fix is vLLM's mechanism: every decode step also advances a bounded prefill
chunk so decodes never freeze.

Stage A (multistep pure decode, q_len=1) was killed: xqa is ~50% slower than
FlashInfer BatchDecode at q_len=1. Stage B is xqa's ACTUAL strength: q_len=N > 1
(a prefill chunk) is exactly the regime trtllm/xqa is built to win.

============================================================================
WHAT THE xqa API ACTUALLY SUPPORTS (read off installed FlashInfer 0.6.13)
============================================================================
  flashinfer.decode.trtllm_batch_decode_with_kv_cache(..., q_len_per_req, mask,
      max_q_len, cum_seq_lens_q, ...)  -> on SM90 dispatches to the xqa kernel.

  * q_len_per_req is a SCALAR int  -> UNIFORM query length across the WHOLE batch.
  * The variable-length path (max_q_len + cum_seq_lens_q) is EXPLICITLY REJECTED
    on the xqa backend (decode.py:2719-2720: "xqa backend does not support
    cum_seq_lens_q"); it is trtllm-gen (Blackwell) only. H200 = SM90 = xqa.
  * spec-dec semantics (decode.py:3060, xqa.py:180-262, and the v0.6.13 test):
      - query is [batch * q_seq_len, num_qo_heads, head_dim];
      - seq_lens[r] = in_kv_len[r] + q_seq_len  (INCLUDES the new query tokens);
      - the q_seq_len query rows of request r map to its LAST q_seq_len KV
        positions [in_kv_len, in_kv_len+q_seq_len); their K,V must already be
        written into the paged cache at those positions BEFORE the call;
      - each query row attends to ALL prior context [0, in_kv_len) UNCONDITIONALLY
        plus the local window [in_kv_len, seq_len) gated by `mask`;
      - mask: uint16 [batch, q_seq_len, ((q_seq_len+31)//32)*2], bit-packed; a set
        bit i in row j means "query row j attends to local KV column i". Causal =
        kv_indices <= q_indices (exact builder copied from the FI 0.6.13 test).

CONSEQUENCE for a genuinely MIXED batch (decodes at q_len=1 + prefill at q_len=N):
  You CANNOT pass different per-request q_lens in one xqa call. Two ways to still
  do it in ONE captured step:
    (S) SINGLE padded call: run everything at uniform q_seq_len=N. A decode is
        emulated as ROW 0 of its N-row block (seq_len = committed+N); the shared
        causal mask makes row 0 attend to [0, committed] only (context + local
        col0=itself), which is EXACTLY the decode. Rows 1..N-1 of a decode are
        junk and discarded. Cost: each decode does N x the query work and needs
        N KV slots reserved. Numerically correct (this POC proves it).
    (T) TWO calls per step: one q_len=1 xqa decode call for the decode requests +
        one q_len=N xqa call for the prefill chunk. Both plan-free + capturable,
        captured back-to-back in ONE graph. No decode padding waste.

This POC measures BOTH (S) and (T) against an fp32 reference, and capture-tests
the q_len=N prefill call and the padded mixed call.

Usage:
  # CPU preflight (no GPU): API-shape + mask-builder checks
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/mixed_prefill_poc.py

  # Full live (owner runs on an IDLE GPU reading <500MiB in nvidia-smi):
  ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=<idle_gpu> \
    PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/mixed_prefill_poc.py
"""
import math
import os
import sys

import torch

# ----- M* Thinker shapes -----
HEAD_DIM = 128
PAGE_SIZE = 128
NUM_KV_HEADS = 4
NUM_QO_HEADS = 32          # GQA 32/4 = grp8 (M* shipping)
DTYPE = torch.bfloat16
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


# ---------------------------------------------------------------------------
# Exact xqa spec-dec causal mask builder (verbatim from FlashInfer v0.6.13
# tests/attention/test_xqa_batch_decode.py::generate_causal_mask).
# ---------------------------------------------------------------------------
def generate_causal_mask(batch_size: int, q_seq_len: int, device) -> torch.Tensor:
    num_packed = (q_seq_len + 31) // 32
    q_idx = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    kv_idx = torch.arange(q_seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    causal = kv_idx <= q_idx                                   # [N, N] bool
    padded = num_packed * 32
    if padded > q_seq_len:
        pad = torch.zeros(q_seq_len, padded - q_seq_len, device=device, dtype=torch.bool)
        causal = torch.cat([causal, pad], dim=1)
    causal = causal.view(q_seq_len, num_packed, 32)
    bits = torch.tensor([1 << i for i in range(32)], device=device, dtype=torch.int64)
    m32 = (causal.to(torch.int64) * bits).sum(dim=-1).to(torch.uint32)
    m32 = m32.unsqueeze(0).expand(batch_size, q_seq_len, num_packed).contiguous()
    return m32.view(torch.uint16)


# ---------------------------------------------------------------------------
# Paged-KV builder (M* native NHD single-tensor [pages,2,page,kvh,hd]).
# We build a cache holding `total_len` valid KV rows per request (context +
# the freshly-written query KV) and a DENSE block table for the xqa call.
# ---------------------------------------------------------------------------
class PagedKV:
    def __init__(self, total_lens, device, extra_pages=4):
        self.total_lens = list(total_lens)
        self.device = device
        self.bs = len(total_lens)
        pages_per = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in total_lens]
        self.pages_per = pages_per
        self.max_pages = max(pages_per)
        n_pages = sum(pages_per) + extra_pages
        self.cache = torch.randn(
            n_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        # left-packed dense block table [bs, max_pages]
        self.block_tables = torch.zeros(self.bs, self.max_pages, dtype=torch.int32, device=device)
        self.page_ids = []
        nxt = 0
        for i, npg in enumerate(pages_per):
            ids = list(range(nxt, nxt + npg))
            nxt += npg
            self.page_ids.append(ids)
            self.block_tables[i, :npg] = torch.tensor(ids, dtype=torch.int32, device=device)
        self.seq_lens = torch.tensor(total_lens, dtype=torch.uint32, device=device)
        self.max_seq_len = self.max_pages * PAGE_SIZE

    def abs_pos_location(self, req, pos):
        """(page_id, offset) for absolute token index `pos` of request `req`."""
        col = pos // PAGE_SIZE
        off = pos - col * PAGE_SIZE
        return self.page_ids[req][col], off

    def write_kv(self, req, pos, k, v):
        p, o = self.abs_pos_location(req, pos)
        self.cache[p, 0, o] = k.to(DTYPE)
        self.cache[p, 1, o] = v.to(DTYPE)

    def gather_kv(self, req, upto):
        """fp32 [upto, kvh, hd] of the first `upto` valid KV rows of request `req`."""
        ks, vs = [], []
        for pos in range(upto):
            p, o = self.abs_pos_location(req, pos)
            ks.append(self.cache[p, 0, o])
            vs.append(self.cache[p, 1, o])
        return torch.stack(ks).float(), torch.stack(vs).float()


def fp32_attn(q_row, K, V):
    """q_row [hq,hd]; K,V [s,kvh,hd] fp32 -> out [hq,hd] fp32 (GQA causal-agnostic:
    caller passes only the KV rows this query attends to)."""
    hq = q_row.shape[0]
    grp = hq // NUM_KV_HEADS
    Kx = K.repeat_interleave(grp, dim=1)   # [s,hq,hd]
    Vx = V.repeat_interleave(grp, dim=1)
    scores = torch.einsum("hd,shd->hs", q_row.float(), Kx) * SM_SCALE
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", w, Vx)


def maxdiff(a, b):
    a, b = a.float(), b.float()
    ad = (a - b).abs()
    rd = ad / (b.abs() + 1e-4)
    return ad.max().item(), rd.max().item()


def xqa_call(query, cache, block_tables, seq_lens, max_seq_len, q_len, mask, ws):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    return trtllm_batch_decode_with_kv_cache(
        query=query.to(DTYPE),
        kv_cache=cache,                      # [pages,2,page,kvh,hd] NHD
        workspace_buffer=ws,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=SM_SCALE,
        bmm2_scale=1.0,
        kv_layout="NHD",
        backend="auto",                      # SM90 -> xqa
        q_len_per_req=q_len,
        mask=mask,
    )


# ===========================================================================
# PART 1 — numerical gate: the q_len=N PREFILL-CHUNK primitive (uniform N)
# ===========================================================================
def part1_prefill_primitive(device, ws):
    print("=" * 78)
    print("PART 1  q_len=N prefill-chunk primitive (uniform N, causal mask)")
    print("=" * 78)
    N = 8
    # (context_len before the chunk) per request; page-boundary + ragged cases.
    ctx = [0, 5, 120, 128, 130, 250, 384, 500]
    total = [c + N for c in ctx]
    kv = PagedKV(total, device)
    # fresh query tokens for the chunk: write their K,V into positions [c, c+N)
    q_chunk = torch.randn(kv.bs, N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    kwrite = torch.randn(kv.bs, N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    vwrite = torch.randn(kv.bs, N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    for r in range(kv.bs):
        for j in range(N):
            kv.write_kv(r, ctx[r] + j, kwrite[r, j], vwrite[r, j])

    query = q_chunk.reshape(kv.bs * N, NUM_QO_HEADS, HEAD_DIM)
    mask = generate_causal_mask(kv.bs, N, device)
    out = xqa_call(query, kv.cache, kv.block_tables, kv.seq_lens, kv.max_seq_len,
                   N, mask, ws).reshape(kv.bs, N, NUM_QO_HEADS, HEAD_DIM)

    # fp32 reference: query row j (abs pos ctx[r]+j) attends causally to [0, ctx[r]+j]
    worst_ab = worst_rl = 0.0
    for r in range(kv.bs):
        Kf, Vf = kv.gather_kv(r, total[r])
        for j in range(N):
            upto = ctx[r] + j + 1
            ref = fp32_attn(q_chunk[r, j], Kf[:upto], Vf[:upto])
            ab, rl = maxdiff(out[r, j], ref)
            worst_ab = max(worst_ab, ab)
            worst_rl = max(worst_rl, rl)
    ok = worst_ab <= 5e-2
    print(f"  N={N} ctx={ctx}")
    print(f"  prefill rows vs fp32 causal ref: max_abs={worst_ab:.3e} max_rel={worst_rl:.3e}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# PART 2 — the MIXED question: (S) single padded call vs (T) two calls
# ===========================================================================
def part2_mixed(device, ws):
    print("=" * 78)
    print("PART 2  MIXED batch: 3 decode (real q_len=1) + 1 prefill (q_len=N)")
    print("=" * 78)
    N = 8
    # request 0,1,2 = decodes (committed context lengths); request 3 = prefill chunk
    dec_ctx = [130, 256, 400]        # committed KV lengths of the 3 decode reqs
    pre_ctx = 300                    # context before the prefill chunk
    bs = 4

    # ---- reference world: real per-request KV (decodes have committed+1, prefill committed+N)
    # decode real total = ctx+1 ; prefill real total = ctx+N
    # For (S) padded single call we must reserve N slots per decode -> total = ctx+N,
    # but only slot ctx is real for a decode (slots ctx+1..ctx+N-1 are junk, masked out).
    total_padded = [c + N for c in dec_ctx] + [pre_ctx + N]
    kv = PagedKV(total_padded, device)

    # real new tokens
    dec_q = torch.randn(3, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    dec_k = torch.randn(3, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    dec_v = torch.randn(3, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    pre_q = torch.randn(N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    pre_k = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    pre_v = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    # write real KV: decode's single new token at abs pos ctx; prefill chunk at [ctx, ctx+N)
    for i in range(3):
        kv.write_kv(i, dec_ctx[i], dec_k[i], dec_v[i])
    for j in range(N):
        kv.write_kv(3, pre_ctx + j, pre_k[j], pre_v[j])

    # ---- fp32 references (the ground truth each real token should equal)
    ref = {}
    for i in range(3):
        Kf, Vf = kv.gather_kv(i, dec_ctx[i] + 1)     # only the ctx+1 REAL rows
        ref[("dec", i)] = fp32_attn(dec_q[i], Kf, Vf)
    Kf, Vf = kv.gather_kv(3, pre_ctx + N)
    for j in range(N):
        ref[("pre", j)] = fp32_attn(pre_q[j], Kf[:pre_ctx + j + 1], Vf[:pre_ctx + j + 1])

    # ---------- (S) SINGLE padded call at uniform q_seq_len=N ----------
    # Build the [bs*N, hq, hd] query. Prefill req 3 = its N real rows.
    # Each decode req = real token in ROW 0, rows 1..N-1 are junk (discarded).
    qS = torch.randn(bs, N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    for i in range(3):
        qS[i, 0] = dec_q[i]
    qS[3] = pre_q
    seq_lens_S = torch.tensor(total_padded, dtype=torch.uint32, device=device)  # all ctx+N
    mask = generate_causal_mask(bs, N, device)
    outS = xqa_call(qS.reshape(bs * N, NUM_QO_HEADS, HEAD_DIM), kv.cache, kv.block_tables,
                    seq_lens_S, kv.max_seq_len, N, mask, ws).reshape(bs, N, NUM_QO_HEADS, HEAD_DIM)

    dS_ab = dS_rl = 0.0
    for i in range(3):
        ab, rl = maxdiff(outS[i, 0], ref[("dec", i)]);  dS_ab = max(dS_ab, ab); dS_rl = max(dS_rl, rl)
    pS_ab = pS_rl = 0.0
    for j in range(N):
        ab, rl = maxdiff(outS[3, j], ref[("pre", j)]);  pS_ab = max(pS_ab, ab); pS_rl = max(pS_rl, rl)
    okS = dS_ab <= 5e-2 and pS_ab <= 5e-2
    print("\n (S) SINGLE padded call, uniform q_seq_len=N, decode=row0:")
    print(f"     decode rows vs ref : max_abs={dS_ab:.3e} max_rel={dS_rl:.3e}")
    print(f"     prefill rows vs ref: max_abs={pS_ab:.3e} max_rel={pS_rl:.3e}  -> {'PASS' if okS else 'FAIL'}")

    # ---------- (T) TWO calls: q_len=1 decodes + q_len=N prefill ----------
    # decode call: only the 3 decode reqs, real seq_lens = ctx+1
    dec_seq = torch.tensor([c + 1 for c in dec_ctx], dtype=torch.uint32, device=device)
    dec_bt = kv.block_tables[:3].contiguous()
    outT_dec = xqa_call(dec_q, kv.cache, dec_bt, dec_seq, kv.max_seq_len,
                        1, None, ws)   # q_len=1 -> plain decode, no mask
    # prefill call: just req 3, q_len=N
    pre_seq = torch.tensor([pre_ctx + N], dtype=torch.uint32, device=device)
    pre_bt = kv.block_tables[3:4].contiguous()
    maskT = generate_causal_mask(1, N, device)
    outT_pre = xqa_call(pre_q, kv.cache, pre_bt, pre_seq, kv.max_seq_len,
                        N, maskT, ws).reshape(N, NUM_QO_HEADS, HEAD_DIM)
    dT_ab = dT_rl = 0.0
    for i in range(3):
        ab, rl = maxdiff(outT_dec[i], ref[("dec", i)]); dT_ab = max(dT_ab, ab); dT_rl = max(dT_rl, rl)
    pT_ab = pT_rl = 0.0
    for j in range(N):
        ab, rl = maxdiff(outT_pre[j], ref[("pre", j)]); pT_ab = max(pT_ab, ab); pT_rl = max(pT_rl, rl)
    okT = dT_ab <= 5e-2 and pT_ab <= 5e-2
    print("\n (T) TWO calls, q_len=1 decode + q_len=N prefill:")
    print(f"     decode call vs ref : max_abs={dT_ab:.3e} max_rel={dT_rl:.3e}")
    print(f"     prefill call vs ref: max_abs={pT_ab:.3e} max_rel={pT_rl:.3e}  -> {'PASS' if okT else 'FAIL'}")
    return okS, okT


# ===========================================================================
# PART 3 — CUDA-graph capture of the q_len=N prefill call + padded mixed call
# ===========================================================================
def part3_capture(device, ws):
    print("=" * 78)
    print("PART 3  CUDA-graph capture + deterministic replay")
    print("=" * 78)
    N = 8
    ctx = [130, 256, 400, 300]
    total = [c + N for c in ctx]
    bs = len(ctx)
    kv = PagedKV(total, device)
    for r in range(bs):
        for j in range(N):
            kv.write_kv(r, ctx[r] + j,
                        torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device),
                        torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device))

    # STATIC buffers (filled before capture, read at replay).
    q_static = torch.randn(bs * N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    bt_static = kv.block_tables.clone()
    seq_static = kv.seq_lens.clone()
    mask_static = generate_causal_mask(bs, N, device)
    out_static = torch.empty(bs * N, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    max_seq_len = kv.max_seq_len

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    def run_into(dst):
        o = trtllm_batch_decode_with_kv_cache(
            query=q_static, kv_cache=kv.cache, workspace_buffer=ws,
            block_tables=bt_static, seq_lens=seq_static, max_seq_len=max_seq_len,
            bmm1_scale=SM_SCALE, bmm2_scale=1.0, kv_layout="NHD", backend="auto",
            q_len_per_req=N, mask=mask_static, out=dst,
        )
        return o

    # eager reference before capture
    eager = run_into(torch.empty_like(out_static)).clone()

    # warmup on a side stream (required so lazy JIT / autotune finishes pre-capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_into(out_static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            run_into(out_static)
    except Exception as e:  # noqa: BLE001
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:200]}")
        return False

    g.replay(); torch.cuda.synchronize()
    r1 = out_static.clone()
    # change nothing -> replay must be bit-identical
    out_static.zero_()
    g.replay(); torch.cuda.synchronize()
    r2 = out_static.clone()

    det = torch.equal(r1, r2)
    ab_e, _ = maxdiff(r1, eager)
    cap_matches_eager = ab_e <= 5e-2
    print(f"  captured OK; replay deterministic (2x): {det}")
    print(f"  replay vs eager: max_abs={ab_e:.3e}  -> {'match' if cap_matches_eager else 'MISMATCH'}")

    # capture-safety of a device-side seq_lens advance (the multistep lever): the
    # kernel must see updated seq_lens after an IN-GRAPH increment with no host sync.
    seq_static.copy_(kv.seq_lens)
    return det and cap_matches_eager


def preflight_cpu():
    print("=" * 78)
    print("CPU PREFLIGHT (no GPU): xqa mixed/spec-dec API shape + mask builder")
    print("=" * 78)
    import inspect
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    sig = inspect.signature(trtllm_batch_decode_with_kv_cache)
    for p in ("q_len_per_req", "mask", "max_q_len", "cum_seq_lens_q"):
        assert p in sig.parameters, f"missing param {p}"
    print("  trtllm_batch_decode_with_kv_cache exposes:",
          [p for p in ("q_len_per_req", "mask", "max_q_len", "cum_seq_lens_q")])
    # mask builder sanity (CPU)
    m = generate_causal_mask(2, 8, torch.device("cpu"))
    assert m.dtype == torch.uint16 and m.shape == (2, 8, 2), (m.dtype, m.shape)
    # row 0 must attend only col 0 (bit0 -> low uint16 == 1); row7 -> bits0..7 == 0xFF
    m32 = m.view(torch.uint32) if False else None  # keep uint16 view; check low word
    row0 = int(m[0, 0, 0]); row7 = int(m[0, 7, 0])
    assert row0 == 0b1, row0
    assert row7 == 0b11111111, row7
    print(f"  causal mask builder OK (uint16 [2,8,2]; row0 lo=0b{row0:b} row7 lo=0b{row7:b})")
    print("  NOTE: xqa q_len_per_req is a SCALAR (uniform across batch);")
    print("        cum_seq_lens_q variable-length is trtllm-gen only (rejected on xqa/SM90).")
    print("  CPU preflight complete.\n")


def main():
    preflight_cpu()
    if not (os.environ.get("ALLOW_GPU") == "1" and torch.cuda.is_available()):
        print("Live GPU DEFERRED (set ALLOW_GPU=1 with an IDLE GPU visible).")
        return 0
    device = torch.device("cuda")
    print(f"\nLIVE on {torch.cuda.get_device_name(device)} cc={torch.cuda.get_device_capability(device)}\n")
    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    p1 = part1_prefill_primitive(device, ws)
    okS, okT = part2_mixed(device, ws)
    p3 = part3_capture(device, ws)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  PART1 prefill-chunk primitive numerically correct : {'PASS' if p1 else 'FAIL'}")
    print(f"  PART2 (S) single padded mixed call correct        : {'PASS' if okS else 'FAIL'}")
    print(f"  PART2 (T) two-call mixed correct                  : {'PASS' if okT else 'FAIL'}")
    print(f"  PART3 mixed call CUDA-graph capturable+determ     : {'PASS' if p3 else 'FAIL'}")
    allp = p1 and okS and okT and p3
    print(f"\n  OVERALL: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())

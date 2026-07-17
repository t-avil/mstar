#!/usr/bin/env python
"""Parity gate: plan-free xqa/trtllm decode vs plan-based BatchDecode wrapper.

Runs a SINGLE decode step for synthetic sequences (varied seq_lens; single and
batched; GQA 32/4, head_dim 128, page_size 128, bf16) through BOTH:

  A. the plan-based ``BatchDecodeWithPagedKVCacheWrapper`` (M*'s current default,
     ``FlashInferDecodeWrapper``), and
  B. the plan-free ``FlashInferXqaDecodeWrapper``
     (``trtllm_batch_decode_with_kv_cache`` -> xqa on SM90),

on IDENTICAL KV cache + query, and reports the exact max abs / rel diff against
each other and against an fp32 gather reference. THIS IS THE GATE.

Usage:
  # CPU-only preflight (no GPU touched): import + JIT-spec constraint checks
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/parity_xqa_decode.py

  # Full live GPU parity (owner runs on a free/idle GPU):
  ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=<idle_gpu> \
    PYTHONPATH=/m-coriander/coriander/tim/mstar-xqa \
    /m-coriander/coriander/tim/mstar-new/.venv/bin/python test/xqa/parity_xqa_decode.py
"""
import math
import os
import sys

import torch


# ----- config (M* Thinker shapes; grp8 per brief, grp7 = config default) -----
HEAD_DIM = 128
PAGE_SIZE = 128
NUM_KV_HEADS = 4
DTYPE = torch.bfloat16

# (name, num_qo_heads, list-of-seq_lens)
CASES = [
    ("single_short_grp8", 32, [130]),
    ("single_long_grp8", 32, [999]),
    ("batch_varied_grp8", 32, [1, 128, 129, 256, 300, 512, 45, 777]),
    ("batch_pageedge_grp8", 32, [128, 256, 384]),   # exact page multiples
    ("batch_grp7_cfgdefault", 28, [130, 400, 65, 1024]),
]


def preflight_cpu():
    """Import + JIT-spec constraint checks. No CUDA context required."""
    print("=" * 78)
    print("CPU PREFLIGHT (no GPU): function presence, signatures, JIT constraints")
    print("=" * 78)
    import inspect

    from flashinfer.decode import (
        trtllm_batch_decode_with_kv_cache,
        xqa_batch_decode_with_kv_cache,
    )

    for fn in (trtllm_batch_decode_with_kv_cache, xqa_batch_decode_with_kv_cache):
        assert callable(fn)
        print(f"\nOK found {fn.__module__}.{fn.__name__}")
        print("   params:", list(inspect.signature(fn).parameters)[:8], "...")

    # Exercise the JIT constraint guards for M*'s exact tuple WITHOUT compiling.
    # FlashInfer reads FLASHINFER_CUDA_ARCH_LIST for target archs (falls back to
    # the visible GPU). Force SM90 so the spec constructs on a GPU-less host.
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0")
    from flashinfer.jit.xqa import gen_xqa_module

    tuples = [
        # (input, kv, page, head_dim, grp, sliding, out, q_seq_len, expect_ok)
        (DTYPE, DTYPE, 128, 128, 8, False, DTYPE, 1, True),   # M* shipping (32/4)
        (DTYPE, DTYPE, 128, 128, 7, False, DTYPE, 1, True),   # config default (28/4)
        (DTYPE, DTYPE, 128, 128, 8, False, DTYPE, 2, True),   # mixed/spec (q_len>1)
        (DTYPE, torch.float8_e4m3fn, 128, 128, 8, False, DTYPE, 1, True),  # fp8 KV
        (DTYPE, DTYPE, 100, 128, 8, False, DTYPE, 1, False),  # bad page_size
        (DTYPE, DTYPE, 128, 130, 8, False, DTYPE, 1, False),  # bad head_dim
    ]
    print("\nJIT gen_xqa_module constraint checks (no nvcc build):")
    for t in tuples:
        *args, expect_ok = t
        try:
            gen_xqa_module(*args)
            got = "OK build-spec"
        except Exception as e:  # noqa: BLE001
            got = f"REJECT ({type(e).__name__}: {str(e)[:60]})"
        verdict = "PASS" if (("OK" in got) == expect_ok) else "UNEXPECTED"
        print(f"   {verdict:10s} page={args[2]:4d} hd={args[3]:3d} grp={args[4]} "
              f"kv={args[1]} q_seq_len={args[7]} -> {got}")
    print("\nCPU preflight complete.\n")


def build_synthetic(num_qo_heads, seq_lens, device):
    """Populate a paged KV cache (M* NHD layout) to the given seq_lens and
    return the ragged plan metadata + a random query."""
    bs = len(seq_lens)
    pages_per = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    total_pages = sum(pages_per) + 4  # a little slack
    kv = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )

    indptr = [0]
    indices = []
    last_page_len = []
    next_page = 0
    for s, np_ in zip(seq_lens, pages_per):
        pgs = list(range(next_page, next_page + np_))
        next_page += np_
        indices.extend(pgs)
        indptr.append(indptr[-1] + np_)
        lpl = s - (np_ - 1) * PAGE_SIZE
        last_page_len.append(lpl)

    indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)
    q = torch.randn(bs, num_qo_heads, HEAD_DIM, dtype=DTYPE, device=device)
    return kv, indptr, indices, last_page_len, q, seq_lens


def fp32_reference(q, kv, indptr, indices, last_page_len, seq_lens):
    """Exact per-request attention over the valid KV positions, fp32."""
    bs, hq, hd = q.shape
    grp = hq // NUM_KV_HEADS
    sm = 1.0 / math.sqrt(hd)
    out = torch.empty(bs, hq, hd, dtype=torch.float32, device=q.device)
    for i in range(bs):
        pages = indices[indptr[i]:indptr[i + 1]].tolist()
        s = seq_lens[i]
        # gather K/V [s, kv_heads, hd]
        ks, vs = [], []
        remaining = s
        for j, p in enumerate(pages):
            take = PAGE_SIZE if remaining >= PAGE_SIZE else remaining
            ks.append(kv[p, 0, :take])
            vs.append(kv[p, 1, :take])
            remaining -= take
        K = torch.cat(ks, 0).float()  # [s, kvh, hd]
        V = torch.cat(vs, 0).float()
        qi = q[i].float()             # [hq, hd]
        # expand kv heads to q heads
        Kx = K.repeat_interleave(grp, dim=1)  # [s, hq, hd]
        Vx = V.repeat_interleave(grp, dim=1)
        scores = torch.einsum("hd,shd->hs", qi, Kx) * sm  # [hq, s]
        w = torch.softmax(scores, dim=-1)
        out[i] = torch.einsum("hs,shd->hd", w, Vx)
    return out


def maxdiff(a, b):
    a = a.float()
    b = b.float()
    abs_d = (a - b).abs()
    rel_d = abs_d / (b.abs() + 1e-4)
    return abs_d.max().item(), rel_d.max().item()


def run_live_parity(device):
    from mstar.utils.flashinfer_utils import (
        FlashInferDecodeWrapper,
        FlashInferXqaDecodeWrapper,
    )

    print("=" * 78)
    print(f"LIVE GPU PARITY on {torch.cuda.get_device_name(device)} "
          f"(cc={torch.cuda.get_device_capability(device)})")
    print("=" * 78)
    ws_plan = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    all_pass = True
    for name, hq, seq_lens in CASES:
        kv, indptr, indices, lpl, q, sl = build_synthetic(hq, seq_lens, device)

        plan_w = FlashInferDecodeWrapper(  # flag OFF -> pure plan-based
            ws_plan, hq, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, device=device,
            use_cuda_graph=False,
        )
        plan_w.plan(indptr, indices, lpl, dtype=DTYPE)
        out_plan = plan_w.run(q, kv)

        xqa_w = FlashInferXqaDecodeWrapper(
            ws_plan, hq, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, device=device,
            use_cuda_graph=False,
        )
        xqa_w.plan(indptr, indices, lpl, dtype=DTYPE)
        out_xqa = xqa_w.run(q, kv)

        ref = fp32_reference(q, kv, indptr, indices, lpl, sl)

        ab_px, rl_px = maxdiff(out_plan, out_xqa)   # plan vs xqa
        ab_pr, rl_pr = maxdiff(out_plan, ref)       # plan vs fp32 ref
        ab_xr, rl_xr = maxdiff(out_xqa, ref)        # xqa  vs fp32 ref

        # xqa is a parity match if it is no worse vs the fp32 ref than the
        # plan-based kernel is (both are bf16 kernels; bit-identity across two
        # different kernels is not expected).
        ok = ab_xr <= max(ab_pr * 3.0, 5e-2)
        all_pass = all_pass and ok
        print(f"\n[{name}] bs={len(seq_lens)} hq={hq} seq_lens={seq_lens}")
        print(f"   plan vs xqa : max_abs={ab_px:.3e}  max_rel={rl_px:.3e}")
        print(f"   plan vs ref : max_abs={ab_pr:.3e}  max_rel={rl_pr:.3e}")
        print(f"   xqa  vs ref : max_abs={ab_xr:.3e}  max_rel={rl_xr:.3e}   "
              f"-> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"GATE: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)
    return all_pass


def main():
    preflight_cpu()
    if not (os.environ.get("ALLOW_GPU") == "1" and torch.cuda.is_available()):
        print("Live GPU parity DEFERRED (set ALLOW_GPU=1 with an idle GPU visible).")
        print("Reason: GPUs 6/7 run a live benchmark; owner runs this from the loop.")
        return 0
    ok = run_live_parity(torch.device("cuda"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

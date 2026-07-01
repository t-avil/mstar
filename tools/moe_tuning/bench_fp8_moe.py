#!/usr/bin/env python
"""Validate block-fp8 w8a8 MoE vs bf16: cosine + speedup at decode shapes."""
import torch
from mstar.utils.fused_moe.runner import fused_experts
from mstar.utils.fused_moe.fp8 import fused_experts_fp8, per_block_cast_to_fp8_weight

E, TOPK, H, I = 128, 8, 2048, 768
DT = torch.bfloat16
BATCHES = [1, 2, 4, 8, 16, 32]


def inputs(B):
    torch.manual_seed(100 + B)
    hs = torch.randn(B, H, device="cuda", dtype=DT) * 0.1
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=DT) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=DT) * 0.02
    g = torch.randn(B, E, device="cuda")
    tw, tid = torch.topk(torch.softmax(g, -1), TOPK, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(DT)
    return hs, w1, w2, tw, tid


def bench(fn, iters=80, reps=3):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


print(f"# torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"{'B':>3} {'bf16_ms':>9} {'fp8_ms':>9} {'speedup':>8} {'cosine':>9} {'maxrel':>9}")
for B in BATCHES:
    hs, w1, w2, tw, tid = inputs(B)
    w1q, w1s = per_block_cast_to_fp8_weight(w1)
    w2q, w2s = per_block_cast_to_fp8_weight(w2)
    o_bf = fused_experts(hs, w1, w2, tw, tid)
    o_fp = fused_experts_fp8(hs, w1q, w1s, w2q, w2s, tw, tid)
    cos = torch.nn.functional.cosine_similarity(o_bf.flatten().float(), o_fp.flatten().float(), dim=0).item()
    denom = o_bf.abs().float().mean().clamp_min(1e-6)
    maxrel = ((o_bf - o_fp).abs().float().max() / denom).item()
    t_bf = bench(lambda: fused_experts(hs, w1, w2, tw, tid))
    t_fp = bench(lambda: fused_experts_fp8(hs, w1q, w1s, w2q, w2s, tw, tid))
    print(f"{B:>3} {t_bf:>9.4f} {t_fp:>9.4f} {t_bf/t_fp:>7.2f}x {cos:>9.5f} {maxrel:>9.3f}")

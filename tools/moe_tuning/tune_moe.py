#!/usr/bin/env python
"""Offline autotune for M*'s Qwen3-Omni Thinker MoE grouped-GEMM (bf16).

Sweeps Triton tile / warp / stage configs for the real decode shapes
(E=128, top_k=8, hidden=2048, moe_inter=768) and reports the best config per
decode batch B, vs the shipped static `get_default_config` (16x32x64).

Times the full `fused_experts` (moe_align + gate_up GEMM + SwiGLU + down GEMM +
sum-reduce) — the real per-layer MoE decode latency. Numerically identical
across configs; pure kernel-speed search.

Run:  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<mstar-opt> python tune_moe.py
"""
import itertools, json, sys, time, traceback
import torch

import mstar.utils.fused_moe.runner as runner
from mstar.utils.fused_moe.runner import fused_experts

DEV = "cuda"
DT = torch.bfloat16
E, TOPK, HIDDEN, INTER = 128, 8, 2048, 768
BATCHES = [1, 2, 4, 8, 16, 32]

# shipped baseline (kernels.py get_default_config, M<=E branch)
BASELINE = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}

# search grid
GRID = dict(
    BLOCK_SIZE_M=[16, 32, 64],
    BLOCK_SIZE_N=[32, 64, 128, 256],
    BLOCK_SIZE_K=[64, 128, 256],
    GROUP_SIZE_M=[1],
    num_warps=[4, 8],
    num_stages=[2, 3, 4],
)


def make_inputs(B):
    torch.manual_seed(1234 + B)
    hs = torch.randn(B, HIDDEN, device=DEV, dtype=DT) * 0.1
    w1 = torch.randn(E, 2 * INTER, HIDDEN, device=DEV, dtype=DT) * 0.02
    w2 = torch.randn(E, HIDDEN, INTER, device=DEV, dtype=DT) * 0.02
    gate = torch.randn(B, E, device=DEV, dtype=torch.float32)
    tw, tid = torch.topk(torch.softmax(gate, dim=-1), TOPK, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(DT)
    return hs, w1, w2, tw, tid


def bench(cfg, hs, w1, w2, tw, tid, iters=60, reps=3):
    runner.get_default_config = lambda **k: dict(cfg)  # monkeypatch
    # warmup / compile
    for _ in range(15):
        fused_experts(hs, w1, w2, tw, tid)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fused_experts(hs, w1, w2, tw, tid)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)  # ms/call
    return best


def main():
    print(f"# torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"# {len(combos)} configs/batch x {len(BATCHES)} batches")
    results = {}
    for B in BATCHES:
        hs, w1, w2, tw, tid = make_inputs(B)
        base = bench(BASELINE, hs, w1, w2, tw, tid)
        rows = []
        for i, cfg in enumerate(combos):
            try:
                t = bench(cfg, hs, w1, w2, tw, tid)
                rows.append((t, cfg))
            except Exception as ex:
                if "out of resource" not in str(ex).lower() and "shared memory" not in str(ex).lower():
                    pass  # skip invalid/oom configs silently
            if (i + 1) % 20 == 0:
                print(f"  B={B}: {i+1}/{len(combos)}  best={min(rows)[0]:.4f}ms" if rows else f"  B={B}: {i+1}", flush=True)
        rows.sort(key=lambda r: r[0])
        best_t, best_cfg = rows[0]
        results[B] = {"baseline_ms": base, "best_ms": best_t, "speedup": base / best_t,
                      "best_cfg": best_cfg, "top5": [(round(t, 4), c) for t, c in rows[:5]]}
        print(f"B={B:2d}  baseline={base:.4f}ms  best={best_t:.4f}ms  "
              f"speedup={base/best_t:.2f}x  cfg={best_cfg}", flush=True)
    with open("/m-coriander/coriander/tim/tune_moe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n# ==== SUMMARY (best config per batch) ====")
    for B in BATCHES:
        r = results[B]
        print(f"B={B:2d}  {r['speedup']:.2f}x  {r['best_cfg']}")


if __name__ == "__main__":
    main()

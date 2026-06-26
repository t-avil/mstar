#!/usr/bin/env python
"""Fast native-only encoder A/B: times the native Qwen3-Omni vision & audio
encoder forward under each varlen backend (dense / per_segment / padded), across
batch sizes, with mean±std. Random weights (perf is value-independent), so no
checkpoint load and no slow HF baseline -> iterates in ~1 min.

Usage: python bench_encoder_fast.py [--repeats 8]
"""
import sys, time, argparse, statistics as st
# Hard-block flash-attn so we exercise the SDPA varlen fallbacks (this H200 path).
sys.modules.setdefault("flash_attn", None)
import torch
from functools import partial

DTYPE = torch.bfloat16
DEV = "cuda:0"
BATCHES = [1, 4, 8, 16, 32]
import mstar.model.qwen3_omni.components.audio_encoder as AE


def measure(fn, repeats, n_iter, n_warmup):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / n_iter * 1000.0)
    return out


def summarize(xs):
    return {"mean": round(st.mean(xs), 3), "std": round(st.pstdev(xs), 3)}


def build():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeAudioEncoderConfig)
    from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder
    vcfg = Qwen3OmniMoeVisionEncoderConfig()
    acfg = Qwen3OmniMoeAudioEncoderConfig()
    acfg.n_window = 50            # published Qwen3-Omni-30B audio frontend
    acfg.n_window_infer = 800     # (required: defaults produce invalid windowing)
    vis = NativeQwen3OmniVisionEncoder(vcfg).to(DEV, DTYPE).eval()
    aud = NativeQwen3OmniAudioEncoder(acfg).to(DEV, DTYPE).eval()
    return vis, vcfg, aud, acfg


def vision_input(cfg, n):
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    grid = torch.tensor([[1, 26, 28]] * n, dtype=torch.long, device=DEV)
    npatch = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    return torch.randn(npatch, rows, device=DEV, dtype=DTYPE), grid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name()} dtype={DTYPE}")
    vis, vcfg, aud, acfg = build()
    frames = 3000
    backends = ["dense", "per_segment", "adaptive"]

    for kind in ("VISION (ms/img, 728 patches)", "AUDIO (ms/req, 30s)"):
        print(f"\n===== {kind} =====")
        print(f"{'batch':>6s} " + " ".join(f"{b:>16s}" for b in backends))
        for n in BATCHES:
            if kind.startswith("VISION"):
                pv, g = vision_input(vcfg, n)
                call = lambda: vis(pv, grid_thw=g)
                ni, nw, denom = 5, 3, n
            else:
                lens = torch.full((n,), frames, dtype=torch.long, device=DEV)
                feats = torch.randn(acfg.num_mel_bins, int(lens.sum()), device=DEV, dtype=DTYPE)
                call = lambda: aud(feats, feature_lens=lens)
                ni, nw, denom = 5, 3, n
            cells = []
            for bk in backends:
                AE._VARLEN_BACKEND = bk
                try:
                    with torch.no_grad():
                        s = [v / denom for v in measure(call, args.repeats, ni, nw)]
                    cells.append(f"{st.mean(s):7.2f}±{st.pstdev(s):4.2f}")
                except Exception as e:
                    cells.append(f"ERR:{str(e)[:10]}")
                torch.cuda.empty_cache()
            print(f"{n:>6d} " + " ".join(f"{c:>16s}" for c in cells))
    print("\n(lower=better; dense=old baseline, per_segment=shipped, padded=audio candidate)")


if __name__ == "__main__":
    main()

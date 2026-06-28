#!/usr/bin/env python
"""Audio-encoder varlen backend matrix: time the NATIVE Qwen3-Omni audio encoder
forward under each varlen backend across batch sizes, emitting every datapoint.

Backends compared (the real served choices on this H200, which HAS flash_attn):
  flash_attn  -> _FLASH_ATTN_AVAILABLE=True  (flash_attn_varlen_func)
  flashinfer  -> _FLASH_ATTN_AVAILABLE=False, _VARLEN_BACKEND=flashinfer (ragged)
  per_segment -> dense SDPA per segment       (O(sum L_i^2), linear in batch)
  padded      -> pad-to-max + 1 batched SDPA   (good for equal-length windows)
  adaptive    -> shape heuristic (dense vs per_segment, tau=5e5)
  dense       -> block-diagonal masked SDPA    (O(total^2), old baseline)

Random weights (perf is value-independent) so no checkpoint load. Audio segment
layout for a 30s clip (frames=3000): ~390 post-CNN tokens -> 4 windows ~[104,104,104,78].

Usage: python bench_audio_backend_matrix.py [--repeats 30] [--warmup 5] [--frames 3000] --out raw.json
"""
import sys, os, time, json, argparse, statistics as st, datetime
import torch

DTYPE = torch.bfloat16
DEV = "cuda:0"  # CUDA_VISIBLE_DEVICES pins the physical device

import mstar.model.qwen3_omni.components.audio_encoder as AE


def build_audio(frames_cfg=None):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig)
    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder
    acfg = Qwen3OmniMoeAudioEncoderConfig()
    acfg.n_window = 50            # published Qwen3-Omni-30B audio frontend
    acfg.n_window_infer = 800     # (defaults produce invalid windowing)
    aud = NativeQwen3OmniAudioEncoder(acfg).to(DEV, DTYPE).eval()
    return aud, acfg


def set_backend(name):
    if name == "flash_attn":
        AE._FLASH_ATTN_AVAILABLE = True
    else:
        AE._FLASH_ATTN_AVAILABLE = False
        AE._VARLEN_BACKEND = name


def seg_structure(aud, acfg, n, frames):
    """Report (n_seg, total_tokens) for this batch layout."""
    lens = torch.full((n,), frames, dtype=torch.long, device=DEV)
    feats = torch.randn(acfg.num_mel_bins, int(lens.sum()), device=DEV, dtype=DTYPE)
    from mstar.model.qwen3_omni.components.audio_encoder import (
        chunk_and_pad_features, get_audio_cu_seqlens)
    padded_feature, chunk_lengths = chunk_and_pad_features(feats, lens, acfg.n_window)
    cu = get_audio_cu_seqlens(chunk_lengths, lens, acfg.n_window_infer, acfg.n_window)
    total = int(cu[-1]); n_seg = cu.shape[0] - 1
    return n_seg, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=3000)  # 30s @ 100 fps mel
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--backends", type=str,
                    default="flash_attn,flashinfer,per_segment,padded,adaptive,dense")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--commit", type=str, default="")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    backends = [b for b in args.backends.split(",")]
    dev_name = torch.cuda.get_device_name()
    print(f"device={dev_name} dtype={DTYPE} frames={args.frames}")

    aud, acfg = build_audio()

    datapoints = []
    seg_info = {}
    for n in batches:
        seg_info[n] = seg_structure(aud, acfg, n, args.frames)
    print("batch layout (n_seg,total):", seg_info)

    print(f"\n{'batch':>6s} " + " ".join(f"{b:>14s}" for b in backends) + "   (ms/req, lower=better)")
    for n in batches:
        lens = torch.full((n,), args.frames, dtype=torch.long, device=DEV)
        feats = torch.randn(acfg.num_mel_bins, int(lens.sum()), device=DEV, dtype=DTYPE)
        n_seg, total = seg_info[n]
        cells = []
        for bk in backends:
            set_backend(bk)
            call = lambda: aud(feats, feature_lens=lens)
            try:
                with torch.no_grad():
                    for _ in range(args.warmup):
                        call()
                    torch.cuda.synchronize()
                    per_req = []
                    for it in range(args.repeats):
                        t0 = time.perf_counter()
                        call()
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - t0) * 1000.0
                        per_req.append(ms / n)
                        datapoints.append({
                            "backend": bk, "batch": n, "iter": it,
                            "phase": "measure", "value": round(ms / n, 5),
                            "value_total_ms": round(ms, 5),
                            "n_seg": n_seg, "total_tokens": total,
                        })
                    cells.append(f"{st.median(per_req):6.3f}±{st.pstdev(per_req):5.3f}")
            except Exception as e:
                cells.append(f"ERR:{str(e)[:8]}")
                datapoints.append({"backend": bk, "batch": n, "phase": "error",
                                   "error": str(e)[:200]})
            torch.cuda.empty_cache()
        print(f"{n:>6d} " + " ".join(f"{c:>14s}" for c in cells))

    out = {
        "benchmark": "audio_encoder_varlen_backend_matrix",
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": args.commit,
        "device": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   "gpu_name": dev_name},
        "units": "ms_per_request",
        "note": "value = forward_ms/batch (per-request encoder forward latency)",
        "frames": args.frames, "warmup_iters": args.warmup, "repeats": args.repeats,
        "seg_structure": {str(k): {"n_seg": v[0], "total_tokens": v[1]}
                          for k, v in seg_info.items()},
        "backends": backends, "batches": batches,
        "status": "complete",
        "datapoints": datapoints,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}  ({len(datapoints)} datapoints)")


if __name__ == "__main__":
    main()

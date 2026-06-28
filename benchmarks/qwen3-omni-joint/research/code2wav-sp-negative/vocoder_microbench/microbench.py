"""Code2Wav vocoder forward microbenchmark: SP-OFF vs SP-ON.

Isolates the vocoder forward (code embedding + pre_transformer + upsample/decoder
conv stack) and times it across codec-frame chunk sizes for:

  * sp_off    : production single-device path (torch.compile'd forward).
  * sp_single : SP on, nshard=2, single device (compute split, no xfer).
  * sp_xdev   : SP on, nshard=2, devices=cuda:0,cuda:1 (real cross-device SP).

The production serving chunk is codec_left_context(25)+codec_chunk(25)=50 frames;
larger sizes show whether/where the 2-way compute split overcomes the loss of
CUDA-graph capture + cross-device copy overhead. Random weights: latency is a
property of shapes/ops, not weight values.

Writes raw.json (every datapoint, units=ms, phase tag) to this dir.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

CHUNKS = [50, 100, 200, 400, 800]
WARMUP = 8
MEASURE = 50


def _clear_sp_env():
    for k in ("MSTAR_CODE2WAV_SP", "MSTAR_CODE2WAV_SP_NSHARD",
              "MSTAR_CODE2WAV_SP_HALO", "MSTAR_CODE2WAV_SP_DEVICES",
              "MSTAR_CODE2WAV_SP_COMPILE"):
        os.environ.pop(k, None)


def build(sp_env: dict, device: str, seed: int = 0):
    _clear_sp_env()
    os.environ.update(sp_env)
    from mstar.model.qwen3_omni.config import Code2WavConfig
    from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
    cfg = Code2WavConfig()
    torch.manual_seed(seed)
    m = Qwen3OmniMoeCode2Wav(cfg).to(device=device, dtype=torch.float32).eval()
    m.consolidate()
    return m, cfg


def time_forward(model, cfg, T: int, device: str):
    Q = cfg.num_quantizers
    torch.manual_seed(7)
    codes = torch.randint(0, cfg.codebook_size, (1, Q, T), device=device)
    pos = torch.arange(T, device=device).unsqueeze(0)
    samples = []
    with torch.no_grad():
        for i in range(WARMUP + MEASURE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.forward(codes, pos)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            samples.append((i, "warmup" if i < WARMUP else "measure", dt))
    return samples


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 2
    dev = "cuda:0"
    ngpu = torch.cuda.device_count()
    variants = [
        ("sp_off", {}, dev),
        ("sp_single", {"MSTAR_CODE2WAV_SP": "1", "MSTAR_CODE2WAV_SP_NSHARD": "2",
                       "MSTAR_CODE2WAV_SP_HALO": "32",
                       "MSTAR_CODE2WAV_SP_COMPILE": "1"}, dev),
    ]
    if ngpu >= 2:
        variants.append(
            ("sp_xdev", {"MSTAR_CODE2WAV_SP": "1", "MSTAR_CODE2WAV_SP_NSHARD": "2",
                         "MSTAR_CODE2WAV_SP_HALO": "32",
                         "MSTAR_CODE2WAV_SP_DEVICES": "cuda:0,cuda:1",
                         "MSTAR_CODE2WAV_SP_COMPILE": "1"}, dev))

    datapoints = []
    summary = {}
    for vname, env, vdev in variants:
        model, cfg = build(env, vdev)
        for T in CHUNKS:
            s = time_forward(model, cfg, T, vdev)
            for (i, phase, dt) in s:
                datapoints.append({"variant": vname, "chunk_frames": T,
                                   "iter": i, "phase": phase, "value": dt})
            meas = [dt for (_, ph, dt) in s if ph == "measure"]
            med = statistics.median(meas)
            summary[(vname, T)] = med
            print(f"{vname:10s} T={T:4d}  median={med:.3f} ms  "
                  f"p10={sorted(meas)[len(meas)//10]:.3f} "
                  f"min={min(meas):.3f}")
        del model
        torch.cuda.empty_cache()

    # speedup table vs sp_off
    print("\n== median ms (lower=better); speedup = sp_off/variant ==")
    print(f"{'chunk':>6} {'sp_off':>10} {'sp_single':>10} {'sp_xdev':>10} "
          f"{'spd_single':>10} {'spd_xdev':>10}")
    for T in CHUNKS:
        off = summary.get(("sp_off", T))
        sg = summary.get(("sp_single", T))
        xd = summary.get(("sp_xdev", T))
        spd_sg = off / sg if sg else float("nan")
        spd_xd = off / xd if xd else float("nan")
        print(f"{T:6d} {off:10.3f} {sg:10.3f} "
              f"{(xd if xd else float('nan')):10.3f} {spd_sg:10.3f} {spd_xd:10.3f}")

    out = {
        "benchmark": "code2wav_vocoder_microbench",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": os.popen(
            "git -C /home/tim/code2wav-sp-wt rev-parse HEAD").read().strip(),
        "device": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "gpu_name": torch.cuda.get_device_name(0), "ngpu": ngpu},
        "units": "ms",
        "warmup_iters": WARMUP,
        "measure_iters": MEASURE,
        "chunk_frames": CHUNKS,
        "note": "vocoder forward latency; production serving chunk = 50 frames "
                "(codec_left_context 25 + codec_chunk 25). sp_off uses compiled "
                "single forward; SP-on disables CUDA-graph capture in serving.",
        "datapoints": datapoints,
        "status": "complete",
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "raw.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(here, 'raw.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

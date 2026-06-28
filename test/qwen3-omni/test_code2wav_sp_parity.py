"""Parity gate for Code2Wav frame-dim sequence parallelism (SP).

A vocoder seam is *audible*, so SP-on must produce a waveform that is
numerically identical to the single-pass (SP-off) vocoder. This test loads the
SAME random raw weights into an SP-off model and an SP-on model, runs both on
the SAME codec tokens, and checks:

  * full-waveform max-abs diff + cosine similarity, and
  * a BOUNDARY-FOCUSED max-abs diff in a window around every shard seam
    (the overlap-add / halo region), which is where a receptive-field
    mis-alignment would show up first.

Two modes are exercised:
  * single-device shard mode  (MSTAR_CODE2WAV_SP=1, no _DEVICES) -- proves the
    halo / seam math in isolation from cross-device transfer.
  * cross-device              (MSTAR_CODE2WAV_SP_DEVICES=cuda:0,cuda:1).

Random weights are sufficient: parity of sharded-vs-single is a property of the
*conv arithmetic*, independent of the weight values. Both paths run eager
(MSTAR_CODE2WAV_SP_COMPILE=0 / direct ``_forward_single`` call) so the only
possible difference is the sharding, not torch.compile fp reordering.

Run directly:
    PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0,1 \
      python test/qwen3-omni/test_code2wav_sp_parity.py
Exit code 0 == all parity checks passed.
"""

from __future__ import annotations

import os
import sys

import torch

# Thresholds. With identical raw weights and eager fp32 on both paths, the only
# residual difference is fp32 reduction-order across different conv input
# lengths -- empirically ~1e-6. A real seam (missing receptive-field context)
# produces O(1e-2..1e0) diffs in the boundary window, so these are strict.
COS_MIN = 0.9999
MAXABS_MAX = 1e-3
BOUNDARY_HALF_WIN = 4096  # samples each side of a seam to scrutinise


def _clear_sp_env() -> None:
    for k in (
        "MSTAR_CODE2WAV_SP",
        "MSTAR_CODE2WAV_SP_NSHARD",
        "MSTAR_CODE2WAV_SP_HALO",
        "MSTAR_CODE2WAV_SP_DEVICES",
        "MSTAR_CODE2WAV_SP_COMPILE",
    ):
        os.environ.pop(k, None)


def _build(sp_env: dict, device: str, seed: int):
    """Construct a Code2Wav model with the given SP env active at __init__."""
    _clear_sp_env()
    os.environ.update(sp_env)
    from mstar.model.qwen3_omni.config import Code2WavConfig
    from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav

    cfg = Code2WavConfig()
    torch.manual_seed(seed)
    model = Qwen3OmniMoeCode2Wav(cfg).to(device=device, dtype=torch.float32)
    model.eval()
    return model, cfg


def _seam_positions(T: int, nshard: int, up: int, C: int) -> list[int]:
    """Cumulative output-sample boundaries between consecutive shards.

    Shard 0 keeps ``up*b0 - C`` samples; interior/last shard k keeps
    ``up*(b_k - b_{k-1})``. The seam after shard k is the running sum.
    """
    bounds = [(k * T) // nshard for k in range(nshard + 1)]
    seams: list[int] = []
    pos = 0
    for k in range(nshard):
        a, b = bounds[k], bounds[k + 1]
        if b <= a:
            continue
        kept = (up * b - C) if a == 0 else up * (b - a)
        pos += kept
        if k < nshard - 1:
            seams.append(pos)
    return seams


def _compare(out_single: torch.Tensor, out_sp: torch.Tensor,
             T: int, nshard: int, up: int, C: int, label: str) -> bool:
    a = out_single.float().reshape(-1).cpu()
    b = out_sp.float().reshape(-1).cpu()
    ok = True
    if a.shape != b.shape:
        print(f"  [{label}] FAIL shape mismatch single={tuple(a.shape)} sp={tuple(b.shape)}")
        return False

    diff = (a - b).abs()
    max_abs = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    seams = _seam_positions(T, nshard, up, C)
    boundary_max = 0.0
    for s in seams:
        lo = max(0, s - BOUNDARY_HALF_WIN)
        hi = min(diff.numel(), s + BOUNDARY_HALF_WIN)
        if hi > lo:
            boundary_max = max(boundary_max, diff[lo:hi].max().item())

    print(f"  [{label}] len={a.numel()} seams={seams}")
    print(f"  [{label}] full max_abs={max_abs:.3e} cos={cos:.8f} "
          f"boundary_max={boundary_max:.3e}")

    if cos < COS_MIN:
        print(f"  [{label}] FAIL cos {cos:.8f} < {COS_MIN}")
        ok = False
    if max_abs > MAXABS_MAX:
        print(f"  [{label}] FAIL max_abs {max_abs:.3e} > {MAXABS_MAX:.1e}")
        ok = False
    if boundary_max > MAXABS_MAX:
        print(f"  [{label}] FAIL boundary seam {boundary_max:.3e} > {MAXABS_MAX:.1e} "
              f"(AUDIBLE SEAM)")
        ok = False
    if ok:
        print(f"  [{label}] PASS")
    return ok


def run_case(T: int, nshard: int, halo: int, devices: str | None,
             seed: int = 0) -> bool:
    base_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    label = (f"T={T} nshard={nshard} halo={halo} "
             f"dev={devices or 'single'}")
    print(f"== case {label} ==")

    # 1) SP-off reference model (eager single forward).
    off, cfg = _build({}, base_dev, seed)
    # 2) SP-on model with identical RAW weights (copy before consolidate).
    sp_env = {
        "MSTAR_CODE2WAV_SP": "1",
        "MSTAR_CODE2WAV_SP_NSHARD": str(nshard),
        "MSTAR_CODE2WAV_SP_HALO": str(halo),
        "MSTAR_CODE2WAV_SP_COMPILE": "0",
    }
    if devices:
        sp_env["MSTAR_CODE2WAV_SP_DEVICES"] = devices
    sp, _ = _build(sp_env, base_dev, seed)
    sp.load_state_dict(off.state_dict())

    off.consolidate()
    sp.consolidate()

    up = int(off.total_upsample)
    C = off._sp_boundary_loss

    Q = cfg.num_quantizers
    torch.manual_seed(1234 + seed)
    codes = torch.randint(0, cfg.codebook_size, (1, Q, T), device=base_dev)
    pos = torch.arange(T, device=base_dev).unsqueeze(0)

    with torch.no_grad():
        out_single = off._forward_single(codes, pos)
        out_sp = sp.forward(codes, pos)  # routes to _forward_sp when SP enabled

    return _compare(out_single, out_sp, T, nshard, up, C, label)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; SP parity test requires a GPU.")
        return 2
    ngpu = torch.cuda.device_count()
    print(f"visible GPUs: {ngpu}")

    results: list[tuple[str, bool]] = []

    # --- Single-device shard mode: proves halo/seam math in isolation. ---
    # Real serving chunk is codec_left_context(25)+codec_chunk(25)=50 frames;
    # also exercise larger T and more shards, and a non-divisible T. All use
    # the production default halo (32); the conv-stack left receptive field is
    # ~4 codec frames (measured: halo>=4 is bit-exact), so 32 has 8x margin.
    single_cases = [
        (50, 2, 32),
        (128, 2, 32),
        (200, 4, 32),
        (97, 3, 32),    # non-divisible frame count
    ]
    for T, n, h in single_cases:
        results.append((f"single T={T} n={n} h={h}", run_case(T, n, h, None)))

    # --- Negative control: halo=1 is BELOW the receptive field and MUST
    # produce a seam. This proves the boundary check actually has teeth; the
    # gate passes only when this case is detected as a seam (run_case False).
    print("== negative control: halo=1 MUST show a seam ==")
    seam_detected = not run_case(50, 2, 1, None)
    results.append(("neg-control halo=1 seam detected", seam_detected))

    # --- Cross-device (cuda:0, cuda:1) if a second GPU is visible. ---
    if ngpu >= 2:
        xdev_cases = [
            (50, 2, 32),
            (128, 2, 32),
            (200, 2, 32),
        ]
        for T, n, h in xdev_cases:
            results.append(
                (f"xdev T={T} n={n} h={h}",
                 run_case(T, n, h, "cuda:0,cuda:1")))
    else:
        print("only 1 GPU visible -- skipping cross-device cases")

    print("\n==== SUMMARY ====")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("=================")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

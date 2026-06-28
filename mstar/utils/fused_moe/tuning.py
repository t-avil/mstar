"""Device-tuned tile-size configs for the fused MoE Triton kernel.

The default config picker in :func:`mstar.utils.fused_moe.kernels.get_default_config`
is a crude two-branch heuristic (one tile shape for ``M <= E`` decode and one
for prefill).  sglang and vLLM instead ship per-device, per-shape tile configs
that were autotuned offline and keyed by the decode batch size ``M``.  Picking
those tiles is purely a *scheduling* change -- the GEMM math is identical, so the
output is numerically equivalent (cos-sim ~1.0) -- but it materially improves the
grouped-GEMM throughput at decode batch sizes, which is the dominant cost of the
Qwen3-Omni Thinker/Talker MoE.

This module loads those JSON configs (ported from sglang for the H200, the
deployment target) and is gated behind ``MSTAR_FUSED_MOE_TUNED``.

Flag contract
-------------
``MSTAR_FUSED_MOE_TUNED`` (default OFF):
    When unset / falsey, :func:`load_tuned_config` always returns ``None`` and
    the caller keeps the existing heuristic -- i.e. behavior is unchanged.  Set
    to ``1`` / ``true`` / ``yes`` / ``on`` to enable tuned tile selection after
    the parity + decode-throughput gate has passed on the target device.

The config files live next to this module under ``configs/`` and follow
sglang's naming convention ``E={num_experts},N={moe_intermediate_size},
device_name={torch.cuda.get_device_name()}.json``.  Each file maps a string
batch-size bucket (``"1"``, ``"2"``, ... ``"4096"``) to a tile dict
(``BLOCK_SIZE_M/N/K``, ``GROUP_SIZE_M``, ``num_warps``, ``num_stages``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "configs"


def tuned_configs_enabled() -> bool:
    """Whether ``MSTAR_FUSED_MOE_TUNED`` opts in to tuned tile selection.

    Default OFF: only ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive)
    enable it.
    """
    raw = os.environ.get("MSTAR_FUSED_MOE_TUNED")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _device_name() -> Optional[str]:
    """``torch.cuda.get_device_name()`` with spaces -> underscores.

    Returns ``None`` when CUDA is unavailable so the loader degrades to the
    heuristic instead of raising on a CPU-only box.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name().replace(" ", "_")
    except Exception:  # pragma: no cover -- torch import / driver issues
        return None


@functools.lru_cache(maxsize=32)
def _load_config_file(num_experts: int, n_inter: int, device_name: str) -> Optional[Dict[str, Any]]:
    """Read and parse one ``E=..,N=..,device_name=..json`` file (cached)."""
    fname = f"E={num_experts},N={n_inter},device_name={device_name}.json"
    path = _CONFIG_DIR / fname
    if not path.is_file():
        logger.debug("No tuned MoE config %s; falling back to heuristic.", fname)
        return None
    try:
        with path.open() as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:  # pragma: no cover -- corrupt file
        logger.warning("Failed to read tuned MoE config %s: %s", path, e)
        return None
    # Normalize bucket keys to int for nearest-bucket selection.
    return {int(k): v for k, v in raw.items()}


def _pick_bucket(buckets: Dict[int, Any], m: int) -> Dict[str, Any]:
    """Pick the smallest tuned bucket >= ``m`` (sglang/vLLM convention).

    Falls back to the largest available bucket when ``m`` exceeds every key.
    """
    keys = sorted(buckets)
    for k in keys:
        if m <= k:
            return buckets[k]
    return buckets[keys[-1]]


def load_tuned_config(
    M: int,
    E: int,
    n_inter: int,
    device_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the tuned tile config for this shape/batch, or ``None``.

    Parameters
    ----------
    M : int
        Decode/prefill batch size (number of tokens).
    E : int
        Number of routed experts.
    n_inter : int
        ``moe_intermediate_size`` (NOT ``2 * inter``).  This matches the ``N``
        in the sglang/vLLM config filenames.
    device_name : str, optional
        Override the auto-detected device (used by CPU-side unit tests).

    Returns
    -------
    dict or None
        A tile dict with the keys the Triton launch expects
        (``BLOCK_SIZE_M/N/K``, ``GROUP_SIZE_M`` and, optionally,
        ``num_warps`` / ``num_stages``), or ``None`` when tuned configs are
        disabled, the device is unknown, or no matching file exists.
    """
    if not tuned_configs_enabled():
        return None
    dev = device_name if device_name is not None else _device_name()
    if dev is None:
        return None
    buckets = _load_config_file(E, n_inter, dev)
    if not buckets:
        return None
    return dict(_pick_bucket(buckets, M))

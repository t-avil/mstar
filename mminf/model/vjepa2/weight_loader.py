"""Load HuggingFace V-JEPA 2 checkpoints into our component modules.

HF V-JEPA 2 checkpoints use a top-level ``encoder.*`` / ``predictor.*`` key
layout.  Our component classes are named to match, so we just use
``load_weights_from_file`` / ``load_weights_from_hf_shards`` with a
``ModuleAndPrefix`` per component — no key renames needed for the standard
variants (vitl/h/g at 256, vitg at 384, V-JEPA 2.1 *if* it uses the same
top-level naming).

AC checkpoint support is deferred — its upstream ``.pt`` format uses a
different key schema (``module.predictor_embed.*``, etc.) and isn't available
as a direct safetensors shard on HF in the same layout.  Added in a follow-up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mminf.model.utils import (
    ModuleAndPrefix,
    load_weights_from_file,
    load_weights_from_hf_shards,
)

logger = logging.getLogger(__name__)


def _find_checkpoint(repo_dir: Path) -> tuple[Path, bool]:
    """Return ``(path, is_sharded)`` for the main checkpoint in a HF repo dir.

    Prefers the sharded index if present, else falls back to the single
    ``model.safetensors``.  Raises ``FileNotFoundError`` if neither exists.
    """
    index = repo_dir / "model.safetensors.index.json"
    if index.exists():
        return index, True
    single = repo_dir / "model.safetensors"
    if single.exists():
        return single, False
    raise FileNotFoundError(
        f"No V-JEPA 2 checkpoint found in {repo_dir}: expected either "
        "model.safetensors.index.json (sharded) or model.safetensors (single)."
    )


def load_vjepa2_hf_weights(
    repo_dir: str | Path,
    encoder_module: torch.nn.Module | None,
    predictor_module: torch.nn.Module | None,
    device: str = "cpu",
    enforce_missing_keys: bool = True,
) -> None:
    """Load HF V-JEPA 2 weights into the supplied encoder / predictor modules.

    Pass ``None`` for a module you don't want to load (e.g. when only the
    encoder lives on this worker's GPU).  ``encoder_module.state_dict()`` is
    expected to have the same keys as HF under the ``encoder.`` prefix;
    ``predictor_module.state_dict()`` similarly under ``predictor.``.
    """
    repo_dir = Path(repo_dir)
    path, is_sharded = _find_checkpoint(repo_dir)

    modules: list[ModuleAndPrefix] = []
    if encoder_module is not None:
        modules.append(
            ModuleAndPrefix(
                module=encoder_module,
                prefix="encoder",
                enforce_missing_keys=enforce_missing_keys,
            )
        )
    if predictor_module is not None:
        modules.append(
            ModuleAndPrefix(
                module=predictor_module,
                prefix="predictor",
                enforce_missing_keys=enforce_missing_keys,
            )
        )
    if not modules:
        logger.warning("load_vjepa2_hf_weights called with no modules to load")
        return

    if is_sharded:
        load_weights_from_hf_shards(repo_dir=repo_dir, modules=modules, device=device)
    else:
        load_weights_from_file(safetensors_file=str(path), modules=modules, device=device)


def download_vjepa2_snapshot(model_path_hf: str, cache_dir: str | None = None) -> Path:
    """Resolve the local directory for an HF V-JEPA 2 snapshot, downloading
    if absent.  Thin wrapper over :func:`huggingface_hub.snapshot_download`."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=model_path_hf, cache_dir=cache_dir)
    return Path(local)

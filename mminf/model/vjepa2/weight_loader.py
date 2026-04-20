"""Load V-JEPA 2 checkpoints into our component modules.

Two checkpoint families are supported:

* **HF safetensors** (``facebook/vjepa2-{vitl,vith,vitg,vitg-384}-fpc64-*``) —
  top-level ``encoder.*`` / ``predictor.*`` key layout; our component classes
  match, so ``load_vjepa2_hf_weights`` just threads them through
  :func:`mminf.model.utils.load_weights_from_file` /
  :func:`load_weights_from_hf_shards` with a ``ModuleAndPrefix`` per component
  — no key renames.

* **Upstream ``.pt``** (``facebook/vjepa2-ac-vitg/original/model.pth`` or the
  raw S3 mirror at ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt``)
  — a single ``torch.save`` blob ``{"encoder": ..., "predictor": ...}`` where
  the encoder uses the upstream ``module.backbone.*`` layout (fused ``qkv``,
  ``blocks.X.``, top-level ``norm.``) and the AC predictor uses the upstream
  ``module.*`` layout (fused ``qkv``, ``predictor_blocks.X.``).

  The AC predictor's class in this repo was ported *to* the upstream key
  layout deliberately, so its side of the rename is a trivial
  ``module.``/``backbone.`` strip.  The encoder is HF-keyed in our tree, so
  we apply the same renames the HF conversion script does
  (``transformers/.../convert_vjepa2_to_hf.py::convert_encoder_keys``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import torch

from mminf.model.utils import (
    ModuleAndPrefix,
    load_weights_from_file,
    load_weights_from_hf_shards,
)

logger = logging.getLogger(__name__)


# Public S3 mirror of the upstream V-JEPA 2-AC checkpoint (same file HF's
# ``upload_original_ckpts`` in ``convert_vjepa2_to_hf.py`` pushes under
# ``original/model.pth``).  ~11.7 GB, no auth required.
VJEPA2_AC_VITG_S3_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


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


# ----------------------------------------------------------------------
# V-JEPA 2-AC — upstream ``.pt`` loader
# ----------------------------------------------------------------------


def _rename_upstream_encoder_keys(
    state_dict: Mapping[str, torch.Tensor],
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Translate the upstream V-JEPA 2 encoder state_dict onto our HF-keyed
    :class:`VJEPA2Encoder`.

    Mirrors the transform in
    ``transformers/src/transformers/models/vjepa2/convert_vjepa2_to_hf.py::convert_encoder_keys``,
    minus the outer ``encoder.`` prefix the HF model uses (our
    ``VJEPA2Encoder`` is already the ``encoder`` sub-module).

    Transforms applied, in order, per key:
      1. strip ``module.backbone.``
      2. ``blocks.X.``               → ``layer.X.``
      3. ``attn.``                   → ``attention.``
      4. ``patch_embed.``            → ``embeddings.patch_embeddings.``
      5. top-level ``norm.``         → ``layernorm.``  (NOT block-internal norm1/norm2)
      6. split fused ``attention.qkv.{weight,bias}`` into
         ``attention.{query,key,value}.{weight,bias}`` along dim 0
      7. drop ``pos_embed`` (AC encoder is RoPE-only; upstream may still ship
         a zero-init pos_embed for compatibility but our port has no slot
         for it)

    Args:
        state_dict: upstream encoder state_dict (``state["encoder"]`` from
            the ``.pt``).
        hidden_size: encoder embedding dim — needed to slice fused qkv.

    Returns:
        A new dict with keys matching ``VJEPA2Encoder.state_dict()``.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, val in state_dict.items():
        key = raw_key.replace("module.backbone.", "")
        # Drop RoPE-encoder pos_embed (our VJEPA2Encoder has no position
        # embedding slot — it uses 3D RoPE inside attention instead).
        if key == "pos_embed":
            continue
        if key.startswith("blocks."):
            key = key.replace("blocks.", "layer.", 1)
        if ".attn." in key:
            key = key.replace(".attn.", ".attention.")
        if key.startswith("patch_embed."):
            key = key.replace("patch_embed.", "embeddings.patch_embeddings.", 1)
        if key.startswith("norm."):
            key = key.replace("norm.", "layernorm.", 1)

        # Split fused qkv (upstream stacks Q/K/V along dim 0).
        if key.endswith("attention.qkv.weight"):
            prefix = key[: -len("qkv.weight")]
            q = val[0:hidden_size, :]
            k = val[hidden_size : 2 * hidden_size, :]
            v = val[2 * hidden_size : 3 * hidden_size, :]
            out[prefix + "query.weight"] = q
            out[prefix + "key.weight"] = k
            out[prefix + "value.weight"] = v
            continue
        if key.endswith("attention.qkv.bias"):
            prefix = key[: -len("qkv.bias")]
            q = val[0:hidden_size]
            k = val[hidden_size : 2 * hidden_size]
            v = val[2 * hidden_size : 3 * hidden_size]
            out[prefix + "query.bias"] = q
            out[prefix + "key.bias"] = k
            out[prefix + "value.bias"] = v
            continue

        out[key] = val
    return out


def _rename_upstream_ac_predictor_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Translate the upstream V-JEPA 2-AC predictor state_dict onto our
    :class:`VisionTransformerPredictorAC`.

    Upstream wraps the module under DDP (``module.``) and sometimes under an
    extra ``backbone.`` attribute depending on how the checkpoint was saved.
    Since our AC predictor class was deliberately ported to preserve upstream
    internal naming (fused ``qkv``, ``predictor_blocks.X.``,
    ``predictor_embed``, etc.), the translation is just a prefix strip —
    matches ``_clean_backbone_key`` in ``vjepa2/src/hub/backbones.py``.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, val in state_dict.items():
        key = raw_key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def download_vjepa2_ac_upstream_pt(
    model_path_hf: str,
    cache_dir: str | None = None,
) -> Path:
    """Fetch just the upstream ``original/model.pth`` from an HF AC repo.

    Uses :func:`huggingface_hub.snapshot_download` with ``allow_patterns`` so
    we don't pull the entire repo (it also ships converted-config metadata we
    don't use).  If HF rate-limits or the file is missing there, the caller
    can fall back to :data:`VJEPA2_AC_VITG_S3_URL` directly.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=model_path_hf,
        cache_dir=cache_dir,
        allow_patterns=["original/*"],
    )
    pt_path = Path(local) / "original" / "model.pth"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Expected upstream checkpoint at {pt_path}; HF repo {model_path_hf} "
            "does not ship an 'original/model.pth'.  Fall back to the direct S3 URL "
            f"{VJEPA2_AC_VITG_S3_URL} or inspect the repo contents."
        )
    return pt_path


def load_vjepa2_ac_upstream_weights(
    pt_path: str | Path,
    encoder_module: torch.nn.Module | None,
    predictor_module: torch.nn.Module | None,
    device: str = "cpu",
    hidden_size: int | None = None,
) -> None:
    """Load weights from an upstream V-JEPA 2-AC ``.pt`` into the supplied modules.

    Args:
        pt_path: path to ``vjepa2-ac-vitg.pt`` (or the HF-hosted copy under
            ``original/model.pth``).
        encoder_module: instance of :class:`VJEPA2Encoder` to populate, or
            ``None`` if the encoder lives on a different worker.
        predictor_module: instance of :class:`VisionTransformerPredictorAC`
            to populate, or ``None``.
        device: target device for tensors after load_state_dict.  Weights
            are first loaded into CPU RAM (via ``torch.load``) and then
            moved; pinning each rank's load to its own ``cuda:X`` here lets
            the OS disk cache amortize multi-rank reads.
        hidden_size: encoder embedding dim.  Required when
            ``encoder_module`` is not None (used to split fused qkv); ignored
            otherwise.  If None, falls back to
            ``encoder_module.config.hidden_size``.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"V-JEPA 2-AC checkpoint not found: {pt_path}")

    if encoder_module is None and predictor_module is None:
        logger.warning("load_vjepa2_ac_upstream_weights called with no modules to load")
        return

    # ``weights_only=True`` is both the modern default and protects against
    # arbitrary pickle code in the ``.pt`` (the upstream checkpoint is
    # trusted — Meta publishes it — but the flag is still best practice).
    logger.info("Loading upstream V-JEPA 2-AC checkpoint from %s", pt_path)
    blob = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "encoder" not in blob or "predictor" not in blob:
        raise ValueError(
            f"Unexpected checkpoint structure at {pt_path}: expected a dict with "
            "'encoder' and 'predictor' keys (upstream V-JEPA 2-AC format)."
        )

    target_device = torch.device(device) if isinstance(device, str) else device

    if encoder_module is not None:
        if hidden_size is None:
            hidden_size = getattr(encoder_module.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "hidden_size must be provided (or encoder_module.config.hidden_size must exist) "
                "to split the upstream fused qkv weights."
            )
        renamed = _rename_upstream_encoder_keys(blob["encoder"], hidden_size=hidden_size)
        if target_device.type != "cpu":
            renamed = {k: v.to(target_device) for k, v in renamed.items()}
        missing, unexpected = encoder_module.load_state_dict(renamed, strict=False, assign=True)
        if missing:
            raise KeyError(f"Missing keys when loading AC encoder weights: {missing}")
        if unexpected:
            logger.debug("Ignored %d unexpected AC-encoder keys: %s", len(unexpected), unexpected[:8])
        # Free upstream-layout encoder state once it's been remapped +
        # assigned — otherwise the ~5.8 GB original CPU blob stays pinned
        # in memory until the caller drops the reference to ``blob``.
        blob["encoder"] = None

    if predictor_module is not None:
        renamed = _rename_upstream_ac_predictor_keys(blob["predictor"])
        if target_device.type != "cpu":
            renamed = {k: v.to(target_device) for k, v in renamed.items()}
        missing, unexpected = predictor_module.load_state_dict(renamed, strict=False, assign=True)
        if missing:
            raise KeyError(f"Missing keys when loading AC predictor weights: {missing}")
        if unexpected:
            logger.debug("Ignored %d unexpected AC-predictor keys: %s", len(unexpected), unexpected[:8])
        blob["predictor"] = None

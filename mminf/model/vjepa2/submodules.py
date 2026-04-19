"""NodeSubmodule wrappers for the V-JEPA 2 graph nodes.

Three submodules:

  VJepa2EncoderSubmodule   - ViT 3D-patch video encoder.
  VJepa2PredictorSubmodule - masked latent predictor (single forward per call).
  VJepa2ACPredictorSubmodule - action-conditioned predictor (ditto).

All three are stateless (no KV cache, no per-iteration state) and are
dispatched through ``EncoderDecoderEngine``.  ``preprocess`` stacks
per-request tensors into a batch when shapes match; ``forward`` runs the
underlying nn.Module and emits a single output tensor per request on the
downstream edge.

The masked-vs-AC choice is made by ``VJepa2Model.get_submodule`` based on
``config.predictor_kind``.  Both predictor submodules emit the same output
edge name (``predicted_hidden``) so downstream consumers don't need to
branch.
"""

from __future__ import annotations

import logging

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.base import NodeSubmodule
from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2Config

logger = logging.getLogger(__name__)


def _normalize_video_frames(frames: torch.Tensor) -> torch.Tensor:
    """Bring a video tensor to ``[1, T, C, H, W]``.

    Accepts ``[T, C, H, W]`` or ``[B, T, C, H, W]``.  Values are assumed to
    already be in the range expected by the encoder (i.e., preprocessing
    happened upstream).  This helper only normalizes the batch dim.
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    if frames.dim() != 5:
        raise ValueError(f"Expected video frames of shape [T,C,H,W] or [B,T,C,H,W]; got {tuple(frames.shape)}")
    return frames


class VJepa2EncoderSubmodule(NodeSubmodule):
    """ViT 3D-patch video encoder.

    Preprocessing stacks ``video_frames`` tensors across requests (requires
    matching shapes) into a single ``pixel_values_videos`` batch.
    ``forward`` runs the encoder and emits ``encoder_hidden``.
    """

    def __init__(self, encoder: VJEPA2Encoder, config: VJepa2Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager | None,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        frames = [_normalize_video_frames(inp["video_frames"][0]) for inp in per_request_inputs]
        # Same-shape requirement for batched dispatch is enforced by can_batch.
        pixel_values_videos = torch.cat(frames, dim=0)
        # Track per-request batch sizes so forward can split the stacked output.
        batch_sizes = [f.size(0) for f in frames]
        return {
            "pixel_values_videos": pixel_values_videos,
            "_batch_sizes": torch.tensor(batch_sizes, dtype=torch.long),
        }

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        pixel_values_videos: torch.Tensor,
        _batch_sizes: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self.encoder(pixel_values_videos)  # [B_total, N, D]
        if _batch_sizes is None or _batch_sizes.numel() <= 1:
            return {"encoder_hidden": [hidden]}
        # Split along the batch dim back to per-request tensors
        splits = torch.split(hidden, _batch_sizes.tolist(), dim=0)
        return {"encoder_hidden": list(splits)}

    def can_batch(self, batch: NodeBatch) -> bool:
        # Batchable when every request has the same video-frames shape.
        shapes = []
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors[rid]
            if "video_frames" not in inputs or not inputs["video_frames"]:
                return False
            t = inputs["video_frames"][0]
            if t.dim() == 4:
                shapes.append(tuple(t.shape))
            elif t.dim() == 5:
                shapes.append(tuple(t.shape[1:]))
            else:
                return False
        return len(set(shapes)) == 1


def _build_default_masks(
    encoder_hidden: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build full-context / full-target masks given encoder hidden states.

    Matches HF's default in ``VJEPA2Model.forward`` when the caller doesn't
    supply masks: ``torch.arange(N).unsqueeze(0).repeat(B, 1)``.
    """
    b, n, _ = encoder_hidden.shape
    ids = torch.arange(n, device=encoder_hidden.device).unsqueeze(0).repeat(b, 1)
    return [ids], [ids]


class VJepa2PredictorSubmodule(NodeSubmodule):
    """Masked latent predictor (non-AC).

    Expects ``encoder_hidden`` from the preceding encoder node.  Optionally
    accepts pre-built ``context_mask`` and ``target_mask`` edges; when
    absent, falls back to full-context / full-target defaults.  Output is
    the predicted hidden tensor at the target positions.
    """

    def __init__(self, predictor: VJEPA2Predictor, config: VJepa2Config):
        super().__init__()
        self.predictor = predictor
        self.config = config

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager | None,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        if len(per_request_inputs) != 1:
            raise NotImplementedError(
                "VJepa2PredictorSubmodule cross-request batching is not yet "
                "supported (would need identical mask shapes).  Use one "
                "request per forward pass for now."
            )
        inputs = per_request_inputs[0]
        encoder_hidden = inputs["encoder_hidden"][0]
        out: dict[str, torch.Tensor] = {"encoder_hidden": encoder_hidden}
        if "context_mask" in inputs:
            out["context_mask"] = inputs["context_mask"][0]
        if "target_mask" in inputs:
            out["target_mask"] = inputs["target_mask"][0]
        return out

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        encoder_hidden: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if context_mask is None or target_mask is None:
            ctx_list, tgt_list = _build_default_masks(encoder_hidden)
        else:
            ctx_list = [context_mask]
            tgt_list = [target_mask]
        predicted = self.predictor(encoder_hidden, ctx_list, tgt_list)
        return {"predicted_hidden": [predicted]}

    def can_batch(self, batch: NodeBatch) -> bool:
        # Defer true cross-request batching (requires matching masks) —
        # each request gets its own forward.
        return False


class VJepa2ACPredictorSubmodule(NodeSubmodule):
    """Action-conditioned predictor (V-JEPA 2-AC).

    Additional required inputs vs the masked predictor: ``actions``,
    ``states``, and optionally ``extrinsics``.  Each is per-timestep, shape
    ``[T, action_embed_dim]`` (or ``[T, action_embed_dim - 1]`` for
    extrinsics), delivered as a GraphEdge by the model's ``process_prompt``
    / ``get_initial_forward_pass_args``.  Output is ``predicted_hidden``
    with the same shape as ``encoder_hidden``.
    """

    def __init__(self, predictor: VisionTransformerPredictorAC, config: VJepa2Config):
        super().__init__()
        self.predictor = predictor
        self.config = config

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager | None,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        if len(per_request_inputs) != 1:
            raise NotImplementedError("VJepa2ACPredictorSubmodule cross-request batching is not yet supported.")
        inputs = per_request_inputs[0]
        out: dict[str, torch.Tensor] = {
            "encoder_hidden": inputs["encoder_hidden"][0],
            "actions": inputs["actions"][0],
            "states": inputs["states"][0],
        }
        if "extrinsics" in inputs:
            out["extrinsics"] = inputs["extrinsics"][0]
        return out

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        encoder_hidden: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if extrinsics is not None and extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)
        predicted = self.predictor(encoder_hidden, actions, states, extrinsics=extrinsics)
        return {"predicted_hidden": [predicted]}

    def can_batch(self, batch: NodeBatch) -> bool:
        return False

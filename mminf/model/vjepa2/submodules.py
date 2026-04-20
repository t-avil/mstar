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

    Phase 1: one request per forward pass (``can_batch`` inherits ``False``
    from ``NodeSubmodule``).  Cross-request batching is a Phase 3
    optimization that requires fixing the engine's batched path —
    ``EncoderDecoderEngine._execute_batched`` currently passes a
    ``dict[rid, inputs]`` where preprocess expects ``list[NameToTensorList]``,
    and skips ``request_info`` at forward time.  See the Phase-3 tracking
    note in the plan; the V-JEPA 2 encoder is natively batchable (stateless,
    fixed shapes, torch.compile-friendly), so the optimization will be
    worth doing once the engine path is fixed.
    """

    def __init__(self, encoder: VJEPA2Encoder, config: VJepa2Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager | None = None,
    ) -> dict[str, torch.Tensor]:
        assert len(per_request_inputs) == 1, (
            "VJepa2EncoderSubmodule runs one request at a time; cross-request "
            "batching is deferred to Phase 3 (needs engine-path fix)."
        )
        frames = _normalize_video_frames(per_request_inputs[0]["video_frames"][0])
        return {"pixel_values_videos": frames}

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        pixel_values_videos: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        logger.info(
            "VJepa2EncoderSubmodule.forward: input shape=%s dtype=%s device=%s",
            tuple(pixel_values_videos.shape),
            pixel_values_videos.dtype,
            pixel_values_videos.device,
        )
        hidden = self.encoder(pixel_values_videos)
        logger.info("VJepa2EncoderSubmodule.forward: output shape=%s", tuple(hidden.shape))
        return {"encoder_hidden": [hidden]}


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
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager | None = None,
    ) -> dict[str, torch.Tensor]:
        assert len(per_request_inputs) == 1, "VJepa2PredictorSubmodule runs one request at a time."
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
        logger.info(
            "VJepa2PredictorSubmodule.forward: encoder_hidden shape=%s",
            tuple(encoder_hidden.shape),
        )
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if context_mask is None or target_mask is None:
            ctx_list, tgt_list = _build_default_masks(encoder_hidden)
        else:
            ctx_list = [context_mask]
            tgt_list = [target_mask]
        predicted = self.predictor(encoder_hidden, ctx_list, tgt_list)
        logger.info("VJepa2PredictorSubmodule.forward: output shape=%s", tuple(predicted.shape))
        return {"predicted_hidden": [predicted]}


class VJepa2RolloutPredictorSubmodule(NodeSubmodule):
    """Masked-predictor rollout submodule — Phase 2 autoregressive anticipation.

    Parity target: upstream
    ``vjepa2/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py``
    ``AnticipativeWrapper.forward`` (lines 172-224).  That Python loop calls
    the (stateless) predictor once per rollout step, feeding a sliding window
    of the previous step's prediction back as context:

        x_pred_input = x_full                          # initial encoder output
        for _ in range(num_steps):
            x_pred = predictor(x_pred_input, ctxt_positions, tgt_positions)
            x_pred_input = cat([x_pred_input[:, N_pred:], x_pred], dim=1)
            x_accumulate = cat([x_accumulate, x_pred], dim=1)

    In mminf this becomes a ``DynamicLoop`` whose section is a single node
    wrapping this submodule.  On each iter:

      * The loop-back ``encoder_hidden`` carries the sliding window state.
        Iter 0 receives it from the preceding ``video_encoder`` node; iter
        k > 0 receives the previous iter's emitted ``encoder_hidden``.
      * ``predicted_hidden`` is emitted both as a (dangling) section output
        — which lets the Loop's cache_outputs machinery pick it up — and
        into the Loop's ``accumulated_outputs`` for client delivery.

    The predictor nn.Module is the same instance used by
    ``VJepa2PredictorSubmodule`` — weights are loaded once and shared.  Only
    the preprocess/forward rollout-awareness differs.

    Shares Phase 1's single-request sequential path (``can_batch=False``
    inherited from ``NodeSubmodule``).  Same Phase-3 batching story as the
    encoder — enabling it requires the engine fix documented in the plan.
    """

    def __init__(
        self,
        predictor: VJEPA2Predictor,
        config: VJepa2Config,
        num_output_frames: int = 2,
        frames_per_second: int = 4,
        anticipation_seconds: float = 1.0,
    ):
        super().__init__()
        self.predictor = predictor
        self.config = config
        # ``num_output_frames`` is rounded up to tubelet_size — below
        # tubelet_size the math (N_pred = grid^2 * (nof // tubelet)) yields
        # zero target tokens, which would make the predictor call degenerate.
        self.num_output_frames = max(int(num_output_frames), config.tubelet_size)
        self.frames_per_second = int(frames_per_second)
        self.anticipation_seconds = float(anticipation_seconds)

    # ------------------------------------------------------------------
    # Rollout geometry (derived from config + hyperparams; shape-static)
    # ------------------------------------------------------------------

    @property
    def _grid_size(self) -> int:
        return self.config.grid_size

    @property
    def _n_pred(self) -> int:
        """Number of predicted tokens per iteration.

        Matches ``AnticipativeWrapper`` line 206:
            N_pred = grid_size**2 * (num_output_frames // tubelet_size)
        """
        return self._grid_size * self._grid_size * (self.num_output_frames // self.config.tubelet_size)

    @property
    def _anticipation_steps(self) -> int:
        """Discrete (tubelet-aligned) time offset between the context window
        and the first predicted token.  From ``AnticipativeWrapper`` line 202.
        """
        return int(self.anticipation_seconds * self.frames_per_second / self.config.tubelet_size)

    # ------------------------------------------------------------------
    # NodeSubmodule ABC
    # ------------------------------------------------------------------

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager | None = None,
    ) -> dict[str, torch.Tensor]:
        assert len(per_request_inputs) == 1, (
            "VJepa2RolloutPredictorSubmodule runs one request at a time; "
            "cross-request batching is a Phase 3 optimization."
        )
        inputs = per_request_inputs[0]
        # ``encoder_hidden`` arrives from either the preceding video_encoder
        # (iter 0) or the self-loop-back (iter k > 0) — both paths deliver a
        # single tensor under the same edge name.
        return {"encoder_hidden": inputs["encoder_hidden"][0]}

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        encoder_hidden: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """One rollout step.  Runs the masked predictor, then slides the
        encoder_hidden window forward so iter k+1's context is the
        concatenation of the last (N - N_pred) context tokens plus the
        N_pred newly-predicted tokens.
        """
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        b, n_ctxt, _ = encoder_hidden.shape
        device = encoder_hidden.device

        n_pred = self._n_pred
        if n_pred >= n_ctxt:
            raise ValueError(
                f"Rollout requires n_pred ({n_pred}) < encoder context length "
                f"({n_ctxt}); reduce num_output_frames or the anticipation "
                f"horizon."
            )

        # Fixed every iter (context window always holds N tokens); matches
        # the upstream wrapper's arange(N).
        ctxt_positions = torch.arange(n_ctxt, device=device).unsqueeze(0).repeat(b, 1)
        skip = n_ctxt + self._grid_size * self._grid_size * self._anticipation_steps
        tgt_positions = (torch.arange(n_pred, device=device) + skip).unsqueeze(0).repeat(b, 1)

        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)
        logger.info(
            "VJepa2RolloutPredictorSubmodule.forward: iter=%d encoder_hidden=%s n_pred=%d skip=%d",
            iter_idx,
            tuple(encoder_hidden.shape),
            n_pred,
            skip,
        )

        predicted = self.predictor(encoder_hidden, [ctxt_positions], [tgt_positions])

        # Slide the window: drop oldest n_pred tokens, append predicted.
        # This is the upstream
        #   x_pred_input = cat([x_pred_input[:, N_pred:, :], x_pred], dim=1)
        next_encoder_hidden = torch.cat([encoder_hidden[:, n_pred:, :], predicted], dim=1)

        # Per-request early-exit.  The DynamicLoop's ``max_iters`` is a
        # config-level upper bound (``max_rollout_horizon``); the caller's
        # ``rollout_horizon`` — snapshotted into ``step_metadata`` by
        # ``VJepa2Model.get_initial_forward_pass_args`` — shortens the loop
        # here.  ``iter_idx`` is the count BEFORE this forward, so after this
        # call the loop will have completed ``iter_idx + 1`` iterations.
        rollout_horizon = int(request_info.step_metadata.get("rollout_horizon", 0) or 0)
        if rollout_horizon > 0 and iter_idx + 1 >= rollout_horizon:
            logger.info(
                "VJepa2RolloutPredictorSubmodule.forward: reached horizon H=%d at iter=%d; stopping rollout_loop.",
                rollout_horizon,
                iter_idx,
            )
            request_info.register_loop_stop("rollout_loop")

        logger.info(
            "VJepa2RolloutPredictorSubmodule.forward: predicted=%s next_encoder_hidden=%s",
            tuple(predicted.shape),
            tuple(next_encoder_hidden.shape),
        )
        return {
            "encoder_hidden": [next_encoder_hidden],
            "predicted_hidden": [predicted],
        }


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
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
        cache_manager: BatchedCacheManager | None = None,
    ) -> dict[str, torch.Tensor]:
        assert len(per_request_inputs) == 1, "VJepa2ACPredictorSubmodule runs one request at a time."
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

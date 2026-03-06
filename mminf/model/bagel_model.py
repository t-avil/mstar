"""
BagelModel: Model implementation for BAGEL (ByteDance) unified multimodal model.

BAGEL uses a Qwen2 LLM with MoT (Mixture-of-Transformers) architecture,
SigLIP2 ViT for image understanding, and FLUX VAE for image generation.
The LLM itself serves as the denoiser for rectified flow image generation
(no separate diffusion model).
"""

import torch
import torch.nn as nn

from mminf.communication.tensors import NameToTensorList
from mminf.graph.base import (
    GraphPointer,
    GraphSection,
    GraphStage,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mminf.model.base import STREAM_OUT, CurrentForwardMetadata, Model, StageSubmodule


# ---------------------------------------------------------------------------
# StageSubmodule wrappers
# ---------------------------------------------------------------------------


class TextEmbSubmodule(StageSubmodule):
    """Wraps language_model.model.embed_tokens: token IDs -> embeddings."""

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, text_inputs: torch.Tensor) -> NameToTensorList:
        return {"text_emb": [self.embed_tokens(text_inputs)]}


class ViTEncoderSubmodule(StageSubmodule):
    """Wraps vit_model + connector + vit_pos_embed: pixel patches -> features.

    Expects preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> NameToTensorList:
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen.item() if max_seqlen.dim() == 0 else max_seqlen,
        )
        features = self.connector(features)
        pos_emb = self.vit_pos_embed(packed_position_ids)
        features = features + pos_emb
        return {"vit_emb": [features]}


class VAEEncoderSubmodule(StageSubmodule):
    """Wraps VAE encode + patchify + vae2llm + time_embedder + latent_pos_embed.

    Encodes an image tensor to VAE latents, patchifies them, and projects
    into the LLM hidden dimension with positional and timestep embeddings.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        vae2llm: nn.Linear,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel

    def forward(
        self,
        padded_images: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        patchified_h: torch.Tensor,
        patchified_w: torch.Tensor,
    ) -> NameToTensorList:
        latent = self.vae_model.encode(padded_images)

        p = self.latent_patch_size
        h, w = patchified_h.item(), patchified_w.item()
        # Patchify: [batch, C, H, W] -> [num_patches, patch_dim]
        packed_latent = []
        for lat in latent:
            lat = lat[:, :h * p, :w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            lat = torch.einsum("chpwq->hwpqc", lat).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(lat)
        packed_latent = torch.cat(packed_latent, dim=0)

        # Project to hidden dim with timestep and position embeddings
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        return {"vae_emb": [packed_latent]}


class LMHeadSubmodule(StageSubmodule):
    """Wraps lm_head: hidden states -> logits -> sampled token."""

    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> NameToTensorList:
        logits = self.lm_head(hidden_states)
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}


class FlowProjSubmodule(StageSubmodule):
    """Wraps llm2vae projection + Euler step update.

    Extracts velocity from LLM hidden states via the llm2vae linear
    projection. The Euler integration step (x_t = x_t - v_t * dt) is
    performed here using the timestep schedule carried in kwargs.

    During CFG, this stage receives 3 velocity tensors (main, cfg_text,
    cfg_img) and combines them via the CFG formula before the Euler step.
    """

    def __init__(self, llm2vae: nn.Linear):
        super().__init__()
        self.llm2vae = llm2vae

    def forward(self, hidden_states: torch.Tensor) -> NameToTensorList:
        v_t = self.llm2vae(hidden_states)
        return {"latents": [v_t]}


class VAEDecoderSubmodule(StageSubmodule):
    """Wraps VAE decoder: latent grid -> pixel image."""

    def __init__(
        self,
        vae_model: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def forward(
        self,
        latents: torch.Tensor,
        image_h: torch.Tensor,
        image_w: torch.Tensor,
    ) -> NameToTensorList:
        H, W = image_h.item(), image_w.item()
        p = self.latent_patch_size
        h = H // self.latent_downsample
        w = W // self.latent_downsample

        # Unpatchify: [num_patches, patch_dim] -> [1, C, H_latent, W_latent]
        latent = latents.reshape(1, h, w, p, p, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.latent_channel, h * p, w * p
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return {"image_output": [image]}


# ---------------------------------------------------------------------------
# BagelModel
# ---------------------------------------------------------------------------


class BagelModel(Model):
    """
    BAGEL unified multimodal model (ByteDance).

    Architecture: Qwen2 LLM with MoT + SigLIP2 ViT + FLUX VAE.
    The LLM serves as both the autoregressive text model and the denoiser
    for rectified flow image generation (no separate diffusion model).

    Stages:
        text_emb      (enc_dec) - Token embedding
        vit_encoder   (enc_dec) - ViT + connector for image understanding
        vae_encoder   (enc_dec) - VAE encode for image conditioning
        LLM           (ar)      - Qwen2 with MoT (shared self-attention)
        lm_head       (enc_dec) - Token prediction head
        flow_proj     (flow)    - Velocity extraction + Euler step
        vae_decoder   (enc_dec) - VAE decode to pixels

    Phases:
        prefill   - Process interleaved text + images into KV cache
        decode    - Autoregressive text token generation
        image_gen - Flow matching denoising loop + VAE decode
    """

    def __init__(
        self,
        bagel_model=None,
        vae_model=None,
        tokenizer=None,
        new_token_ids: dict | None = None,
        num_timesteps: int = 50,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
    ):
        self.bagel_model = bagel_model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale

        # Special token IDs
        token_ids = new_token_ids or {}
        self.boi_token_id = token_ids.get("start_of_image")
        self.eoi_token_id = token_ids.get("end_of_image")
        self.eos_token_id = token_ids.get("eos_token_id")
        self.bos_token_id = token_ids.get("bos_token_id")

        # Build submodule wrappers (None when no real model is provided)
        self._text_emb_sub = None
        self._vit_encoder_sub = None
        self._vae_encoder_sub = None
        self._lm_head_sub = None
        self._flow_proj_sub = None
        self._vae_decoder_sub = None

        if bagel_model is not None:
            self._text_emb_sub = TextEmbSubmodule(
                bagel_model.language_model.model.embed_tokens
            )
            self._lm_head_sub = LMHeadSubmodule(
                bagel_model.language_model.lm_head
            )
            self._flow_proj_sub = FlowProjSubmodule(bagel_model.llm2vae)

            if hasattr(bagel_model, "vit_model"):
                self._vit_encoder_sub = ViTEncoderSubmodule(
                    bagel_model.vit_model,
                    bagel_model.connector,
                    bagel_model.vit_pos_embed,
                )

            if vae_model is not None:
                self._vae_encoder_sub = VAEEncoderSubmodule(
                    vae_model,
                    bagel_model.vae2llm,
                    bagel_model.time_embedder,
                    bagel_model.latent_pos_embed,
                    bagel_model.latent_patch_size,
                    bagel_model.latent_channel,
                )
                self._vae_decoder_sub = VAEDecoderSubmodule(
                    vae_model,
                    bagel_model.latent_patch_size,
                    bagel_model.latent_channel,
                    bagel_model.latent_downsample,
                )

    # -----------------------------------------------------------------------
    # Model ABC implementation
    # -----------------------------------------------------------------------

    def get_stage_engine_types(self) -> dict[str, str]:
        return {
            "text_emb": "enc_dec",
            "vit_encoder": "enc_dec",
            "vae_encoder": "enc_dec",
            "LLM": "ar",
            "lm_head": "enc_dec",
            "flow_proj": "flow",
            "vae_decoder": "enc_dec",
        }

    def get_phase_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            Parallel([
                GraphStage(
                    name="text_emb",
                    input_ids=["text_inputs"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="text_emb"),
                    ],
                ),
                GraphStage(
                    name="vit_encoder",
                    input_ids=["vit_inputs"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="vit_emb"),
                    ],
                ),
                GraphStage(
                    name="vae_encoder",
                    input_ids=["vae_inputs"],
                    outputs=[
                        GraphPointer(next_stage="LLM", name="vae_emb"),
                    ],
                ),
            ]),
            GraphStage(
                name="LLM",
                input_ids=["text_emb", "vit_emb", "vae_emb"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        output_modality="text",
                        name="prefill_done",
                        is_new_token=True,
                    ),
                ],
            ),
        ])

        decode = Sequential([
            GraphStage(
                name="text_emb",
                input_ids=["text_inputs"],
                outputs=[
                    GraphPointer(next_stage="LLM", name="text_emb"),
                ],
            ),
            GraphStage(
                name="LLM",
                input_ids=["text_emb"],
                outputs=[
                    GraphPointer(next_stage="lm_head", name="hidden_states"),
                ],
            ),
            GraphStage(
                name="lm_head",
                input_ids=["hidden_states"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        output_modality="text",
                        name="new_token",
                        is_new_token=True,
                    ),
                ],
            ),
        ])

        image_gen = Sequential([
            Loop(
                section=Sequential([
                    GraphStage(
                        name="LLM",
                        input_ids=["latents"],
                        outputs=[
                            GraphPointer(
                                next_stage="flow_proj", name="hidden_states"
                            ),
                        ],
                    ),
                    GraphStage(
                        name="flow_proj",
                        input_ids=["hidden_states"],
                        outputs=[
                            GraphPointer(next_stage="LLM", name="latents"),
                        ],
                    ),
                ]),
                n_iters=self.num_timesteps - 1,
                outputs=[
                    GraphPointer(next_stage="vae_decoder", name="latents"),
                ],
            ),
            GraphStage(
                name="vae_decoder",
                input_ids=["latents"],
                outputs=[
                    GraphPointer(
                        next_stage=STREAM_OUT,
                        output_modality="image",
                        name="image_output",
                        back_to_conductor=True,
                    ),
                ],
            ),
        ])

        return dict(
            prefill=prefill,
            decode=decode,
            image_gen=image_gen,
        )

    def get_initial_forward_metadata(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
    ) -> CurrentForwardMetadata:
        return CurrentForwardMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            phase="prefill",
            is_prefill=True,
            kwargs={
                "mode": "und",
                "num_timesteps": self.num_timesteps,
                "cfg_text_scale": self.cfg_text_scale,
                "cfg_img_scale": self.cfg_img_scale,
            },
        )

    def get_forward_pass_inputs(
        self,
        metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        prev_forward_metadata: CurrentForwardMetadata = None,
    ) -> list[GraphPointer]:
        pointers: list[GraphPointer] = []

        if metadata.is_prefill:
            # Prefill: all three encoders receive their inputs.
            # Signal-only pointers (empty tensor_info) for absent modalities.
            text_ptr = GraphPointer(next_stage="text_emb", name="text_inputs")
            text_ptr.tensor_info = persist_signals.get("text_inputs", [])
            pointers.append(text_ptr)

            vit_ptr = GraphPointer(next_stage="vit_encoder", name="vit_inputs")
            vit_ptr.tensor_info = persist_signals.get("vit_inputs", [])
            pointers.append(vit_ptr)

            vae_ptr = GraphPointer(next_stage="vae_encoder", name="vae_inputs")
            vae_ptr.tensor_info = persist_signals.get("vae_inputs", [])
            pointers.append(vae_ptr)

        elif metadata.phase == "decode":
            # Decode: the previously generated token feeds text_emb
            text_ptr = GraphPointer(next_stage="text_emb", name="text_inputs")
            text_ptr.tensor_info = persist_signals.get("new_token", [])
            pointers.append(text_ptr)

        elif metadata.phase == "image_gen":
            # Image gen: initial noise latents feed the LLM
            latent_ptr = GraphPointer(next_stage="LLM", name="latents")
            latent_ptr.tensor_info = persist_signals.get("latents", [])
            pointers.append(latent_ptr)

        return pointers

    def update_for_next_forward(
        self,
        metadata: CurrentForwardMetadata,
        new_tokens: dict[str, list[int]],
    ) -> CurrentForwardMetadata:
        if metadata.is_prefill:
            # After prefill -> always transition to decode
            metadata.is_prefill = False
            metadata.phase = "decode"
            metadata.output_modalities = ["text"]
            metadata.kwargs["mode"] = "und"
            return metadata

        if metadata.phase == "decode":
            tokens = new_tokens.get("new_token", [])
            if self.boi_token_id is not None and self.boi_token_id in tokens:
                # BOI token detected -> switch to image generation
                metadata.phase = "image_gen"
                metadata.output_modalities = ["image"]
                metadata.kwargs["mode"] = "gen"
            elif self.eos_token_id is not None and self.eos_token_id in tokens:
                # EOS token -> request complete
                metadata.kwargs["done"] = True
            # else: stay in decode phase
            return metadata

        if metadata.phase == "image_gen":
            # After image generation -> back to decode
            metadata.phase = "decode"
            metadata.output_modalities = ["text"]
            metadata.kwargs["mode"] = "und"
            return metadata

        return metadata

    def get_submodule(self, stage_name: str) -> torch.nn.Module | None:
        if self.bagel_model is None:
            return None  # dummy mode
        submodule_map = {
            "text_emb": self._text_emb_sub,
            "vit_encoder": self._vit_encoder_sub,
            "vae_encoder": self._vae_encoder_sub,
            "LLM": self.bagel_model.language_model,
            "lm_head": self._lm_head_sub,
            "flow_proj": self._flow_proj_sub,
            "vae_decoder": self._vae_decoder_sub,
        }
        return submodule_map.get(stage_name)

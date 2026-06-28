"""Parity: batched Thinker ``prefill_vision`` == per-request single prefills.

The I2T-TTFT optimization (``MSTAR_BATCH_VISION_PREFILL=1``) lets the Thinker
``prefill_vision`` walk batch more than one image per step: ``preprocess``
``torch.cat``'s the per-request vision embeddings, concatenates each layer's
deepstack features along the token dim (request-offset layout in one buffer
per layer), and plans a *block-diagonal* causal attention over the summed
token count — exactly mirroring ``prefill_audio`` / ``prefill_text``.

Because the attention is block-diagonal (request ``i``'s tokens never attend
to request ``j``'s) this batched forward MUST produce, for each request, the
same hidden states as prefilling that request alone at ``B=1``. This test
pins that invariant: for ``B=2`` synthetic images it compares the per-request
slices of one batched forward against two independent ``B=1`` forwards within
bf16 tolerance.

It exercises the inner ``Qwen3OmniThinkerModel.forward`` directly (not the
submodule's ``forward_batched``) so it does not need the CUDA-graph runner's
static ``qo_indptr`` buffer — the comparison is purely eager batched vs eager
single, which is where the deepstack-key and per-request pos-advance changes
live.

Run locally::

    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
    MSTAR_BATCH_VISION_PREFILL=1 \
      pytest test/modular/test_qwen3_omni_batch_vision_prefill_parity.py -v -s
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.kv_store import PositionInfo, TransferEngineInfo  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402
from mstar.utils.sampling import SeenTokenMask  # noqa: E402

QWEN3_OMNI_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _hf_cache_has_qwen3_omni() -> bool:
    """True if the Qwen3-Omni snapshot is already on local disk.

    Mirrors ``test_prefill_cuda_graph.py`` so this test self-skips on machines
    without the ~60 GB download instead of trying to fetch it.
    """
    candidates: list[Path] = []
    for env_key in ("HF_HOME", "HF_HUB_CACHE"):
        if env_key in os.environ:
            base = Path(os.environ[env_key])
            candidates.extend([base, base / "hub"])
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    target = "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
    return any((base / target).exists() for base in candidates)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not _hf_cache_has_qwen3_omni(),
        reason=f"{QWEN3_OMNI_REPO} not in local HF cache; run "
               f"`huggingface-cli download {QWEN3_OMNI_REPO}`",
    ),
]


class _StubTransferEngine:
    """Minimal stand-in for the Mooncake TransferEngine (no cross-worker IO)."""

    def __init__(self):
        self.registered: list[tuple[int, int]] = []

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered.append((ptr, nbytes))
        return 0

    def unregister_memory(self, ptr: int) -> int:  # noqa: ARG002
        return 0

    def get_async_reader(self, device):  # noqa: ARG002
        return None

    def batch_transfer_sync_read(self, *args, **kwargs):
        raise RuntimeError("stub: no transfers expected in this test")


@pytest.fixture(scope="session")
def thinker_engine():
    """Bring up the Thinker submodule on GPU inside a KVCacheEngine.

    No CUDA-graph capture: the parity comparison runs the eager inner model,
    so we only need the engine's per-request KV cache machinery
    (``_create_cache_manager``).
    """
    from mstar.engine.kv_cache_engine import KVCacheEngine
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")

    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None, "Thinker submodule failed to load"

    kv_cfgs = [
        c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes
    ]
    assert len(kv_cfgs) == 1, f"expected 1 Thinker KV config, got {len(kv_cfgs)}"
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 64  # tiny: this test prefills only a few hundred tokens

    engine = KVCacheEngine(autocast_dtype=torch.bfloat16)
    transfer_info = TransferEngineInfo(
        my_entity_id="vision_parity_test",
        my_session_id="vision_parity_session",
        transfer_engine=_StubTransferEngine(),
    )
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )
    yield engine, thinker, device
    engine.shutdown()


def _make_vision_input(
    submodule, device: torch.device, seed: int, grid_hw: tuple[int, int],
) -> ARNodeInputs:
    """Synthesize one image's ``prefill_vision`` ARNodeInputs.

    Vision embeds + deepstack features are drawn from the model's own
    ``embed_tokens`` (run on random in-vocab IDs) rather than raw ``randn`` so
    the hidden-state magnitudes stay in-distribution — the batched and single
    forwards invoke different-sized FlashInfer/GEMM kernels, and wildly OOD
    activations would let benign bf16 reduction-order differences compound
    across the 30+ MoE layers into a false mismatch.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    cfg = submodule.config
    hidden = cfg.thinker_hidden_size
    merge_sq = cfg.vision.spatial_merge_size ** 2
    t, (h, w) = 1, grid_hw
    assert (h * w) % merge_sq == 0, "grid must be divisible by spatial_merge_size^2"
    vision_len = (t * h * w) // merge_sq
    num_deepstack = len(cfg.vision.deepstack_visual_indexes)

    embed_layer = submodule.model.model.embed_tokens
    safe_vocab_max = 10000

    def _embed(n: int, scale: float) -> torch.Tensor:
        ids = torch.randint(
            0, safe_vocab_max, (n,), dtype=torch.long, device=device, generator=g,
        )
        with torch.no_grad():
            return (embed_layer(ids).to(torch.bfloat16) * scale).contiguous()

    vision_embeds = _embed(vision_len, 1.0)
    # deepstack features are *added* into hidden states at several layers; keep
    # them small so the residual stream stays in a realistic range.
    deepstack = [_embed(vision_len, 0.05) for _ in range(num_deepstack)]
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=device)

    inputs = {
        "vision_embeds": [vision_embeds],
        "image_grid_thw": [grid_thw],
        "deepstack": deepstack,
        "video_second_per_grid": [],
    }
    seen = SeenTokenMask.new(f"seed_{seed}", vocab_size=None, device=device)
    return submodule.prepare_inputs(
        graph_walk="prefill_vision",
        fwd_info=None,
        inputs=inputs,
        seen_token_mask=seen,
        pos_info={"main": PositionInfo()},
    )


def _forward_hidden(
    engine, submodule, request_ids: list[str], inputs: list[ARNodeInputs],
) -> torch.Tensor:
    """Run the eager inner Thinker model and return packed hidden states.

    Returns ``(total_tokens, hidden)`` — the final normed hidden states for the
    whole (possibly batched) prefill, in request-concatenation order.
    """
    cache_mgr = engine._create_cache_manager(request_ids, "Thinker")
    per_request_info = {
        rid: CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="prefill_vision",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={},
            step_metadata={"audio_output": True, "is_last_prefill": True},
        )
        for rid in request_ids
    }
    engine_inputs = ModelInputsFromEngine(
        request_ids=request_ids,
        per_request_info=per_request_info,
        cache_manager=cache_mgr,
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            packed = submodule.preprocess(
                graph_walk="prefill_vision",
                engine_inputs=engine_inputs,
                inputs=inputs,
            )
            deepstack = submodule._collect_deepstack_kwargs(packed)
            cos_sin_3d = (packed["cos_3d"], packed["sin_3d"])
            hidden, _layer_0, _layer_n = submodule.model(
                input_embeds=packed["input_embeds"],
                cache_handle=cache_mgr,
                cos_sin_3d=cos_sin_3d,
                mrope_section=packed["mrope_section"],
                mrope_pos_advance=packed.get("mrope_pos_advance"),
                deepstack_visual_embeds=deepstack,
            )
    return hidden


def _rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    ref_scale = max(ref.abs().max().item(), 1e-6)
    return (actual - ref).abs().max().item() / ref_scale


def test_batched_vision_prefill_matches_single(thinker_engine, monkeypatch):
    """B=2 batched prefill_vision == two B=1 prefills, per request."""
    monkeypatch.setenv("MSTAR_BATCH_VISION_PREFILL", "1")
    engine, submodule, device = thinker_engine

    # Two differently-sized images so the per-request boundaries are real.
    inp0 = _make_vision_input(submodule, device, seed=1, grid_hw=(8, 8))
    inp1 = _make_vision_input(submodule, device, seed=2, grid_hw=(8, 12))
    len0, len1 = inp0.input_seq_len, inp1.input_seq_len

    # Batched B=2 forward (single block-diagonal prefill).
    rids = [f"req_{uuid.uuid4().hex[:8]}" for _ in range(2)]
    hidden_batched = _forward_hidden(engine, submodule, rids, [inp0, inp1])
    assert hidden_batched.shape[0] == len0 + len1

    # Two independent B=1 forwards.
    h0 = _forward_hidden(engine, submodule, [f"req_{uuid.uuid4().hex[:8]}"], [inp0])
    h1 = _forward_hidden(engine, submodule, [f"req_{uuid.uuid4().hex[:8]}"], [inp1])

    err0 = _rel_err(hidden_batched[:len0], h0)
    err1 = _rel_err(hidden_batched[len0:len0 + len1], h1)

    tol = 2e-2  # bf16: batched vs single invoke different GEMM/attention sizes
    assert err0 < tol, f"request 0 hidden mismatch: rel_err={err0:.4g} >= {tol}"
    assert err1 < tol, f"request 1 hidden mismatch: rel_err={err1:.4g} >= {tol}"

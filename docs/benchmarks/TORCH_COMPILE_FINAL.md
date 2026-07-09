# TORCH_COMPILE_FINAL — final static audit of torch.compile in M*

Scope: worktree `mstar-b1fix` @ `opt/prep-h2d` (the winning build), 2026-07-06.
Static audit only (GPUs occupied); every claim carries a file:line. Measured
compile-related wins to date: CDT +3.4% at i2t B32 (env-only), custom-ops
compiled-thinker break count 816 → ~165 (opt/custom-ops work, now on-branch).

## 1. Compile call-sites (what is compiled, how)

| # | Site | Module | Mode | dynamic | fullgraph |
|---|------|--------|------|---------|-----------|
| 1 | `mstar/engine/cuda_graph_runner.py:618` | Thinker `forward_batched` (AR decode capture, per NUM_SLOTS) | `max-autotune-no-cudagraphs` | False | False |
| 2 | `mstar/engine/cuda_graph_runner.py:2490` | `forward_batched` (piecewise/unrolled capture) | `max-autotune-no-cudagraphs` | False | False |
| 3 | `mstar/engine/kv_cache_engine.py:221,226` | language_model `forward` + `forward_batched` | **default** (no mode=) | None | False |
| 4 | `mstar/engine/stateless_engine.py:528` | codec/stateless `forward` | default | False | False |
| 5 | `mstar/model/qwen3_omni/components/code2wav.py:472` | Code2Wav vocoder `forward` | default | False | default |

- fp8 experts are pre-quantized BEFORE compile (`cuda_graph_runner.py:614-617`,
  `compile_ops.py:99-106`) so lazy-quant never traces — this is why the
  freezing×fp8 hazard does not exist here (no `freezing` anywhere in the tree).
- `-no-cudagraphs` is deliberate: M* owns CUDA-graph capture manually
  (SGLang-style; `compile_ops.py:5-7`). Full `max-autotune` would collide.
- `stateless_engine.py:538-540`: codec `forward_batched` intentionally eager —
  ~30s one-shot varlen trace cost dwarfs the win.
- NOT compiled: Talker + code-predictor (eager; `components/attention.py:60-63`
  sets `layer_idx=None` for them), audio encoder (`audio_encoder.py:188`
  `@torch.compiler.disable`), Code2Wav `pre_transformer` (`code2wav.py:308`),
  dense-depth forward (`components/thinker.py:308`).

Global dynamo config (`mstar/engine/__init__.py:3-6`): `recompile_limit=84`,
`allow_unspec_int_on_nn_module=True`, `specialize_int=False`,
`float32_matmul_precision('high')`. No `torch._inductor.config.*` in code —
all Inductor tuning is env-driven at launch.

## 2. Custom ops + graph-break state

`mstar/engine/compile_ops.py` is present on this branch (gated
`MSTAR_CUSTOM_OPS=1`): `mstar::run_attention` (:81), `mstar::fused_experts_fp8`
(:108), `mstar::apply_rope` (:152), each with `register_fake`. Wiring:
`components/attention.py:86-91` (opaque attention op inside compile),
`components/thinker.py:225-232` (skips the per-layer disabled `set_layer_idx`
break), `model/components/moe.py:250-278` (fp8 MoE folds to the op).

Remaining break surface, all outside the compiled hot body by design:
`thinker.py:261` `advance_seq_lens(pos_id_ns=...)` (once per forward, after the
layer loop), ~16 `@torch.compiler.disable` sites in `engine/cache_manager.py`,
plus disables in `utils/attention.py`, `utils/flashinfer_utils.py`,
`utils/sampling.py`. No `.item()`/`.tolist()` inside the compiled thinker/moe
bodies. `model/components/norm.py:53` is compile-aware dispatch, not a break.

## 3. Launch env (the certified recipe)

`launch_mstar_best.sh` sets: `TORCHINDUCTOR_FX_GRAPH_CACHE=1`,
`TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache_cdt`,
`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`, `TORCHDYNAMO_CACHE_SIZE_LIMIT=128`,
`MSTAR_CUSTOM_OPS=1`, `MSTAR_MOE_FP8=1`. Note `TORCHDYNAMO_CACHE_SIZE_LIMIT`
(env) and `recompile_limit=84` (code) are independent knobs set via different
mechanisms. Without the cache dir, cold CDT compile ≈ 40 min.

## 4. Correctness flags (human follow-up)

1. **KVCacheEngine vs runner compile-mode mismatch** — same Thinker, two
   recipes (site 3: default mode, `dynamic=None`, applied after capture at
   `kv_cache_engine.py:294-296`; sites 1–2: max-autotune, `dynamic=False`).
   Likely a redundant un-autotuned cache entry; confirm which path is live
   under the launch config and align.
2. **Silent `pos_id_ns` drop on replay** (`thinker.py:255-260`): vision-prefill
   `pos_id_ns` is not honored on the CUDA-graph replay path (runner calls
   no-arg `advance_seq_lens()` at `cuda_graph_runner.py:552`). Latent bug the
   moment vision prefill is ever captured.

## 5. Unexploited options (ranked by feasibility × expected value)

1. **Inductor autotune knobs beyond CDT** — `max_autotune_gemm_backends`,
   epilogue-fusion flags, `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE`: env-only,
   zero code change, cheap to sweep on a free box. Best next experiment.
2. **Align/dedupe the KVCacheEngine compile** (flag #1) — possible free win if
   the default-mode entry is the live one (it would be un-autotuned today).
3. **`dynamic=False` pinning on the KVCacheEngine path** — decode shapes are
   static per capture bucket; targeted A/B.
4. **Compile the Talker / code-predictor** — infra exists (`apply_rope` op,
   `run_attention`); missing piece is `layer_idx` threading to Talker.
   Highest ceiling of the list (speech decode floor), medium effort.
5. **Regional compile of DecoderLayer** — could cut recompile cost across
   buckets, but risks re-adding the per-layer boundary custom-ops removed.
   Speculative.
6. **AOTInductor** for static-shape submodules (Code2Wav, codec) — kills
   first-call trace cost only; varlen Thinker prefill remains the hard case.
7. **Pin critical inductor flags in code** (`engine/__init__.py`) — hardening
   against launch-path drift, not a perf win.

## 6. Verdict

The compile frontier for the TEXT decode path is exhausted at the knob level:
max-autotune + CDT + FX cache + custom-ops is the shipped ceiling, and the
remaining break (`thinker.py:261`) fires once per forward outside the hot
loop (measured low payoff). What's left is either env-sweepable autotune
knobs (#1) or structural work (#4 Talker compile, EngineCore rewrite) —
consistent with the closed host-side frontier at B32.

# FP8 for M* (Qwen3-Omni 30B-A3B, Hopper H200)

Status: **scaffolded, default OFF, CPU-validated. Needs GPU validation.**
Branch: `exp/fp8-quant` (from `integration-mnew` = M*-new).

Three env-gated FP8 sub-levers. With every flag unset the code path is
byte-for-byte the existing bf16 path (verified by `test_fp8_utils.py` and
by the structure of the edits — every fp8 branch is guarded).

| Env flag             | Lever                         | ROI / safety | Status               |
|----------------------|-------------------------------|--------------|----------------------|
| `MSTAR_FP8_KV`       | fp8 (e4m3) paged KV cache     | highest / low | implemented (GPU-validate) |
| `MSTAR_FP8_WEIGHTS`  | fp8 weights + `scaled_mm` GEMM| high / medium | scaffolded hook      |
| `MSTAR_FP8_ATTN`     | fp8 attention compute         | medium / high-risk | gate only (design) |

---

## Feasibility verdict

**Runtime (from `/home/tim/mstar/.venv`):** torch 2.9.1+cu129, `torch._scaled_mm`
present, `torch.float8_e4m3fn` / `e5m2` present, flashinfer-python 0.6.13,
flash_attn 2.8.3 (FA2 — **no FA3**), triton 3.5.1.

**No existing fp8 support in M*** — the only `quantiz*` hits are the audio
RVQ codec (unrelated) and a docstring in `utils/fused_moe/kernels.py` that
lists an `fp8_w8a8` branch the bf16-only build does not actually ship. So
everything here is new.

**No fp8 Qwen3-Omni checkpoint** — the model ships bf16 safetensors
(`iter_safetensors_shards`). All weight fp8 must therefore be **online /
dynamic** quantization (scales computed at load), not a pre-quantized
checkpoint load. KV fp8 is intrinsically online (K/V cast at write time).

### Sub-lever 1 — FP8 KV cache (e4m3): **FEASIBLE, implemented**
- FlashInfer 0.6.13 `BatchDecodeWithPagedKVCacheWrapper.plan` /
  `BatchPrefillWithPagedKVCacheWrapper.plan` accept separate
  `q_data_type` and `kv_data_type`, and `.run(...)` accepts
  `q_scale` / `k_scale` / `v_scale`. So bf16 query against an fp8 KV cache
  with per-tensor dequant scales is natively supported — no custom kernel.
- `kv_cache_engine.load_model` already had a `kv_cache_type` parameter; the
  KV tensor is allocated with it (`kv_cache_engine.py:130`). `torch.zeros`
  supports fp8 dtypes, and `CPUPagePool` already takes `kv_cache_dtype`, so
  offload works too.
- The only real change is that the old code **conflated the query dtype and
  the KV storage dtype** in the FlashInfer wrappers (`flashinfer_utils.py`:
  one `self.dtype` drove q-cast, kv-cast, and `q_data_type`). Split into
  `dtype` (compute/query, bf16) vs `kv_dtype` (storage, e4m3) + scales.

### Sub-lever 2 — FP8 weights + `scaled_mm`: **FEASIBLE, scaffolded**
- Every TP Linear funnels through `F.linear` in
  `model/components/distributed/linear.py` (Column/Merged/QKV/Row). A single
  `_linear()` helper now dispatches to `torch._scaled_mm` when
  `MSTAR_FP8_WEIGHTS` is set and fp8 scaled_mm is available, else bf16.
- MoE experts (`utils/fused_moe`) are bf16-only today; the fp8_w8a8 grouped
  GEMM is the larger follow-up (see "MoE" below) — **not** wired yet.
- Open risk to resolve on GPU: lazy weight quantization must happen before
  CUDA-graph capture (otherwise the graph bakes the bf16 path); the
  intended production wiring is one-shot quantization at load.

### Sub-lever 3 — FP8 attention compute: **gate only (design)**
- FA3 fp8 is not installed (only FA2 2.8.3). FlashInfer *prefill* does have
  an fp8 path (`fp8_enabled`, `scale_q`), so fp8 attention is reachable via
  FlashInfer, but it also quantizes Q and is the highest quality risk for
  the smallest, prefill-skewed gain. Only the `MSTAR_FP8_ATTN` gate exists;
  no hot-path wiring. Recommend last, after KV+weights land.

---

## Exact code locations (what changed)

- **`mstar/utils/fp8_utils.py`** (new) — env gates, dtype constants
  (E4M3 max 448, E5M2 max 57344), `kv_cache_storage_dtype`,
  `fp8_compute_dtype`, `kv_scales`, `amax_scale`, `quantize_to_fp8` /
  `dequantize_fp8`, `quantize_weight_fp8`, `fp8_linear` (scaled_mm + CPU
  fallback). All CPU-safe.
- **`mstar/utils/flashinfer_utils.py`** — both wrappers gained
  `kv_dtype` / `k_scale` / `v_scale` on `plan`; `set_kv_cache` scales+casts
  K/V to the storage dtype; `run` passes `k_scale`/`v_scale` to FlashInfer
  for fp8. bf16 path unchanged (`kv_dtype` defaults to `dtype`).
- **`mstar/engine/cache_manager.py`** — `BatchedCacheManager` detects an
  fp8 cache (`is_fp8_dtype(kv_cache.dtype)`), records `k_scale`/`v_scale`
  from `kv_scales()`, and the plan paths split compute vs storage dtype
  (`fp8_compute_dtype`) and forward the scales.
- **`mstar/worker/engine_manager.py`** — selects the KV storage dtype via
  `kv_cache_storage_dtype(autocast_dtype)` for `KVCacheEngine` only.
- **`mstar/model/components/distributed/linear.py`** — `_linear()` fp8
  dispatch for all four parallel Linear classes.
- **`test/modular/test_fp8_utils.py`** (new) — CPU tests for the above.

The CUDA-graph path needs **no** change: the runner builds the same
wrappers and the same `BatchedCacheManager` (which reads `kv_cache.dtype`),
so the fp8 dtype/scales flow through `plan_attention` automatically. The
query input buffers stay bf16 (autocast), which is exactly what fp8 KV
wants.

---

## Expected impact

### KV memory (Thinker: 28 layers, 4 KV heads, head_dim 128)
Per-token KV (K+V) = `layers * 2 * kv_heads * head_dim * bytes`:
- bf16: 28·2·4·128·2 = **57,344 B/token (56 KiB)**
- e4m3: **28,672 B/token (28 KiB)** — **2.0x reduction**

(Talker, 20 layers / 2 KV heads / head_dim 64: 10 KiB → 5 KiB, same 2x.)

In the KV-bound decode regime, halving per-token KV roughly **doubles the
number of concurrent sequences / the max batch** that fit in the same KV
budget. Throughput (tokens/s) scales with effective batch until the decode
GEMMs/attention become compute-bound, so expect a **large decode-throughput
gain at high concurrency** (most of it from bigger B, plus less HBM traffic
per attention step). Conservative target: **1.5–2x decode throughput** at
the batch sizes that previously hit the KV ceiling.

### Decode GEMM throughput (weights fp8)
H200 fp8 tensor-core peak is ~2x bf16. Decode is memory-bound at small B,
so the win grows with B: expect **up to ~2x on the large GEMMs** (qkv_proj,
o_proj, MoE up/down) at B≥16, less at B=1–4. MoE experts are the biggest
GEMMs in 30B-A3B (128 experts, top-8) and are where weights-fp8 pays off
most — but that needs the fused fp8 grouped GEMM, not just `_linear`.

### Attention fp8
Mostly a prefill lever; secondary to KV+weights. Defer.

---

## Quality risk + parity gate

FP8 changes numerics, so every lever ships behind a parity gate:

1. **Logit cos-sim vs bf16** — run identical prompts through bf16 and each
   fp8 config; require per-step `cos_sim(logits_fp8, logits_bf16)` ≥ 0.999
   (KV), ≥ 0.998 (weights). e4m3 KV is usually near-lossless once scaled;
   the `amax`/quantize round-trip in `fp8_utils` measures ~2.6% per-tensor
   RMS on random data, which softmax/accumulation largely averages out.
2. **Greedy token-match rate** — ≥ 99% identical next-tokens on a fixed S2T
   set; report first-divergence position.
3. **S2S audio A/B** — generate the same utterances bf16 vs fp8, compare
   (a) transcribed-text WER delta and (b) a perceptual/codec metric
   (e.g. UTMOS or mel-spectral distance). The Talker/code2wav path is more
   sensitive than text; treat any audible degradation as a blocker.
4. **Scale calibration** — defaults `k_scale=v_scale=1.0` are *plumbing*
   defaults (valid only if |K|,|V| < 448). Run a short calibration pass to
   measure K/V amax and set `MSTAR_FP8_KV_K_SCALE` / `_V_SCALE`; QK-norm
   keeps K well-bounded, V is the one to watch. If V overflows at 1.0,
   raise its scale (or use e5m2 for V via `MSTAR_FP8_KV_DTYPE=e5m2`).

---

## Implemented vs needs-GPU

**Implemented + CPU-tested:** env gates; KV storage-dtype selection;
compute/storage dtype split; scale math (amax/quantize/dequantize/clamp);
`fp8_linear` with CPU fallback; FlashInfer wrapper fp8 plumbing
(plan/run/set_kv_cache signatures); cache-manager + engine-manager wiring;
linear-layer fp8 hook.

**Needs GPU to validate / finish:**
- FlashInfer fp8 KV decode+prefill actually run on H200 (scale convention,
  cuda-graph capture with fp8 cache, accuracy).
- `torch._scaled_mm` weight path: SM89+ check, shape constraints (K,N % 16),
  and moving weight quantization to load-time (pre-capture).
- MoE fp8 grouped GEMM (`utils/fused_moe`) — the high-value weights win.
- `MSTAR_FP8_ATTN` hot-path wiring.

---

## GPU validation commands

Pin the same GPUs every run (see workspace conventions); confirm idle via
`nvidia-smi`; wrap each run in `timeout`; one config per process.

```bash
# 0) unit gate (no GPU)
pytest test/modular/test_fp8_utils.py -q

# 1) correctness / parity gate — fp8 KV vs bf16, fixed prompts
#    (cos-sim logits + greedy token-match; expect ≥0.999 / ≥99%)
MSTAR_FP8_KV=0 timeout 1200 <eval-harness> --task s2t --emit-logits ref_bf16.pt
MSTAR_FP8_KV=1 timeout 1200 <eval-harness> --task s2t --emit-logits fp8_kv.pt
python perf_testing/compare_logits.py ref_bf16.pt fp8_kv.pt   # cos-sim, token-match

# 2) S2S audio A/B (Talker + code2wav most sensitive)
MSTAR_FP8_KV=0 timeout 1800 <s2s-harness> --out audio_bf16/
MSTAR_FP8_KV=1 timeout 1800 <s2s-harness> --out audio_fp8/
python perf_testing/audio_ab.py audio_bf16/ audio_fp8/   # WER delta + mel/UTMOS

# 3) throughput sweep — bf16 vs fp8 KV across B and modality
for MODE in s2t i2t s2s; do
  for B in 8 16 32; do
    MSTAR_FP8_KV=0 timeout 1800 <bench> --mode $MODE --batch $B --tag bf16_${MODE}_b$B
    MSTAR_FP8_KV=1 timeout 1800 <bench> --mode $MODE --batch $B --tag fp8kv_${MODE}_b$B
  done
done
# expect fp8 KV to admit larger B before the KV ceiling and raise tok/s at B=16,32

# 4) (after KV passes) weights fp8, then KV+weights stacked
MSTAR_FP8_WEIGHTS=1 timeout 1800 <bench> --mode s2t --batch 16 --tag fp8w_s2t_b16
MSTAR_FP8_KV=1 MSTAR_FP8_WEIGHTS=1 timeout 1800 <bench> --mode s2t --batch 32 --tag fp8kvw_s2t_b32

# clocks/persistence teardown + idle check after every run (workspace rules)
nvidia-smi --query-gpu=index,persistence_mode --format=csv
```

Record raw datapoints to `raw.json`, generate charts from the shared style,
commit only complete runs on the bench branch (workspace git workflow).

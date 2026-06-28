# Qwen3-Omni MoE optimization — feasibility, design, scaffold

Branch: `exp/moe-kernels` (based on `integration-mnew` = M*-new).
Target: Qwen3-Omni Thinker (30B total, ~3B active MoE) and Talker MoE on 2x H200.
Scope of this change: investigation + env-gated scaffold + design + CPU-safe tests.
No GPU runs performed here.

## 1. Feasibility verdict

**The headline lever in the task brief — "swap a per-expert Python loop for a
fused/grouped GEMM" — is ALREADY DONE in M*-new.** The Thinker and Talker MoE
already dispatch through a fused grouped-GEMM Triton kernel (ported from sglang),
gated only by whether `sgl-kernel` is importable. On this box it is, so the fused
path is already the default. Re-implementing a batched-bmm grouped path would be
a regression, not a win.

The remaining, still-real decode-throughput levers, in feasibility order:

1. **Device-tuned Triton tile configs (IMPLEMENTED, env-gated, default OFF).**
   The current tile picker is a crude two-branch heuristic; sglang/vLLM ship
   per-device, per-shape autotuned tiles. Numerically equivalent, low risk,
   meaningful decode speedup. This is the primary win wired up here.
2. **fp8 grouped GEMM (DeepGEMM) (DESIGNED, not implemented).** `deep_gemm` is
   installed with exactly the masked grouped-GEMM decode primitive. Largest
   GEMM win but needs fp8 weights + quant + a quality gate.
3. **Expert parallelism across the 2 GPUs (DESIGNED, not implemented).** Current
   multi-GPU MoE is tensor-parallel (all-reduce per layer); EP would cut
   redundant compute. Flag recognized, design below.
4. **Host-device sync in routing (NO ACTION NEEDED).** The active (fused) router
   path is already sync-free (softmax+topk on device, `moe_align_block_size` on
   device). Only the naive fallback has `.nonzero()`/`.where()` syncs.

## 2. Current MoE kernel (file:line + quote)

Router + dispatch live in `mstar/model/components/moe.py`. Selection:

```python
# mstar/model/components/moe.py  (original _dispatch)
def _dispatch(...):
    """Pick fused-Triton if available, otherwise the naive loop."""
    if _HAS_FUSED and hidden_states.is_cuda:
        return _fused_experts(hidden_states, gate_up_proj, down_proj,
                              routing_weights, selected_experts)
    return dispatch_experts_fused(...)   # naive per-expert loop fallback
```

`_HAS_FUSED` comes from importing `mstar.utils.fused_moe` (needs `sgl_kernel`):

```python
# mstar/model/components/moe.py:101-109
try:
    from mstar.utils.fused_moe import fused_experts as _fused_experts
    from mstar.utils.fused_moe.align import has_sgl_kernel
    _HAS_FUSED = has_sgl_kernel()
except Exception as e:
    _fused_experts = None
    _HAS_FUSED = False
```

The fused kernel itself (`mstar/utils/fused_moe/runner.py:fused_experts`) is a
real grouped GEMM: `moe_align_block_size` (sgl_kernel CUDA op) permutes tokens
into per-expert, `BLOCK_SIZE_M`-padded blocks, then one `fused_moe_kernel`
grouped GEMM for gate+up, a Triton SwiGLU, a second grouped GEMM for down (with
the routing weight folded in), and `moe_sum_reduce_triton` over top-k. This is
the standard sglang/vLLM fused-MoE design — not a Python loop.

TP path (`ParallelSparseMoeBlock._dispatch_tp`, `moe.py`): experts are sharded
along the intermediate dim; `fused_experts(reduce_results=False)` → all-reduce →
`moe_sum_reduce_triton`. So multi-GPU today = **tensor parallel**, not expert
parallel.

Thinker MoE config (`mstar/model/qwen3_omni/config.py`): `hidden_size=2048`,
`num_experts=128`, `num_experts_per_tok=8`, `moe_intermediate_size=768`,
`norm_topk_prob=True`. Talker: `hidden=1024`, `inter=384`, `E=128`, `top_k=8`,
shared expert + sigmoid gate, `norm_topk_prob=False`.

Installed kernel libs (`/home/tim/mstar/.venv`): `triton 3.5.1`,
`sgl_kernel 0.3.21`, `deep_gemm` (DeepGEMM, fp8 grouped GEMM), `flashinfer`,
`triton_kernels`.

## 3. The crude tile picker (what we improve)

```python
# mstar/utils/fused_moe/kernels.py:get_default_config (original)
if M <= E:   # E == 128, so all decode batches hit this
    return {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1}
return {"BLOCK_SIZE_M":64,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":32,"GROUP_SIZE_M":8}
```

Every decode step uses a single fixed tile with no `num_warps`/`num_stages`
tuning. The H200-tuned config for the Thinker shape (E=128, N=768) varies tiles
and pipeline depth per batch size, e.g. M=1: `BLOCK_SIZE_N=64`, `num_stages=5`;
M=16: `BLOCK_SIZE_K=256`. Deeper software pipelining at small M is exactly where
the heuristic leaves throughput on the table.

## 4. Proposed optimization (implemented vs stubbed)

### IMPLEMENTED — tuned tile configs, `MSTAR_FUSED_MOE_TUNED` (default OFF)
- New `mstar/utils/fused_moe/tuning.py`: `load_tuned_config(M,E,n_inter,...)`
  reads `configs/E=<E>,N=<inter>,device_name=<dev>.json`, picks the smallest
  batch bucket ≥ M (clamps to max), returns `None` when the flag is off / device
  unknown / file missing.
- `configs/E=128,N=768,device_name=NVIDIA_H200.json` (Thinker) and
  `...,N=384,...` (Talker), ported from sglang.
- `get_default_config` consults the tuned loader first (looked up by
  `n_inter = N // 2`, since it is called with the fused `2*inter` width), and
  falls back to the existing heuristic.
- **Numerically equivalent**: only GEMM tiling/pipelining changes, math is
  identical → cos-sim ~1.0. Off by default; flip on after the parity + decode
  bench gate.

### IMPLEMENTED — `MSTAR_FUSED_MOE` tri-state override (default auto = unchanged)
- `auto` (default): fused when available + CUDA, else naive. No behavior change.
- `off`: force the naive loop (parity baseline / bisect). Single-GPU path only;
  the TP path requires the fused kernel.
- `on`: force fused; raise if unavailable instead of silently degrading.

### STUBBED / DESIGN — `MSTAR_MOE_EXPERT_PARALLEL` (default OFF)
Recognized in `moe.py` (`_expert_parallel_requested`, warns once at parallel
block construction); not wired. Proposed design: shard the 128 experts across
the 2 GPUs (64 each) instead of slicing every expert's intermediate dim.
Replace the per-layer all-reduce with an all-to-all (dispatch tokens to the
owning GPU, combine after). Pros: ~halves redundant expert FLOPs and weight
memory traffic per GPU vs TP; better at small decode batch. Cons: all-to-all
latency + load imbalance across experts; needs a dispatch/combine path
(DeepEP-style) M* does not have yet. Bigger lift than (1)–(2).

### DESIGN — fp8 grouped GEMM (DeepGEMM)
`deep_gemm.m_grouped_fp8_gemm_nt_masked` / `..._contiguous` are the masked /
contiguous grouped fp8 GEMMs purpose-built for decode MoE. Plan: per-block
cast experts to fp8 (`per_block_cast_to_fp8`) at load, per-token-cast
activations at runtime, run the two grouped GEMMs in fp8 with bf16 accumulate.
Expected ~1.5–2x on the expert GEMMs (the MoE bottleneck) on H200. Risk:
quality (needs cos-sim + task-metric gate vs bf16) and a quant path for the
fused `gate_up_proj`/`down_proj` checkpoint layout. Highest payoff, highest
risk — sequence it after the tuned-tile win is benchmarked.

## 5. Expected decode-throughput gain (decode dominates every path)

Decode is the dominant cost across all serving paths — S2T/I2T token rate and
S2S/I2S frame rate are all gated by per-step decode latency, and MoE dispatch is
the largest single component of that step for this architecture. So a decode-step
MoE speedup propagates to **all** paths' tok/s and frame rate.

- Tuned tiles (implemented): typically ~10–30% on the MoE GEMM step at decode
  batch sizes from the sglang/vLLM autotune deltas; net per-token decode gain is
  smaller (MoE is a fraction of the step) but positive and free. **To be
  measured on H200 before enabling by default.**
- fp8 grouped GEMM (designed): ~1.5–2x on the expert GEMMs → the largest decode
  lever, pending quality gate.
- Expert parallel (designed): reduces redundant compute/memory per GPU vs TP at
  decode; gain depends on all-to-all overhead and expert balance.

All figures are estimates from baseline kernel data; none are measured here (no
GPU runs). The numbers must be confirmed by the GPU validation below.

## 6. Risk

- Tuned tiles: very low. Pure scheduling; equivalence-tested; default OFF.
  Tiles were tuned for triton 3.2.0; installed is 3.5.1 — validate before
  enabling (tiles are still valid, may not be optimal).
- `MSTAR_FUSED_MOE` override: low. Default `auto` preserves current behavior.
- fp8 / EP: not implemented; no runtime risk until built. Both need quality and
  performance gates before exposure.

## 7. GPU validation commands (run on H200; not run here)

Parity (must pass before enabling tuned tiles by default):

```bash
cd /home/tim/exp/moe-wt
# fused vs naive cos-sim/atol parity (Thinker + Talker shapes)
.venv/bin/python -m pytest test/modular/test_qwen3_omni_fused_moe.py -q
# CPU-safe logic (loader, mode parsing, permutation) — runs anywhere
.venv/bin/python -m pytest test/modular/test_qwen3_omni_moe_tuning.py -q
# tuned tiles produce the same output as the heuristic (numerical equivalence)
MSTAR_FUSED_MOE_TUNED=1 .venv/bin/python -m pytest \
    test/modular/test_qwen3_omni_fused_moe.py -q
```

Decode-throughput A/B (heuristic vs tuned), single fixed GPU set, per the
benchmarking conventions (hard timeout, monitor, cleanup):

```bash
# baseline (heuristic tiles)
CUDA_VISIBLE_DEVICES=0,1 MSTAR_FUSED_MOE_TUNED=0 timeout 1800 <decode-bench-cmd>
# tuned tiles
CUDA_VISIBLE_DEVICES=0,1 MSTAR_FUSED_MOE_TUNED=1 timeout 1800 <decode-bench-cmd>
# force-naive sanity (slow path still correct)
CUDA_VISIBLE_DEVICES=0,1 MSTAR_FUSED_MOE=off  timeout 1800 <decode-bench-cmd>
```

Compare decode tok/s (S2T/I2T) and frame rate (S2S/I2S) across the three; only
enable `MSTAR_FUSED_MOE_TUNED` by default if parity holds and decode throughput
improves.

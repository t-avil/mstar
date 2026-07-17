# XQA in-graph multistep decode — design + capture-safety notes

Plan-free xqa/trtllm paged decode (`MSTAR_XQA_DECODE=1`) makes N decode steps
capturable in ONE CUDA graph: the kernel takes a device-resident `seq_lens`
(uint32) and a dense `block_tables` (int32 `[bs, W]`) and does no host sync, so
`seq_lens` can be advanced *in-graph* between K forward passes
(`MSTAR_XQA_MULTISTEP=K`, K>=2).

Stage-A primitives live in `mstar/utils/flashinfer_utils.py`:
`multistep_write_locations` (page-roll write-slot math), `multistep_pages_needed`
(pre-reserve columns), `ingraph_greedy_token`, `trim_after_eos`. Their pure index
math is validated on CPU by `test/xqa/multistep_cpu_checks.py`.

The captured K-loop body (per step) is: forward (attention read at current
`seq_lens` + `set_kv_cache` write at `kv_cache_locations`) -> sample ->
`record_token_ingraph(step, tok)` -> `advance_step_ingraph()`.

## Capture-safety fix (2026-07-17)

`FlashInferXqaDecodeWrapper.advance_step_ingraph()` failed at
`self._seq_lens_buf[:n] += 1` (flashinfer_utils.py). Reproduced standalone in
seconds — no server boot — by `test/xqa/capture_repro.py` (tiny wrapper,
`use_cuda_graph=True`, `enable_multistep(2)`, synthetic seq_lens/block_tables,
`torch.cuda.graph` capture of a K-loop calling `advance_step_ingraph` + a dummy
forward).

**Actual illegal op (root cause).** The write-location buffer `_seq_lens_buf` is
`torch.uint32` (the dtype the trtllm/xqa kernel requires). PyTorch has **no CUDA
add kernel for uint32** — `self._seq_lens_buf[:n] += 1` raises
`NotImplementedError: "ufunc_add_CUDA" not implemented for 'UInt32'` and, in the
server's torch/driver build, surfaces as an async `cudaErrorInvalidValue` pinned
to that line. It fails **even in eager mode**, so the advance never reached a
valid capture.

**On the allocation hypothesis.** The original `multistep_write_locations`
allocates new tensors every call (`torch.arange`, advanced-index gather,
`torch.stack`, div/mod intermediates). This was *suspected* to break capture, but
a direct probe showed that in this PyTorch build those allocations capture and
replay fine — the caching allocator serves them from the graph's private memory
pool. Allocations were **not** the illegal op here. We still eliminated every
allocation from the advance region (below) as defense-in-depth: some
torch/driver combos do restrict in-capture allocation, and an allocation-free
in-place form is strictly more robust for capture and re-capture.

**Fix (capture-safe, allocation-free, uint32-safe).** Pre-allocate persistent
scratch buffers ONCE at wrapper init (cuda-graph branch): `_ms_len` (int64
running length — the source of truth), `_ms_last`, `_ms_col`, `_ms_off` (int64),
`_ms_page` (int32 `[bs,1]`). `plan()` seeds `_ms_len` from `seq_lens` (outside
capture, allocation fine). `advance_step_ingraph()` now uses ONLY in-place ops
into those buffers:

1. `_ms_len[:n] += 1` — int64 add (CUDA-supported, unlike uint32).
2. `_seq_lens_buf[:n].copy_(_ms_len[:n])` — int64 -> uint32 cast-copy for the kernel.
3. `last/col/off` via `torch.sub/div(...,out=)`; `page` via `torch.gather(...,out=)`.
4. Scatter into the static `kv_cache_locations[:n,0]` / `[:n,1]` via `copy_`.

Zero tensor allocations inside the capture region; identical page-roll semantics
to `multistep_write_locations` (kept unchanged, still exercised by the CPU
checks).

**Validation.**
- `test/xqa/capture_repro.py`: capture SUCCEEDS; replay advances `seq_lens` and
  `kv_cache_locations` correctly (matches CPU brute-force, incl. a page-boundary
  roll); two replays are byte-identical (deterministic). Eager advance also now
  works (uint32 bug fixed).
- `test/xqa/multistep_cpu_checks.py`: all checks still PASS
  (`multistep_write_locations` untouched, byte-identical).

**Remaining capture risks to watch when booting the full server**
(`MSTAR_XQA_DECODE=1 MSTAR_XQA_MULTISTEP=2`): the repro captures the advance +
KV-write machinery and a *dummy* forward, not the real attention. Still to
confirm under real capture: (a) `trtllm_batch_decode_with_kv_cache` captures
cleanly (it is plan-free/no host sync by design, but unverified in-graph here);
(b) the sampler / `record_token_ingraph` path introduces no host sync; (c) any
`.item()`/`.cpu()`/`max_seq_len` recompute in the real forward stays outside the
captured region; (d) `_ms_len` / `_seq_lens_buf` / `kv_cache_locations` are
re-seeded by `plan()` before every captured batch (the graph re-runs the same
in-place `+= 1`).

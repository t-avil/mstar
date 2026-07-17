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

## §6.5 Emit / stop seam — consume all K tokens per replay (2026-07-17)

The capture emits K tokens per replay but the emit path consumed only one, so
the client saw every K-th token (decimated / garbled text). The seam wires the
existing (already list-aware) transport to carry all K, in order, with
single-step-identical stop handling.

**Data flow (unchanged plumbing, now fed K tokens).**
`_remap_multistep_tokens` returns per rid `new_tokens` = the K in-order token
views and `new_token` = the last (feed anchor). The Thinker decode node has two
text edges: `new_token` -> `EMIT_TO_CLIENT` (api server iterates
`graph_edge.tensor_info` in order -> one detokenized chunk per token; worker
`_send_outputs` extends `num_output_tokens` by the tensor count) and
`text_inputs` -> next Thinker step. Both are already list-aware, so emitting K
is a matter of populating the `new_token` edge with the K token list while
feeding only the last to `text_inputs`. Ordering is preserved end to end: the K
views are built in generation order, ride one `result_tensors` message, and the
api server appends chunks in arrival order (no loop-index re-sort).

**Stage 1 — coherence (this commit).** `ThinkerSubmodule.postprocess` (runs on
the GPU thread — pure list reshaping, no host sync): when `new_tokens` is
present, set `new_token = list(new_tokens)` (all K -> client) and
`text_inputs = <last>` (feed). Single step (no `new_tokens`) is byte-identical.
Validated by an `--ignore-eos --output-len N` run: mid-K EOS is not yet trimmed,
so this isolates "is the K-token generation coherent" from "is stop handling
correct". The conductor's token-accurate `num_output_tokens >= max_output_tokens`
governs fixed length (overshoot <= K-1 per replay); the client's `output-len`
truncation makes the visible length exact.

**Stage 2 — stop handling (this commit).** Adds `multistep_keep_count`
(flashinfer_utils), `ThinkerSubmodule.check_stop` scanning all K tokens for EOS,
`ThinkerSubmodule.trim_multistep_emit` (first-EOS-inclusive keep count, computed
from the host copy the worker already made for check_stop — no extra sync), and
`kv_cache_engine.trim_multistep_for_batch` invoked by the worker right after
`check_stop_for_batch`, trimming the routed `new_token` list before store/route
so post-EOS tokens are never stored, emitted, or counted. Same round-trip, no
async deferral. Single step (`len(new_token) <= 1`) and submodules without the
hook are untouched.

**Speech gate.** `forward_batched_multistep` returns only `__batched_tokens__` /
`__batched_logits__` — it drops `thinker_states`/`thinker_mask`, which the Talker
needs. So multistep is text-out only; audio-output requests must stay
single-step. This is enforced by topology/config (only text-out serving sets
`MSTAR_XQA_MULTISTEP`), matching the eiv2 topology-per-modality rule. The emit
seam does not re-introduce speech data — it only reshapes the text token edges.

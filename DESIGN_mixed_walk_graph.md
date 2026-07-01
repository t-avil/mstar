# Graphed mixed replay — implementation plan (the stubbed piece)

Status after port to 4c33b33 (`opt/mixed-walk`): mixed prefill+decode runs via
`KVCacheEngine._execute_mixed_eager` (one FlashInfer varlen prefill forward, NO
CUDA graph). `CudaGraphRunner.run_mixed` (cuda_graph_runner.py:1225) is a stub
that logs "eager fallback". This is functionally correct and gave the measured
**S2T 1.26–1.66×** win, but the eager penalty caps the win and causes the
**i2t B32 collapse (4.7× slower)**: at high batch nearly every step is mixed →
nearly every step runs eager (un-graphed 30B Thinker), and vision-heavy prefill
chunks make the eager forward dominate. Graphing the mixed step removes this.

## Why graphing fixes both

The measured decode-step profile (see [[project_mstar_vllm_decode_gap]]) is
~66% fwd + plan + comm; the captured decode graph collapses hundreds of
per-layer launches into one replay. The eager mixed path forfeits that. A
captured mixed graph restores it for mixed steps too → S2T win grows and the
i2t eager penalty disappears.

## Design (mirrors the existing decode capture)

The mixed `CudaGraphKey(mixed=True, num_decode=D, num_prefill_tokens=P)` already
exists (bs = D+1, num_tokens = D+P; P from a FIXED bucket set). Implement:

1. **MIXED CudaGraphConfig** — declare capture shapes as the cross product of
   decode buckets D ∈ {1,2,4,8,16,32} and prefill buckets P ∈ {128,256,512}
   (18 graphs; bounded, matches `pad_prefill_tokens_to_bucket`). Gate the P set
   behind `MSTAR_MIXED_PREFILL_BUCKETS` to tune capture cost vs coverage.

2. **Static buffers per (D,P)** — one varlen layout: qo_indptr =
   [0,1,..,D, D+P0]. The D decode rows are length-1; the prefill chunk is the
   single length-P segment (max_prefill_requests currently small). Reuse
   `build_mixed_varlen_layout` to fill kv_indptr / kv_indices / positions into
   pre-allocated static tensors.

3. **Capture loop** — for each (D,P): warmup pair then `torch.cuda.graph`
   capture of the Thinker `forward_batched` over the mixed layout, exactly as
   `_capture_slots` does for decode. The **FlashInfer prefill wrapper** (not the
   decode wrapper) serves the mixed qo_indptr; its `plan()` runs OUTSIDE the
   graph and updates static buffers via `.copy_()` (same pattern as decode
   plan, cache_manager.py).

4. **`run_mixed` replay** — replace the eager-fallback body: pad the actual
   prefill chunk up to the nearest P bucket (`pad_prefill_tokens_to_bucket`),
   snap D to the nearest decode bucket (pad decode rows with dummies as decode
   capture already does), copy inputs into the (D,P) graph's static buffers,
   run `plan()` on the prefill wrapper, replay. Fall back to
   `_execute_mixed_eager` only when (D,P) exceeds the largest captured bucket.

5. **Route** — in `execute_forward`, when `mixed_plan` is present and a captured
   (D,P) graph exists, call `run_mixed` (graphed) instead of
   `_execute_mixed_eager`.

## Validation order (needs GPU)

1. Server-validate the current EAGER port on 4c33b33 (confirm S2T win + i2t
   behavior reproduce on the new base).
2. Implement graphed replay; re-run the s2t/i2t B=1/8/32 A/B. Expect S2T win to
   grow past 1.66× and i2t B32 collapse to resolve.
3. Interleaved off-vs-on for a contention-robust final number, then port back to
   a mixed-walk-only branch (strip encoder-coalesce) for review.

## Risk / notes

- Capture memory: 18 mixed graphs × slots × wrapper sets. Start `MSTAR_NUM_SLOTS=1`
  for mixed if memory-tight.
- max_prefill_requests > 1 multiplies the P-shape space; keep it at 1 for the
  first graphed version (matches current default).
- The prefill wrapper `plan()` per mixed step is the remaining un-hidden cost
  (~same as decode plan); overlap later if it shows up in the profile.

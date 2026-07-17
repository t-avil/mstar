# Stage-B POC: mixed decode + bounded-prefill in one xqa call

**Verdict: the core primitive WORKS — numerically correct AND CUDA-graph capturable.**
There is ONE API constraint that shapes the integration (xqa's query length is
**uniform per call**), but it does **not** block a single captured step — it just
picks between two equivalent assembly strategies. This de-risks Stage B: the
attention layer can co-admit a bounded prefill chunk with decodes in one captured
step, which is the mechanism the i2t B32 TTFT tail needs.

Script: `test/xqa/mixed_prefill_poc.py` (standalone; no server, no engine).
Run live on H200 (SM90) GPU 0, FlashInfer 0.6.13, M* shapes (hd128, page128,
bf16, GQA 32/4 = grp8). Actual output at the bottom.

---

## 1. Is the mixed primitive numerically correct?  YES

| Test | what it proves | max_abs vs fp32 ref | gate |
|---|---|---|---|
| **PART 1** q_len=N=8 prefill chunk, uniform, causal mask; ctx incl. page-boundary + ragged (`[0,5,120,128,130,250,384,500]`) | the prefill-chunk primitive itself | **8.0e-3** | PASS |
| **PART 2 (S)** single padded call: 3 decode + 1 prefill in ONE q_seq_len=8 call | mixed-in-one-call | dec **5.4e-3** / pre **3.0e-3** | PASS |
| **PART 2 (T)** two calls: q_len=1 decode + q_len=N prefill | mixed-as-two-calls | dec **5.4e-3** / pre **3.0e-3** | PASS |

All within a few e-3 — same bf16 class as the already-passed single-step xqa
parity gate. (max_rel reads high, ~3–8, but that is the near-zero-denominator
artifact: `abs/( |ref| + 1e-4 )` on tiny output elements. max_abs is the gate.)

The fp32 reference is exact per-request causal attention: query row `j` (absolute
position `ctx+j`) attends to KV `[0, ctx+j]`. The xqa output matches it, so the
kernel's spec-dec semantics are confirmed empirically (not just from docs).

## 2. Is it CUDA-graph capturable?  YES

**PART 3**: the mixed q_seq_len=8 call captured in `torch.cuda.graph` with static
buffers (`query`, `block_tables`, `seq_lens`, **`mask`**, `out`) and replayed:

- capture succeeded (no in-capture host sync / alloc / uint32-op failure);
- **2 replays bit-identical** (`torch.equal` True);
- **replay == eager, max_abs = 0.0e+00** (exact).

The plan-free entry point takes only tensors + Python ints, so it traces cleanly,
exactly like the single-step xqa path. The `mask` is just another static device
tensor. No capture-safety issue of the Stage-A class appeared.

---

## 3. The exact API shape (what a mixed batch looks like)

`flashinfer.decode.trtllm_batch_decode_with_kv_cache(...)` → on SM90 dispatches to
the **xqa** kernel (`decode.py:2708-2711`). Relevant args:

```
query        : [batch * q_len_per_req, num_qo_heads, head_dim]   bf16
kv_cache     : [num_pages, 2, page_size, num_kv_heads, head_dim] NHD  (M* native, no reshape)
block_tables : [batch, max_pages_per_seq]  int32   (DENSE, left-packed, zero-padded)
seq_lens     : [batch]  uint32   = in_kv_len + q_len_per_req   (INCLUDES the new query tokens)
max_seq_len  : python int
q_len_per_req: python int   (SCALAR — uniform query length for the WHOLE batch)
mask         : [batch, q_len_per_req, ((q_len_per_req+31)//32)*2]  uint16, bit-packed
```

**spec-dec semantics (confirmed):** the `q_len_per_req` query rows of request `r`
map to its **last `q_len_per_req` KV positions** `[in_kv_len, in_kv_len+q_len)`.
Their K,V must be written into the paged cache at those positions **before** the
call. Each query row attends to **all prior context `[0, in_kv_len)`
unconditionally** + the **local window** `[in_kv_len, seq_len)` gated by `mask`.

**mask layout (exact, from FI 0.6.13 `generate_causal_mask`):** bit `i` of row `j`
set ⇒ query row `j` attends local KV column `i`. Causal = `kv_idx <= q_idx`. The
mask is **shared across the batch** (same `[q_len, mask_row]` broadcast to all
requests). uint16, `mask_size_per_row = ((q_len+31)//32)*2` (2 for q_len≤32).

## 4. THE constraint that shapes integration (be honest)

**xqa's `q_len_per_req` is a single scalar → every request in one call shares the
same query length.** You **cannot** pass per-request q_lens (decode=1, prefill=N)
directly. FlashInfer's variable-length path (`max_q_len` + `cum_seq_lens_q`)
exists but is **explicitly rejected on the xqa backend** (`decode.py:2719-2720`:
"xqa backend does not support cum_seq_lens_q") — it is **trtllm-gen (Blackwell)
only**. H200 is SM90 → xqa → no variable q_len.

This does **not** block a single captured step. Two equivalent ways to assemble a
mixed step, both proven here (PART 2), both capturable:

- **(S) Single padded call**, uniform `q_seq_len = N` (the prefill chunk size).
  A decode is emulated as **row 0** of an N-row block with `seq_len = committed+N`;
  the shared causal mask makes row 0 attend to `[0, committed]` only (context +
  local col 0 = itself) = exactly the decode; rows 1..N-1 are junk, discarded.
  **Cost:** each decode does **N× the query FLOPs** (N query rows) and needs N KV
  slots reserved (N-1 junk, masked out). At B32 with the current bounded N this is
  a real but bounded waste on the decode side.
- **(T) Two calls per step**: one `q_len=1` xqa decode call for all decode
  requests + one `q_len=N` xqa call for the prefill chunk, captured back-to-back
  in ONE graph. **No decode padding waste**; two kernel launches instead of one.

Both are plan-free, device-`seq_lens`, and capturable. **(T) is the recommended
default** (no decode FLOP inflation, no junk-KV reservation, cleaner scheduler
bookkeeping); **(S)** is available if a single fused launch ever measures faster
than two launches for a given (bs, N). The decision is now a perf A/B, not a
feasibility question — which is the whole point of this POC.

> Note: Stage A killed **q_len=1 multistep** because xqa is ~50% slower than
> FlashInfer BatchDecode at q_len=1. Strategy **(T)**'s decode call is also
> q_len=1 xqa, so the decode leg inherits that per-step disadvantage. This does
> not affect Stage B's *feasibility* (proven here) but it means the **win must
> come from unfreezing decodes during prefill** (queue-wait elimination), not from
> the decode kernel being faster. If the q_len=1 decode-leg cost is material, (S)
> (which runs decode as extra rows of the same prefill-shaped kernel) or keeping
> FlashInfer BatchDecode for a pure-decode step and only switching to xqa on steps
> that carry a prefill chunk are both open levers.

## 5. Concrete next-step integration plan

Goal: at i2t B32, the 32 image-prefills currently run as **separate walks that
freeze all decodes** (82% of the TTFT tail is walk-serialization queue-wait). Fix
= co-admit a **bounded** prefill slice with each decode step so decodes never
freeze (vLLM's mechanism).

1. **Scheduler: admit a bounded prefill chunk per decode step.** Add a per-step
   budget `MSTAR_MIXED_PREFILL_CHUNK = N` (start 128–512 tokens; already the
   PREFILL_BUCKET regime). Each decode step pulls up to N unprocessed prefill
   tokens from the head-of-queue image-prefill request and marks them "in flight".
   Reuse the existing bounded-prefill budget plumbing
   (`MSTAR_MIXED_BUDGET_TOKENS` / `PREFILL_CHUNK` from the throttle work).
2. **Assemble the mixed batch** where `run_attention` is dispatched
   (`cache_manager.run_attention`, `cache_manager.py:653`; single call site
   `attention.py:78`). Two-call strategy (T):
   - **decode leg**: existing `FlashInferXqaDecodeWrapper.run` (q_len=1) for the
     `bs` decode requests — already validated + captured.
   - **prefill leg**: a new `run_prefill_chunk(q_chunk, mask, seq_lens)` on the
     same wrapper calling `trtllm_batch_decode_with_kv_cache(..., q_len_per_req=N,
     mask=causal_mask)`. Build the causal mask **once** (static; it only depends on
     N) — reuse `generate_causal_mask` from the POC.
   - Write the chunk's K,V into KV positions `[committed, committed+N)` **before**
     the prefill leg (the wrapper's `set_kv_cache`, extended to write N rows).
3. **Capture both legs in one graph.** `_capture_one_basic_batched`
   (`cuda_graph_runner.py:679`): capture `[decode-attn (q=1)] then
   [prefill-attn (q=N)]` with static `seq_lens` / `block_tables` / `mask` buffers.
   PART 3 proves the prefill leg captures; the decode leg is the already-captured
   single-step path. Key the graph on `(bs, N)`.
4. **Dense page table + reserved pages.** Reuse `build_dense_block_tables` +
   `multistep_pages_needed`-style reservation so the prefill chunk's N new tokens
   always land in pre-allocated pages (no in-graph alloc). The prefill request
   needs `ceil((committed+N)/page)` pages reserved before capture.
5. **Gate + A/B.** Flag `MSTAR_MIXED_PREFILL=1`, default off, plan-based +
   pure-xqa-decode paths kept as fallback. Then the perf question: does
   co-admission collapse the B32 TTFT tail (target: into the committed vLLM band
   8.03–8.50 req/s i2t B32) without regressing decode ITL? Decide (S) vs (T) by
   measured launch cost. **Do not benchmark vLLM** — compare to committed values.

**Bottom line:** the primitive is real, correct, and capturable on M*'s exact
config with zero new dependencies. The only design fork is single-padded-call vs
two-call assembly, and that is now a perf A/B, not a feasibility risk. Stage B is
cleared to proceed to scheduler + capture integration.

---

### Actual live output (H200 SM90, GPU 0, FI 0.6.13)

```
PART 1  q_len=N prefill-chunk primitive (uniform N, causal mask)
  N=8 ctx=[0, 5, 120, 128, 130, 250, 384, 500]
  prefill rows vs fp32 causal ref: max_abs=8.025e-03 max_rel=8.159e+00  -> PASS
PART 2  MIXED batch: 3 decode (real q_len=1) + 1 prefill (q_len=N)
 (S) SINGLE padded call, uniform q_seq_len=N, decode=row0:
     decode rows vs ref : max_abs=5.407e-03 max_rel=3.251e+00
     prefill rows vs ref: max_abs=3.018e-03 max_rel=3.869e+00  -> PASS
 (T) TWO calls, q_len=1 decode + q_len=N prefill:
     decode call vs ref : max_abs=5.407e-03 max_rel=3.251e+00
     prefill call vs ref: max_abs=3.018e-03 max_rel=3.869e+00  -> PASS
PART 3  CUDA-graph capture + deterministic replay
  captured OK; replay deterministic (2x): True
  replay vs eager: max_abs=0.000e+00  -> match
VERDICT
  PART1 prefill-chunk primitive numerically correct : PASS
  PART2 (S) single padded mixed call correct        : PASS
  PART2 (T) two-call mixed correct                  : PASS
  PART3 mixed call CUDA-graph capturable+determ     : PASS
  OVERALL: PASS
```

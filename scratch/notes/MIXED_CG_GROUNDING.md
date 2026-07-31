# Grounding: CUDA-graph capture for the MIXED prefill+decode walk

All anchors are on commit 6bdcb90 (exp/mixed-walk-piggyback). Your worktree is forked
from it, so the SAME file:line anchors apply in your worktree. Paths below say
`piggyback-wt` but use YOUR worktree path.

The codebase is ALREADY SCAFFOLDED for this. Mixed graph capture + replay are stubbed.
Do NOT use a ragged FlashInfer wrapper — this codebase uses
`flashinfer.BatchPrefillWithPagedKVCacheWrapper` via `FlashInferPrefillWrapper`. The
"mixed qo_indptr" idea applies verbatim, just paged not ragged.

## Existing scaffolding you build on
- `mstar/engine/mixed_walk.py` (210 lines, pure CPU): `MixedWalkLayout` dataclass (:34-64);
  `build_mixed_varlen_layout(...)` (:67-189) — builds qo_indptr `[0,1,..,D, D+P0, D+P0+P1,..]`
  (decodes first, 1 token each, then prefill chunks), kv_seq_lens, 3-row M-RoPE.
  `DEFAULT_MIXED_PREFILL_BUCKETS=(64,128,256,512)` (:197);
  `pad_prefill_tokens_to_bucket(n, buckets)->int|None` (:200-210) smallest bucket>=n else None.
- `CudaGraphKey` (cuda_graph_runner.py:96-114) ALREADY has `mixed:bool=False, num_decode:int=0,
  num_prefill_tokens:int=0` (default-inert, hashes identically today).
- `run_mixed(...)` STUB at cuda_graph_runner.py:1220-1261 — currently logs + returns None,
  NOT called by the engine yet.
- Engine dispatch: `KVCacheEngine.execute_forward` (kv_cache_engine.py:1011-1060); mixed branch
  at :1025-1037 calls `_execute_mixed_eager` directly (priority mixed>cuda_graph>batched>seq).
- `_execute_mixed_eager` (kv_cache_engine.py:477-660) — the working eager path to keep as fallback.

## The pattern to MIRROR (FLASH_INFER_PACKED prefill graph)
- Config classes: `mstar/engine/cuda_graph_config.py` — `CudaGraphConfigType{BASIC_BATCHED,
  FLASH_INFER_PACKED}` (:10-12); `FlashInferPackedCudaGraphConfig` (:74-103) holds
  `packed_seq_len_to_inputs` dict, `causal_attention`, `zero_padding_input`;
  `get_total_tokens=list(num_token_to_inputs.keys())`.
- Capture driver: `warmup_and_capture` (cuda_graph_runner.py:238-298) loops configs largest-first,
  builds `CudaGraphKey`, dispatches on `config.get_config_type()` to `_capture_one_basic_batched`
  / `_capture_one_flashinfer_packed` (:264-272).
- `_capture_one_flashinfer_packed` (:622-687) — closest analog. Pulls `template_dict =
  config.num_token_to_inputs[key.num_tokens]`, interns via `_intern_static_buffer` (:393-428),
  builds persistent wrappers `_create_persistent_wrappers` (:300-373; is_decode=(total_tokens==bs),
  so mixed total>bs -> prefill wrapper auto-selected), re_prepare hook calls
  `cache_manager.plan_attention(...)`+`plan_rope(...)` (:655-664). Shared per-slot core
  `_capture_slots` (:536-620): 2 warmup forwards, `torch.cuda.graph(g, pool=self.memory_pool)`.
- Replay template: `_run_flashinfer_packed` (:1479-1684): swap real states onto dummy slots ->
  pad to padded_bs with zero_padding_input -> `submodule.preprocess(...)` re-plans persistent
  wrappers -> `static_buf[:n].copy_(real)` -> `graph.replay()` -> `advance_seq_lens()` (Python,
  post-replay) -> `_sample_and_remap` dummy->real. Dispatcher `run(...)` (:1145-1218),
  `reserve_slot` (:914-955).
- Routing/padding: `_get_key_for` (:785-810) -> `_get_padded_batch_size` (:818-834 bisect over
  capture sizes) + `_get_padded_num_tokens` (:836-847).
- Capture sizes: decode `[1,2,4,8,16,32]` (submodules.py:1068); `PREFILL_TOKEN_BUCKETS=
  [128,256,512,1024,2048]` (:897), `PREFILL_CAPTURE_BATCH_SIZES=[1,2,4]` (:898).
- Config source: `Thinker.get_cuda_graph_configs` (submodules.py:999-1127); packed builder
  `_build_prefill_text_packed(num_tokens, device)` (:910-947).

## FlashInfer plan (flashinfer_utils.py)
- `FlashInferPrefillWrapper.__init__` (:90-160) pre-allocs static buffers in cuda-graph mode
  (`_qo_indptr_buf[bs+1]`, `_paged_kv_indptr_buf`, `_paged_kv_indices_buf`,
  `_paged_kv_last_page_len_buf`, token_to_page/cache len max_total_tokens).
- `plan(qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, causal, dtype)`
  (:162-256) runs OUTSIDE the captured region; in graph mode updates persistent buffers via
  `.copy_()`. CRITICAL: pass CPU int32 tensors (FlashInfer does indptr.to('cpu'); GPU tensor =>
  sync). Same persistent wrapper object across capture and replay.
- Eager plan-building to mirror: `BatchedCacheManager._plan_attention_impl`
  (cache_manager.py:229-424): accumulates qo_indptr from seq_lens, builds CPU int32 tensors
  (:314-317), is_decode=all(sl==1) picks wrapper. For mixed pass seq_lens=[1]*D + prefill_lens.

## Parity infra
- `test/integration/test_prefill_cuda_graph.py`: parity test (:316-437) — assert key in
  runner.graphs; run eager per-rid vs runner.run; compare top-5 next-token agreement
  (top_k_matches==bs) + hidden-state rel err <5e-3. Determinism test (:440-494) 3 replays
  torch.equal. NOTE post-refactor static_outputs lives on `CudaGraphSlot` (:63), read
  `runner.graphs[key].slots[0].static_outputs[...]` not on CudaGraphData.
- Layout CPU tests: `test/modular/test_qwen3_omni_mixed_walk.py` (test_build_mixed_varlen_layout_*
  :210-247). Reserved GPU parity name: `test_mixed_step_logits_match_separate_steps`.

## Warmup hook
- `KVCacheEngine.warmup` (kv_cache_engine.py:234-291) builds CudaGraphRunner +
  `runner.warmup_and_capture()`. No new top-level call needed — adding a MixedCudaGraphConfig to
  `get_cuda_graph_configs` makes the existing loop capture it.

## The 7-item implementation skeleton (full bucketed = the reference)
1. New `MixedCudaGraphConfig` + `CudaGraphConfigType.MIXED` in cuda_graph_config.py. `get_total_tokens(bs)`
   enumerates the (num_decode, num_prefill_bucket) grid; carry causal_attention + zero_padding_input.
2. `_build_mixed_packed(num_decode, num_prefill_tokens, device)` in submodules.py (mirror
   `_build_prefill_text_packed`); append `MixedCudaGraphConfig(...)` to get_cuda_graph_configs.
3. `_capture_one_mixed` in cuda_graph_runner.py (mirror `_capture_one_flashinfer_packed`), wire into
   warmup_and_capture dispatch (:264-272). Set mixed=True,num_decode,num_prefill_tokens on the key.
4. `_get_mixed_key_for(num_decode, num_prefill_tokens, ...)`: bucket num_decode->bs
   (_get_padded_batch_size) and num_prefill_tokens->pad_prefill_tokens_to_bucket; None => eager fallback.
5. Implement `run_mixed` body: build_mixed_varlen_layout -> lookup bucketed slot -> plan persistent
   wrapper on CPU qo_indptr/kv -> .copy_() flat tokens + M-RoPE into static buffers -> graph.replay()
   -> advance_seq_lens -> _sample_and_remap. Return None ONLY on a true miss (=> eager fallback).
6. Engine dispatch: in mixed branch (kv_cache_engine.py:1025-1037) try `runner.run_mixed(...)` first,
   fall back to `_execute_mixed_eager` on None. Split decode/prefill rids per decode-first order
   (worker.py:862-863).
7. Parity test `test_mixed_step_logits_match_separate_steps` in test_qwen3_omni_mixed_walk.py.

## REQUIRED toggle for the A/B (all four arms must honor this)
- Capture+use the mixed graph when `MSTAR_MIXED_WALK=1` AND `os.environ.get("MSTAR_MIXED_CG","1")!="0"`.
- When `MSTAR_MIXED_CG=0`, skip mixed capture and force run_mixed->None so the eager path runs.
  This lets the harness A/B graph-vs-eager on the SAME branch.

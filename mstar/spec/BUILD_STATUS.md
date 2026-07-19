# N-gram speculative decoding (#17) — build status

## Validated WITHOUT any full model boot (per the "POC/tests before booting" directive)
- **Stage 0 drafter+accept**: 11 unit tests pass (`mstar/spec/test_ngram_drafter.py`) — recurrence, most-recent-wins, longer-n-preference, k-truncate, no-match, accept prefix. Pure logic.
- **Verify-forward attention numerics**: 18/18 varlen-parity tests pass (`test/modular/test_qwen3_omni_varlen_backend_parity.py`, ninja on PATH) — the multi-query (q_len=K+1) varlen path the verify forward reuses computes correct attention. Plus M*'s paged-prefill wrapper already runs every i2t prefill through this exact code, so the paged verify forward is numerically sound.

## Design (from deep-read): the multi-query verify forward ALREADY EXISTS
`cache_manager.py:363` selects the prefill wrapper for any seq_len>1; `causal=True` = the lossless verify mask (no xqa, no custom mask). Verify = a uniform `thinker_mixed`. New code needed: (1) per-rid token-history accumulator (worker keeps none), (2) `thinker_verify` branch = `lm_head` over ALL K+1 rows (submodules.py ~2104), (3) accept (done), (4) seq_len rollback (set seq_len/position_id_start to before+(A+1), leave pages), (5) bypass q_len=1 spec chain (use packed path). Full edit points in the design doc.

## Remaining = INTEGRATION (needs 1 boot to certify real-Thinker token-parity)
Stage 1 (B1 eager verify) -> Stage 2 (batched + KV rollback + ordered-emit + MSTAR_NGRAM_ASSERT shadow) -> Stage 3 (capture, only if eager host cost dominates). Cheapest validation: extend the tiny-shape paged-KV POC for accept/rollback bookkeeping; full boot only for the end-to-end token-parity assertion (B1 greedy). Metric: accept-rate + tok/s + GPU-idle (target: fill the 44%-mean decode idle, modality-agnostic).

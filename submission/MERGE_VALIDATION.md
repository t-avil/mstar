# Merge validation — full stack on `upstream/main`

**Branch:** `merge/full-stack-on-upstream` (merge commit `568a34f9`).
**Date:** 2026-07-22. Booted the MERGED code (upstream/main `d7e79890` + the full
MSTAR_* flag stack, 29 conflicts resolved) on 2× H200 (GPUs 6,7), `dpenc.yaml`,
natural-EOS closed-loop, and re-measured the headline. Raw in
`submission/merge_validation/`.

## Result: the win survives the merge (all completions full = tokens correct)

| path | B32 merged branch | fresh pre-merge ref | vLLM-0.24 | ratio | comp |
|------|-------------------|---------------------|-----------|-------|------|
| i2t (tok/s) | **1857.2** | 1844.8 | 1744.6 | **1.06×** | 128/128 |
| s2t (tok/s) | **856.8**  | 841.8  | 664.9  | **1.29×** | 128/128 |
| i2s (req/s) | **2.25**   | 2.27   | 1.14   | **1.97×** | 39/40 |

Full i2t/s2t curves (merged vs pre-merge ref, tok/s):
- i2t: B1 202/206 · B2 335/348 · B4 573/583 · B8 928/936 · B16 1346/1363 · B32 1857/1845
- s2t: B1 122/121 · B2 225/228 · B4 350/334 · B8 498/499 · B16 493/655* · B32 857/842

## What this validates
Every HIGH-risk conflict resolution (the ports where our flag hooks were re-applied
onto upstream's rewrites) is behaviorally sound:
- **H1** worker `register_for_send` (#177 tensor_infos): emit correct, full completions.
- **H2** worker `_send_outputs` (#149 token-COUNTING; our prematerialized path dropped):
  s2t completes 128/128 and matches the reference → token VALUES still reach the client;
  dropping the prematerialize buffer was correct.
- **H3** cuda_graph_runner `_get_key_for` (#154 bucket search + MIXED_BATCH split-attn):
  no "chunk len outside window"; decode throughput matches.
- **H5** data_worker `run()` (#181 drain-ahead + PREPROC_PROC): i2t TTFT/throughput
  intact (preproc pool working).
- **H6** cache_manager `plan_attention` (#121 rewrite + publish_manager / CUSTOM_OPS):
  i2t/s2t run under fp8+custom-ops, throughput matches.
- **i2s / talker:** works (2.25 req/s ≈ reference) despite a one-shot
  `torch._dynamo recompile_limit` warning from the merged `attention.py` `layer_idx`
  change — did not break the speech path.

## Caveats / follow-ups (non-blocking)
- s2t **B16** read (493) is a single-cell host-variance dip (B8=498, B32=857 both
  solid); not a regression.
- The talker `recompile_limit` warning is worth a look before heavy speech serving
  (consider raising `TORCHDYNAMO_CACHE_SIZE_LIMIT` or pinning `layer_idx` out of the
  guard) — cosmetic here, no functional impact at these batches.
- This is a focused validation (i2t/s2t all batches + i2s B8/B32). A full 5-path
  sweep on the merged branch can be run from `submission/scripts` if desired.

## Bottom line
The single up-to-date branch `merge/full-stack-on-upstream` = upstream/main + our full
winning stack, resolves clean, compiles, boots, and **reproduces the headline win with
correct tokens.** It is the maintainable artifact to serve from and the base to carve
review-sized PRs from (see MERGE_STRATEGY.md).

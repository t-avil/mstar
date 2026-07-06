# PR_DECOMPOSITION — how to upstream the M* delta as small, decoupled PRs

Companion to FEATURES_SINCE_ENCODERS.md (feature detail + SHAs) and
TORCH_COMPILE_FINAL.md. Design principle: every unit below is flag-gated and
byte-identical when OFF (unless marked), so each PR is independently
revertable and A/B-able. Several were literally authored standalone off
`12ce776` before landing on the stack — the decoupling is proven, not
hypothetical.

Ordering = recommended merge order: (ease of review × measured value ×
dependency depth). Sizes are the feature's own diff slice.

## Tier 1 — small, measured wins, zero dependencies (merge first)

1. **Compile-aware RMSNorm** — `opt/compile-fix` `1733fab`, **+14 lines**,
   no flag. Pure-torch RMSNorm under `torch.compiler.is_compiling()`; kills
   ~311 graph breaks/boot. Eager path untouched; only risk is ULP drift vs
   FlashInfer. Trivial review.
2. **Sampler config cache V2** — `opt/cfgcache-v2` `ded928d`, **+227/−25,
   one file** (`utils/sampling.py`), flag `MSTAR_SAMPLER_CFG_CACHE_V2`.
   Kills the 22%-of-wall pageable-H2D thrash. Measured ~+2–5% B32.
   Byte-identical output (shadow-verified). Recommend default-ON after soak.
3. **Device-side decode pos_ids** — `opt/prep-h2d` tip `35a9358`+`620de91`,
   **+65/−9**, flag `MSTAR_PREP_DEVICE_POS`. Kills the 24%-of-B1-wall
   pos_ids H2D. Measured +3.1% B2. Input-build only, capture-safe.
   (Technically touches the cfgv2 gather hoist — rebase trivially if #2
   hasn't merged yet.)
4. **Sidecar Stage-2 check_stop** — standalone `opt/sidecar-checkstop`
   `5414a5a`, **+452/−77** (worker.py + test), flag
   `MSTAR_SIDECAR_CHECKSTOP` + `_SHADOW` verify mode. Measured +2.3% B32.
   Ships its own shadow-equality test.
5. **Decode buckets 24/28** — `exp/bucket24` `845faff`, **+5/−1**,
   unconditional. One-line capture-size list edit + review note on capture
   memory (+2 graphs/slot).
6. **Prefill buckets 384/768/1536** — `exp/prefill-buckets` `08e7123`,
   **+4/−1**, unconditional. Same shape; +3 prefill captures.
7. **GC/GIL tuning** — `opt/integration-v4` `52d9116`, **+27**, flags
   `MSTAR_GC_TUNE`, `MSTAR_PY_SWITCH_INTERVAL_SEC`. Self-contained
   Worker.__init__/run() edits.

## Tier 2 — medium, measured or mechanism-proven, shallow deps

8. **torch.library custom ops** — `opt/custom-ops` `12ce776`, **+310/−10**,
   flag `MSTAR_CUSTOM_OPS`. Graph breaks 816→~165; prerequisite for any
   further compile work (see TORCH_COMPILE_FINAL.md). Reviewer focus: the
   active-manager global and `register_fake` correctness. Include the
   launch-env note (FX_GRAPH_CACHE + CACHE_DIR + DYNAMO_CACHE_SIZE_LIMIT=128,
   else boot-time regression).
9. **Block-fp8 MoE + fused topk** — from `opt/decode-v2` (`ca97ba2`,
   `22b3cd6`, `2239005`, `856d103`), **~+570** (new fp8.py + moe.py), flags
   `MSTAR_MOE_FP8`(`_TALKER`), `MSTAR_FUSED_TOPK`. Measured 1.13× e2e.
   Reviewable independently of #8 (the custom-op fold is a 30-line follow-up).
10. **Merged multimodal prefill (vision)** — `opt/prefill-merge` `41200ec`,
    **+789/−105**, flag `MSTAR_MERGED_PREFILL`. Byte-identical OFF; rides the
    existing prefill_vision capture. Document the strategy-swap gate
    (CHUNKED_PREFILL_V2_VISION off).
11. **Merged prefill (audio twin)** — `opt/prefill-merge-audio` `a2d2f47`,
    **+683/−123**, flag `MSTAR_MERGED_PREFILL_AUDIO`. Depends conceptually on
    #10 (shares `_maybe_merge_prefill_schedule`); rebase onto it.
12. **Speech floor pair** — `opt/speech-floor` `01856e8`, **+843/−8**, flags
    `MSTAR_FAST_CHECKSTOP_TALKER`, `MSTAR_CODEC_CHUNK_EMIT`. Two separable
    commits if reviewers prefer; both walk-gated.
13. **Emit/postprocess fast paths** — from `opt/decode-v2`
    (`a06c08f`,`c2b9b5b`,`4ce0125`,`de4a741`,`9193cd7`), flags
    `MSTAR_INLINE_EMIT`/`BATCH_EMIT`/`FAST_POSTPROC`/`FAST_CHECKSTOP`.
    Split into 2 PRs: emit paths; memoized postprocess (+289/−9).
14. **Scheduler pack + fairness backoff** — `opt/sched-pack` `74a985c`,
    **+96/−11**, flags `MSTAR_SCHED_PACK`(`_PEEK_CAP`).

## Tier 3 — large but coherent subsystems

15. **Side-stream prefill** — from `opt/decode-v2` (`b9de820`,`31ea1bc`,
    `c6f97ae`), flag `MSTAR_SIDE_PREFILL`. Concurrency review (2nd stream +
    engine locks) — needs a careful reviewer, ships with drain/reap paths.
16. **Chunked prefill V2** — `exp/chunked-prefill-v2` `23d07ee`, +1234/−104,
    flags `MSTAR_CHUNKED_PREFILL_V2` family. Value is workload-dependent
    (long prompts only — falsified for short-span i2t/s2t sets).
17. **Captured mixed batch** — `exp/mixed-batch-p2` `4d9d6e3`, +1823/−30,
    flags `MSTAR_MIXED_BATCH` family. Depends hard on #16. Ships WALK_STATS
    (worth extracting as its own small observability PR if #16/#17 stall).
18. **V2 budget policy** — `opt/v2-policy` `56e65f8`, +446/−288, flags
    `MSTAR_MIXED_BUDGET_TOKENS` family. Depends on #17.
19. **Direct feed / multistep decode** — `exp/direct-feed` `6644913` (+107)
    then `exp/two-step-decode` `6bf6136` (+638/−139), flags
    `MSTAR_DIRECT_FEED`, `MSTAR_MULTISTEP_DECODE`. Direct-feed is shippable
    alone (and async-sched's prerequisite); multistep is capture-surgery —
    review with the runner owner.

## Do NOT upstream (measured dead/parked — keep branches as evidence)

- `opt/sidecar-batch` `8b775a3` — washed-negative live (GIL-shade absorbed
  into the await-GPU wait).
- `opt/async-sched` `18b1244` — parked: non-identical (+9% length), wash.
- `opt/w2-retest` `a2788a9` / `exp/step-txn` `b98de66` — closed: 4/4 boot
  failures, EV 2–5% < cost.
- `opt/admit-jitter` `0c68268`, `opt/prefill-gather` `092d18f` — falsified
  for the target workload (guard self-suppresses / readiness serializes).
  Mechanically sound; revisit only for open-loop arrival workloads.
- `opt/mixed-walk` `4334810` / `exp/combined-coalesce-piggyback` `0671191` —
  superseded eager line; reference implementation only.

## Env-only "PR" (docs/infra, no code)

20. **Launch recipe** — `launch_mstar_best.sh` env block:
    `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` (+3.4% B32),
    `TORCHINDUCTOR_FX_GRAPH_CACHE=1`, `TORCHINDUCTOR_CACHE_DIR`,
    `TORCHDYNAMO_CACHE_SIZE_LIMIT=128`, plus the winning MSTAR_* set
    (HANDOFF_V8 §1). Consider pinning critical inductor flags in
    `engine/__init__.py` per TORCH_COMPILE_FINAL.md §5.7.

## Suggested first batch (one afternoon of review)

PRs 1+2+3+4 ≈ 760 added lines across three files + tests, all flag-gated,
all with measured wins summing to the bulk of the B1/B2/B32 host-side gains.
Then 8 (custom-ops) to unlock the compile roadmap, then 9 (fp8 MoE) for the
largest single e2e multiplier.

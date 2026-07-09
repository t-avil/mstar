# Review: Qwen3-Omni native encoders (#131) + benchmark showcase

Reviewed by orchestrator + 5 sub-agents (code, tests, bench-audit, 2× research) and an
independent GPU replication on GPUs 6,7. Date: 2026-06-30.

Branches:
- Code PR: `t-avil:mstar:pr/native-encoders` (1 commit, base `upstream/main` = ae7d173)
- Showcase: `t-avil:mstar:pr/native-encoders-benchmarks-v3`

TL;DR
- **Code PR: mergeable with 2 tiny fixes + 1 doc fix.** Contracts, weight loading, parity all
  genuinely hold. The real weakness is *test coverage of the production CUDA-graph path*, not the code.
- **⚠ But the PR ships the benchmarked optimizations OFF by default** (CUDA_GRAPH=0, varlen=adaptive,
  no compile dynamic, codec=25) — the benchmarked `1f66ce6` had them ON. Merge ≠ benchmarked perf unless
  you flip those defaults or document the required env vars. See "CRITICAL" in Part A.
- **Showcase: the numbers are NOT fabricated** (byte-identical regen, reported figure is the *lower*
  of two runs, real per-request variance). But the *provenance and reproduce path are broken enough
  that a skeptic could reject the evidence* until fixed. These are all fixable with no re-runs.
- **The "as good as it gets" story holds up** and is well-evidenced. The i2t TTFT loss vs vLLM is a
  real, explained, structural trade-off — not a bug.

================================================================================
# PART A — CODE PR (the thing you're shipping). Priority #1.
================================================================================

## ⚠ CRITICAL — the PR ships the benchmarked optimizations OFF by default
The encoder *code* in `pr/native-encoders` is ~11 lines different from the benchmarked `1f66ce6`
(`opt/combined-lowrisk`) — but those lines are **default flag values**, and they decide behavior:

| Setting | PR (ships) | Benchmarked 1f66ce6 |
|---|---|---|
| `MSTAR_ENCODER_CUDA_GRAPH` (audio_encoder.py:444, vision_encoder.py:206) | **`0` OFF** | `1` ON |
| `MSTAR_VARLEN_BACKEND` (audio_encoder.py:228) | **`adaptive` (SDPA)** | `flashinfer` |
| `MSTAR_ENCODER_CG_WARMUP` (both encoders) | **`""` none** | `1,2,4,8` |
| `torch.compile` (stateless_engine.py:522) | `dynamic` unset | `dynamic=True` |
| `codec_chunk_frames` (config.py:291) | **`25`** | `15` |

So a person who merges this PR and runs it gets native encoders with **CUDA-graph disabled and SDPA
(not FlashInfer) varlen** — it does NOT reproduce the benchmark out of the box. `1f66ce6` is a parallel
branch (merge-base = README commit `2e6465a`); its ~30 extra commits consolidate to these few default
flips in the shipped files. **Decide and state explicitly which you intend:**
- (A) Flip the PR defaults to match the benchmark (CUDA_GRAPH=1, VARLEN_BACKEND=flashinfer,
  CG_WARMUP=1,2,4,8, dynamic=True, codec=15) so "merge = benchmarked perf", OR
- (B) Keep conservative defaults and document that the benchmarked numbers require those env vars.
Either is fine — but right now BENCHMARK.md implies the PR == the numbers, and by default it doesn't.
(Note: the code-review sub-agent misread the gate default as `"1"`; the shipped file is `"0"` — verified
by direct diff `pr/native-encoders`↔`1f66ce6`.)

## VERDICT: Mergeable with fixes. Two 1-line code fixes + one docstring fix + the defaults decision above.

## MUST-FIX before sharing
1. **deepstack empty-list regression** — `submodules.py:360`
   `"deepstack": deepstack if deepstack else [torch.tensor([])]`
   The old wrapper (`:282`) used `deepstack if deepstack is not None else ...`. Truthiness makes an
   empty list `[]` (a model with `deepstack_visual_indexes = []`) fall to the sentinel `[tensor([])]`
   (len 1), which then trips `assert len(deepstack_list) == num_deepstack` in the Thinker (`:731`).
   Dormant for Qwen3-Omni-30B (`[8,16,24]`, num=3) but a silent regression for any no-deepstack port.
   Fix: restore `is not None`.
2. **Dead field** — `vision_encoder.py:201` `self._cg_pool = None` is assigned, never read. Delete.
3. **False docstring** — `audio_encoder.py:4` says the encoder "is decoupled from `transformers` at
   inference time", but `:28` still does `from transformers.activations import ACT2FN`. Vision imports
   3 helpers from `transformers.models.qwen3_omni_moe` too (`:29`, disclosed at `:14`). Either inline
   the helpers or fix the claim. A reviewer who greps will call this out.

## NITPICKS (optional, but cheap)
- `zip(..., strict=False)` in both `forward_batched` (`submodules.py:166,348`): on a token-count bug,
  tail requests are silently dropped (missing `rid` keys). Use `strict=True` in a serving path.
- Comments cite bench-branch files (`README_qwen3_omni_encoders.md`, `exp_audioenc/raw.json`,
  `bench_audio_backend_matrix.py`) — `audio_encoder.py:52,114,124`. Invisible from the PR diff.
- `vision_encoder.py:219-220`: O(D) `.index()` scan per block on `deepstack_visual_indexes`; a
  `{layer: merger}` dict in `__init__` reads cleaner. (Perf irrelevant at 3 entries.)
- `can_batch` docstring (audio) says a multi-clip request "falls back to sequential forward" — actually
  the whole batch defers, not just that request.

## WHAT'S GENUINELY GOOD (verified, not assumed)
- **Output contract preserved.** Keys exactly `{"audio_embeds":[t]}` and
  `{"vision_embeds":[t],"deepstack":[t1,t2,t3]}`, list-wrapped; deepstack tensors shaped
  `(merged_tokens, out_hidden)` exactly as the Thinker's `full_deepstack[mm_mask,:]=...` needs.
- **Weight loading is clean.** Native attribute names mirror HF exactly; CI test asserts
  `load_state_dict(strict=False)` → zero missing / zero unexpected for both encoders. Same prefixes
  (`thinker.audio_tower`, `thinker.visual`) — no new remap needed.
- **cu_seqlens** int32/on-device for both; fed to `flash_attn_varlen_func` correctly.
- **Batched token-count arithmetic** proven correct algebraically (audio `_feat_extract_output_lengths`
  is additive across the 100-frame boundary; vision `sum(T*H*W)//merge_sq`).
- **Conv3d→matmul** substitution is exact in fp32 (kernel==stride==patch, same C-contiguous layout).
- **CUDA-graph capture safety**: per-graph FlashInfer state (no replan on replay), `_fi_override`
  set/cleared in `finally`, `.clone()` on replay output prevents static-buffer aliasing.
- **HF fallback retained** behind `MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER=0` — satisfies acceptance
  criterion #3 (HF path relegated to fallback, not in the hot path by default).

## TEST COVERAGE — the actual soft spot (acceptance criterion #1 only *partially* covered)
Eager-path parity is real: real HF oracle, real weights, real image/audio, cos>0.9999 for audio (single
clip) and vision (4 resolutions, all 3 deepstack levels); fp32 bit-exactness grounds the bf16 bar;
per-layer residual hooks in CI. **But:**
- **No parity test for the production path.** `MSTAR_ENCODER_CUDA_GRAPH=1` + FlashInfer varlen is never
  exercised by either test file. The FlashInfer head-dim padding, per-key graph capture, and
  `copy_()+replay()` mechanics are unvalidated for numerics. This is the most likely silent-regression
  vector and it's exactly the path you ship.
- **No multi-image vision parity vs HF** — single image only; a wrong multi-image `cu_seqlens` boundary
  (cross-image attention bleed) is structurally invisible to current tests.
- **Audio batch test has no HF ground truth** (native-vs-native only; a systematic bug appears in both).
- **No max-abs-error check** — cos + global relL2 are bulk-dominated; a few catastrophic tokens pass.
- **Skip-on-exception too broad**: `_resolve_checkpoint` `except Exception: return None`, and the
  double `pytestmark` (`:37` overwritten at `:61`) → a broken checkout can go green-with-0-tests.

Recommendation: add one parity test that runs the encoder with `MSTAR_ENCODER_CUDA_GRAPH=1` and asserts
eager==graph (and both==HF) plus a `.abs().max()` bound; add one multi-image case. Cheap, closes the gap.

================================================================================
# PART B — BENCHMARK SHOWCASE (pr/native-encoders-benchmarks-v3)
================================================================================

## HONESTY VERDICT: the data is NOT fabricated. Lead with this.
Independent evidence it's genuine:
- `make_numbers.py` regenerates `NUMBERS.md` **byte-identical**; all 4 charts regenerate at **identical
  byte sizes** from the committed `raw_*.json` via the committed scripts → artifacts really come from
  this data + these scripts, not hand-editing.
- The reported `mstar_new` figure is the **lower** of two runs (agg 68.94 tok/s vs the ingest
  datapoint's 71.65) — the opposite of cherry-picking.
- `mstar_old`/`mstar_new_chunked` carry 710 real per-request points each with realistic variance.
- `verify_not_worse.py` genuinely recomputes from the data; the ITL trade-off is openly disclosed.
- My smoke replication ran `sweep.sh` end-to-end on GPUs 6,7 for all 4 paths (B=1): server up → 4 runs
  (`comp/fail 1/0`) → artifacts persisted → clean teardown, GPUs freed. The harness works.

So this is **sloppy provenance, not lying.** But the following would let a hostile reviewer dismiss it:

## SUPER RED FLAGS (fix before showing anyone who'll push back)
1. **Documented commits are wrong.**
   - `BENCHMARK.md` says `mstar_old = ae7d173`. Every `mstar_old` aggregate's provenance says
     **`git_commit: 9ee1369`** (5 commits ahead of ae7d173 — docs + sweep.sh only, no model code, so the
     comparison is still fair, but the hash is wrong and reads as a provenance lie).
   - `mstar_new` is documented as `1f66ce6` (`opt/combined-lowrisk`), but the `mstar-encoders` worktree
     currently sits at `dbec528`, and `command.txt` never tells the replicator to `git checkout 1f66ce6`.
2. **`command.txt` does not reproduce the committed artifacts.**
   - Committed raw files were built by **`aggregate.py --refine-dir`** (per `provenance.generated_by`),
     reading experiment source roots not referenced anywhere in `command.txt`. The documented
     `ingest_sweep.py` produces a *different* shape: only **6 summary datapoints/system** (I confirmed by
     running it on my smoke output), no per-request rows, and `make_numbers.py` emits a different
     `NUMBERS.md` format than the committed (aggregate.py) one.
   - `ingest_sweep.py` default `--raw-dir` is `benchmarks/qwen3-omni-joint` (wrong dir) — verbatim run
     writes to a nonexistent path.
   - **No vLLM reproduce command at all**, yet vLLM is in every table and chart.
   - The I2S B1/B2 "recheck-folded" cells come from an undocumented recheck dir.
3. **Headline `mstar_new` (and `vllm`) have no per-request datapoints** — only synthetic per-batch points
   with `request_id=None` (the source of the "duplicate key" warnings). Their TTFT/ITL distributions
   cannot be recomputed from `raw.json`; only `mstar_old`/`mstar_new_chunked` can. Datapoints and
   aggregates are also from **different sessions** (env.txt 2026-06-29 vs aggregates generated
   2026-06-28).
4. **Undisclosed variants shipped in the data.** Aggregates contain `mstar_new_chunked` (commit
   f58a805, `MSTAR_CHUNKED_PREFILL=1`), `mstar_new_v1`, `mstar_new_v2` — none mentioned in
   BENCHMARK.md/NUMBERS.md/command.txt. v1 and v2 show identical numbers (label/ingest artifact). Clutter
   that invites "what are these and why aren't they the headline?"
5. **Shared venv confound.** `sweep.sh:186,242` hardcode the `mstar-encoders/.venv` python/PATH for
   **all** systems including `mstar_old`. PYTHONPATH points at the old worktree (pure-Python is old), but
   compiled extensions (.so / CUDA kernels / FlashInfer) load from the *new* venv. If any "new" speedup
   lives in a compiled ext, `mstar_old` silently benefits. Document the venv, ideally use per-system venvs.
6. **A verify script is a rubber-stamp.** `verify_s2s_itl.py` always prints its PASS verdict; no
   `sys.exit(1)` path — passes on empty/missing/garbage data. (The other 3 verify scripts are real.)
7. **Charts silently drop anomalous points.** `make_proof_charts.py` `ITL_OUTLIER_FRAC=0.40` omits any
   ITL < 40% of its predecessor with "no marker/annotation". This hides the degenerate `mstar_old` S2T
   ITL (0.18–0.67 ms — `mstar_old` likely isn't truly streaming), which is *still* shown in NUMBERS.md as
   `new/old` ratios of `0.00x`/`0.06x` with no caveat. Either footnote the mstar_old non-streaming ITL or
   exclude that metric for that system — don't silently drop from the chart while leaving it in the table.

## NITPICKS
- `env.txt` is hand-merged (prose under `=== date (UTC) ===`) and shows `no nvcc` / `no torch` — the
  capture used system python, so toolkit + framework CUDA versions (2 of CLAUDE.md's 3) were never
  recorded. `requirements.txt` likely system-only.
- Summary datapoints for S2S/I2S (and vllm) are all mislabeled `"batch": 1` for every batch size.
- `aggregate.py` hardcodes `DEFAULT_STYLE = /home/tim/exp_3way/chartstyle.mplstyle` — others get
  matplotlib defaults, not the shared style (the committed `benchmarks/chartstyle.mplstyle` is correct;
  point aggregate.py at it).
- No explicit completion marker / sentinel in `raw_*.json` (provenance is a partial substitute).
- Persistence mode reads `Enabled` on the GPUs and there's no documented post-run clock/persistence
  check. (My run didn't lock clocks; this is pre-existing node state — but CLAUDE.md wants it recorded.)
- `command.txt` should add `--raw-dir .` (or a cwd note), the `git checkout 1f66ce6` step, the vLLM
  launch command, and the recheck step — or just say "artifacts were generated by `aggregate.py
  --refine-dir`; `command.txt` shows the sweeps, not the post-processing."

## What I verified actually works (positives to keep)
- Full regen chain reproducible & deterministic; harness runs end-to-end with clean GPU teardown;
  `verify_not_worse` / `verify_itl_consistency` / `verify_s2s_itl`(content) / `verify_data_integrity`
  all execute and (3 of 4) genuinely compute; `provenance` + `token_count_source` disclosure
  (own-tokenizer tok/s caveat) is good practice.

================================================================================
# PART C — "AS GOOD AS IT GETS" DEFENSE + QUIZ PREP
================================================================================

## The one-sentence thesis
M*'s throughput/ITL wins come from keeping every Thinker/Talker step on a pre-captured, fixed-shape CUDA
graph; that *requires* separating prefill from decode, which *is* why first-token latency on i2t trails
vLLM. It's one trade-off dial, not an oversight.

## The i2t TTFT gap (verified from raw_image_to_text.json)
| B | M*-new TTFT | vLLM TTFT | gap | M*-new req/s | vLLM req/s | M*-new ITL | vLLM ITL |
|---|---|---|---|---|---|---|---|
| 1 | 0.215s | 0.150s | vLLM 1.43× | 0.755 | 0.373 (M* 2.0×) | 0.0068 | 0.0123 (M* 1.8×) |
| 8 | 0.344s | 0.187s | vLLM 1.84× | 2.52 | 1.02 (M* 2.5×) | 0.0151 | 0.0360 (M* 2.4×) |
| 32| 0.576s | 0.206s | vLLM 2.80× | 4.46 | 2.55 (M* 1.75×)| 0.0341 | 0.0574 (M* 1.68×)|
- Gap exists at every batch and **widens with load** (signature of prefill-queue serialization).
- M*-new already improved i2t TTFT ~1.87× vs M*-old (B=1) — progress, but the residual gap is structural.
- Contrast: on S2T, M*-new *beats* vLLM TTFT at B=1 (0.097 vs 0.140) because GPU mel + in-process IPC win
  where there's no queue; i2t has no equivalent cheap win (vision encoder is intrinsically big/variable).

## Root cause
M*'s scheduler enforces one walk-type per micro-batch (prefill_vision / prefill_text / decode_text), so
each step stays on its captured graph. A new i2t request must traverse prefill_vision → prefill_text →
first decode before token 1. vLLM's v1 engine mixes prefill+decode in one step (continuous batching) via
"record-around-attention" (a single shape-polymorphic compiled artifact, only attention outside the
recording), so a new request piggybacks onto the next decode step → flat TTFT under load.

## What was tried to close it → why it didn't land (the defensible record)
| Approach | Result | Why not shipped |
|---|---|---|
| `MSTAR_MIXED_WALK=1` (vLLM-style mixing, eager) | B=32 i2t: **0.19 vs 3.74 req/s — 20× collapse**, 3 errors | mixed-shape step matches no captured graph → all-eager |
| mixed-CG: bucketed / coarse | captured 48/6 graphs but `NodeOutput` wrapping bug → AttributeError crash every mixed step; >15 min warmup | correctness + prohibitive warmup; **B=32 result PENDING** |
| mixed-CG: supergraph / decode-only | captured under wrong key → silent eager fallback (were just the eager baseline) | key bug; no real mixed graph ever ran |
| `MSTAR_ENCODER_ASYNC` (overlap encode w/ decode) | i2t B=32: −30% TTFT, +7.4% req/s (promising) | "flaky in integration testing" (commit 95290f6 flipped default off) → excluded from *lowrisk* |
| `MSTAR_CHUNKED_PREFILL` | design + parity argument done | scheduler re-enqueue is a `NotImplementedError` stub; never GPU-validated |
| wider prefill buckets / encoder-gap | designed | need GPU validation (VRAM headroom / sync-elim) — deferred |

## What "combined-lowrisk" (the shipped M*-new) actually contains
Native encoders; encoder transformer-block CUDA graph (per varlen layout, lazy, ≤16 keys) enabled by the
3-default flip (varlen→flashinfer, ENCODER_CUDA_GRAPH=1, CG_WARMUP=1,2,4,8); `torch.compile(dynamic=True)`
on encoder forward (tail-latency only — microbench says compile does **not** beat eager in steady state;
the big matmuls dominate); vision GPU→CPU sync elimination; GPU mel + GPU image preprocess. Excludes
mixed-walk (harmful), async-encoder (flaky), chunked-prefill (incomplete), wider buckets/encoder-gap
(unvalidated). FlashInfer varlen is *required* for encoder CG because the default SDPA varlen backend
loops in Python over cu_seqlens (uncapturable).

## Crisp quiz answers
- **Why is i2t TTFT worse than vLLM, even at B=1?** B=1: ~65 ms of serialized prefill_vision→prefill_text
  →decode before token 1; vLLM admits into the next decode step immediately. No queue at B=1, so it's pure
  scheduling structure.
- **Why does the gap widen with load?** vLLM continuous-batches new prefills onto decode steps (flat
  TTFT); M* queues prefill slots → grows. Widening = prefill-queue signature.
- **Why not just mix like vLLM?** Done as an experiment: 20× throughput collapse at B=32, because a mixed
  (decode_count, prefill_len) step matches neither captured bucket → eager every step. Mixing *and*
  staying fast needs record-around-attention (a substrate rewrite), not a flag.
- **Did mixed-CG fix it?** No clean result: 2 of 4 variants silently ran eager (wrong graph key), the
  other 2 crashed on a NodeOutput type bug; full grid warmup >15 min; **B=32 post-fix is still PENDING.**
- **What did torch.compile buy?** Not throughput (microbench: doesn't beat eager) — only elimination of
  multi-second recompile stalls on novel shapes, via `dynamic=True`.
- **Why FlashInfer varlen for the encoder graph?** SDPA "adaptive" varlen is a Python for-loop over
  cu_seqlens (uncapturable); FlashInfer ragged-prefill takes cu_seqlens as a tensor, one kernel, no
  branching → capture-safe.
- **Is the i2t TTFT regression acceptable?** Yes: structural + explained + experimentally bounded; M* wins
  the cost/UX-steady-state metrics (≈2× throughput, 1.7–2.4× better ITL that stays flat while vLLM's
  degrades ~5× under load); i2t TTFT already improved 1.87× vs M*-old; and on speech-out paths (S2S/I2S)
  M* beats vLLM on TTFT(audio) 1.9–2.6×.

## Don't over-claim (gaps the research could NOT verify)
- The "torch.compile doesn't win" microbench file couldn't be read directly — cite as plausible, not
  measured.
- mixed-cg experiment code was never committed (all 4 branches = same commit 6bdcb90); bug/warmup details
  come only from `KNOWLEDGE_*.md`.
- The B=32 bucketed-mixed-CG **post-fix result is PENDING** — the whole "would graphed mixing recover
  throughput?" question is open. Say "we don't yet know," not "it doesn't work."
- encoder-gap ~10× gap-reduction and ~15 min warmup are estimates, not measured.
- vLLM "record-around-attention" is from vLLM's docs, not traced into vLLM source.
- The ~3.3 s HF vision-encoder/call figure is an in-code comment, not a standalone benchmark (the *delta*
  M*-old 0.401 → M*-new 0.215 TTFT is real).

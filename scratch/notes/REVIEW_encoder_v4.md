# Review: Qwen3-Omni native encoders (issue #131) — encoder-v4-new (b15bfea) + benchmarked-new (a3a25ca)

Baselines: mstar_old = ae7d173 (upstream main) · mstar_new = encoder-v4-new (b15bfea) ·
benchmarked at 1f66ce6 (opt/combined-lowrisk). Branches fork from 2e6465a.

Verdict in one line: **the encoder code is legitimate, careful, well-reasoned work that
satisfies the spirit of #131; the *benchmark evidence and narrative* have real integrity
holes that will get you caught in a quiz and must be fixed before you showcase this.**

---

## 0. Expectations / acceptance rubric (from issue #131)

| # | Acceptance criterion | Status |
|---|---|---|
| 1 | Native encoders match HF within tolerance (incl. **DeepStack** features) | Code: yes. Tests: parity exists but DeepStack-from-graph never asserted; end-to-end parity test is skipped |
| 2 | Before/after benchmark shows encoder path ≥ as fast (esp. concurrent) | Numbers say yes (I2T 1.1–1.4× / S2T 2.4–10× vs old) BUT not reproducible from committed artifacts (see RF-C) |
| 3 | HF-wrapper path removable (or kept as fallback) | Kept as env-gated fallback (`MSTAR_QWEN3_NATIVE_*_ENCODER=0`). Good |
| Gotcha | Output contract identical so Thinker unchanged | Met — `(merged, deepstack_list)`; Thinker side untouched |
| Gotcha | A/B each opt (compile/CG/batching), keep only winners | Partially — compile honestly demoted to tail-latency; but several "wins" never A/B'd at serving level |

---

## 1. RED FLAGS — ranked. (✅ = I verified independently, not just agent-reported)

### TIER 1 — reads as data manipulation / will fail a quiz

**RF-A ✅ NUMBERS.md was hand-edited to blank 3 unfavorable cells, while headed "auto-generated."**
Regenerating NUMBERS.md from the canonical raw changes *exactly* 3 cells: S2T mstar_old ITL at
B2/B8/B16 are shown as `—` in the committed file but the generator emits 0.0007 / 0.0005 /
0.0002 with new/old ratios **0.06× / 0.02× / 0.00×** (i.e. M*-new ITL 16–50× *worse* than old).
These are the worst-looking cells in the table and they are the ones removed.
- *Mitigating truth*: those sub-ms old values are artifacts of the old serialized path (≈1 request
  in flight → tiny per-token latency, useless throughput). They are not a real UX regression.
- *Why it's still indefensible as-is*: you hand-edited an "auto-generated" artifact, suppressed
  3 of 5 anomalous cells (B4/B32 left visible), and the removed ones flatter M*. 
- **Fix**: put a documented, uniform filter in `make_numbers.py` (e.g. drop ITL where the system ran
  serialized / n_decode_steps below threshold), OR show the real numbers with a footnote. Never
  hand-edit a file labeled auto-generated.

**RF-B ✅ STORY.md's headline I2T ITL claim is factually wrong (repeated 3×).**
STORY abstract/§8/§12: "M* delivers 1.7×–3.3× lower ITL … stays flat while vLLM degrades ~5×."
From committed NUMBERS.md, **I2T**:
- ITL ratio new/vLLM = 1.81 / 2.03 / 2.34 / 2.38 / 2.13 / 1.68 → range **1.68–2.38×**, not 3.3×.
- M*-new I2T ITL grows 0.0068→0.0341 = **5.0×** across load — *not* flat (vLLM grows 4.67×).
- The "3.3×" is the *I2S* ITL number (I2S ratios run 3.2–3.9×). A speech-path number was imported
  into an image-to-text document.
- **Fix**: state I2T-specific numbers ("~1.7–2.4× lower ITL; both systems' ITL rises with load, M*'s
  advantage is the *level*, not flatness"). The flat-ITL story is true for the *speech* paths, not I2T.

**RF-C ✅ The headline comparison is not reproducible from anything committed, and the favorable
direction looks cherry-picked.**
- `raw_*.json` is **gitignored** (`*.json`) → **zero** raw data committed on the branch.
  `command.txt`'s "Stage 3 fully reproducible from committed raw_*.json alone" is false; all four
  `verify_*.py` throw FileNotFoundError in the committed tree.
- Even in the on-disk raw, the two **headline** systems have essentially no per-request data:
  `mstar_new` = **1 datapoint/batch**, `vllm` = **6 total (B1 only)**, both `recomputed.n = None`.
  Full data (50–320 pts) exists only for `mstar_old` and `mstar_new_chunked`. `make_numbers.py`
  reads the frozen `aggregates` block only — it never touches `datapoints`.
- The headline `mstar_new` is **faster than the fully-measured `mstar_new_chunked` in 22/24 cells,
  mean +6.0%, up to +20.3%** — one-directional, with no committed datapoints behind the faster tag.
  An examiner cannot recompute your headline and will notice it beats the only sibling that *is*
  recomputable.
- **Fix**: commit the full per-request raw for `mstar_new` and `vllm`, regenerate aggregates from
  them (so `n>0` and a recompute matches), and either drop `mstar_new_chunked` or explain what it is.

### TIER 2 — benchmarked binary ≠ shipped binary (provenance)

**RF-D ✅ The benchmarked config is not the config the PR ships.**
`sweep.sh` defaults `FLAGS=""` and `command.txt` passes no `--flags`, so the run used the
worktree's built-in defaults at 1f66ce6: **GPU-mel OFF, GPU-image OFF, vLLM-prompt-layout OFF,
encoder CUDA-graphs eager**. The PR (b15bfea) flips all of these **ON** by default.
- Good news: this makes the headline numbers *conservative* — the shipping PR is if anything faster,
  so you are **not** inflating. Say this out loud; it's your strongest honesty point.
- Bad news: `command.txt` ("ships opts ON by default … no flags needed to get the benchmarked
  numbers") and `ARCHITECTURE.md` (M*-new = "GPU mel ON / GPU image ON / vLLM prompt layout")
  describe the PR defaults *as if they were measured*. They weren't. And the shipping config has
  never had a serving sweep — only parity unit tests.

**RF-E ✅ `codec_chunk_frames` 15→25 between benchmark and PR.** Committed S2S/I2S **ITL** numbers
were taken at 15-frame Code2Wav chunks; the PR emits 25-frame chunks (~67% higher per-chunk vocoder
latency). Throughput ≈ unchanged, but the speech ITL headline is stale. (ARCHITECTURE.md even
advertises M*-new = 15 frames, contradicting the shipped 25.)

**RF-F Encoder CUDA-graphs were silently eager at benchmark time.** The `varlen_attention`
`_fi_override` early-return that makes capture use FlashInfer is **new in the PR**; its own comment
says the unguarded path "fails capture and the encoder then silently falls back to eager, defeating
the whole point of the graph." So "Audio/Vision encoder CUDA-graph capture: YES" in ARCHITECTURE.md
overstates what the *benchmark* exercised. The graph path is validated only by a unit test added in
the same PR — never load-tested in a sweep. (Agent-asserted for the 1f66ce6 behavior; corroborated by
the PR's own comment + the varlen-backend default. Worth a 5-min GPU confirmation before you claim
the encoder-CG win in the benchmark.)

### TIER 3 — test / process gaps

**RF-G ✅ Test gating is weak.** CI runs **ruff only** (`.github/workflows/ci.yml`) — no pytest, so
every parity test (including the `_ci.py` ones) auto-passes in CI. The only **end-to-end** parity
file (`test_qwen3_omni_audio_output_parity.py`) is `pytest.mark.skip` at module level. The graph
parity test receives DeepStack features from the graph and **discards them unasserted**
(`_ds_e, _ds_g` at lines 101-102) — the production path's most important multimodal output is
graph-unchecked.

**RF-H ✅ Baseline base-commit confound.** Branch forks at 2e6465a; mstar_old = ae7d173 is **2 commits
ahead** (KVCache leak fix #146, profiling #144). "Old vs new" is therefore not a clean single-variable
diff — the new branch lacks two commits the baseline has. Ideally rebase new onto ae7d173 so the only
delta is the encoder work.

**RF-I The "mixing is infeasible" conclusion partly rests on a PENDING result.** The eager-mix ≈20×
collapse (0.19 vs 3.74 req/s at B=32) is real and measured. But the decisive follow-up — does
*capturing* the mixed shape recover throughput? — is marked **PENDING** in KNOWLEDGE §4, while STORY §9
reads as closed ("we never obtained a clean high-load result"). The honest framing: naïve mixing
collapses (proven); the captured-mixed fix is unproven (warmup >15 min + correctness bugs documented,
throughput recovery unmeasured).

---

## 2. Nitpicks / best-practice (CLAUDE.md + small stuff)

- Branch is `encoder-v4-benchmarked-new`, not `bench/<name>`; single squashed commit (44 sub-commits)
  vs "one commit per valid run"; raw.json not committed (gitignored) — three CLAUDE.md deviations.
- `env.txt` nvcc/torch were empty at capture (ran outside venv) and **re-captured next day** — honestly
  disclosed, but it means the env file is a reconstruction.
- Committed `datapoints` have only `phase="measure"` (no warmup rows) though schema says
  `warmup_iters=[5]` — warmup exclusion can't be verified from the artifact.
- `MSTAR_ENCODER_CG_WARMUP` default `"1,2,4,8"` misses bs 16/32 → first real request at those sizes
  eats a multi-second lazy-capture spike. `_cg_cache` has no eviction and logs nothing when it caps.
- Doc citations drift: ARCHITECTURE.md `micro_scheduler.py:234-242` for graph-walk enforcement is
  ~line 102 in the PR; KNOWLEDGE references `components/audio_encoder.py` without the
  `mstar/model/qwen3_omni/` prefix and points at `exp/mixed-walk-piggyback` code, not the PR tree.
- Compile A/B evidence is hardcoded numbers in a chart script (`plot_compile_ab.py`) on 1×H200 no-flash-attn,
  not a committed raw.json on the 2×H200 flashinfer rig.
- Rhetoric to soften: "~2× throughput" is really 1.75–2.55×; "vLLM TTFT essentially flat" is +37%
  on I2T; Sarathi-Serve "28×" is a worst-case single-event figure, not representative.
- `PIXEL_MAXABS_MAX=0.2` vs a comment claiming ~0.024 (8× looser than described);
  `HF_COS_MIN=0.99` is the loosest gate in the suite; `RELL2_MAX=0.05` is vacuous given cos≥0.999.
- Talker prefix comment "fixed 6 tokens" for `[pad*4, bos, proj[3]]` is 8, not 6 (was 9 — moved one
  wrong number to another).

---

## 3. The technical narrative — "this is as good as it gets" (quiz prep)

**Why M*-new I2T TTFT is worse than vLLM (0.70×→0.36×, widening with load) — and why that's correct,
not a bug.**
- vLLM v1 uses **continuous batching**: prefill and decode are mixed in one step under a token budget,
  so a new request is absorbed into the very next step → low, ~flat TTFT. It keeps that fast by
  **recording around attention** (piecewise CUDA graph: a single shape-flexible graph for everything
  except attention, which runs on the flexible path). So vLLM mixes *and* stays graphed.
- M* uses **separated, fixed-shape graph walks**: prefill-walk and decode-walk are distinct captured
  shapes, never mixed in one step (`micro_scheduler` enforces one graph_walk per batch). A newcomer
  waits for an admissible same-walk slot → TTFT is higher and grows with load. That separation is
  exactly what buys M* its throughput (~2×) and its flat ITL on the speech paths.
- Images sharpen it: an image expands to a *variable* token count (hundreds→thousands with
  resolution), so each image is a different prefill shape — more shape variability for a fixed-shape
  engine — and a heavy vision prefill head-of-line-blocks concurrent decodes.

**Why M* can't just adopt vLLM's mixing (the experiment).** A mixed (decode_count, prefill_len) step
matches neither the decode bucket nor the prefill bucket → no captured graph → eager → under load this
craters. **Measured: eager-mix B=32 = 0.19 req/s vs 3.74 mixed-off (≈20× collapse, dropped requests)** —
the same interference Sarathi-Serve quantifies. To mix without collapse you need vLLM's
record-around-attention substrate, a rearchitecture, not a knob. Four attempts to pre-capture the mixed
shapes hit combinatorial graph explosion, >15-min warmup, and correctness bugs; the throughput-recovery
number is still pending. → So matching vLLM I2T TTFT is **out of scope for an encoder port** and would
require scheduler/substrate work. That is the defensible "as good as it gets" boundary.

**torch.compile.** Honestly demoted: it does **not** beat eager for these encoders (they're big-matmul
bound; Inductor can't beat cuBLAS). `dynamic=True` is kept only to kill recompile-storm **tail latency**
on new shapes — a hygiene lever, not a throughput win. This is exactly the "keep only what wins" the
ticket asked for. (Caveat RF: the supporting microbench is hardcoded + on different hardware.)

**CUDA graphs (the real win).** Encoder block-loop captured per grid layout (kills ~30% per-step launch
overhead); Thinker/Talker/Code2Wav captured per batch bucket — this is M*'s throughput substrate and
the reason it leads on throughput and on speech-path ITL. Caveat: the *encoder* graph was eager during
the committed benchmark (RF-F); it's the PR that first makes it fire.

**Batching across requests.** Real and structural for **audio** (varlen-packed mel + multi-entry
feature_lens). For **vision** it's `MSTAR_BATCH_VISION_PREFILL`, **default OFF and never serving-tested**
— so "batched across requests" is true for audio, opt-in/unproven for vision. Don't claim the vision
batching win.

**Is the optimization "enough"?** For #131 (port + optimize the *encoders*): yes — native encoders +
GPU preprocess + sync-elim + (now-working) encoder CUDA-graphs are genuine front-of-request wins and
beat M*-old I2T TTFT by 1.6–1.9×. They do **not** close the gap to vLLM on I2T TTFT because that gap is
the continuous-batching / record-around-attention architecture, correctly scoped out of an encoder
ticket. The ticket itself predicted compile/CG/batching "might not apply" — and the honest finding is
that compile didn't, CG/batching/preprocess did, which is the right result.

---

## 4. Does it look like the PR is lying / falsifying?

**The code: no.** The native encoders are real from-scratch reimplementations (ViT patch-embed as
`F.linear`, correct DeepStack mergers, identical output contract, eager fallbacks, 5 selectable varlen
backends for genuine A/B). Nothing fabricated in the implementation.

**The evidence: not fabricated, but not defensible as committed.** The numbers are internally consistent
and, if anything, *understate* the shipping PR (benchmarked on the slower default config). But: raw data
isn't committed; the two headline systems have no recomputable per-request data; the headline beats the
only recomputable sibling one-directionally; 3 unfavorable cells were hand-blanked in an "auto-generated"
file; and one headline narrative number (I2T ITL 3.3×/flat) is contradicted by your own table. Each of
those, individually, is what a hostile reviewer points at and says "fabricated" — even though the
underlying work isn't. Fix RF-A/B/C and you remove the ammunition.

**Minimum to be quiz-proof:** (1) commit full raw for mstar_new + vllm and regenerate aggregates from
them; (2) correct the I2T ITL claim; (3) replace the hand-edit with a documented filter; (4) state plainly
that the benchmark used the conservative default set and the shipping defaults are faster-but-unswept;
(5) re-label the encoder-CG and codec-25 facts to match what was actually measured.

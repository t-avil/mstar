# Finalization — single-config #131 hand-off

**Branch:** `submission/single-config-v2` (off `submission/single-config`).
**Date:** 2026-07-22. Reviewer-facing status + honesty ledger + what's left.

## TL;DR
The headline is **real and defensible after wording fixes**. An independent audit
recomputed all five B32 ratios exactly from the committed raw and confirmed the
competitor series is byte-identical to the `benchmarks` branch (owner rule intact —
no fresh vLLM race). The embarrassing bits were **wording/sampling, not fraud**, and
are fixed here. One item is **blocked on GPU availability**: a fresh equal-n M* rebench
(the node was claimed by another user's sglang job mid-session).

## What changed on this branch (6 commits, all small)
1. `fix(replicate)`: **boot script didn't run** — `$FLAGS` expanded into command
   position (`MSTAR_FAST_POSTPROC=1: command not found`). Now `env $FLAGS`. A reviewer
   following REPLICATE.md would have hit this on line 1.
2. `fix(conductor)`: docstring claimed a **random fallback when `output_modalities`
   is None** — false; None → rank 0 deterministically. Corrected (behavior unchanged).
3. `test(parity)`: added the missing case — the audio-merge gate **declines for a
   vision+text (i2t) schedule** (the single boot enables the flag globally). 9 passed.
4. `docs`: removed the **self-contradiction** — WRITEUP Limitation #1 said "TWO configs,
   cannot win everywhere," refuting the single-config headline; rewrote to the dpenc
   reality + the ~4% i2t cost vs pure encoff. Fixed the false **"n=256"** claim.
5. `docs`: `MERGE_STRATEGY.md` — see below.
6. `fix(replicate)`: boot reused an **empty Inductor cache dir**, forcing a ~25 min
   recompile every boot; use the persistent default so warm kernels are reused.

## Honesty ledger (the "big complaints" — surface these, don't bury them)
- ★ **i2t B32 = 1.04× is the weak claim.** It is real (numbers scale monotonically,
  not noise) but it is (a) a 4% win the WRITEUP itself elsewhere calls "fluke-level
  tie," and (b) the ONLY cell measured at unequal n (dpenc n=256 vs vLLM n=128).
  **Lead the text story with s2t 1.27×** (comfortably above the 11.9% s2t cross-version
  band); present i2t as "parity/slight edge." The pending fresh rebench measures i2t
  B32 at **equal n=128** to settle it honestly.
- **Speech "2–3×" is req/s, not audio-seconds/s** (~1.5× on audio-sps — length
  confound). Already disclosed; keep it prominent.
- **t2s ~2.0×** is the weakest speech path (low edge of the band). Disclosed.
- **`upmain` s2t is degenerate** (~60 tok/s flat B4–B32 — it doesn't batch), which
  inflates "3–8× over main." That's main being broken, not M* being fast; don't lean
  on the over-main multiple. Doesn't touch the vLLM headline.
- **Low-batch bars rest on n=12–24** — directional only; defend at B32 (n=96–128).
- Two headline win cells carried **1 failed request** (i2s B32 95/96, t2s B32 95/96);
  tiny effect on a rate metric but note it.
- **Parity is design/unit-level, not a runtime cross-build token diff.** The shipped
  build runs fp8 + custom-ops = bounded-rounding, NOT bit-exact. Say so; don't imply
  the winning build is byte-identical.

## Merge strategy (see MERGE_STRATEGY.md for the verified detail)
- Base `origin/main` is **14 commits behind `upstream/main`** (our 5 are docs only).
- **The core #131 win is ONE clean commit — `4216adc9`** (conductor.py + dpenc.yaml);
  it auto-merges onto upstream with **zero conflict markers**. Open that as PR-1.
- All 29 conflicts are in the **default-off perf-flag machinery** (worker.py/cuda
  graph runner) colliding with upstream's cosmos3 (#121) + piecewise (#154) rewrites →
  split into PR-2 (ordered-emit fix), PR-3 (preproc pool), PR-4+ (decode stack, rebase
  + re-bench AFTER #154). `encoders-implemented-md` is an orphan docs branch — exclude.
- One rebase hazard to hand-check: verify the modality route fires on upstream's new
  `_instance_ranks` branch, not only the old flat `else`.

## Charts
`python submission/scripts/gen_charts.py` reproduces all 5 PNGs **byte-identical** to
the committed ones from committed raw (no GPU). These are committed-data charts; the
fresh-data versions land after the rebench.

## The one blocker
Fresh M* rebench (dpenc equal-n=128, then upmain) needs **2 idle whole GPUs**. As of
this writing the entire node (all 8 GPUs) is running another user's `sglang` server,
so per the no-co-location rule the rebench is paused. The loop is polling; it boots the
moment a clean pair frees. vLLM-0.24 stays the committed reference — re-run it through
your own pipeline before defending those cells (owner rule).

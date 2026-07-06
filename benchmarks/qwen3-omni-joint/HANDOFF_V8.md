# HANDOFF_V8 — M* vs vLLM-Omni campaign (state as of 2026-07-06)

Supersedes HANDOFF_V7.md. Single entry point for the next agent. Companion docs:
EXPERIMENTS.md (THE knowledge base, ~80 entries), GOAL_MATRIX.md rev-5 (per-cell
endgame), FEATURES_SINCE_ENCODERS.md (full feature set of the code delta),
PR_DECOMPOSITION.md (how to upstream it as small PRs), TORCH_COMPILE_FINAL.md
(final compile audit), VLLM_RELIABILITY.md (8 logged vLLM failure events).

## 0. OWNER RULE (standing, 2026-07-05)

Do NOT benchmark, boot, or race vLLM. The user runs all vLLM measurements
themselves. Compare only against the COMMITTED vLLM values on the benchmarks
branch (h2h_* raw results.json; i2t B32 fresh-boot live band 8.03–8.50 req/s).
M*-side A/Bs are unrestricted.

## 1. THE WINNING BUILD (one command)

```
/m-coriander/coriander/tim/launch_mstar_best.sh
```

Boots worktree /m-coriander/coriander/tim/mstar-b1fix @ opt/prep-h2d (620de91)
with the certified env. Lineage: encoders base → merge-config → cfgv2 (ded928d)
→ checkstop (7ef5150) → stack-n2 → prep-h2d. Flags (full list in the script):

- MSTAR_SAMPLER_CFG_CACHE_V2=1 — kills per-step sampler-config cache thrash
  (pageable H2D + stream sync was 22% of B32 wall; sampling.py:419).
- MSTAR_SIDECAR_CHECKSTOP=1 — Stage-2 check_stop offload, deferred-consume.
- MSTAR_PREP_DEVICE_POS=1 — prepare_inputs pos_ids kept device-side
  (submodules.py:620 pageable H2D was 24% of B1 wall).
- MSTAR_MERGED_PREFILL=1 — prefill_text+prefill_vision one-walk merge.
- TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 with
  TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache_cdt
  (+3.4% B32; cold compile without the cache ≈ 40 min, with cache ≈ boot-normal).

Certified numbers at ship: B32 peak 8.95 (all-time high, above ref-band max),
B1 1.17×, B2 1.04–1.05× on the canonical pair.

## 2. SCOREBOARD vs COMMITTED refs (the only comparison allowed)

- Speech 12/12 GREEN, 2.1–2.9×.
- s2t 6/6 GREEN, 1.08–3.06× (small-batch ratios inflated by vLLM answer-moding
  on interrogative audio; M* transcripts at parity on 534 pairs).
- i2t: B1 1.17–1.21×, B2 1.04–1.08×, B4 1.11×, B8 1.22×, B16 1.10× — GREEN.
  B32 = band-parity: mean 7.98–8.37, peaks 8.72–8.95 vs band 8.03–8.50.
  Not a stable ≥1.05×; the residual is proven structural (see §3).

## 3. WHY B32 IS CLOSED (host-side)

The last 10%-of-wall is worker.py:3193 `completion_event.synchronize()` — the
await-GPU gate at the top of _postprocess_batch. The main thread is AHEAD of
the GPU there: structural GPU-bound time. Every host-side lever was measured:
sends washed (sidecar-batch B/A ~0.94), deferral = V1 identity-fail (parked),
polling burns the gpu-thread GIL shade. Remaining upside is kernel-below-
Inductor or the EngineCore rewrite — a scope decision for the owner.

## 4. WHAT CHANGED SINCE V7

1. prep-h2d landed (+3.1% B2) and CDT landed (+3.4% B32, env-only).
2. sidecar-batch WASHED-NEGATIVE and struct-pack DEAD pre-build (pickle wins).
3. Frontier formally closed with the §3 mechanism.
4. launch_mstar_best.sh shipped + self-certified.
5. sweep_mstar_v4/ staged (median representative cells from the 07-05/06
   iteration) and the v4 charts regenerated in the original 2×2 4-metric
   square (gen_v4_charts.py; per-metric layered merge v2→v3→v4 so no
   committed point is ever dropped; speech uses audio-side TTFT/ITL).
6. All 25+ opt/* and exp/* branches verified pushed to fork
   (git@github.com:t-avil/mstar.git).
7. FEATURES_SINCE_ENCODERS.md, PR_DECOMPOSITION.md, TORCH_COMPILE_FINAL.md
   written (this session).

## 5. STANDING ORDER (next actionable work)

The user wants the FULL 24-cell sweep re-run on the newest build the moment
GPUs free up (M*-only, vs committed refs): boot via launch_mstar_best.sh, run
{i2t,s2t,s2s,i2s} × {B1..B32}, commit raw per-cell results, update the charts'
solid blue line. Canonical pair is 6,7 (pair 4,5 quarantined — 4 straight boot
deaths; pair 0,1 reads ~10% low). Check `nvidia-smi` idle + ≥~200G free RAM
before boot (Ray plasma holds ~290G of /dev/shm; boots during measurement have
killed servers). Max 2 M* servers on the box, never boot during measurement.

## 6. MEASUREMENT DISCIPLINE (unchanged, binding)

Adjacent-pair A/Bs with ABBA ordering (legacy harness had +5–12% B-favoring
bias); criterion warm-in (p99/med<1.7 + ±3% ×2 cells); soft-cell rejection via
per-arm robust-z; tok/req parity gate; req/s for cross-system claims; grade
via ab_verdict.py; commit raw with `git add -f` (a *.json gitignore rule
silently emptied four raw commits once — always `git ls-files` verify).
Worktree PYTHONPATH trap: export PYTHONPATH=<worktree> or spawned GPU workers
load the wrong code and invalidate the A/B.

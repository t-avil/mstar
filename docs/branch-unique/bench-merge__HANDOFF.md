# Qwen3-Omni on M\* (#131) — Handoff & One-Week Experiment Plan

Goal of #131: **port the Qwen3-Omni encoders to native M\*** and show M\* is a superior
serving platform vs vLLM-Omni (and not a regression vs M\*-old/HF-wrapper). This doc is a
self-contained handoff: what's done, the *honest* performance picture, every pushed branch,
the methodology that must be followed, and an ordered plan to finish the ticket with great
results + benchmarks + parity checks using agents + 8×H200.

Companion docs (same dir, `main`): `FINDINGS.md` (full investigation), `EXPERIMENTS.md`
(experiment matrix). Conventions: `/home/tim/CLAUDE.md` (GPU hygiene, git workflow) — non-negotiable.

---

## TL;DR — the honest status
- **Encoder parity (acceptance #1): ✅ DONE.** Native vision+audio encoders are numerically
  identical to HF (= vLLM): cos=1.000000 fp32, ≥0.9999 bf16; 0 missing/unexpected weights.
  18-case backend-equivalence unit test green.
- **M\* is faster than vLLM on essentially every axis once measured fairly.** At equal workload:
  ITL ~2×, E2E ~1.6×, throughput ~1.6×, I2S RTF ~1.8×, S2S RTF (with codec_chunk) beats vLLM.
- **The one apparent loss — TTFT ~2× — was a MEASUREMENT ARTIFACT**, not a real deficit (see
  Methodology §, "device isolation"). Isolated, M\* TTFT ≈ 105–120 ms ≈ vLLM's ~118 ms.
- **Remaining work is mostly: validate the one in-flight build (Code2Wav SP), re-run the
  headline sweeps under correct isolation, and finish the #131 batch/throughput proof + charts.**

---

## Verified B=1 performance (fair = isolated, single server, seed=42, ×50)
| Metric | M\*-new | vLLM | Verdict |
|---|---|---|---|
| ITL (S2T/I2T) | **0.007 s** | 0.012–0.016 | M\* ~2× ✅ |
| E2E @ forced 256 tok | **2.02 s** | 3.27 s | M\* ~1.6× ✅ |
| Throughput @ 256 tok | **127 tok/s** | 78 | M\* ~1.6× ✅ |
| TTFT (isolated) | **0.105–0.120 s** | 0.118 | tie/slight win ✅ |
| I2S RTF | **0.087** | 0.158 | M\* ~1.8× ✅ |
| S2S RTF (codec_chunk 15) | **0.167** | 0.193 | M\* win ✅ |

Caveat retained: vLLM emits ~2× longer S2S/I2S audio (it answers vs M\* transcribes) — use
**median RTF + length-independent TTFT/ITL**, OR run M\* with `MSTAR_VLLM_PROMPT_LAYOUT=1` (below)
to match audio length. The pre-isolation "M\* TTFT 0.28 s" numbers in old `fair-b1` artifacts are
**invalid (co-located)** — do not cite them.

---

## Pushed branches (fork `git@github.com:t-avil/mstar.git`)
| Branch | State | What / how to use |
|---|---|---|
| `main` | ✅ | `FINDINGS.md`, `EXPERIMENTS.md`, `HANDOFF.md`. No benchmark artifacts (CLAUDE.md). |
| `vllm-layout` @ `09e96b8` | ✅ proven | ⭐ Token+position parity w/ vLLM. `MSTAR_VLLM_PROMPT_LAYOUT=1` → M\* answers like vLLM (FIX1 system-dup, FIX2 audio M-RoPE h/w). Carries the 18-case backend parity test. Default OFF = byte-identical. |
| `codec-chunk` @ `086539f` | ✅ win | codec_chunk 25→15 — S2S beats vLLM. Keep chunk ≥ left_context. Config default change. |
| `gpu-img-preprocess` @ `0cb3f98` | ✅ correct | `MSTAR_GPU_IMAGE_PREPROCESS=1` — on-device resize/patchify, 7–100× faster/img, parity cos≥0.999983. Only moves TTFT on large (~3000px) images. |
| `ttft-profile` @ `6c3f663` | ✅ tool | `MSTAR_NODE_TIMING=1` — per-stage TTFT instrumentation + `mstar/utils/ttft_trace.py`. Use to measure any TTFT change. |
| `merge-prefill-walks` @ `7c7cd55` | ✅ negative | `MSTAR_MERGE_PREFILL_WALKS=1` — correct walk merge, **no TTFT win** (round-trip ~0). Keep as a clean simplification; not a perf lever. 11-case parity test. |
| `code2wav-sp` (WIP) | ⚠️ **UNVALIDATED** | Frame-dim sequence parallelism for the vocoder. Agent interrupted **before** the parity gate + A/B. **Validate before trusting** (Phase 1). |
| `benchmarks` (aggregation) | ✅ | All bench artifacts (raw.json + charts). Never merge to main. |

---

## CRITICAL methodology — read before running anything
1. **DEVICE ISOLATION IS THE #1 RULE.** Never co-locate two model servers on the same node when
   benchmarking M\*. M\*'s TTFT path crosses 4 Python processes with 1 ms poll/sleep loops; a busy
   neighbor deschedules those threads and inflates ONLY M\*'s TTFT (decode is a captured CUDA graph
   and is robust). This single mistake produced the bogus "M\* loses TTFT 2×". One benchmark = its
   own GPUs **and** no competing server thrashing the host CPU. If you must run M\* and vLLM to
   compare, run them **sequentially** or on well-separated NUMA with confirmed-idle CPUs.
2. **Use one fixed GPU set per project; confirm idle (`nvidia-smi`) before launch; never silently
   fall back to whatever's free.** Pin `CUDA_VISIBLE_DEVICES`.
3. **Servers: launch detached** via `setsid` (reparented to init so the harness can't reap them),
   unique `--socket-path-prefix` per server (avoid the `/tmp/mstar` ZMQ collision). Wrap runs in
   `timeout`; cleanup process groups on every exit path; monitor `nvidia-smi`; free GPUs the instant
   a run finishes. Reuse `/home/tim/launch_mstar.sh`.
4. **Only complete runs get committed** (one commit per valid run, on its bench branch; merge to
   `benchmarks`). Never commit benchmark artifacts on `main`.
5. **Every code change is env-gated, default OFF, byte-identical baseline.** After any encoder/model
   change, keep `test/modular/test_qwen3_omni_varlen_backend_parity.py` (18) + the encoder-vs-HF
   parity green. A speedup only "counts" if it is **≥10% over BOTH M\*-old and vLLM** with parity green.
6. **Clocks/persistence:** node has `persistence_mode=Enabled` (pre-existing, admin-only — we can't
   change it). Record "clocks unlocked" in env capture; treat cross-run variance accordingly.

---

## The one-week plan (ordered; each phase = 1+ agent, parallel where independent)

### Phase 0 — Lock the fair baseline (0.5 day) ⟵ do first
Re-run the full B=1 headline set (S2T, I2T, I2S, S2S) for M\*-new, M\*-old, vLLM **under correct
isolation** (each system its own GPU pair, no host-CPU contention, sequential if needed). Two passes:
(a) natural length, (b) `MSTAR_VLLM_PROMPT_LAYOUT=1` + identical `max_tokens` (matched audio length).
Produce the clean 3-way table (median RTF + TTFT + ITL + throughput + audio_dur). This replaces all
invalid co-located numbers and is the reference every later A/B is measured against.

### Phase 1 — Validate / land Code2Wav SP (1–2 days) ⟵ the I2S headline
On `code2wav-sp`: (1) **parity gate first** — sharded vs single-device waveform, per-sample max-abs +
cosine, *boundary-focused* (a vocoder seam is audible; this is the correctness wall). (2) If parity
holds, A/B I2S (long audio) + S2S, B=1 ×50, flag OFF/ON, expect toward ~half-vocoder → widen I2S RTF.
(3) If a seam appears, fix the halo width / overlap-add before benchmarking. Commit only when parity +
A/B are green; then bench branch + charts. This is the biggest remaining *new* win and is **M\*-only**.

### Phase 2 — TTFT polish (1 day, optional — TTFT is already ~tied)
From the profile, the only real TTFT levers left are small: **collapse the 1 ms poll/sleep loops + the
~28 ms first-token emit→client SHM read into an event-driven push**, and **mel-extraction → GPU**
(~part of 18 ms preprocess). Each is tens of ms. Gate, parity, A/B with `MSTAR_NODE_TIMING`. Skip the
disproven levers (merge-walks, prefill bucket padding, encoder bucketed-cudagraph — profile showed
little headroom). Pursue only if you want a *clear* TTFT win rather than a tie.

### Phase 3 — Batch / throughput: the #131 "superior platform" proof (1–2 days)
B=1 is launch-bound and structurally ties M\*-new/old; **M\*'s real win is at batch.** Run the async
pipelining + continuous-batching sweep B=4→32 (all 4 paths) M\*-new vs vLLM — confirm the paper's
~2–2.5× throughput. Then the **encoder batch sweep native vs HF via A2T** (text out, no talker): M\*-old's
dense O(n²) HF encoder degrades (→2.0 RTF @ B=32 S2S) while native varlen holds — this is acceptance #2
and the cleanest "native > HF" evidence. Charts per CLAUDE.md.

### Phase 4 — Encoder backend matrix (0.5 day, widens the batch story)
`MSTAR_VARLEN_BACKEND` × batch matrix (flash_attn / flashinfer / per_segment / padded / adaptive) on the
isolated A2T encoder-forward — the `adaptive` τ=5e5 heuristic is likely miscalibrated for audio's many
~50-tok windows. Pick a better default backend curve. Keep the 18-case parity green.

### Phase 5 — Finalize #131 deliverables (0.5 day)
Per CLAUDE.md git workflow: push `bench/qwen3-omni-{i2s,s2s,s2t,i2t}-{mstar-new,mstar-old,vllm}`, each
merged to `benchmarks`; the 4 comparison charts (paths × {RTF/TTFT, throughput}); a final results table
in FINDINGS.md. Land the validated wins (codec_chunk ✅, Code2Wav SP if green) as defaults; keep the rest
env-gated. Write the #131 PR summary: parity proof, fair B=1 table, batch throughput proof, parity tests.

---

## Run-an-experiment template (copy this shape)
```
1. nvidia-smi → confirm the chosen pair is idle (≈4 MiB). Pin CUDA_VISIBLE_DEVICES.
2. git worktree off base 43ffffa (or the relevant branch); implement env-gated (default OFF).
3. Parity FIRST: 18-case varlen + encoder-vs-HF + any change-specific parity. Must be green.
4. Launch server: setsid daemon, unique --socket-path-prefix, timeout wrapper, cleanup trap.
5. Warmup (tag/discard) → measured ×50, B=1, seed=42, closed-loop. Capture EVERY datapoint.
6. raw.json (units, warmup_iters, phase tag per datapoint) + env.txt + command.txt + requirements.txt.
7. Free GPUs immediately. Commit ONLY if complete + ≥10% over both baselines + parity green.
8. Bench branch → push → merge to `benchmarks` → push. Never on main.
```

## Acceptance checklist for #131
- [x] Native encoders == HF (parity test, cos≈1.0) — **met**
- [x] Backend-equivalence regression test (18 cases) — **met**
- [ ] M\*-new batch sweep B=1→32, native > HF at batch (Phase 3/4)
- [ ] Fair isolated 3-way B=1 table, all 4 paths (Phase 0)
- [ ] Code2Wav SP validated + landed (Phase 1) — *the standout M\*-only win*
- [ ] Throughput ~2× vLLM at batch confirmed (Phase 3)
- [ ] Bench branches + 4 charts + PR summary (Phase 5)

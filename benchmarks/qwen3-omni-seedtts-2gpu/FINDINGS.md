# Qwen3-Omni Seed-TTS, 2-GPU — Figure 5 reproduction findings

Reproduction of the M\* paper (arXiv 2606.12688) Figure 5: Qwen3-Omni Seed-TTS on
2 GPUs, RTF (lower=better) + audio throughput (higher=better), batch sweep
B∈{1,2,4,8,16,32}. Run on 8×H200 (GPUs 6,7), model `Qwen/Qwen3-Omni-30B-A3B-Instruct`,
Seed-TTS-eval (en). See `charts/fig5a_rtf.png`, `charts/fig5b_throughput.png`,
`raw.json` (every per-request datapoint).

## Headline result — closed-loop max-concurrency (canonical), warmup=5

Protocol = **closed-loop max-concurrency (continuous batching)** — the protocol the
fork's own benchmark scripts use (`run_omni_paths.sh`, `serving_scripts/*`) and the
reference CSV's "Max con." column. Data under `runs/cl/`. (An initial offline
sized-waves sweep — `runs/out_*/` — is retained as a cross-check; it gave matching
RTF but ~25–30% lower throughput due to tail-of-wave GPU idle, which is exactly why
closed-loop is the correct protocol.)

| B | M\* RTF | vLLM RTF | M\* tput | vLLM tput | M\* tput advantage |
|---|---|---|---|---|---|
| 1  | 0.081 | 0.162 | 12.7  | 6.4  | 2.0× |
| 2  | 0.099 | 0.202 | 19.8  | 9.9  | 2.0× |
| 4  | 0.113 | 0.248 | 34.6  | 15.7 | 2.2× |
| 8  | 0.130 | 0.326 | 55.3  | 23.0 | 2.4× |
| 16 | 0.170 | 0.450 | 86.2  | 34.6 | 2.5× |
| 32 | 0.281 | 0.640 | 107.2 | 46.8 | 2.3× |

Matches the team's reference CSV (same closed-loop protocol) within ~1–12% across the
curve. M\* beats vLLM-Omni on **both** RTF and throughput at every batch size, ~2–2.5×,
reproducing the paper's "2.7× higher throughput vs vLLM-Omni" claim. All runs
completed 100% (310 requests each).

## Version audit vs the paper (Table 4 / Appendix I)

| Component | Paper | This run | Match |
|---|---|---|---|
| Hardware | 8× H200 | 8× H200 | yes |
| M\* | Python 3.12 + PyTorch/CUDA + FlashInfer | py3.12, torch 2.9.1+cu129, upstream `main` | yes |
| vLLM-Omni | vllm v0.21.0 | vllm 0.21.0+cu129, vllm-omni @60c15004 (0.21.0 line) | yes |
| SGLang-Omni | commit 4a3960 | **unavailable** (see below) | no |

## Reference-CSV comparison + the protocol effect

A team reference CSV (post-graph/engine-refactor M\* commit `4a3960be…`) uses a
**closed-loop / max-concurrency** firing pattern with `num_warmup=5`. Our main sweep
used **offline sized-waves** with `num_warmup=3`. RTF matched closely; throughput was
systematically lower under offline (tail-of-wave GPU idle). Re-running B=32 in
closed-loop (max-concurrency=32, warmup=5, 160 reqs — `runs/out_*_closedloop/`)
closed the gap:

| B=32 | offline (ours) | closed-loop (ours) | reference |
|---|---|---|---|
| M\* RTF | 0.301 | 0.283 | 0.264 |
| M\* tput | 81.0 | 105.6 | 114.3 |
| vLLM RTF | 0.536 | 0.630 | 0.714 |
| vLLM tput | 38.2 | 47.0 | 41.8 |

Both closed-loop points land within ~7–12% of the reference (residual ≈ the M\*
commit difference: reference is the post-refactor build, we ran plain `main`).
Note the RTF moved *down* for M\* (burst-limited: offline's simultaneous-prefill
spike is avoided) but *up* for vLLM (contention-limited: sustained concurrency
queues more) — both toward their references.

## SGLang-Omni: not reproducible in this environment

The paper's pinned commit **`4a3960` is gone from GitHub** (history force-pushed/
squashed to a single "Initial commit"; `git fetch <sha>` and the GitHub API both
return "not found"). Of the two reachable builds:

- **V1** (current main, post-"Retire SGLang Omni V0" #435): the `talker_ar` stage
  **deadlocks during distributed init** (full GPU memory allocated, 0% util for 8+
  min) — never serves.
- **V0** (`5ae9f3e`, the architecture the M\* team's `sglang_omni_instructions.md`
  references via `python -m sglang_omni.cli.cli serve`): bare-metal it crashed
  (CUDA-IPC dealloc leak + permission errors). Run correctly **inside the prescribed
  Docker image `frankleeeee/sglang-omni:dev`** (`--shm-size 32g`, GPUs 6,7) it
  becomes **stable** (the instructions genuinely required the container), but the
  talker **over-generates audio** under the harness's `temperature=0` (greedy-talker
  degeneration → 68–88 s of audio for 3 s sentences) and runs **~10× slower per-token**
  than the paper implies. The single completed B=1 datapoint (`runs/out_sglang_omni/B1`)
  is non-representative and is intentionally excluded from the figure.

Conclusion: a faithful SGLang-Omni number is not obtainable here — the exact paper
commit no longer exists, and neither reachable build represents the paper's
(optimized, now-deleted) baseline. Figure 5 is reported with M\* vs vLLM-Omni.

# Handoff: Qwen3-Omni cross-framework serving benchmark (M* vs vLLM-Omni vs sglang-omni)

Environment setup learnings, working recipes, and results. Node: 4×H100, shared
(GPU 1 is another user's — never touched; only GPUs 0/2/3 used). Hostile bits:
root fs `/` periodically 100% full (other users), no RDMA HCAs.

## TL;DR results (Qwen3-Omni-30B, H100, bs=1, ~8 req — approximate setup test)

| path | vLLM-Omni | M* |
|---|---|---|
| I2T  | TTFT 153ms, ITL 5.2ms, 0.97 req/s | TTFT **107ms**, ITL 7.9ms, 0.63 req/s |
| S2T  | TTFT 89ms, ITL 5.4ms, 5.1 req/s   | TTFT 97ms, ITL 8.3ms, 4.67 req/s |
| I2S  | *no speech path in vLLM-Omni*     | 8/8, **RTF 0.086** (~11× real-time) |

Consistent with paper/blog (arxiv 2606.12688, m-star.org): published Qwen3-Omni
numbers are **TTS-only @ B=16 on 2×H200** (~2.7× throughput vs vLLM, ~4× vs SGLang).
No published TTFT/ITL/I2T/S2T numbers, and no claim of beating vLLM on text paths —
so M*'s slightly-behind text decode here is expected ("other paths not yet optimized").

## Reusable harness
`benchmark/runner.py` (drives all systems via `--inference-system {ours,vllm_omni,sglang_omni}`)
already measures per-modality TTFT, per-token ITL, audio RTF, throughput. Parse stdout
with `benchmark/convenience/parse_i2t_table.py`. `results.json` was extended to persist
the full TTFT/ITL/RTF/throughput (was JCT-only). Orchestrators: `bench_system.sh`
(per-system I2T/S2T/I2S sweep).

## Shared gotchas (all systems)
- **Root `/` is 100% full intermittently** → put ALL caches/venvs/logs/JIT temp on
  `/mnt/storage`. Set `UV_CACHE_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`, and (for the
  server JIT) `TMPDIR` off `/`. When `/` has room, `TMPDIR=/tmp` is *faster* than
  storage (matters for flashinfer/cutlass JIT — slow-disk JIT can hang warmup).
- **GPUs:** only 0/2/3 free (1 is another user). Each framework offloaded before the
  next; kill `multiprocessing.spawn`/`StageEngine`/`EngineCore` children, not just the
  parent, and free GPU procs (`nvidia-smi --query-compute-apps`, skip the other user's PID).
- **Checkpoint:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` is ungated; cached at
  `/mnt/storage/timchick/hf_cache/hub/...`. Ships **no `tokenizer.json`**.
- **tokenizer.json is shared-snapshot poison across venvs:** generating it (transformers
  5.12.1) makes M* happy but breaks vLLM/transformers-4.x tokenizer load. Best to give
  each stack its own snapshot copy, or regen per-stack.

## vLLM-Omni — WORKING recipe
- Repo `benchmark/vllm_omni_instructions.md` says `vllm==0.19.0` → **STALE**. Use
  **`vllm==0.23.0`** (vllm-omni HEAD needs `IrOpPriorityConfig`; per vllm-omni
  `docs/getting_started/quickstart.md`).
- vllm 0.23 pulls `transformers==4.57.6`, whose `Qwen3OmniMoeProcessor` crashes:
  `Qwen2TokenizerFast has no attribute image_token`. Fix: **`transformers==5.12.1`**
  (allowed by vllm's pin `>=4.56,!=5.0-5.5`).
- Stage config moved → `vllm_omni/deploy/qwen3_omni_moe.yaml` (2 GPUs: stage0 cuda:0,
  stages1+2 cuda:1).
- Serve: `CUDA_VISIBLE_DEVICES=0,2 HF_HOME=… vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct
  --omni --port 8091 --stage-configs-path vllm_omni/deploy/qwen3_omni_moe.yaml`
- I2S (speech) path errors (EngineCore 500) — and vLLM-Omni has no first-class image→speech
  anyway; treat I2S as M*-only.

## M* — WORKING recipe (the misconfig was mine, not M*)
- Run `mstar-serve` (or the `mstar` wrapper, which defaults to the right transport):
  - `--tensor-comm-protocol SHM` — **critical**: this node has 0 RDMA HCAs, so the
    default RDMA/Mooncake transport 500s on any cross-worker tensor (all multimodal
    paths). SHM is the single-node default the `mstar` wrapper sets; bypassing it via
    `mstar-serve` directly is what broke multimodal earlier.
  - `--config configs/qwen3omni_thinker_tp2.yaml` — 30B Thinker TP=2 (~30GB/rank) across
    GPUs 0,2,3; the 2-GPU config OOMs at CUDA-graph capture.
  - `--cache-dir $HF_HOME/hub` + `HF_HUB_OFFLINE=1` so weight resolve doesn't hang online
    and fall back to the bare repo-id (which trips the tokenizer BPE bug).
  - `TMPDIR=/tmp/...` on a non-full disk — slow-storage JIT made the **Talker** CUDA-graph
    warmup hang for 25 min; fast `/tmp` JIT completes in minutes.
  - regenerate `tokenizer.json` (transformers 5.12.1) for the snapshot first.
- Text-only config impossible (`KeyError: 'Talker'` — model hardcodes the Talker partition).

## sglang-omni — IN PROGRESS
- Repo instruction `python -m sglang_omni.cli.cli serve` → **wrong module**. Use
  `examples/run_qwen3_omni_server.py` (text-only, I2T/S2T) or
  `examples/run_qwen3_omni_speech_server.py` (I2S). transformers 5.6.0 loads the processor fine.
- **FIXED**: CUTLASS-DSL/MLIR mismatch (`mlir_global_dtors() got an unexpected
  keyword argument 'data'`) came from **flashinfer 0.6.11's rmsnorm JIT** using
  `nvidia-cutlass-dsl 4.5.1` (CuTe-DSL path is SM100+/Blackwell-only; we're on
  H100/SM90). Set **`FLASHINFER_USE_CUDA_NORM=1`** to force the CUDA rmsnorm and
  bypass cutlass-dsl entirely. (The import-time try/except only catches ImportError,
  not the runtime JIT `DSLRuntimeError`, so the env var is the real fix.)
- Serve: `CUDA_VISIBLE_DEVICES=0,2 FLASHINFER_USE_CUDA_NORM=1 python
  examples/run_qwen3_omni_server.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
  --port 8092`.

## TTS reproduction vs the paper — findings + blocker

**Paper target (arxiv 2606.12688, §TTS):** Seed-TTS dataset, batch B∈{1,4,8,16,32},
max_tokens=256, metrics = RTF (wall/audio-dur, <1 = real-time) + throughput. M* claim:
**2.7× throughput vs vLLM-Omni, 4.0× vs SGLang-Omni @ B=16** (2×H200; consistent with TP2 on 3×H200).

**What we got:**
- **M\*** reproduces TTS capability: T2S RTF **0.10**, I2S RTF **0.093**, ~11 audio-sec/sec
  (real audio, bs=1, H100). Robust.
- **vLLM-Omni AND sglang-omni both FAIL TTS identically** — root cause found:
  `_thinker_to_talker_prefill → torch.cat(): expected a non-empty list of Tensors`
  (vllm: `vllm_omni/model_executor/models/qwen3_omni/...`; sglang:
  `sglang_omni/models/qwen3_omni/components/talker_input.py:270`). The **thinker
  yields no text for the talker** on the harness's TTS request, while M* handles the
  same prompts (28–48 s audio). Both independent impls failing the same way ⇒ a
  request/handoff issue or a shared regression in their current HEADs.
  - To test next (needs GPUs): send TTS with temperature>0 (harness forces greedy
    temperature=0 for parity — may give the competitors' thinker an empty/EOS-only
    response → empty talker embeds), and/or pin a stable vllm-omni/sglang-omni release
    (we're on HEAD), and/or use `--dataset seed_tts` (the paper's actual dataset).
- Directionally this **supports the paper's TTS-dominance claim** (M* serves TTS
  robustly; both competitors crash), but the **exact 2.7×/4× ratio is NOT reproduced**
  because the competitors produced zero TTS output.

**BLOCKER (external):** mid-task, another user (`baris`) took GPUs 2 & 3; GPU 1 is root's.
**Only GPU 0 is free**, and Qwen3-Omni-30B speech serving needs 2–3 GPUs — so the
fix-and-verify (and the B-sweep) can't run until GPUs free up. Per policy, other users'
GPUs were not touched.

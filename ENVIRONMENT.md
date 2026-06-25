# Environment changes (cross-framework Qwen3-Omni benchmarking)

This file records every environment/disk change made to benchmark M\* vs
vllm-omni vs sglang-omni on the I2T / S2T / I2S paths, so they can be reproduced
or reverted. Created 2026-06-25.

## GPU policy
- **Use only GPUs 0, 2, 3** (idle). **GPU 1 is another user's job (PID 98474,
  ~74 GB) — never touched.** All runs pin a single free GPU via `CUDA_VISIBLE_DEVICES`.

## Downloads
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` (ungated) → `/mnt/storage/timchick/hf_cache`
  (`HF_HOME`). ~60 GB. Disk: `/mnt/storage` had 1.9 TB free.

## New virtualenvs (under /mnt/storage/timchick/venvs, not /home)
- `venvs/mstar` — pre-existing project env (torch 2.9.1+cu130, transformers 5.12.1,
  flash-attn 2.8.3). Used for M\* + the encoder micro-benchmarks.
- `venvs/vllm-omni` — NEW. `uv venv --python 3.12`; `vllm==0.19.0
  --torch-backend=auto`; `vllm-omni` cloned to `/mnt/storage/timchick/vllm-omni`
  and `pip install -e .`.
- `venvs/sglang-omni` — NEW (planned). py3.11; `sglang-omni` cloned via SSH.

## flash-attn
- The mstar venv's flash-attn (2.8.3) is **kept** — M\*'s real Qwen3-Omni serving
  path and vllm/sglang all need it for representative TTFT/ITL numbers. (Earlier
  encoder micro-benchmarks deliberately *excluded* it via a `sys.modules` block;
  the serving benchmarks here do not.)

## WORKING M\* server recipe (after debugging — see history below)

```bash
# 1. one-time: generate the fast tokenizer.json the repo doesn't ship
HF_HOME=/mnt/storage/timchick/hf_cache python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('<SNAPSHOT_PATH>', trust_remote_code=True)"   # creates tokenizer.json

# 2. serve: TP=2 Thinker across 3 free GPUs, TMPDIR on storage (/ is 100% full!)
setsid env CUDA_VISIBLE_DEVICES=0,2,3 HF_HOME=/mnt/storage/timchick/hf_cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR=/mnt/storage/timchick/tmp TEMP=/mnt/storage/timchick/tmp TMP=/mnt/storage/timchick/tmp \
  mstar-serve --config configs/qwen3omni_thinker_tp2.yaml --port 8011 \
  --cache-dir /mnt/storage/timchick/hf_cache/hub \
  --socket-path-prefix /tmp/mstar_timchick_bench/ --upload-dir /tmp/mstar_uploads_timchick/ &
```

Four independent issues had to be cleared (all environment, not M\* code):
1. **Tokenizer**: transformers 5.12.1 / tokenizers 0.22.2 reject the Qwen2 *slow*
   tokenizer loaded by bare repo-id; the repo ships no `tokenizer.json`. Loading
   once by explicit snapshot path generates `tokenizer.json` → fast path works.
2. **HF cache path**: download lives at `HF_HOME/hub/...`; pass `--cache-dir
   $HF_HOME/hub` so the server's `snapshot_download(cache_dir=…)` finds it; and
   `HF_HUB_OFFLINE=1` so `_resolve_local_hf_snapshot` doesn't hang online and
   fall back to the bare repo-id.
3. **Memory**: the 2-GPU config puts the 30B Thinker (60 GB) on one GPU →
   OOM at CUDA-graph capture. Use `qwen3omni_thinker_tp2.yaml` (Thinker TP=2,
   ~30 GB/rank) across **GPUs 0,2,3** (1 is the other user's).
4. **`/tmp` is 100% full** on this box (root fs) → flashinfer's nvcc JIT fails
   with "Invalid argument". Redirect `TMPDIR/TEMP/TMP` to `/mnt/storage`.

A pre-existing stale `mstar serve` wrapper process (not matched by
`pkill -f mstar-serve`) was also masking fixes by re-emitting old errors — kill
`mstar serve` (space) AND `mstar-serve` (hyphen) AND `multiprocessing.spawn`.

### (historical) BLOCKER — M\* 30B server won't start (tokenizer/cache env bug)

Bringing up `mstar serve qwen3_omni` fails before model load with:

```
transformers/models/qwen2/tokenization_qwen2.py:62  BPE(...)
ValueError: `vocab` and `merges` must be both be from memory or both filenames
```

Root cause (diagnosed, not an M\* code bug):
- transformers **5.12.1** + tokenizers **0.22.2**: the Qwen2 *slow* tokenizer's
  BPE constructor raises this when a Qwen3-Omni snapshot (which ships **no
  `tokenizer.json`**) is loaded **by bare repo-id + `cache_dir`** or via the
  broken `use_fast=True` conversion. Loaded by the **explicit local snapshot
  path**, it works fine.
- `Qwen3OmniModel` resolves weights via `_resolve_local_hf_snapshot(repo_id,
  cache_dir)` → `snapshot_download(local_files_only=False)`. On this box that
  call **hangs/raises** (unauthenticated HF is slow), so the `except` returns the
  **bare repo-id**, which then hits the tokenizer bug. Also, the HF cache lives at
  `HF_HOME/hub/...` but `--cache-dir X` makes `snapshot_download` look at
  `X/models--...` — a *different* path.

What works **standalone** (verified):
```
HF_HOME=/mnt/storage/timchick/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('<HF_HOME>/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f...', trust_remote_code=True)"
```
i.e. `cache_dir=$HF_HOME/hub` **and** `HF_HUB_OFFLINE=1` makes `_resolve` return
the complete snapshot path and the tokenizer loads. But launching via the
`mstar serve` CLI, the offline env doesn't reach the worker subprocess, so it
goes online → hangs/raises → bare repo-id → the BPE error recurs.

Suggested fixes (for whoever continues):
1. Run `mstar-serve` directly (not the `mstar` wrapper) with
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `--cache-dir $HF_HOME/hub`, OR
2. Pin `tokenizers<0.21` / a transformers build where the slow Qwen2 BPE-from-files
   path works, OR
3. Generate a valid `tokenizer.json` (fast) in the snapshot (the in-process
   `use_fast=True` conversion is itself broken here, so do it in a clean env).

## Install state
- vllm-omni: was still building at stop time (`venvs/vllm-omni`).
- sglang-omni: install **completed** (`venvs/sglang-omni`, `/mnt/storage/timchick/sglang-omni`).
- No cross-framework serving numbers were produced — the M\* server blocker
  gates the comparison, and **no numbers were fabricated**.

## Cleanup / revert
- Remove `/mnt/storage/timchick/venvs/{vllm-omni,sglang-omni}`,
  `/mnt/storage/timchick/{vllm-omni,sglang-omni}`, and
  `/mnt/storage/timchick/hf_cache` to reclaim disk.

## Multimodal serving: blocked by tensor-transport on this node (NOT M\* code)

After all four fixes above, the M\* 30B server **loads and serves TEXT correctly**
(`/v1/chat/completions` returns proper output). But the **multimodal** paths
(I2T/S2T/I2S), which must ship encoder output tensors across workers, are blocked:

- **Default RDMA / Mooncake**: server reaches READY, text works, but image/audio
  requests 500 with `RuntimeError: Mooncake read failed. Status: -1`. Startup logs
  `Topology discovery complete. Found 0 HCAs` → no RDMA on this node; the TCP
  fallback transport is broken here.
- **`--tensor-comm-protocol SHM`** (the `mstar` wrapper's documented single-node
  default — bypassed because we ran `mstar-serve` directly): fixes the transport
  but the **Talker** partition's CUDA-graph/flashinfer-JIT warmup **hangs** (GPUs
  idle, one stuck `nvcc`, no progress for 25 min) — likely the JIT compiling to
  the slow `/mnt/storage` TMPDIR, or an SHM-path setup hang.
- A text-only config (drop Talker/Code2Wav) fails fast: the model hardcodes the
  Talker partition (`KeyError: 'Talker'`).

Net: cross-framework multimodal serving numbers were **not** produced — the M\*
server's multimodal transport doesn't work on this specific node (no HCA + 100%-
full root fs + slow-storage JIT). No numbers were fabricated. To finish on
working infra: use the `mstar` wrapper (SHM by default) on a node with a normal
`/tmp` and either RDMA HCAs or a working SHM Talker warmup, then run
`benchmark/run_omni_paths.sh SYSTEM=ours URL=…`.

## Benchmark-infra fixes made (reusable, ruff-clean)
- `benchmark/dataset.py`: `ethz/food101` now loads without `trust_remote_code`
  (newer `datasets` rejects it) — Parquet path, with fallback.
- `benchmark/runner.py`: `results.json` now persists the full TTFT / ITL / RTF /
  throughput from `AggregateMetrics` (was JCT-only) + the real `inference_system`
  tag — additive, existing consumers unaffected.

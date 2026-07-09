---
name: mstar-bench-env
description: "Environment quirks when launching mstar Qwen3-Omni servers + benchmarks in /home/tim worktrees (SHM protocol, ninja PATH, offline datasets)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 135734de-bce9-4315-a5b5-c8ae24647f89
---

Three recurring gotchas when standing up an mstar Qwen3-Omni server + benchmark.runner in a /home/tim worktree (venv at .venv, entrypoint is `mstar.api_server.entrypoint --config <yaml> --port N`, NOT `mstar.entrypoints.api_server` with a positional config):

1. **Mooncake/RDMA fails to load** — default `--tensor-comm-protocol RDMA` raises `GLIBCXX_3.4.30 not found` for mooncake/engine.so. Use `--tensor-comm-protocol SHM`. Also pass a unique `--socket-path-prefix` and `--upload-dir` so concurrent servers from other agents don't collide.
2. **ninja not on PATH** — FlashInfer JIT-compiles a kernel on the first real prefill and calls `ninja`, which lives at `.venv/bin/ninja` but isn't on PATH when you invoke `.venv/bin/python` directly. Server loads fine, then dies at first request with `FileNotFoundError: ninja`. Fix: prepend `PATH="<worktree>/.venv/bin:$PATH"`.
3. **Datasets want network** — as of 2026-07-01 HF data was consolidated onto the coriander pool to keep `/home` free: `HF_HOME=/m-coriander/coriander/hf` (the old `/home/tim/hf_datasets` no longer exists — do NOT point there). If a dataset hits a FileLock/PermissionError writing under the shared pool, point the client's HF cache at a writable subdir you own. librispeech `validation` split isn't cached and triggers a slow full HF download (the `datasets` lib also rejects `trust_remote_code` now); pre-decoded wavs still live at `/home/tim/tmp/libri_wavs`, so reuse those (kept on `/home` deliberately — small, reusable input). Keep `/home` with ~50G+ free; large data goes on `/m-coriander/coriander` (see CLAUDE.md "Disk / home free space").

**Why:** these are env/packaging issues, not code bugs; they recur every fresh worktree.
**How to apply:** bake SHM + PATH + HF_HOME into every server/bench launch command in this workspace. See [[feedback-benchmark-metrics]].

**UPDATE 2026-07-08 (measurement gotchas that wasted hours):**
- **HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets got EVICTED (down to 193K, images
  gone) → ALL i2t/s2t requests fail with server error "expected modalities ['text'], received
  []" (empty inputs).** FIX: use HF_DATASETS_CACHE=/m-coriander/coriander/hf (food101 IS cached
  at /m-coriander/coriander/hf/datasets/ethz___food101; the read-only "Ignored error writing
  commit hash [Errno 13]" is a harmless WARNING). Always verify a run has 0 "modalities" errors.
- **Heavy warmup (e.g. --num-warmup 12 --num-requests 384) can trip a data_worker KeyError race
  in the server** → use light canonical warmup (--num-warmup 2..10, --num-requests ~48-96).
- **Server churn leaves ORPHANED tim procs holding GPU memory** (lab_kill doesn't always fully
  reap) → next boot OOMs on top of zombies (OOM referencing a visible-device index). After
  killing, VERIFY `nvidia-smi -i <gpus>` <200MB; if not, reap: for each gpu, kill -9 the
  compute-app pids that are user=tim. NEVER kill other users' procs (e.g. naomi on GPUs 4,5).
- **Shared box:** GPUs 0 and 4,5 are other users' — only use free NUMA-local pairs (2,3 node0;
  6,7 node1). numactl --cpunodebind must match the pair's NUMA node (2,3→0, 6,7→1). Under
  contention boots stretch to 20-30min (from ~7-10). Delegating long boots to subagents burns
  their turn limits; orchestrate long-boot A/Bs from the persistent main loop instead.

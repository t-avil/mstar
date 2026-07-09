---
name: project_mstar_talker_code2wav_colocation
description: "Talker+Code2Wav colocation can't be done by a yaml; the default already colocates them; fusing the codec edge needs a code change"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

Config-only colocation of Talker + Code2Wav (to kill per-frame IPC on the
codec_tokens streaming edge, paper §4.2) is a NO-OP in M*. Three reasons, all
verified on CPU 2026-07-03 while building the speech host-floor bundle
(opt/speech-floor):

1. M* spawns ONE process per rank (`conductor._launch_workers`). worker_id =
   f"worker_{rank}". Every worker graph on a rank shares that one process.
2. The default `qwen3omni_2gpu.yaml` (the qwen3_omni default in cli DEFAULT_CONFIGS)
   already puts Talker AND Code2Wav on rank 0 -> already one process -> the
   codec_tokens edge is already intra-process.
3. `_divide_into_worker_graphs` (mstar/model/base.py) forces every
   `consumes_stream` node into its OWN worker graph (the `not ...consumes_stream`
   guards in the Sequential/Parallel merge). Both Talker and Code2Wav are
   streaming consumers, so no node_groups yaml can fuse them into one worker
   graph. `configs/qwen3omni_2gpu_taco.yaml` builds a byte-identical worker-graph
   decomposition to `qwen3omni_2gpu.yaml`.

**Why:** the campaign "colocation config" item (#15) assumed separate node_groups
on the same rank = separate workers with cross-worker IPC. They're the same
process already.
**How to apply:** to actually remove the residual codec-edge cost, do a CODE
change (fused in-graph Talker->Code2Wav, or make Code2Wav non-streaming), not a
yaml. To MEASURE whether the edge costs anything, force a split across ranks
(Talker rank0 / Code2Wav rank1) and A/B vs colocated — see SMOKE.md on the branch.

Related: [[project_mstar_bench_env]], [[feedback_mstar_worktree_pythonpath]].

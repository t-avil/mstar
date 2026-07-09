---
name: benchmark-full-metrics
description: "All benchmarks must capture TTFT, ITL, RTF, JCT, and throughput — never run with the old runner that only reports JCT"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 135734de-bce9-4315-a5b5-c8ae24647f89
---

Every benchmark run (M*-old, M*-new, vLLM — all systems) must capture the full metric set: TTFT, ITL, RTF, JCT, and throughput (both text token/s and audio sec/s where applicable). All individual datapoints in JSON.

**Why:** The user expects apples-to-apples comparison across all systems. The old runner at upstream main only reports JCT and req/s. Previous runs wasted time producing incomplete data.

**How to apply:** When benchmarking against old server code (e.g. upstream main), copy the updated benchmark/runner.py, base.py, dataset.py from the optimization branch into the worktree. The runner instruments TTFT/ITL client-side from streaming timing — it doesn't need server-side changes. Always verify after B=1 that results.json contains non-None ttft, itl, and throughput fields.

---
name: mstar-new-label-convention
description: mstar_new on benchmark branch = current shipping-candidate build; label moves forward as optimizations land
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 135734de-bce9-4315-a5b5-c8ae24647f89
---

The `mstar_new` system label on the benchmark branch always represents the current best/shipping-candidate build. When a new optimization proves out, mstar_new absorbs it — the label stays, the build advances. Old data gets overwritten (commit history tracks previous runs).

**Why:** Keeps the benchmark branch a clean 3-way comparison (mstar_new vs mstar_old vs vllm) without accumulating variant labels like mstar_new_chunked, mstar_new_gpumel, etc.

**How to apply:** When a new optimization is validated (e.g. chunked prefill), run a full sweep with the combined flags, label it `mstar_new`, and commit to the bench branch — replacing the previous mstar_new data. Record the exact build in provenance metadata (git_commit, flags) for traceability. Related: [[benchmark-full-metrics]].

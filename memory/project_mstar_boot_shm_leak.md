---
name: project_mstar_boot_shm_leak
description: "M* server boot OOM at set_device is an orphaned /dev/shm leak (host RAM), NOT a loader/GPU race"
metadata: 
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

M* servers fail to boot with `worker_N failed to initialize: CUDA error: out of
memory` at `torch.cuda.set_device(self.device)` (`worker.py:448`) **on an empty
143GB GPU**. This is NOT a GPU-capacity or rank-placement race (HANDOFF_V8
misdiagnosed it and floated a node reboot). Real cause: **host-RAM exhaustion**.
CUDA context creation needs pinned host memory; the launcher pins
`numactl --membind=1`, and GPUs 6,7's sysfs `numa_node` reads empty so the
launcher fallback forces node 1 — the starved node.

Root: the SHM tensor-comm protocol leaves **orphaned /dev/shm segments** from every
dead server. Seen 2026-07-06: tim owned **477k /dev/shm entries = 289 GB**
(`torch_*`, `cuda.shm.*`, `shm_chatcmpl-*_lockfile.lock`), host free 46G, swap 100%.

**Fix (no reboot):** reap tim-owned dead segments only:
`find /dev/shm -maxdepth 1 -user tim -exec rm -rf {} +` (use rm -rf, not -delete —
non-empty dir segments survive -delete) plus `rm -rf /home/tim/tmp/sk_lab_* sk_qb_*`.
Result: /dev/shm 295G→3G, host free 46G→343G, node1 free 14.6G→243G, boots fine.
NEVER touch other users' /dev/shm (atindra/naomi/stephenduan/etc).

**Preflight before any boot:** `df -h /dev/shm` and
`find /dev/shm -maxdepth 1 -user tim | wc -l`; if huge, reap first. See
[[project_mstar_decode_bottleneck]], [[project_mstar_bench_env]].

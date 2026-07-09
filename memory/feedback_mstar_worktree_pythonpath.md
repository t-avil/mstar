---
name: mstar-worktree-pythonpath
description: "mstar worktrees need PYTHONPATH=<worktree> on every launch — editable install + spawn silently loads a DIFFERENT worktree's code, invalidating A/B benchmarks"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 135734de-bce9-4315-a5b5-c8ae24647f89
---

When running an mstar server/benchmark from a worktree under /home/tim (venv symlinked from /home/tim/mstar-encoders/.venv), ALWAYS set `PYTHONPATH=<that-worktree-abs-path>` on every server launch command.

**Why:** the shared `.venv` has an editable install whose finder (`__editable___mstar_0_1_0_finder.py`) hard-maps `mstar` → a fixed directory (observed: `/home/tim/exp/combined-wt/mstar`). The api_server parent process picks up the cwd's `mstar/` via sys.path[0], but the conductor + workers are launched with `mp.get_context("spawn")`, and a spawned process resets sys.path[0] to the *spawn bootstrap script dir*, not the cwd — so the editable finder wins and the GPU workers silently load the WRONG worktree's engine code. An A/B I ran without PYTHONPATH was invalid: both baseline and `MSTAR_ASYNC_ENCODER=1` executed identical combined-wt code, so the flag did nothing and the diff was pure noise. A teammate caught it. `PYTHONPATH=<worktree>` makes the spawned child resolve to the worktree (verified: spawn-test child reported the worktree path + the new symbol present).

**How to apply:** prepend `PYTHONPATH=/home/tim/tmp/<worktree>` (alongside the SHM/PATH/HF_HOME flags in [[mstar-bench-env]]). To CERTIFY the right code ran, add a one-time log on the modified path and grep the server log for it after the run — don't trust that the symlinked venv loaded your edits. Also: spawned mstar workers + torch inductor compile_worker children survive parent death and `kill -- -PGID` (different process groups); clean up with `pkill -9 -f "<worktree>/.venv/bin/python"` and then kill any lingering pid still holding GPU memory.

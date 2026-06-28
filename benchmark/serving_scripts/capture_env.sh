#!/usr/bin/env bash
# capture_env.sh <out_dir> [WORKTREE]
# Programmatic environment capture per /home/tim/CLAUDE.md "Environment capture".
# Records OS, GPU/driver, all three CUDA versions (driver-max, nvcc toolkit, the
# CUDA torch was built against), packages, and git state into <out_dir>/env.txt
# (+ requirements.txt). Never hand-written -- always regenerated per run.
set -uo pipefail
d="${1:?usage: capture_env.sh <out_dir> [worktree]}"
WT="${2:-$PWD}"
mkdir -p "$d"
{
  echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
  echo "=== uname ==="; uname -a
  echo "=== os-release ==="; cat /etc/os-release 2>/dev/null || true
  echo "=== CUDA_VISIBLE_DEVICES ==="; echo "${CUDA_VISIBLE_DEVICES:-unset}"
  echo "=== MSTAR optim env ==="
  env | grep -E '^MSTAR_' | sort || echo "(none)"
  echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
  echo "=== nvidia-smi query ==="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,persistence_mode \
    --format=csv 2>/dev/null || true
  echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "=== torch cuda ==="
  python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('cudnn',torch.backends.cudnn.version())" 2>/dev/null || echo "no torch"
  echo "=== git (worktree $WT) ==="
  git -C "$WT" rev-parse --abbrev-ref HEAD 2>/dev/null || true
  git -C "$WT" rev-parse HEAD 2>/dev/null || true
  git -C "$WT" status --short 2>/dev/null || true
} > "$d/env.txt" 2>&1

if command -v uv >/dev/null && [ -f "$WT/uv.lock" ]; then
  uv pip freeze > "$d/requirements.txt" 2>/dev/null || pip freeze > "$d/requirements.txt" 2>/dev/null || true
else
  pip freeze > "$d/requirements.txt" 2>/dev/null || true
fi
echo "captured env -> $d/env.txt"

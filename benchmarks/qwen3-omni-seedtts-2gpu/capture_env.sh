#!/usr/bin/env bash
# capture_env.sh <session_dir>
# CLAUDE.md environment capture: OS, GPU, driver, all CUDA versions, packages, git.
set -euo pipefail
d="$1"
mkdir -p "$d"
{
  echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
  echo "=== uname ==="; uname -a
  echo "=== os-release ==="; cat /etc/os-release 2>/dev/null || true
  echo "=== CUDA_VISIBLE_DEVICES ==="; echo "${CUDA_VISIBLE_DEVICES:-unset}"
  echo "=== nvidia-smi ==="; nvidia-smi 2>/dev/null || echo "no nvidia-smi"
  echo "=== nvidia-smi query ==="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,persistence_mode \
    --format=csv 2>/dev/null || true
  echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "=== torch cuda ==="
  python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('cudnn',torch.backends.cudnn.version())" 2>/dev/null || echo "no torch"
  echo "=== git ==="; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo "=== clocks/persistence note ==="
  echo "persistence_mode is system-wide On (set by box admins, not this benchmark)."
  echo "Paper Figure 5 does not lock clocks; no clock lock applied here. Clocks recorded as-is (unlocked)."
} > "$d/env.txt" 2>&1

if command -v uv >/dev/null && [ -f uv.lock ]; then
  uv pip freeze > "$d/requirements.txt" 2>/dev/null || pip freeze > "$d/requirements.txt"
else
  pip freeze > "$d/requirements.txt" 2>/dev/null || echo "(no pip freeze available yet)" > "$d/requirements.txt"
fi
echo "captured env -> $d/env.txt"

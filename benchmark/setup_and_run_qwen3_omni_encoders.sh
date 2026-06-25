#!/usr/bin/env bash
#
# From-zero setup + run for the Qwen3-Omni native-encoder benchmark.
#
# Builds a fresh, isolated Python 3.12 environment, installs the Qwen3-Omni
# runtime the way the project documents it (docs/installation.rst), and runs
# benchmark/qwen3_omni_encoders.py on a GPU that is actually free — never one
# already busy with someone else's work.
#
# Deliberately does NOT install flash-attn. The benchmark forces the SDPA
# attention path (flash-attn is blocked at import), so the native-vs-HF
# comparison is apples-to-apples and the setup stays simple/portable. To compare
# the production flash-attn path instead, install the wheel per
# docs/installation.rst ("flash-attn (Qwen3-Omni)") and drop the sys.modules
# block at the top of the benchmark.
#
# Usage:
#   benchmark/setup_and_run_qwen3_omni_encoders.sh                 # auto-pick a free GPU
#   GPU=3 benchmark/setup_and_run_qwen3_omni_encoders.sh           # force a specific GPU
#   VENV=/tmp/mstar-bench benchmark/setup_and_run_qwen3_omni_encoders.sh
#   SKIP_INSTALL=1 benchmark/setup_and_run_qwen3_omni_encoders.sh  # reuse an existing VENV
#
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/benchmark) ----------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV="${VENV:-${REPO_ROOT}/.venv-bench}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
OUT="${OUT:-benchmark/artifacts}"
# A GPU is "free" if it uses <= this many MiB and <= this %% utilization.
MAX_MEM_MIB="${MAX_MEM_MIB:-2000}"
MAX_UTIL="${MAX_UTIL:-10}"

# --------------------------------------------------------------------------- #
# 1. Pick a free GPU (respect other users — never grab a busy card)
# --------------------------------------------------------------------------- #
pick_free_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found." >&2; exit 1; }
  local free=()
  while IFS=',' read -r idx util mem _; do
    idx="${idx// /}"; util="${util// /}"; mem="${mem// /}"
    if [[ "${util}" -le "${MAX_UTIL}" && "${mem}" -le "${MAX_MEM_MIB}" ]]; then
      free+=("${idx}")
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  echo "Free GPUs (<=${MAX_MEM_MIB} MiB, <=${MAX_UTIL}%% util): ${free[*]:-none}" >&2
  [[ ${#free[@]} -gt 0 ]] || { echo "ERROR: no free GPU. Set GPU=<idx> to override." >&2; exit 1; }
  echo "${free[0]}"
}

if [[ -n "${GPU:-}" ]]; then
  GPU_IDX="${GPU}"
  echo "Using caller-pinned GPU ${GPU_IDX}."
else
  GPU_IDX="$(pick_free_gpu)"
  echo "Auto-selected free GPU ${GPU_IDX}."
fi

# --------------------------------------------------------------------------- #
# 2. Build the environment (uv if available, else python venv + pip)
# --------------------------------------------------------------------------- #
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Creating env with uv at ${VENV} ..."
    uv venv --python 3.12 --seed "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    # --torch-backend=auto pins the torch 2.9 build matching this box's CUDA.
    # .[qwen3_omni] pulls transformers / flashinfer / sgl-kernel / etc.
    # flash-attn is intentionally omitted (see header).
    UV_TORCH_BACKEND=auto uv pip install -e ".[qwen3_omni]"
    uv pip install matplotlib
  else
    echo "uv not found; falling back to python -m venv + pip at ${VENV} ..."
    python3.12 -m venv "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    pip install --upgrade pip
    pip install -e ".[qwen3_omni]"
    pip install matplotlib
  fi
else
  echo "SKIP_INSTALL=1 — reusing existing env at ${VENV}."
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

# --------------------------------------------------------------------------- #
# 3. Run the benchmark on the chosen free GPU only
# --------------------------------------------------------------------------- #
echo "Running benchmark on GPU ${GPU_IDX} (SDPA backend, no flash-attn) ..."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="${GPU_IDX}" python -m benchmark.qwen3_omni_encoders \
  --device cuda:0 --iters "${ITERS}" --warmup "${WARMUP}" --out "${OUT}"

echo
echo "Artifacts written to ${OUT}/ :"
ls -1 "${OUT}"/qwen3_omni_* 2>/dev/null || true

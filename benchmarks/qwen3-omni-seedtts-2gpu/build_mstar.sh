#!/usr/bin/env bash
# Staged M* install for Qwen3-Omni on a CUDA 12.8 box (torch 2.9 / py3.12 / cu12).
# Staged because `.[all]` backtracks to numba 0.53.1 (broken on py3.12).
set -uo pipefail
cd /home/tim/mstar
export PATH="$HOME/.local/bin:$PATH"
export UV_TORCH_BACKEND=auto
source .venv/bin/activate
ts(){ date -u +%Y%m%dT%H%M%SZ; }

echo "[$(ts)] STEP 1: core install (-e .)"
uv pip install --torch-backend=auto -e . || { echo "CORE FAILED"; exit 11; }
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" || { echo "TORCH IMPORT FAILED"; exit 12; }

echo "[$(ts)] STEP 2: qwen3_omni extra"
if ! uv pip install --torch-backend=auto -e ".[qwen3_omni]"; then
  echo "[$(ts)] qwen3_omni failed (likely numba backtrack); retry with modern numba/llvmlite floor"
  uv pip install --torch-backend=auto -e ".[qwen3_omni]" "numba>=0.60" "llvmlite>=0.43" || { echo "QWEN3_OMNI EXTRA FAILED"; exit 13; }
fi

echo "[$(ts)] STEP 3: flash-attn prebuilt wheel (cu12 / torch2.9 / cp312)"
CU=$(python -c "import torch;print(torch.version.cuda)")
echo "torch.version.cuda=$CU"
if [[ "$CU" == 12.* ]]; then
  uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" || { echo "FLASH-ATTN FAILED"; exit 14; }
else
  uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" || { echo "FLASH-ATTN FAILED"; exit 14; }
fi

echo "[$(ts)] VERIFY imports"
python - <<'PY'
import importlib
for m in ["torch","mstar","flash_attn","flashinfer","sgl_kernel","qwen_omni_utils","transformers","datasets"]:
    try:
        importlib.import_module(m); print("OK ",m)
    except Exception as e:
        print("FAIL",m,"->",repr(e)[:120])
PY
which mstar && mstar --help >/dev/null 2>&1 && echo "mstar CLI OK"
echo "[$(ts)] DONE M* build (rc=$?)"

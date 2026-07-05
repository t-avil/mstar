#!/usr/bin/env bash
# One-command launch of the BEST M* build (2026-07-05 iteration close).
# Build: opt/prep-h2d @620de91 = opt/stack-n2 (merge-config + cfgv2 + checkstop)
#        + prep-h2d fix. Env adds CDT-tuned Inductor cache.
# Numbers vs committed vLLM refs: B1 1.21x, B2 1.08x, B4 1.11x, B8 1.22x,
# B16 1.10x, B32 8.37 mean/8.72 peak (band 8.03-8.50); s2t/s2s/i2s all won.
# Usage: launch_mstar_best.sh [gpus=6,7] [port=8321] [ttl_h=8]
set -uo pipefail
GPUS=${1:-6,7}; PORT=${2:-8321}; TTL=${3:-8}
exec bash /m-coriander/coriander/tim/lab_server.sh best /m-coriander/coriander/tim/mstar-b1fix "TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache_cdt TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHDYNAMO_CACHE_SIZE_LIMIT=128 MSTAR_CUSTOM_OPS=1 MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 MSTAR_PREFILL_CHUNK_TOKENS=512 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_SAMPLER_CFG_CACHE_V2=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1 MSTAR_EMIT_SIDECAR=1 MSTAR_SIDECAR_CHECKSTOP=1 MSTAR_MIXED_BUDGET_TOKENS=512 MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_MERGED_PREFILL=1 MSTAR_PREP_DEVICE_POS=1" configs/qwen3omni_2gpu_encoff.yaml "$GPUS" "$PORT" "$TTL"

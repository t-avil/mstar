#!/usr/bin/env bash
# CANDIDATE DELIVERABLE: godv9 build (full host-floor/MoE flag stack) on the
# SPEECH-FRIENDLY base topology (configs/qwen3omni_2gpu.yaml: rank1=encoders+
# Thinker, rank0=Talker+Code2Wav alone). Hypothesis: text tok/s parity (levers
# are Thinker-scoped) AND speech 2-3x (Talker/Code2Wav GPU uncontended).
# Same flag stack as launch_mstar_best.sh; only the config yaml differs.
# Usage: launch_mstar_base_fullstack.sh [gpus=6,7] [port=8322] [ttl_h=6]
set -uo pipefail
GPUS=${1:-6,7}; PORT=${2:-8322}; TTL=${3:-6}
exec bash /m-coriander/coriander/tim/lab_server.sh basefs /m-coriander/coriander/tim/mstar-godv9 \
"TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_CACHE_DIR=/m-coriander/coriander/tim/inductor_cache_cdt TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHDYNAMO_CACHE_SIZE_LIMIT=128 MSTAR_CUSTOM_OPS=1 MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 MSTAR_PREFILL_CHUNK_TOKENS=512 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_SAMPLER_CFG_CACHE_V2=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1 MSTAR_EMIT_SIDECAR=1 MSTAR_SIDECAR_CHECKSTOP=1 MSTAR_MIXED_BUDGET_TOKENS=512 MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_MERGED_PREFILL=1 MSTAR_PREP_DEVICE_POS=1 MSTAR_PREP_DEVICE_POS_BATCHED=1" \
configs/qwen3omni_2gpu.yaml "$GPUS" "$PORT" "$TTL"

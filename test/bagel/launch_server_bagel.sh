#!/bin/bash

WHO=naomi

# CACHE_DIR=/mnt/storage/$WHO/mminf/bagel/
CACHE_DIR=/m-coriander/coriander/$WHO/mminf_cache/bagel/
DEVICES=3,4,5

TENSOR_PROTOCOL=TCP # Needed for coriander!!
# TENSOR_PROTOCOL=RDMA # faster

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/bagel_cfg_parallel.yaml \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_$WHO/ \
    --upload-dir /tmp/mminf_uploads_$WHO/ \
    --port 8001 \
    --mooncake-port 8081 \
    --tensor-comm-protocol $TENSOR_PROTOCOL
    # --log-level DEBUG

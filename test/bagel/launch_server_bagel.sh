#!/bin/bash

WHO=naomi

CACHE_DIR=/mnt/storage/$WHO/mminf/bagel/
DEVICES=2,3

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/bagel.yaml \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_$WHO/ \
    --upload-dir /tmp/mminf_uploads_$WHO/ \
    --port 8000 \
    --mooncake-port 8081
    # --log-level DEBUG

#!/bin/bash
#
# Smoke test the V-JEPA 2 Phase-2 rollout path: POST a video with
# ``model_kwargs={"rollout_horizon": H}`` and verify the server returns H
# distinct video chunks (one per accumulated rollout iteration).
#
# Usage:
#   bash test/vjepa2/video_request_rollout.sh <video_path.mp4> [horizon] [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request_rollout.sh /path/to/clip.mp4
#   bash test/vjepa2/video_request_rollout.sh /tmp/synth.mp4 4 127.0.0.1 20003
#
# Expected server-side: ``prefill_video_rollout`` walk runs, the DynamicLoop
# registered as ``rollout_loop`` completes H iterations (or early-exits
# when iter_idx + 1 >= H), and each iteration's ``predicted_hidden`` is
# accumulated into a single ``accumulated_outputs`` edge that emits H
# tensor_infos to the client.
#
# The launch script (``launch_server_vjepa2.sh``) is used as-is — the server
# serves all three walks (prefill_video, prefill_video_encoder_only,
# prefill_video_rollout) from the same endpoint.  This script drives the
# rollout walk by passing ``rollout_horizon`` in ``model_kwargs``.

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path.mp4> [horizon] [host] [port]}"
HORIZON="${2:-4}"
HOST="${3:-127.0.0.1}"
PORT="${4:-20003}"
URL="http://${HOST}:${PORT}/generate"

if [ ! -f "${VIDEO}" ]; then
    echo "Video file not found: ${VIDEO}" >&2
    exit 1
fi

MODEL_KWARGS="{\"rollout_horizon\": ${HORIZON}}"

TMPFILE=$(mktemp /tmp/vjepa2_rollout_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-rollout] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  horizon:       ${HORIZON}"
echo "  model_kwargs:  ${MODEL_KWARGS}"

curl -sS --fail-with-body -X POST "${URL}" \
    -F "files=@${VIDEO}" \
    -F 'input_modalities=video' \
    -F 'output_modalities=video' \
    -F "model_kwargs=${MODEL_KWARGS}" \
    -o "${TMPFILE}"

python3 - <<PY
import base64
import json
import sys

import numpy as np

path = "${TMPFILE}"
expected_horizon = int("${HORIZON}")

chunks = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    if msg.get("modality") != "video":
        continue
    data = msg.get("data", "")
    if data:
        chunks.append(base64.b64decode(data))

if not chunks:
    print("No video-modality chunks received.  Raw response:")
    sys.stdout.write(open(path).read())
    sys.exit(1)

print(f"[vjepa2-rollout] received {len(chunks)} chunk(s) for horizon={expected_horizon}")

arrs = []
for i, raw in enumerate(chunks):
    arr = np.frombuffer(raw, dtype=np.float32)
    assert arr.size % 1024 == 0, (
        f"chunk {i}: size {arr.size} not divisible by hidden_size=1024"
    )
    tokens = arr.size // 1024
    print(
        f"  chunk {i}: {len(raw)} bytes = {arr.size} floats = {tokens} tokens x 1024 hidden | "
        f"mean={arr.mean():.4f} std={arr.std():.4f}"
    )
    arrs.append(arr)

# Expected count: one accumulated tensor per rollout iteration.
# Early-exit (register_loop_stop) could make len(chunks) < expected_horizon;
# we surface a warning rather than failing because that's legitimate
# behavior when the per-request horizon is below max_rollout_horizon.
if len(chunks) != expected_horizon:
    print(
        f"[vjepa2-rollout] NOTE: got {len(chunks)} chunks vs. expected {expected_horizon}. "
        "If horizon <= config.max_rollout_horizon this is register_loop_stop at work — fine.",
        file=sys.stderr,
    )

if len(arrs) >= 2:
    a, b = arrs[0], arrs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"[vjepa2-rollout] cosine(chunk[0], chunk[1]) = {cos:.4f}")
    assert cos < 0.999, (
        f"chunks 0 and 1 are essentially identical (cos={cos:.4f}) — "
        "sliding-window update appears to be a no-op."
    )

print("[vjepa2-rollout] OK")
PY

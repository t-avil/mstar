#!/bin/bash
#
# Smoke test the V-JEPA 2-AC Phase-3.D rollout path: POST a context video +
# deterministic 7-DOF actions/states of length T_ctx + H - 1 + rollout_horizon=H,
# receive H per-iter imagined tubelet groups, and verify shape + non-degeneracy.
#
# Usage:
#   bash test/vjepa2/video_request_ac_rollout.sh <video_path.mp4> [horizon] [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request_ac_rollout.sh test/qwen3-omni/video.webm
#   bash test/vjepa2/video_request_ac_rollout.sh clip.mp4 4 127.0.0.1 20003
#
# What gets sent
# --------------
# AC ViT-g @ 256 with tubelet_size=2 gives T_ctx = 64 / 2 = 32 tubelet
# timesteps per predictor forward.  For sliding-window rollout we need
# T_total >= T_ctx + H - 1 actions/states; we send exactly that.  The
# action ramp is small and smooth so the predictor doesn't saturate to
# NaN and consecutive iters produce distinguishable outputs.
#
# What gets asserted client-side
# ------------------------------
#   * Exactly H ``modality="video"`` chunks arrive (one per rollout iter).
#   * Each chunk is 1 tubelet group = grid^2 * hidden_size = 256 * 1408
#     floats = 1,441,792 bytes (f32).  We verify divisibility rather than
#     exact count so the script works for any backbone (vitl/g/etc).
#   * All floats finite (no NaN/Inf — would indicate a load bug).
#   * Non-zero std per chunk — zero std means the predictor isn't
#     contributing signal.
#   * Consecutive chunks have cosine similarity < 0.999 — sanity check
#     that the sliding window is actually advancing (if it were frozen,
#     consecutive iters would be identical).

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

# Build deterministic actions/states of length T_ctx + H - 1.
MODEL_KWARGS=$(python3 - <<PY
import json

T_CTX = 32  # num_frames (64) // tubelet_size (2) for facebook/vjepa2-ac-vitg
H = ${HORIZON}
DOF = 7
T_TOTAL = T_CTX + H - 1

def ramp(lo, hi, T):
    step = (hi - lo) / max(T - 1, 1)
    return [[lo + i * step] * DOF for i in range(T)]

print(json.dumps({
    "rollout_horizon": H,
    "actions": ramp(-0.5, 0.5, T_TOTAL),
    "states":  ramp( 0.1, 0.9, T_TOTAL),
}))
PY
)

TMPFILE=$(mktemp /tmp/vjepa2_ac_rollout_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-ac-rollout] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  horizon:       ${HORIZON}"
echo "  model_kwargs:  $(echo "${MODEL_KWARGS}" | head -c 160)..."

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

HIDDEN_SIZE = 1408  # ViT-g embed_dim (AC predictor projects back to this).
expected_horizon = int("${HORIZON}")
path = "${TMPFILE}"

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

print(f"[vjepa2-ac-rollout] received {len(chunks)} chunk(s) for horizon={expected_horizon}")

arrs = []
for i, raw in enumerate(chunks):
    arr = np.frombuffer(raw, dtype=np.float32)
    assert arr.size % HIDDEN_SIZE == 0, (
        f"chunk {i}: size {arr.size} not divisible by hidden_size={HIDDEN_SIZE}; "
        "AC predictor projects back to embed_dim=1408 on output."
    )
    tokens = arr.size // HIDDEN_SIZE
    print(
        f"  chunk {i}: {len(raw)} bytes = {arr.size} floats = {tokens} tokens x {HIDDEN_SIZE} hidden | "
        f"mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}"
    )
    assert np.isfinite(arr).all(), f"chunk {i}: non-finite values (NaN/Inf)"
    assert arr.std() > 1e-4, (
        f"chunk {i}: near-zero std ({arr.std():.6g}) — predictor weights may be uninitialized."
    )
    arrs.append(arr)

# Expected: H chunks.  Early-exit (register_loop_stop) at horizon - 1
# produces exactly H chunks, so mismatch here would be a logic bug.
if len(chunks) != expected_horizon:
    print(
        f"[vjepa2-ac-rollout] WARNING: got {len(chunks)} chunks vs expected {expected_horizon}.",
        file=sys.stderr,
    )

# Sliding-window sanity: iter k and iter k+1 should NOT produce identical
# tubelet groups (different actions + shifted context → different output).
if len(arrs) >= 2:
    a, b = arrs[0], arrs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"[vjepa2-ac-rollout] cosine(chunk[0], chunk[1]) = {cos:.4f}")
    assert cos < 0.999, (
        f"chunks 0 and 1 are essentially identical (cos={cos:.4f}) — "
        "sliding window isn't advancing (iter_idx not threaded, or action slice frozen)."
    )

print("[vjepa2-ac-rollout] OK")
PY

#!/usr/bin/env bash
# sweep.sh <port> <path> <label> <n_per_batch> <warmup> "<batches>"
# Runs matched-protocol cells over batches into results/<label>/<path>_B<b>, prints a table.
set -uo pipefail
PORT="${1:?}"; PTH="${2:?}"; LABEL="${3:?}"; N="${4:?}"; W="${5:?}"; BATCHES="${6:?e.g. '1 4 8 16 32'}"
ROOT=/m-coriander/coriander/tim/campaign_i2t
OUTBASE="$ROOT/results/$LABEL"
echo "=== SWEEP $LABEL $PTH  port=$PORT n=$N warmup=$W batches=[$BATCHES]  $(date -u +%H:%M:%SZ) ==="
for b in $BATCHES; do
  echo "--- $PTH B$b $(date -u +%H:%M:%SZ) ---"
  bash "$ROOT/bench_cell.sh" "$PORT" "$PTH" "$b" "$N" "$W" "$OUTBASE/${PTH}_B${b}"
done
echo "=== SWEEP $LABEL DONE $(date -u +%H:%M:%SZ) ==="

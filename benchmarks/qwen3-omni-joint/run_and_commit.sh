#!/usr/bin/env bash
# run_and_commit.sh — Full automated pipeline: sweep → ingest → charts → commit → push
#
# Usage:
#   ./run_and_commit.sh --system mstar_new --gpus 5,6 --port 8162 \
#       --paths s2t,i2t,s2s,i2s \
#       --flags "MSTAR_GPU_MEL=1 MSTAR_CHUNKED_PREFILL=1 ..." \
#       --worktree /home/tim/path/to/code \
#       --sweep-output /home/tim/tmp/sweep_mnew_v2
#
# What it does:
#   1. Runs benchmark/sweep.sh (launches server, runs all paths×batches, tears down)
#   2. Runs ingest_sweep.py (upserts results into raw_*.json)
#   3. Runs make_proof_charts.py (regenerates all charts from raw JSON)
#   4. Runs make_numbers.py (regenerates NUMBERS.md)
#   5. Commits and pushes to bench/qwen3-omni-joint + merges into benchmarks
#
# Prerequisites:
#   - Must be run from the bench-wt (benchmarks) or bench-sweep-wt worktree
#   - benchmark/sweep.sh must exist in the --worktree
#   - Python with matplotlib available at PYTHON path below
set -euo pipefail

PYTHON="/home/tim/mstar-encoders/.venv/bin/python"
BENCH_DIR="benchmarks/qwen3-omni-joint"

# ── Parse args (pass through to sweep.sh, plus our extras) ──
SYSTEM="" GPUS="" PORT="" PATHS="" FLAGS="" WORKTREE="" SWEEP_OUTPUT=""
SKIP_SWEEP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --system)       SYSTEM="$2"; shift 2;;
        --gpus)         GPUS="$2"; shift 2;;
        --port)         PORT="$2"; shift 2;;
        --paths)        PATHS="$2"; shift 2;;
        --flags)        FLAGS="$2"; shift 2;;
        --worktree)     WORKTREE="$2"; shift 2;;
        --sweep-output) SWEEP_OUTPUT="$2"; shift 2;;
        --skip-sweep)   SKIP_SWEEP=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -z "$SYSTEM" ]] && { echo "ERROR: --system required"; exit 1; }
[[ -z "$PATHS" ]]  && { echo "ERROR: --paths required"; exit 1; }
SWEEP_OUTPUT="${SWEEP_OUTPUT:-/home/tim/tmp/sweep_${SYSTEM}_$(date -u +%Y%m%dT%H%M%S)}"

echo "========================================"
echo "run_and_commit.sh"
echo "  system: $SYSTEM"
echo "  paths: $PATHS"
echo "  sweep output: $SWEEP_OUTPUT"
echo "  skip sweep: $SKIP_SWEEP"
echo "========================================"

# ── Step 1: Run sweep ──
if [[ "$SKIP_SWEEP" -eq 0 ]]; then
    [[ -z "$GPUS" ]]     && { echo "ERROR: --gpus required for sweep"; exit 1; }
    [[ -z "$PORT" ]]     && { echo "ERROR: --port required for sweep"; exit 1; }
    [[ -z "$WORKTREE" ]] && { echo "ERROR: --worktree required for sweep"; exit 1; }

    echo ""
    echo "[$(date -u +%H:%M:%S)] Step 1: Running sweep..."
    bash "$WORKTREE/benchmark/sweep.sh" \
        --system "$SYSTEM" \
        --gpus "$GPUS" \
        --port "$PORT" \
        --paths "$PATHS" \
        --worktree "$WORKTREE" \
        --output "$SWEEP_OUTPUT" \
        ${FLAGS:+--flags "$FLAGS"}
    echo "[$(date -u +%H:%M:%S)] Sweep complete."
else
    echo "[$(date -u +%H:%M:%S)] Step 1: Skipping sweep (--skip-sweep)"
fi

# ── Step 2: Ingest ──
echo ""
echo "[$(date -u +%H:%M:%S)] Step 2: Ingesting results..."
"$PYTHON" "$BENCH_DIR/ingest_sweep.py" \
    --sweep-dir "$SWEEP_OUTPUT" \
    --system "$SYSTEM" \
    --paths "$PATHS" \
    --raw-dir "$BENCH_DIR"

# ── Step 3: Charts ──
echo ""
echo "[$(date -u +%H:%M:%S)] Step 3: Generating charts..."
"$PYTHON" "$BENCH_DIR/make_proof_charts.py" "$BENCH_DIR" "$BENCH_DIR/charts"

# ── Step 4: NUMBERS.md ──
echo ""
echo "[$(date -u +%H:%M:%S)] Step 4: Generating NUMBERS.md..."
if [[ -f "$BENCH_DIR/make_numbers.py" ]]; then
    "$PYTHON" "$BENCH_DIR/make_numbers.py" "$BENCH_DIR"
else
    echo "  (no make_numbers.py — skip)"
fi

echo ""
echo "========================================"
echo "Pipeline complete for $SYSTEM"
echo "  Raw JSON updated in $BENCH_DIR/"
echo "  Charts in $BENCH_DIR/charts/"
echo ""
echo "To commit and push:"
echo "  git add $BENCH_DIR/"
echo "  git commit -m 'bench(qwen3-omni): $SYSTEM sweep'"
echo "  git push fork bench/qwen3-omni-joint"
echo "========================================"

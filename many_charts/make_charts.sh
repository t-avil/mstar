#!/usr/bin/env bash
# One command: re-extract every committed data series and render all charts.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/tim/mstar-encoders/.venv/bin/python
$PY extract_data.py
$PY render_charts.py
echo "done — charts/ updated"

#!/usr/bin/env bash
# Regenerate Figures 3-6 and Tables 4-10 of the manuscript from the result files.
# Usage: bash reproduce.sh <results_revision> <results_blockchain> <load-test results> <optimized load-test results> [output folder]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM="${1:-results_revision}"
BC="${2:-blockchain_run/results_blockchain}"
LOAD="${3:-loadtest_run/results_blockchain}"
FAST="${4:-fasttest_run/results_blockchain}"
OUT="${5:-reproduced}"
PY="${PYTHON:-python}"
$PY "$HERE/make_figures.py" --sim "$SIM" --bc "$BC" --out "$OUT/figures"
$PY "$HERE/make_tables.py"  --sim "$SIM" --bc "$BC" --load "$LOAD" --fast "$FAST" --out "$OUT/tables"
echo "Done: $OUT/figures and $OUT/tables"

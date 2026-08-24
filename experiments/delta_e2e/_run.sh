#!/bin/bash
# ============================================================
# Delta-Checkpoint E2E Verification (T4–T6)
# ============================================================
# Usage:
#   bash _run.sh [DEVICE_ID] [--steps N] [--tests T4,T5,T6]
#
# Requires a narrowly scoped passwordless sudo rule, or an already valid
# sudo credential. The script never reads or stores a password.
# ============================================================

set -euo pipefail

DEVICE_ID="${1:-1}"
shift 1 2>/dev/null || true
EXTRA_ARGS=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON="${NPU_NVME_PYTHON:-python}"
ASCEND_SETUP="${ASCEND_SETUP:-/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash}"

OUTPUT_DIR="$REPO/experiments/output/delta_e2e"
SCRIPT="$REPO/experiments/delta_e2e/delta_e2e.py"

echo "============================================================"
echo "Delta-Checkpoint E2E (T4–T6)"
echo "  Device: $DEVICE_ID  |  Args: ${EXTRA_ARGS[*]:-none}"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

set +u
source "$ASCEND_SETUP"
set -u
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"

sudo -n --preserve-env=PATH,LD_LIBRARY_PATH,PYTHONPATH,ASCEND_HOME_PATH \
    "$PYTHON" "$SCRIPT" --device-id "$DEVICE_ID" "${EXTRA_ARGS[@]}"
RC=$?

echo ""
echo "Done (exit=$RC). Results:"
echo "  $OUTPUT_DIR/delta_e2e.json"
exit $RC

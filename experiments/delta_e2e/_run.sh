#!/bin/bash
# ============================================================
# Delta-Checkpoint E2E Verification (T4–T6)
# ============================================================
# Usage:
#   bash _run.sh [DEVICE_ID] [--steps N] [--tests T4,T5,T6]
#
# Requires: echo "your-sudo-pw" > /home/user7/npu-nvme/.sudo_pw
# ============================================================

set -euo pipefail

DEVICE_ID="${1:-1}"
shift 1 2>/dev/null || true

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

PW_FILE="$REPO/.sudo_pw"
if [ ! -f "$PW_FILE" ]; then
    echo "ERROR: sudo password file not found at $PW_FILE"
    echo "  Create it: echo 'your-sudo-pw' > $PW_FILE"
    exit 1
fi
SUDO_PW="$(cat "$PW_FILE" | tr -d '\n')"

OUTPUT_DIR="$REPO/experiments/output/delta_e2e"
SCRIPT="$REPO/experiments/delta_e2e/delta_e2e.py"

echo "============================================================"
echo "Delta-Checkpoint E2E (T4–T6)"
echo "  Device: $DEVICE_ID"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  export PYTHONPATH=$REPO/python:\$PYTHONPATH && \
  $PYTHON $SCRIPT --device-id $DEVICE_ID $@
"

echo ""
echo "Done. Results:"
echo "  $OUTPUT_DIR/delta_e2e.json"

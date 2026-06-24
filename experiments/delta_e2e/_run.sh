#!/bin/bash
# ============================================================
# Delta-Checkpoint E2E Verification
# ============================================================
# Usage:
#   bash _run.sh [DEVICE_ID] [--steps N]
# ============================================================

set -euo pipefail

DEVICE_ID="${1:-1}"
STEPS="${2:-5}"

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

OUTPUT_DIR="$REPO/experiments/output/delta_e2e"
SCRIPT="$REPO/experiments/delta_e2e/delta_e2e.py"

echo "============================================================"
echo "Delta-Checkpoint E2E Verification"
echo "  Device: $DEVICE_ID | Steps: $STEPS"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  export PYTHONPATH=$REPO/python:\$PYTHONPATH && \
  $PYTHON $SCRIPT --device-id $DEVICE_ID --steps $STEPS
"

echo ""
echo "Done. Results:"
echo "  $OUTPUT_DIR/delta_e2e.json"

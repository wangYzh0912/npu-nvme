#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 2b: Recovery Validation — NRMSE vs T curve
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash _run.sh [STEPS] [DEVICE_ID]
#   bash _run.sh 100 1          # Default: 100 steps, device 1
#
# Output:
#   experiments/output/step2b_recovery/step2b_nrmse.json
#   experiments/output/step2b_recovery/step2b_nrmse_curve.png
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

STEPS="${1:-100}"
DEVICE_ID="${2:-1}"

REPO="/home/user7/npu-nvme"
PYTHON="/home/user7/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

OUTPUT_DIR="$REPO/experiments/output/step2b_recovery"
SCRIPT="$REPO/experiments/baselines/step2b_recovery_validation/step2b_recovery.py"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 2b: Recovery Validation — NRMSE vs T"
echo "  Steps: $STEPS  |  Device: $DEVICE_ID"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  $PYTHON $SCRIPT --steps $STEPS --device-id $DEVICE_ID
"

echo ""
echo "Done! Results:"
echo "  $OUTPUT_DIR/step2b_nrmse.json"
echo "  $OUTPUT_DIR/step2b_nrmse_curve.png"

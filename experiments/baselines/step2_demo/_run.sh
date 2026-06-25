#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 2: In-Graph Delta Detection + INT8 Quantization Demo
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash _run.sh [STEPS] [DEVICE_ID]
#   bash _run.sh 2 1          # Default: 2 steps, device 1
#
# Output:
#   experiments/output/step2_demo/step2_validation.json
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

STEPS="${1:-2}"
DEVICE_ID="${2:-1}"

REPO="/home/user7/npu-nvme"
PYTHON="/home/user7/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

OUTPUT_DIR="$REPO/experiments/output/step2_demo"
SCRIPT="$REPO/experiments/baselines/step2_demo/step2_demo.py"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 2: In-Graph Delta + INT8 Quant Demo"
echo "  Steps: $STEPS  |  Device: $DEVICE_ID"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  $PYTHON $SCRIPT --steps $STEPS --device-id $DEVICE_ID
"

echo ""
echo "Done! Results:"
echo "  $OUTPUT_DIR/step2_validation.json"

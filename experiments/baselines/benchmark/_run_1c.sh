#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 1c: SPDK FULL Checkpoint BW Benchmark
# ═══════════════════════════════════════════════════════════════════════
# Requirements: NVMe disk must be formatted (python/format_npu_disk.py).
#   One-time: echo "CGCL_2025_#$" | sudo -S bash -c 'source .../setenv.bash && \
#     /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/python/format_npu_disk.py'
#
# Usage:
#   bash _run_1c.sh [DEVICE_ID]
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

DEVICE_ID="${1:-1}"

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

OUTPUT_DIR="$REPO/experiments/output/benchmark"
SCRIPT="$REPO/experiments/baselines/benchmark/step1c_spdk_bw.py"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 1c: SPDK FULL Checkpoint BW Benchmark (shm_id=80)"
echo "  Device: $DEVICE_ID"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  $PYTHON $SCRIPT --device-id $DEVICE_ID
"

echo ""
echo "Done! Results:"
echo "  $OUTPUT_DIR/step1c_spdk_bw.json"

#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 1b: Device-level PMU Profiling (sample-based ArithmeticUtilization)
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash _run.sh [STEPS] [DEVICE_ID]
#
# Examples:
#   bash _run.sh 12          # 12 steps, device 1 (default)
#   bash _run.sh 20 2        # 20 steps, device 2
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

STEPS="${1:-12}"
DEVICE_ID="${2:-1}"

REPO="/home/user7/npu-nvme"
PYTHON="/root/miniconda3/envs/ms_2.5/bin/python"
SUDO_PW="CGCL_2025_#$"
ASCEND_SETUP="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"

OUTPUT_DIR="$REPO/experiments/output/benchmark"
PROFILING_BASE="$REPO/output/profiling_vec/step1b"
SCRIPT="$REPO/experiments/baselines/benchmark/step1b_pmu.py"
WRAPPER="$REPO/experiments/baselines/benchmark/pmu_wrapper_1b.sh"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 1b: Device-level PMU Profiling (sample-based)"
echo "  Steps: $STEPS  |  Device: $DEVICE_ID"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROFILING_BASE"

# Clean stale output
find "$PROFILING_BASE" -name "PROF_*" -mmin -120 -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "[Phase 1/2] Running training under msprof (sample-based, ArithmeticUtilization)..."
echo "  Profiler output → $PROFILING_BASE"
echo ""

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  msprof --output=$PROFILING_BASE \
    --aic-mode=sample-based \
    --aic-metrics=ArithmeticUtilization \
    --aic-freq=100 \
    --application=$WRAPPER $STEPS $DEVICE_ID
"

# Fix ownership (msprof writes as root)
echo "$SUDO_PW" | sudo -S chown -R user7:user7 "$PROFILING_BASE" 2>/dev/null || true

# Find PROF directory
PROF_DIR=$(ls -dt "$PROFILING_BASE"/PROF_* 2>/dev/null | head -1)
if [ -z "$PROF_DIR" ]; then
  echo "ERROR: No PROF_* directory found under $PROFILING_BASE"
  ls -la "$PROFILING_BASE" || true
  exit 1
fi

echo ""
echo "[Phase 2/2] Parsing msprof output..."
echo "  PROF directory: $PROF_DIR"
echo ""

echo "$SUDO_PW" | sudo -S bash -c "
  source $ASCEND_SETUP && \
  $PYTHON $SCRIPT --parse-only --profiler-dir $PROF_DIR --output $OUTPUT_DIR/step1b_pmu.json
"

echo ""
echo "Done! Results:"
echo "  $OUTPUT_DIR/step1b_pmu.json"
echo "  $OUTPUT_DIR/step1b_benchmark_partial.json"
echo "  $PROF_DIR/"

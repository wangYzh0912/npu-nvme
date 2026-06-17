#!/bin/bash
# _run_phase1b.sh — msprof wrapper for Phase 1b profiling
# Must be run as root (via sudo).
# Usage: sudo bash _run_phase1b.sh <label> <preset> <inject> [steps] [sink]
#
# Examples:
#   sudo bash _run_phase1b.sh V5_baseline gpt2_large 0 16 4
#   sudo bash _run_phase1b.sh V5_inject50 gpt2_large 50 16 4

LABEL="${1:?Need label}"
PRESET="${2:?Need preset}"
INJECT="${3:?Need inject count}"
STEPS="${4:-16}"
SINK="${5:-4}"
REPO="/home/user7/npu-nvme"
SCRIPT="${REPO}/experiments/baselines/phase1b_profile.py"
PROF_DIR="${REPO}/output/profiling_vec/${LABEL}"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# Clean old profile data
rm -rf "$PROF_DIR" 2>/dev/null

# Run under msprof for PMU collection
# No --aic-metrics flag needed — Core Type and Duration are in default CSV output.
exec /usr/local/Ascend/ascend-toolkit/latest/bin/msprof \
    --output="$PROF_DIR" \
    -- "$SCRIPT" \
    --label "$LABEL" --preset "$PRESET" \
    --inject "$INJECT" --steps "$STEPS" --sink "$SINK" \
    --epochs 2

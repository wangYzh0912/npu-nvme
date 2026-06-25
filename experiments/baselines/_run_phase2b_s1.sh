#!/bin/bash
# _run_phase2b_s1.sh — Phase 2b Step 1: Block Delta Detection
# Tests fixed-size block aggregation + delta detection on GPT-2 Small.
#
# Usage:
#   echo "CGCL_2025_#$" | sudo -S bash _run_phase2b_s1.sh pynative 0
#   echo "CGCL_2025_#$" | sudo -S bash _run_phase2b_s1.sh graph 0
#   echo "CGCL_2025_#$" | sudo -S bash _run_phase2b_s1.sh both 0 8 4

MODE="${1:-both}"
LAYER="${2:-0}"
STEPS="${3:-8}"
SINK="${4:-4}"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

exec /home/user7/miniconda3/envs/ms_2.5/bin/python \
  /home/user7/npu-nvme/experiments/baselines/phase2b_step1_block_delta.py \
  --mode "${MODE}" --layer "${LAYER}" --steps "${STEPS}" --sink "${SINK}" \
  --label "L${LAYER}_${MODE}"

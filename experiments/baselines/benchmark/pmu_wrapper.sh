#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# Step 1 PMU Profiler Wrapper
# This script is called by msprof as --application child process.
# It sources the Ascend environment and runs the Python benchmark.
#
# Usage (as root):
#   sudo -E msprof --output=<dir> -- <this_script> <steps> <device_id>
# ════════════════════════════════════════════════════════════════════
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export RANK_ID=0

STEPS=${1:-12}
DEVICE=${2:-1}

exec /home/user7/miniconda3/envs/ms_2.5/bin/python \
    /home/user7/npu-nvme/experiments/baselines/benchmark/step1_benchmark.py \
    --steps "$STEPS" --device-id "$DEVICE" --no-hbm-watch

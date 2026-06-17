#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Step 1b PMU Profiler Wrapper
# Called by msprof --application=. Sources Ascend env, runs step1b_pmu.py.
# Usage (as root): ./pmu_wrapper_1b.sh <steps> <device_id>
# ═══════════════════════════════════════════════════════════════════════
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export RANK_ID=0

STEPS=${1:-12}
DEVICE=${2:-1}

exec /root/miniconda3/envs/ms_2.5/bin/python \
    /home/user7/npu-nvme/experiments/baselines/benchmark/step1b_pmu.py \
    --steps "$STEPS" --device-id "$DEVICE"

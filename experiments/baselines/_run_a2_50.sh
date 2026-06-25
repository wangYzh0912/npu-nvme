#!/bin/bash
# _run_a2_50.sh — self-contained msprof launcher for A2_50
LABEL="$1"
INJECT="$2"
PROF_DIR="/home/user7/npu-nvme/output/profiling_vec/${LABEL}"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

exec /home/user7/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/phase1a_inject.py --inject "${INJECT}" --label "${LABEL}" --steps 16 --sink 4

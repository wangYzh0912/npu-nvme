#!/bin/bash
# Wrapper script for msprof to profile phase1a_train.py
# Usage: sudo ./_run_msprof.sh <label> <inject_count>
LABEL="$1"
INJECT="$2"
PROF_DIR="/home/user7/npu-nvme/output/profiling_vec/${LABEL}"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

exec /root/miniconda3/envs/ms_2.5/bin/python \
  /home/user7/npu-nvme/experiments/baselines/phase1a_train.py \
  --inject "${INJECT}"

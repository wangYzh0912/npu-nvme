#!/bin/bash
# E10 runner script — wraps source + python in a single bash -c
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH
cd /home/user7/npu-nvme
exec /root/miniconda3/envs/ms_2.5/bin/python experiments/baselines/phase5_e10_spdk_delta_e2e.py --steps 10 "$@"

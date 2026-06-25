#!/bin/bash
set -e
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/home/user7/npu-nvme/build_out/lib:$LD_LIBRARY_PATH
PY=/home/user7/miniconda3/envs/ms_2.5/bin/python3
cd /home/user7/npu-nvme

echo "===== R0: BASELINE ====="
$PY experiments/r0_baseline.py 1>/tmp/r0.log 2>/tmp/r0_err.log
echo "R0 exit=$?"
grep "RESULT:" /tmp/r0.log

echo "===== R1: SPDK OFF ====="
NPU_NVME_LISTENER_MODE=off $PY experiments/r1_spdk_off.py 1>/tmp/r1.log 2>/tmp/r1_err.log
echo "R1 exit=$?"
grep "RESULT:" /tmp/r1.log

echo "===== R2: SPDK IDLE ====="
NPU_NVME_LISTENER_MODE=idle $PY experiments/r2_spdk_idle.py 1>/tmp/r2.log 2>/tmp/r2_err.log
echo "R2 exit=$?"
grep "RESULT:" /tmp/r2.log

echo "===== R3: SPDK QPOLL ====="
NPU_NVME_LISTENER_MODE=qpoll $PY experiments/r3_spdk_qpoll.py 1>/tmp/r3.log 2>/tmp/r3_err.log
echo "R3 exit=$?"
grep "RESULT:" /tmp/r3.log

echo "===== R4: SPDK FULL ====="
NPU_NVME_LISTENER_MODE=full $PY experiments/r4_spdk_full.py 1>/tmp/r4.log 2>/tmp/r4_err.log
echo "R4 exit=$?"
grep "RESULT:" /tmp/r4.log

echo "===== PHASE 1 DONE ====="

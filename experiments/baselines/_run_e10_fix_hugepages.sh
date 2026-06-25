#!/bin/bash
# SPDK hugepage reset helper — must run as root
# Frees 8GB of hugepages so SPDK can allocate DMA buffers
echo "=== BEFORE ==="
echo "HugePages_Free=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages)"

# Reduce nr_hugepages to release kernel-held hugepages
echo 4096 > /proc/sys/vm/nr_hugepages 2>/dev/null
sleep 1

echo "=== AFTER ==="
echo "HugePages_Total=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)"
echo "HugePages_Free=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages)"

# Now retry SPDK init
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd /home/user7/npu-nvme
PYTHONPATH=python:$PYTHONPATH /home/user7/miniconda3/envs/ms_2.5/bin/python experiments/baselines/phase5_e10_spdk_delta_e2e.py --steps 10

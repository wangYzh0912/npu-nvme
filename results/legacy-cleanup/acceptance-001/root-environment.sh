#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PATH=/home/user7/miniconda3/envs/ms_2.5/bin:$PATH
cd /models/npu_nvme_exp/user7-stack/checkouts/legacy-cleanup
export PYTHONPATH=$PWD/python:$PWD:/home/user7/.local/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$PWD/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
exec "$@"

#!/usr/bin/env bash
set -euo pipefail
exec /models/npu_nvme_exp/user7-stack/conda/bin/python \
  /home/user7/npu-nvme/scripts/run_user_environment.py --profile candidate -- \
  python /home/user7/npu-nvme/experiments/training/train_qwen3_full_restart.py "$@"

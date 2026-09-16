#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${QWEN_PYTHON:-/models/npu_nvme_exp/user7-stack/conda/bin/python}"
exec "$python" "$repo/scripts/run_user_environment.py" --profile candidate -- \
  python "$repo/experiments/training/train_qwen3_full_restart.py" "$@"

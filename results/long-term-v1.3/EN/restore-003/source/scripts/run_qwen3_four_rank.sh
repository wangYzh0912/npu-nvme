#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${QWEN_PYTHON:-/models/npu_nvme_exp/user7-stack/conda/bin/python}"
out="${QWEN_RUN_OUTPUT:-/models/npu_nvme_exp/user7-stack/qwen3-native-$(date +%Y%m%d-%H%M%S)}"
# A run directory has one owner and is never reused.
mkdir "$out"
mkdir "$out/msrun-log"
cd "$out"
export ASCEND_RT_VISIBLE_DEVICES="${QWEN_DEVICES:-0,1,2,3}"
export RUN_MODE=finetune
export PYTHONUNBUFFERED=1
export HCCL_CONNECT_TIMEOUT=300
export MS_COMPILER_CACHE_PATH="$out/compiler-cache"
port="$("$python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
echo "QWEN_RUN_OUTPUT=$out"
"$python" "$repo/scripts/run_user_environment.py" --profile candidate --inspect > "$out/environment.lock.json"
"$python" "$repo/tools/record_run_manifest.py" --out "$out/run_manifest.json" --repo "$repo" -- \
  "$python" "$repo/scripts/run_user_environment.py" --profile candidate -- \
  python -m mindspore.parallel.cluster.run --worker_num=4 --local_worker_num=4 \
  --master_addr=127.0.0.1 --master_port="$port" --join=True --cluster_time_out=300 \
  --tail_worker_log=4 --log_dir="$out/msrun-log" python "$repo/experiments/training/train_qwen3_full_restart.py" \
  --output "$out" "$@"
"$python" "$repo/scripts/run_user_environment.py" --profile candidate -- \
  python -m mindspore.parallel.cluster.run --worker_num=4 --local_worker_num=4 \
  --master_addr=127.0.0.1 --master_port="$port" --join=True --cluster_time_out=300 \
  --tail_worker_log=4 --log_dir="$out/msrun-log" python "$repo/experiments/training/train_qwen3_full_restart.py" \
  --output "$out" "$@"
check_args=()
argv=("$@")
for ((i=0; i<${#argv[@]}; ++i)); do
  if [[ "${argv[$i]}" == "--resume-run" ]]; then
    check_args=(--source-run "${argv[$((i+1))]}")
  fi
done
exec "$python" "$repo/experiments/training/check_qwen_training_run.py" --output "$out" "${check_args[@]}"

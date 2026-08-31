# 项目执行环境与命令

> 最近核对：2026-08-31。项目目录：`/home/user7/npu-nvme`；分支：
> `exp/ppt-evidence-20260829`；最近提交：`ef01af9`。
>
> 本文件不保存 sudo 密码。密码只从根目录 `.sudo_pw` 读取，文件应保持 `0600`，
> 不要将密码复制到命令行、日志或 Git 提交中。

## 1. 固定环境

| 项目 | 当前值/约束 |
|---|---|
| Python | `/home/user7/miniconda3/envs/ms_2.5/bin/python`，3.9.25 |
| Conda / MindSpore / NumPy | `ms_2.5` / 2.5.0 / 1.26.4 |
| Ascend CANN | `/usr/local/Ascend/ascend-toolkit`，用 `set_env.sh` 初始化 |
| NPU | Ascend 910B3，默认 `--npu 7`，当前 Bus-Id `0000:42:00.0` |
| 裸盘/SPDK | Huawei ES3000 V6 / ES3500P V6 3.84 TB，`0000:83:00.0`，`uio_pci_generic` |
| 文件系统对照 | 同型号 SSD，`0000:84:00.0`，内核 `nvme`；当前 `/models` 为 XFS |
| SPDK | `/home/user7/npu-nvme/third_party/spdk` |
| C 库 | `build_out/lib/libnpu_nvme.so` |

`83:00.0` 和 `84:00.0` 是两块物理盘。裸盘实验只允许 `83:00.0`；P1 必须标注
“同型号双盘 A/B”，不能把两盘绝对值直接解释成软件路径收益。`/dev/uio0` 通常
只有 root 可访问。

## 2. Shell 初始化

```bash
cd /home/user7/npu-nvme
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/user7/miniconda3/etc/profile.d/conda.sh
conda activate ms_2.5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$PWD/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

```bash
python --version
python -c 'import mindspore, numpy; print(mindspore.__version__, numpy.__version__)'
test -f build_out/lib/libnpu_nvme.so
```

## 3. Root shell和权限

硬件实验需要 root，且 root shell 要显式重新加载 CANN 和 Conda。`.sudo_pw` 可能无
末尾换行，统一使用此模板，不要用 `sudo -E` 替代：

```bash
cd /home/user7/npu-nvme
{ cat .sudo_pw; printf '\n'; } | sudo -S -k bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /home/user7/miniconda3/etc/profile.d/conda.sh
  conda activate ms_2.5
  cd /home/user7/npu-nvme
  export PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=/home/user7/npu-nvme/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
  export PYTHONPATH=/home/user7/npu-nvme/python:${PYTHONPATH:-}
  # 在此处放入实际实验命令
'
```

## 4. 每次实验前检查

```bash
cd /home/user7/npu-nvme
npu-smi info
lspci -s 83:00.0 -nnk
lspci -s 84:00.0 -nnk
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS -T /models
readlink -f /sys/bus/pci/devices/0000:83:00.0/driver
ls -l /dev/uio*
pgrep -af 'msprof|p1_fair_io|p3_async_pipeline|p4_training_e2e|p6_aux_injection|vector_engine_profile' || true
```

继续前确认：83 盘为 `uio_pci_generic`、84 盘为 `nvme`、NPU 7 无占用进程、`/models`
可写。禁止对未声明设备执行格式化；SPDK shared-memory ID 不能与残留进程重复。

## 5. SPDK 和 C 构建

```bash
cd /home/user7/npu-nvme
git submodule update --init --recursive
cd third_party/spdk
./configure
make -j"$(nproc)"
cd ../..
export SPDK_ROOT_DIR=/home/user7/npu-nvme/third_party/spdk
export DPDK_MEMPOOL_RING_FIXED_LIB=/home/user7/npu-nvme/build/dpdk_fix/librte_mempool_ring_fixed.a
export SOC_VERSION=Ascend910B3
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSPDK_ROOT_DIR="$SPDK_ROOT_DIR" \
  -DDPDK_MEMPOOL_RING_FIXED_LIB="$DPDK_MEMPOOL_RING_FIXED_LIB" \
  -DSOC_VERSION="$SOC_VERSION"
cmake --build build -j4
cmake --install build
```

## 6. 回归测试

```bash
cd /home/user7/npu-nvme
source /home/user7/miniconda3/etc/profile.d/conda.sh
conda activate ms_2.5
PYTHONPATH=.:python:/home/user7/.local/lib/python3.9/site-packages \
  python -m pytest -q tests/python
```

当前基线约为 `79 passed`。C SPDK smoke test 用 root：

```bash
{ cat .sudo_pw; printf '\n'; } | sudo -S -k bash -c '
  timeout 30s /home/user7/npu-nvme/build/reactor_v0_test
  timeout 30s /home/user7/npu-nvme/build/reactor_v0_spdk_thread_test
'
```

## 7. 正式 P1--P9

```bash
cd /home/user7/npu-nvme
source /home/user7/miniconda3/etc/profile.d/conda.sh
conda activate ms_2.5
python experiments/benchmarks/run_ppt_p1_p9.py --dry-run
```

确认 dry-run 后，通过第 3 节 root 模板运行：

```bash
python experiments/benchmarks/run_ppt_p1_p9.py --npu 7
python experiments/benchmarks/summarize_p1_p9.py --root results/ppt-evidence-20260829
```

正式结果目录为 `results/ppt-evidence-20260829`；编排状态在 `execution_state.json`，
失败目录不要删除。

## 8. 两小时快速趋势

快速轮次使用 GPT-2、单 seed 和短样本，只报告趋势。原始数据在
`/tmp/npu-nvme-quick-trend-20260830`，紧凑结果在 `results/quick-trend-20260830`。

```bash
{ cat .sudo_pw; printf '\n'; } | sudo -S -k bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /home/user7/miniconda3/etc/profile.d/conda.sh
  conda activate ms_2.5
  cd /home/user7/npu-nvme
  export PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=/home/user7/npu-nvme/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
  export PYTHONPATH=/home/user7/npu-nvme/python:${PYTHONPATH:-}
  python experiments/benchmarks/run_quick_trends.py \
    --deadline-minutes 105 --output-root results/quick-trend-20260830 \
    --raw-root /tmp/npu-nvme-quick-trend-20260830 \
    --npu 7 --pci 0000:83:00.0 --resume
'
python experiments/benchmarks/summarize_quick_trends.py --root results/quick-trend-20260830
```

## 9. P6 真实 msprof

真实 NPU 采样也通过第 3 节 root 模板运行：

```bash
python experiments/microbench/vector_engine_profile.py \
  --model gpt2 --device-id 7 --seeds 41 --warmups 2 --steps 10 \
  --output-dir /tmp/npu-nvme-quick-trend-20260830/P6_profile
python experiments/benchmarks/p6_analyze_tree.py \
  --root /tmp/npu-nvme-quick-trend-20260830/P6_profile
python experiments/benchmarks/p6_aggregate.py \
  --source /tmp/npu-nvme-quick-trend-20260830/P6_profile \
  --output results/quick-trend-20260830/P6/profile_summary.json
python experiments/benchmarks/summarize_quick_trends.py --root results/quick-trend-20260830
```

Arithmetic PMU issue ratio 的时间投影不是整颗 NPU Vector 利用率；`hbm.csv` 的 Average
是设备平均带宽，不是 HBM 利用率百分比。

## 10. 单项快速命令

真实 NPU/NVMe 命令套用第 3 节 root 模板；`--allow-fewer-samples` 不得用于正式门禁。

```bash
python experiments/benchmarks/p1_fair_io.py --path all --operations write read \
  --sizes 4194304 --depths 1 4 --total-bytes 268435456 --warmups 2 --samples 8 \
  --allow-fewer-samples --npu 7 --pci 0000:83:00.0 \
  --output-root /tmp/npu-nvme-quick-trend-20260830/P1 \
  --fs-root /models/npu_nvme_exp/quick-trend-20260830

python experiments/benchmarks/p3_async_pipeline.py --model gpt2 \
  --modes serial queue async --chunks 4194304 --depths 1 4 --delays 0 1000 \
  --seeds 41 --warmups 2 --samples 5 --allow-fewer-samples \
  --npu 7 --pci 0000:83:00.0 --output-root /tmp/npu-nvme-quick-trend-20260830/P3

python experiments/benchmarks/p4_training_e2e.py --model gpt2 \
  --modes none sync async --intervals 5 --checkpoints 2 --total-formal-steps 10 \
  --seeds 41 --warmup-steps 2 --chunk-size 4194304 --pipeline-depth 4 \
  --npu 7 --pci 0000:83:00.0 --output-root /tmp/npu-nvme-quick-trend-20260830/P4

python experiments/benchmarks/p6_aux_injection.py --model gpt2 \
  --modes none npu_serial npu_parallel --tasks diff --seeds 41 --warmups 1 --steps 5 \
  --npu 7 --pci 0000:83:00.0 --output-root /tmp/npu-nvme-quick-trend-20260830/P6
```

## 11. P8/P9 最小恢复

```bash
python experiments/benchmarks/p8_p9_incremental.py produce --model-name gpt2 \
  --npu 7 --pci 0000:83:00.0 --seed 41 --steps 10 --model-fraction 0.05 \
  --m-fraction 0.20 --m-encoding fp16 --v-encoding fp16 --v-refresh 4 \
  --full-interval 10 --max-age 4 --keep-last-n 3 --shm-id 12044 \
  --output-root results/quick-trend-20260830/P8
python experiments/benchmarks/p8_p9_incremental.py recover \
  --manifest <recovery_manifest.json> --targets 5,10 --continue-steps 10 \
  --npu 7 --pci 0000:83:00.0 --shm-id 14044 --slot-size-gb 10 \
  --output-root results/quick-trend-20260830/P9
```

只有 fresh-process 哈希、NRMSE、loss 偏差和 generation 检查均通过，才可称为恢复正确。

## 12. 结果和清理

```bash
{ cat .sudo_pw; printf '\n'; } | sudo -S -k chown -R user7:user7 \
  /home/user7/npu-nvme/results/quick-trend-20260830
pgrep -af 'msprof|p1_fair_io|p3_async_pipeline|p4_training_e2e|p6_aux_injection|vector_engine_profile' || true
git status --short
git log -1 --oneline
```

只对明确结果目录执行 `chown`，不要对项目根目录递归修改；提交时不要使用 `git add .`。
P2 残差超过 10% 时不得画精确百分比；P4 的 `throughput_overhead` 和 `step_overhead`
不可互换；P5 RSS 含运行时基线；P6 PMU 投影不等于整机占用；短样本、单 seed、GPT-2
只能报告趋势，不能替代 GPT-2 XL 多 seed 正式门禁。

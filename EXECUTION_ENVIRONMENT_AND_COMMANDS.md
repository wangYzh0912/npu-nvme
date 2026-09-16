# 项目执行环境与命令

> 最近核对：2026-09-11（本轮只读环境/设备与 CPU 测试）。项目目录：
> `/home/user7/npu-nvme`；分支 `exp/ppt-evidence-20260829`；HEAD `ef21f92`，
> 工作树有未提交修改。已 fetch 的 `origin/master=5b164b6`，与本分支分叉 37/2；
> 后续实施入口见 [长期计划](docs/LONG_TERM_DEVELOPMENT_PLAN.md)，
> 核查范围见 [审查报告](docs/DEVELOPMENT_PROGRESS_REVIEW_20260911.md)。
>
> 本文件不保存 sudo 密码。密码只从根目录 `.sudo_pw` 读取，文件应保持 `0600`，
> 不要将密码复制到命令行、日志或 Git 提交中。

## 1. 固定环境

本节分别列出 candidate 与保留的 old 环境。候选栈现已配置，默认仍 old；环境阶段尚未完成恢复和回退验收。

| 候选环境项 | 本机路径/身份 |
|---|---|
| Python | `/models/npu_nvme_exp/user7-stack/conda/bin/python`，3.11.4 |
| CANN | `/models/npu_nvme_exp/user7-stack/cann-install/ascend-toolkit/8.3.RC1` |
| 项目库 | `/models/npu_nvme_exp/user7-stack/project-lib/libnpu_nvme.so` |
| 框架 | Qwen 历史报告为 MindSpore 2.7.1 / MindFormers 1.7.0 |
| 配置/启动器 | `config/user_environments.json` / `scripts/run_user_environment.py`，本地尚未提交 |
| Qwen 设备 | 历史 TP4 为设备 0–3；不是下表旧单卡默认 7 |

可只读检查两套路径；不要把旧环境的 LD_LIBRARY_PATH 直接套给候选栈：

```bash
/home/user7/miniconda3/envs/ms_2.5/bin/python scripts/run_user_environment.py --profile old --inspect
/home/user7/miniconda3/envs/ms_2.5/bin/python scripts/run_user_environment.py --profile candidate --inspect
```

两项本轮均退出 0，仅证明环境识别/依赖解析。候选环境的实际训练进程库路径、
GPT-2/Ours 回归、Qwen 完整态 fresh restart 和 old→candidate→old 回退仍按长期计划验收。

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

上述全集命令是历史入口，本轮未运行全集。当前干净 `5b164b6` 的可移植子集为
`84 passed`，五个排除模块和精确命令见长期计划 §8.5；本地环境/证据/Qwen 三模块为
`23 passed`，命令见长期计划 §9.7。这些数值不是硬编码验收门槛。

以下 C SPDK smoke test 是需独立执行的历史命令，本轮未运行；使用前检查专用区域与占用，用 root：

```bash
{ cat .sudo_pw; printf '\n'; } | sudo -S -k bash -c '
  timeout 30s /home/user7/npu-nvme/build/reactor_v0_test
  timeout 30s /home/user7/npu-nvme/build/reactor_v0_spdk_thread_test
'
```

## 7. 正式 P1--P9

以下为旧实验工作树的既有入口，不是当前下一步自动执行顺序。新主线已整理历史
PPT campaign；先按长期计划完成安全/正确性及比较口径门禁，再选择性迁入、重跑。
G1 会写新 FULL 代并可能淘汰旧槽，G2 会改写 live metadata；两者都不能当只读检查。

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


## C1 已验收入口（2026-09-14）

实施目录 `/models/npu_nvme_exp/user7-stack/checkouts/c1-unified-entry`，分支 `codex/c1-unified-entry`。硬件验收源码 `cb3e21d`：213 软件用例、12 次训练、9 次精确新进程恢复、27 次正式恢复计时通过。入口兼容补丁的 CPU 回归另记，不改变此硬件来源。证据见该工作树 `results/long-term-v1.3/C1/README.md`。原始用户工作树已有修改保留。

CPU dry-run（使用未占用输出目录）：

```bash
cd /models/npu_nvme_exp/user7-stack/checkouts/c1-unified-entry
/home/user7/miniconda3/envs/ms_2.5/bin/python tools/run_c1_acceptance.py --dry-run --out /models/npu_nvme_exp/user7-stack/c1-runs/new-dry-run
```

真实验收沿用上文 root 授权与 old 环境，在 root shell 中执行以下命令；只操作 raw PCI 0000:83:00.0，0000:84:00.0 保持 /models 挂载。命令不会格式化设备。库应使用已验收构建，SHA256 `0f521c87a95b18de2d87f48417abd45b603414e8586ff568c226fc4b36fc3c7d`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/user7/miniconda3/etc/profile.d/conda.sh
conda activate ms_2.5
cd /models/npu_nvme_exp/user7-stack/checkouts/c1-unified-entry
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/python:${PYTHONPATH:-}"
export NPU_NVME_LIBRARY_PATH="$PWD/build_out/lib/libnpu_nvme.so"
export LD_LIBRARY_PATH="$PWD/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
python tools/run_c1_acceptance.py --out /models/npu_nvme_exp/user7-stack/c1-runs/new-acceptance
```

helper 将实际 HEAD 写入外部配置，拒绝未提交源码用于正式验收。重复实验必须使用新目录；超时/未知设备进度保留 quarantine，不能把进程退出当作 DMA 停止。完整用法见 `docs/migrations/C1_UNIFIED_ENTRY.md`。

## B2 开发验证（独立工作树，尚非全阶段验收）

工作树：`/models/npu_nvme_exp/user7-stack/checkouts/b2-d2`，分支 `codex/b2-d2`。
old 库在 `build/libnpu_nvme.so`；candidate 单独用其 CANN 路径构建到 `build-candidate/libnpu_nvme.so`。真机命令须在对应环境的 root shell 中执行，每个输出目录必须全新，串行占用 raw83/NPU7；异常所有者未证明停止前禁止重开设备。

```bash
python tests/hardware/b2_transfer_probe.py --read --high-depth --out /path/to/new-run
python tests/hardware/b2_read_faults.py --output /path/to/new-read-fault-run
python tools/run_gate.py --profile B2 --out /path/to/new-software-run
```

candidate 真机探针显式传 `--library "$PWD/build-candidate/libnpu_nvme.so"`，并使用 candidate Python/CANN 环境。`--high-depth` 覆盖 Host/HBM × 1/4/16MiB × depth1/4/8/16/32/64 共36项；每项记录实际库、能力、字节校验与资源数据。有效调度配置由 `npu_nvme_get_capabilities` 查询；`NPU_NVME_COPY_BYTES`、`NPU_NVME_CHECKSUM_BYTES`、`NPU_NVME_SUBMIT_ITEMS`、`NPU_NVME_QUANTUM_ITEMS`、`NPU_NVME_MAX_PENDING` 仅在初始化时读取，非法值拒绝初始化。限制按每请求/每轮调度解释，不是检查点总容量限制。

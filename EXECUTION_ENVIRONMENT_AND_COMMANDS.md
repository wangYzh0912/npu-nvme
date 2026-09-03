# 项目执行环境与命令

> 最近核对：2026-08-31。基线仓库：`/home/user7/npu-nvme`；本分支 worktree：
> `/home/user7/npu-nvme-io-path-v1`；分支：`exp/io-path-v1`；FULL-only 实现基于
> `ef01af9`。
>
> 下方历史命令中的仓库路径沿用基线；本分支复现 FULL 入口时，将工作目录和
> `PYTHONPATH`/`LD_LIBRARY_PATH` 中的仓库前缀替换为本 worktree。
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

当前基线为 `105 passed`（2026-09-04）。C SPDK smoke test 用 root：

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

单盘单卡 FULL 训练基准统一使用 `run_single_card_full.py`。它会依次启动连续训练
基线、source 保存进程和 fresh restore 进程；source 只有在数据、flush 和 metadata
均完成后才退出，restore 失败或续训 loss 不一致时返回非零退出码。
独立 `none` 进程用于性能对照；恢复后的 loss 正确性以 source 进程从冻结点继续训练的
轨迹为 oracle，避免把 Ascend GRAPH_MODE 在两个独立进程中重新编译产生的浮点归约差异
误判为 checkpoint 损坏。加载后的 model+optimizer digest 仍必须与冻结态字节一致。
若数据已经持久化但恢复或续训门禁失败，结果会保留 `persisted: true` 并以
`restore_verified: false` 区分失败阶段。

```bash
python experiments/benchmarks/run_single_card_full.py --dry-run
python experiments/benchmarks/run_single_card_full.py --model gpt2 \
  --mode serial --checkpoint-steps 10 50 100 --total-steps 110 \
  --seed 41 --npu 7 --pci 0000:83:00.0 \
  --run-dir results/single-card-full/gpt2_seed41_serial
python experiments/benchmarks/run_single_card_full.py --model gpt2_xl \
  --mode serial --checkpoint-steps 10 50 100 --total-steps 110 \
  --seed 41 --npu 7 --pci 0000:83:00.0 \
  --run-dir results/single-card-full/gpt2_xl_seed41_serial
```

`--mode queue` 和 `--mode async` 只改变 FULL 数据面；训练模型、seed、步数、恢复
协议和结果目录结构不变。`--dry-run` 不初始化硬件，`--smoke` 配置应由调用者显式
缩短步数，不能作为正式门禁结果。

## 11. IO-next 分阶段正式入口

IO-1/IO-2 使用 `io_next_campaign.py`。筛选、100-step gate 和 30-checkpoint
正式层分开启动，均支持 `--resume`。`live_async` 只有在完整 FULL HBM 读取能建立
设备侧 update fence 时才允许执行；当前有界 DMA pool 没有聚合 device fence，入口会
以退出码 2 和结构化 `UNSUPPORTED` 结果结束，禁止用 Host polling 代替。

```bash
python experiments/benchmarks/io_next_campaign.py --phases io1 \
  --output-root results/io-next-20260903 --npu 2 --numa-node 4 --resume
python experiments/benchmarks/io_next_campaign.py --phases io1_formal \
  --output-root results/io-next-20260903 --npu 2 --numa-node 4 --resume
python experiments/benchmarks/io_next_campaign.py --phases io2 \
  --output-root results/io-next-20260903 --npu 2 --numa-node 4 --resume
python experiments/benchmarks/io_next_campaign.py --phases io2_formal \
  --output-root results/io-next-20260903 --npu 2 --numa-node 4 --resume
```

IO-3 的一个 child 是同一 source HCCL job 内的多次全局 checkpoint，不是独立短跑。
`--restore-retained` 会在 source 和原 coordinator 都退出后，为每个保留 generation
分别创建 fresh coordinator 和 fresh HCCL world。

```bash
python experiments/benchmarks/io3_hccl_longrun.py \
  --world-sizes 2 4 --seeds 41 42 43 --intervals 10 50 \
  --total-steps 500 --continue-steps 10 100 --keep-last-n 3 \
  --restore-retained --resume
```

IO-4 的 B0/B1 分别是 Host→SPDK 和 HBM→异步 SPDK；B2/B3/B4 分别是
Host→Unix→memory、HBM→Unix→memory、Host→Unix→单 Reactor→SPDK。
B0-B3 不发布 checkpoint，B4 仅作 raw-path 诊断且显式记录
`publishes_generation=false`；发布 generation 的端到端 B5 证据来自 IO-3，必须恢复。

```bash
python experiments/benchmarks/io4_bottleneck_campaign.py \
  --paths B0 B1 B2 B3 B4 --producers 1 2 4 \
  --chunks 1048576 4194304 16777216 --depths 2 4 8 \
  --payloads 268435456 --numa-nodes 4 0 --samples 30 --warmups 10 --resume
python experiments/benchmarks/io_next_report.py \
  --root results/io-next-20260903
```

硬件命令必须在第 2 节环境初始化后运行；root shell 中应在 CANN 的现有
`PYTHONPATH`/`LD_LIBRARY_PATH` 前追加本 worktree，不能覆盖 CANN 的 TBE 路径。
完整 raw 目录默认被 Git 忽略，只提交配置、摘要、恢复结果、报告和 hash 索引。

阶段 2 独立数据面（筛选默认 3 次，正式默认 10 次 warmup + 30 次计入样本）：

```bash
python experiments/benchmarks/s2_async_data_plane.py --screening \
  --npu 7 --pci 0000:83:00.0 --output-root results/stage2-async/screening
python experiments/benchmarks/s2_async_data_plane.py \
  --payloads 268435456 --chunks 4194304 --depths 2 4 --modes queue async \
  --npu 7 --pci 0000:83:00.0 --output-root results/stage2-async/formal
```

阶段 3 正式矩阵复用同一个 fresh-process 基准，不另写训练/恢复实现：

```bash
python experiments/benchmarks/s3_training_io_matrix.py --model gpt2_xl \
  --modes none serial queue async --seeds 41 42 43 --intervals 10 20 50 \
  --chunks 4194304 --depths 4 --total-steps 110 \
  --npu 7 --pci 0000:83:00.0 --output-root results/stage3-training-io/formal
```

阶段 4 故障/背压门禁：

```bash
python tests/hardware/stage4_fault_lifecycle.py --npu 7 --pci 0000:83:00.0 \
  --output results/stage4-fault-lifecycle
python experiments/benchmarks/s4_control_matrix.py \
  --dma-slots 1 2 4 --checkpoint-slots 1 2 4 \
  --delays-ms 0 100 1000 5000 --intervals 1 5 10 \
  --npu 7 --pci 0000:83:00.0 --output-root results/stage4-control/formal
```

阶段 2--4 长时间正式矩阵使用统一入口。入口拒绝 dirty worktree，并在任一阶段失败时
立即返回非零；中断后以相同参数加 `--resume`，只复用 campaign 中已有结果文件且状态为
`pass` 的样本：

```bash
python experiments/benchmarks/run_full_long_validation.py \
  --output-root /tmp/npu-nvme-longrun-$(git rev-parse --short=12 HEAD) \
  --index-path results/full-io-longrun-$(git rev-parse --short=12 HEAD).json
# 中断恢复
python experiments/benchmarks/run_full_long_validation.py --resume \
  --output-root /tmp/npu-nvme-longrun-$(git rev-parse --short=12 HEAD) \
  --index-path results/full-io-longrun-$(git rev-parse --short=12 HEAD).json
```

上述真实 NPU/NVMe 命令均须套用第 3 节 root shell 模板。阶段 2 只有在回读
SHA-256、原始时间线、depth=1 无重叠、depth>=2 有重叠全部成立时才返回 0；阶段 4
只有所有注入点均显式失败且 fresh context 再写读成功时才返回 0。

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

## 12. P8/P9 最小恢复

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

## 13. 结果和清理

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

## 阶段 5/6 多卡入口

阶段 5 先运行单 Reactor 协议和 rank 分区门禁；`--run-c2` 才会启动现有
两 rank 真实训练态 coordinator。该 C2 路径是 Unix-socket host staging，
结果中会显式标注 transport，不能作为共享 DMA slot 或吞吐结论。

```bash
python experiments/benchmarks/stage5_multirank_reactor.py --world-size 2
python experiments/benchmarks/stage5_multirank_reactor.py --world-size 2 --run-c2 \
  --c2-run-dir /tmp/stage5-c2 --shm-id 50100 --coordinator-npu 7
python experiments/benchmarks/stage5_multirank_reactor.py --world-size 4
```

阶段 6 使用统一 `msrun` wrapper；只有子进程自身完成 FULL 保存、退出、fresh
restore 和续训校验时才算 checkpoint 通过，launcher exit 仅表示训练进程退出码。

```bash
python experiments/benchmarks/run_hccl_full.py --world-size 2 --steps 2 \
  --output /tmp/hccl-gpt2-2p --script experiments/benchmarks/gpt2_13b_dist.py --dry-run
python experiments/benchmarks/run_hccl_full.py --world-size 4 --steps 2 \
  --output /tmp/hccl-gpt2-4p --rank-table config/rank_table_4p.example.json

# FULL checkpoint acceptance: rank-local state, one coordinator commit,
# source exit, fresh per-rank restore, and continuation verification.
python experiments/benchmarks/run_hccl_full.py --c2-full --world-size 2 \
  --master-port 8127 --steps 2 --output /tmp/hccl-full-gpt2-2p \
  --rank-table /path/to/rank_table_2p.json -- \
  --pci 0000:83:00.0 --coordinator-npu 7 --slot-size-gb 4 --shm-id 50120
python experiments/benchmarks/run_hccl_full.py --c2-full --world-size 4 \
  --master-port 8128 --steps 2 --output /tmp/hccl-full-gpt2-4p \
  --rank-table /path/to/rank_table_4p.json -- \
  --pci 0000:83:00.0 --coordinator-npu 7 --slot-size-gb 4 --shm-id 50140
```

`--c2-full` 的 source 和 fresh restore 阶段均为真实多 rank HCCL 训练。source 完全
退出后，由唯一 SPDK owner 从 NVMe 读取各 rank shard，经 Unix socket 分发给 fresh
rank；每个 rank 完成全字段 checksum 和 checkpoint-state digest 校验后，在新 HCCL
group 中续训。`--master-port` 用于 source group，restore group 使用其下一个端口；
并行执行多个验收时必须为每组预留连续两个端口并分配不同 `--shm-id`。

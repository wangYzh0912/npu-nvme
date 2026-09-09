# 当前实验入口

从仓库根目录运行，环境见主 README。新结果写入独立的 `experiments/output/<run-id>/`，
记录实际 HEAD/dirty、配置、设备、原始事件与恢复证据；不要覆盖保留的 results。

| 工作负载 | 入口（相对 experiments） | 用途 |
|---|---|---|
| 单卡 FULL | `benchmarks/run_single_card_full.py` | serial/frozen/live，退出后恢复与续训 |
| IO1/IO2 | `benchmarks/io_next_campaign.py` | GPT-2/XL 机制、重复运行与慢盘压力 |
| IO3 | `benchmarks/io3_hccl_longrun.py` | 2/4 rank、HCCL 恢复、保留代际验证 |
| IO4 | `benchmarks/io4_bottleneck_campaign.py` | Host、Unix staging 和 Reactor 瓶颈 |
| INC1 | `microbench/vector_engine_profile.py` | 使用 `--model` 进入真实训练 PMU 模式 |
| INC2 | `benchmarks/inc2_graph_edge_load.py` | 图内 marker/compute/memory/chain 负载 |
| INC3 | `benchmarks/s2_real_trajectory.py` | adjacent/persisted reference 更新统计 |
| 结果处理 | `benchmarks/summarize_inc_minimal.py`、`p6_vector_timeline.py`、`io_next_report.py` | 当前记录校验与统计 |
| 方法对照 | [baselines/repro](baselines/repro/README.md) | 统一 fixture、状态桥、适配器和恢复计时 |

其余保留的 benchmark 模块是上述入口的 I/O、矩阵、环境和时间线依赖。
`baselines/two_phase_common.py` 是现行 ACL 捕获的共享依赖。

```bash
python experiments/benchmarks/io_next_campaign.py --dry-run \
  --phases io1_mechanism --output-root experiments/output/io1-001
python experiments/benchmarks/io3_hccl_longrun.py --dry-run \
  --world-sizes 2 --seeds 41 --output-root experiments/output/io3-001
```

更新观测示例，需目标机和模型/数据准备：

```bash
python experiments/benchmarks/s2_real_trajectory.py \
  --model gpt2_xl --steps 120 --seq-len 129 --seed 41 --npu 7 \
  --block-sizes 65536,262144 --top-k-percents 5,10,20 \
  --sample-windows 1-20,51-70,101-120 --score-dtype float32 \
  --output-root experiments/output/trajectory-001
```

完整状态观测不要使用 `--no-optimizer`。seeds 42/43 独立运行，保存真实数据路径与哈希。
PMU/INC2 参数见各自 `--help`，不能用默认算子微基准替代真实训练。原运行配置见结果
中的 config/environment，绝对路径和设备编号需根据实际机器调整。

## 解释边界

FULL 必须证明持久化、新进程逐字段恢复及续训；派发耗时不能代替持久化时间。
XL live 严格续训失败保留，放宽容差的诊断不等于正式通过。INC PMU 尚无共同设备时钟，
图内负载等效性门禁未过，固定 Top-K 类别覆盖不足；这些观测不证明增量检查点可恢复。
旧 PPT、R0/R1/R2 历史 campaign、重复排障和图表生成入口已移出 master，历史在 Git 中。

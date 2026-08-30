# Quick Trend Round

本轮是约两小时的方向性探测，不替代正式验收矩阵。正式样本量、模型规模和多 seed 覆盖不足时，只报告趋势，不报告最终性能结论。

状态：`complete`；编排记录数：14。

## P1 同规格路径

4 MiB、256 MiB 总逻辑数据、QD 1/4 的快速样本；下表为 4 MiB 组平均值，吞吐按十进制 GB/s。

| 操作/QD | buffered 延迟 ms | O_DIRECT 延迟 ms | SPDK host 延迟 ms | O_DIRECT/SPDK |
|---|---:|---:|---:|---:|
| read/QD1 | 207.2 | 81.2 | 118.8 | 0.684 |
| read/QD4 | 170.8 | 40.9 | 54.2 | 0.754 |
| write/QD1 | 247.1 | 62.5 | 116.6 | 0.536 |
| write/QD4 | 243.4 | 56.2 | 69.9 | 0.804 |

观察：本轮 O_DIRECT 在四个 4 MiB 组合中均低于 SPDK host 延迟；因此不能把这轮快速数据写成‘裸盘路径胜过 O_DIRECT’，只能写成‘路径差异明显，需正式双盘校准’。

## P3 DMA-NVMe 时间轴

只把包含 serial/queue/async 三模式的 4 MiB 组视为完整组；早先中断留下的 1 MiB serial-only 记录保留但不参与结论。

| depth | 注入延迟 ms | async overlap median | async/queue speedup |
|---:|---:|---:|---:|
| 1 | 0 | 0.000 | 0.999 |
| 1 | 1000 | 0.000 | 1.008 |
| 4 | 0 | 0.943 | 1.037 |
| 4 | 1000 | 0.953 | 0.999 |

趋势：depth=1 的 overlap 为 0；depth=4 的两组 overlap median 约 0.943/0.953，但 async 相对已有 queue 仅约 1.00x/1.04x。说明确实存在时间交叠，尚未证明新 async 实现带来稳定端到端收益。

## P4 训练影响

| 模式 | 吞吐 step/s | 吞吐下降 | checkpoint/普通 step - 1 | 前台等待均值 ms | 恢复校验 |
|---|---:|---:|---:|---:|---|
| async | 0.4056 | 7.68% | 1282.86% | 2848.9 | False |
| none | 0.4393 | 0.00% | 0.00% | n/a | True |
| sync | 0.3565 | 18.86% | 2040.19% | 4411.6 | False |

注意：sync/async 都出现 parameter checksum mismatch，故恢复校验为 false；该失败本身是本轮的有效趋势/阻塞证据，不能把 P4 记为通过。

## P5 环形缓冲 RSS

| 槽数 | 分块 MiB | 期望池 MiB | 增量 RSS MiB | host 峰值 RSS MiB | HugePage delta MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 1.0 | 45.6 | 430.7 | 10.0 |
| 1 | 4.0 | 4.0 | 45.7 | 430.7 | 84.0 |
| 1 | 16.0 | 16.0 | 45.7 | 430.7 | 96.0 |
| 2 | 1.0 | 2.0 | 45.6 | 430.7 | 86.0 |
| 2 | 4.0 | 8.0 | 43.7 | 428.7 | 90.0 |
| 2 | 16.0 | 32.0 | 43.7 | 428.7 | 114.0 |
| 4 | 1.0 | 4.0 | 45.7 | 430.7 | 94.0 |
| 4 | 4.0 | 16.0 | 43.7 | 428.7 | 102.0 |
| 4 | 16.0 | 64.0 | 45.7 | 430.7 | 150.0 |

观察：期望池大小严格按 slots×chunk 变化；进程 RSS 还包含 Python/NPU 基线，不能用 RSS 绝对值直接等同于 ring 大小。`pinned_dram_peak` 是 HugeTLB 可用页差值字段，报告中不将其称为已分配 pinned bytes。

## P6 辅助任务与真实利用率

| 模式 | 辅助任务 | 总延迟均值 ms | 前台等待均值 ms | 状态 |
|---|---|---:|---:|---|
| none | diff | 43.0 | 0.0 | pass |
| npu_serial | diff | 104.3 | 56.6 | pass |
| npu_parallel | diff | 105.4 | 55.6 | pass |

辅助 diff 在本轮约增加 61 ms；npu_parallel 与 npu_serial 基本相同，不能仅凭该注入实验声称并行已生效。

真实 msprof 导出摘要：

| 指标组 | Vector 时间线均值 | HBM 设备读 GB/s | HBM 设备写 GB/s |
|---|---:|---:|---:|
| ArithmeticUtilization | 0.018 | 6.705 | 5.424 |
| Memory | unavailable | 6.932 | 5.595 |

Vector 值仅在导出的 PMU 字段可用时解释为时间线投影，不能直接当作整颗 NPU 的百分比占用；HBM 列是设备平均带宽，也不是 HBM 利用率百分比。

## 结论边界

- 本轮目标是快速拿到变化趋势，样本数为 3--8，单 seed，GPT-2；不能替代 GPT-2 XL、多 seed、30 样本正式实验。
- 可保留的趋势：P3 depth=4 出现稳定 DMA/NVMe 时间交叠；P5 池容量随 slots×chunk 配置变化；P4 checkpoint 仍是主要前台代价；P1 三路径性能排序需要重新校准。
- 必须继续修复/复测：P4 checksum mismatch、P6 并行辅助任务无收益，以及 P1 双盘校准和正式置信区间。

原始运行目录：`/tmp/npu-nvme-quick-trend-20260830`；本目录仅提交紧凑摘要。

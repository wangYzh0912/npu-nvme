# P1-P9 实验报告

生成时间：2026-08-30 19:08:38 +0800

编排器状态：`pass`；实验结论按下表逐项判定。本报告只汇总 result.json 中已记录的观测值；缺失项标为“未测”。

## 环境门禁

状态：`pass`

- 无前置阻塞

## 实验汇总

| 实验 | 运行数 | pass | fail/degraded | 关键指标 |
|---|---:|---:|---:|---|
| P1 | 56 | 48 | 8 | latency_p95=376.2, throughput=2.898e+09, write_ratio=未测, recovery_error=未测 |
| P2 | 6 | 0 | 6 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P3 | 5 | 3 | 2 | latency_p95=2.488e+04, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P4 | 3 | 3 | 0 | latency_p95=未测, throughput=0.3719, write_ratio=未测, recovery_error=未测 |
| P5 | 15 | 12 | 3 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P6 | 18 | 14 | 4 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P7 | 1 | 1 | 0 | trajectory coverage_rows=54, jaccard_rows=9 |
| P8 | 4 | 2 | 2 | latency_p95=未测, throughput=未测, write_ratio=mean=0.2558, recovery_error=未测 |
| P9 | 2 | 1 | 1 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=mean=0.001024 |

## 审查后可支持的观察

- P1 同型号双盘完成 A/B 校准，但两盘 O_DIRECT 读取均值相差约 19%；4 MiB 对照的请求规格一致，256 MiB 裸盘逻辑块实际拆成 4 MiB NVMe 命令。结果支持相对 buffered FS 的路径差异，不支持笼统宣称裸盘优于 O_DIRECT。
- P2 六组均为 `degraded`：perf/strace/trace 已保存，但层间时间尚未闭合，禁止绘制精确百分比。
- P3 仅完成 seed 41、4 MiB、depth 4、正常延迟的 serial/queue/async 各 30 样本；async 时间轴重叠率均值约 0.945，能够证明该配置存在真实重叠。CSV queue_depth 不是 NVMe 在途深度，延迟注入也不是设备服务延迟，不能外推完整矩阵。
- P4 保存与恢复功能通过，但性能门槛不通过：原始 step_overhead=5.335 是比例，即约 533.5%，不是 5.3%。独立进程吞吐差3.6%受运行长度混杂，不作为验收结论。
- P5 9 组 DMA ring 均完成；HugePages_Free 的增量随槽数和分块增长，但固定 SPDK hugepage 开销和 1 GiB payload 污染绝对 RSS，当前只能作为趋势证据。
- P6 报告的 2.9%--3.1% 是 ArithmeticUtilization PMU issue ratio 按算子持续时间投影到设备 wall-clock 的值，不是整机 Vector 占用率；host/device 时钟无共同 epoch，step 对齐为估算。hbm.csv 设备平均约为读 19--21 GB/s、写 20--22 GB/s，但缺峰值分母，不能判断是否接近瓶颈。辅助注入只覆盖 seed 41 的 NPU serial/parallel，不能得出存在可免费利用空隙。
- P7 GPT-2 XL seed 42 的 500 步训练中采样早/中/晚各 30 步，覆盖三种分块；缺 seed 41/43，结论仅为单 seed 描述性证据。
- P8 的 25.6% 是对齐后的提交字节核算，不是 SSD SMART/NAND 实际写量，且只有 GPT-2 seed 41、10 步；未达到 `<20%`。P9 两个恢复点的 fresh-process 哈希与误差检查通过，但样本规模不足。

## 验收判定

- 当前可正式使用：P3 单配置真实重叠、P9 两个位置的功能正确性、单所有者压力与一次 NVMe 错误恢复。
- 当前需降级使用：P1、P5、P7、P8。
- 当前不可用于目标结论：P2 精确分层、P4 `<=5%`、P6 Vector 空闲算力、完整 P3/P4/P8/P9 矩阵。

## 可复现入口

```bash
python experiments/benchmarks/run_ppt_p1_p9.py --dry-run
python experiments/benchmarks/run_ppt_p1_p9.py
python experiments/benchmarks/summarize_p1_p9.py
```

# P1-P9 实验报告

生成时间：2026-09-10 01:58:56 +0800

编排器状态：`interrupted_runtime_budget`；实验结论按下表逐项判定。本报告只汇总 result.json 中已记录的观测值；缺失项标为“未测”。

## 环境门禁

状态：`pass`

- 无前置阻塞

## 实验汇总

| 实验 | 运行数 | pass | fail/degraded | 关键指标 |
|---|---:|---:|---:|---|
| P1 | 61 | 61 | 0 | latency_p95=196.2, throughput=5.881e+09, write_ratio=未测, recovery_error=未测 |
| P2 | 10 | 4 | 6 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P3 | 1 | 1 | 0 | latency_p95=3.32e+04, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P4 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P5 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P6 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P7 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P8 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |
| P9 | 0 | 0 | 0 | latency_p95=未测, throughput=未测, write_ratio=未测, recovery_error=未测 |

## 审查后可支持的观察

- P1 同型号双盘完成 A/B 校准，但两盘 O_DIRECT 读取均值相差约 19%；结果支持相对 buffered FS 的路径差异，不支持笼统宣称裸盘优于 O_DIRECT。
- P2 本轮结果包含通过和 degraded 样本；perf/strace/trace 已保存，但层间时间尚未闭合，禁止绘制精确百分比。
- P3 本轮状态为 `interrupted_runtime_budget`，已完成配置数=1；未完成配置不参与完整矩阵结论。
- P4/P5/P6/P7/P8/P9 未在本轮重新执行，历史结果不并入本轮运行计数。

## 验收判定

- 当前可正式使用：G0/G1/G2 正确性门禁、P1/P2 已完成样本，以及 P3 已完成的单配置结果。
- 当前需降级使用：P1、P2、P3（部分）。
- 当前不可用于目标结论：P2 精确分层、P4 `<=5%`、P6 Vector 空闲算力、完整 P3/P4/P8/P9 矩阵。

## 可复现入口

```bash
python experiments/benchmarks/run_ppt_p1_p9.py --dry-run
python experiments/benchmarks/run_ppt_p1_p9.py
python experiments/benchmarks/summarize_p1_p9.py
```

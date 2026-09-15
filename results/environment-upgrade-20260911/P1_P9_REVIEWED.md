# P1-P9 实验报告

生成时间：2026-09-11 00:57:57 +0800

编排器状态：`interrupted_runtime_budget`；环境门禁：`pass`。
主实验、嵌套负载和 smoke 分开统计；运行通过不代表完整实验矩阵或性能目标通过。
不同配置的延迟和吞吐不合并为一个数值。

## 主实验汇总

| 实验 | 主运行数 | pass | degraded | fail/其他 | 辅助运行数 |
|---|---:|---:|---:|---:|---:|
| P1 | 60 | 60 | 0 | 0 | 5 |
| P2 | 6 | 0 | 6 | 0 | 0 |
| P3 | 1 | 1 | 0 | 0 | 0 |
| P4 | 0 | 0 | 0 | 0 | 0 |
| P5 | 0 | 0 | 0 | 0 | 0 |
| P6 | 0 | 0 | 0 | 0 | 0 |
| P7 | 0 | 0 | 0 | 0 | 0 |
| P8 | 0 | 0 | 0 | 0 | 0 |
| P9 | 0 | 0 | 0 | 0 | 0 |

## 本轮证据与边界

- G0_roundtrip：`pass`；host and NPU buffer write/read, alignment/capacity rejection, bounded metadata timeout/recovery。
- G1_full_restart：`pass`；fresh-process FULL load with 772 per-parameter digest checks。
- G2_metadata：`pass`；A/B metadata fallback, superblock corruption rejection, byte restoration。
- P1 包含跨物理盘配置，需按设备、请求规格和持久化边界解释；不直接推断严格软件加速比。
- P2 主实验时间闭合通过 0/6；嵌套 P1 不构成 P2 分层验收。
- P2 未闭合配置不得绘制精确分层百分比。
- P3 主运行 1 个，模式为 serial；阶段状态 `interrupted_runtime_budget`。
- P3 仅报告已完成配置；须在相同配置的 serial/queue/async 对照和真实时间线完整后判断重叠与收益。
- P4, P5, P6, P7, P8, P9 无本轮主实验结果；不导入历史通过标志。
- 环境标识：legacy-unidentified；不同环境分别解释，不混算性能。

## 辅助结果归属

- P1: `P1_busy_smoke/P1/P1_20260910_001937_d68105f8/result.json`（auxiliary directory or role）。
- P1: `P2/P2/P2_20260910_010605_1abb6bb1/nested_p1/P1/P1_20260910_010607_bdc1b60b/result.json`（nested workload）。
- P1: `P2/P2/P2_20260910_010605_1abb6bb1/nested_p1/P1/P1_20260910_010711_029cbf90/result.json`（nested workload）。
- P1: `P2/P2/P2_20260910_010810_4c794afa/nested_p1/P1/P1_20260910_010812_ab9b165d/result.json`（nested workload）。
- P1: `P2/P2/P2_20260910_010810_4c794afa/nested_p1/P1/P1_20260910_010914_bdf5f9a5/result.json`（nested workload）。

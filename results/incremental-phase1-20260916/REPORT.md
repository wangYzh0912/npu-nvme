# 增量检查点第一阶段实验报告

当前状态：执行中，未完成项不计为通过。已完成 7/35 项覆盖。

实验固定 TP4、序列长度 128、micro-batch 1、FP32 权重/BF16 计算、AdamW、种子 42。各组从相同初始 FULL 恢复模型、优化器和随机状态，预热后连续运行 20 个优化器 step，每步保存。主模型 Qwen3-8B，辅模型 Qwen3-4B；两者同属 Qwen3，不据此声称跨模型家族泛化。

后续每配置默认一次有效运行，必要时最多两次。调度策略与初始 FULL 身份分离。失败尝试保留，不进入性能汇总。计时运行与媒体读回、影子 loss 验证运行分开。

## 五组 demo

| 模型 | 组 | 次数 | 20 步训练/s | 全部保存完成/s | 相对 B0 减速 | 实际写入/GB | 训练 loss 完全一致 |
|---|---|---:|---:|---:|---:|---:|---|
| main | B0 | 2 | 20.506 | 20.506 | 0.00% | 0.000 | True |
| main | B1 | 1 | 1387.319 | 1449.158 | 6665.29% | 661.271 | True |
| main | K10 | 1 | 292.710 | 300.700 | 1327.41% | 66.409 | True |
| main | K20 | 1 | 462.672 | 475.138 | 2156.23% | 132.636 | True |
| main | K5 | 1 | 194.118 | 198.266 | 846.62% | 33.252 | True |

写入量包含帧元数据、对齐和提交页；初始化 FULL 单独保留，不混入每步增量比例。主要性能预算为 3%，同时展示 1% 和 5%。单次运行及有波动的基线不能提供精确置信区间。

## Vector/Cube 资源与时间窗口

主模型保留同一次采集的全部 19 个完整区间，以及排除 profiler 启动首区间后的 18 个区间。下表采用后者；阶段边界由内核命名识别。混合任务同时计入两种引擎的任务包络，不能解释为指令同时执行。

| 卡 | Cube-only | Vector-only | 两者任务包络重合 | 两者均无任务 | 更新后候选窗口/μs |
|---|---:|---:|---:|---:|---:|
| 0 | 1.716% | 26.079% | 0.849% | 71.356% | 0.0 |
| 1 | 1.720% | 26.190% | 0.851% | 71.238% | 0.0 |
| 2 | 1.720% | 26.168% | 0.850% | 71.262% | 0.0 |
| 3 | 1.716% | 26.084% | 0.849% | 71.351% | 0.0 |

未获取核数量占用及候选窗口内同步 HBM 带宽，明确标记为缺失。两者均无任务的区间可能包含通信、Host 调度或依赖等待，不作为可插入能力的证明。主模型同一步内尚未发现优化器更新结束后的 Cube-only 候选窗口。跨到下一步前向/反向、并阻止下一次更新直至检测完成，仍可能形成合法窗口；当前同步检测实现未验证这种重叠能力。

## 计算与搬运边界

保留 0.5×、1×、2× 的评分和评分加选择，以及两种粒度的打包和 D2H。真实训练注入放在优化器更新后的合法边界；当前为同步探针。固定循环缓冲、rank-local 选择不能替代真实全模型工作集和 TP 全局选择，其差异在原始记录中保留。standalone 图仅表示独立执行耗时。

## 保存状态保真度

| 组 | 校验 step 数 | 末步相对 L2 | 末步 loss 差 | 最大块年龄 |
|---|---:|---:|---:|---:|
| K10 | 20 | 0.00010939587 | 0.069645479 | 20 |

K10 全局块平均年龄 6.142，未选中过的块 17571 个，其中末步评分为零的块 9477 个。年龄统计不按 TP 片段重复计数；年龄本身不等同于保存误差。

影子状态由实际媒体读回的数据更新；逐块检查映射、覆盖范围及选中值，与设备参考状态交叉核验。误差按真实完整权重计算。固定评估 batch 与训练固定样本相同，衡量状态数值差异，不作为留出集泛化质量指标。未指定质量预算，故只报告误差及组间差异，不宣布质量通过。

## 成本与空间

实际链路分开记录稳定等待、评分、全局选择、打包/D2H、校验、存储提交及参考推进。阶段执行时间不直接相加作为训练关键路径；累计消融与总训练时间共同用于归因。

额外空间包括参考副本、输出缓冲、分数/索引、框架工作区及临时张量。框架峰值与整卡采样峰值分开报告，禁止将不同时间的峰值简单相加。完整训练状态仅做容量预算，不算已完成优化器增量实验。

## 待完成项

- auxiliary-B0
- auxiliary-B1
- auxiliary-K5
- auxiliary-K10
- auxiliary-K20
- fidelity-B1
- fidelity-K5
- fidelity-K20
- p2-score-0.5
- p2-score-1.0
- p2-score-2.0
- p2-score_select-0.5
- p2-score_select-1.0
- p2-score_select-2.0
- p2-copy-0.5-262144
- p2-copy-0.5-4194304
- p2-copy-1.0-262144
- p2-copy-1.0-4194304
- p2-copy-2.0-262144
- p2-copy-2.0-4194304
- p5-ablation-1
- p5-ablation-2
- p5-ablation-3
- p5-ablation-4
- p5-ablation-5
- p5-ablation-6
- auxiliary-profile
- actual-checkpoint-profile

## 候选窗口分布

主模型稳定区间内共 9738 个 Cube-only 任务窗口，总计 360.094 ms，最长 381.500 μs。

长度分位数（μs）：{"0": 10.0, "0.25": 15.75, "0.5": 28.5, "0.75": 39.0, "0.95": 105.75, "1": 381.5}。

这些是任务包络窗口；核数量占用、窗口内带宽与精确计算指令重合均缺失，不能凭平均指标宣布辅助任务可插入。

稳定区间内，约 9.600 秒的 neither 时间与 HCCL 任务包络重合。HCCL 包络也可能包含等待；该重合不能解释为通信硬件持续满载。

PipeUtilization 导出的计数器（样本为任务，不是独立重复）：{"aic_mac_ratio": {"samples": 236060, "minimum": 0.0, "median": 0.0, "maximum": 0.802}, "aiv_vec_ratio": {"samples": 236060, "minimum": 0.0, "median": 0.04, "maximum": 0.681}, "cube_utilization(%)": {"samples": 236060, "minimum": 0.0, "median": 0.0, "maximum": 98.462}}。

HBM 导出表只有本次采集汇总：[{"Device_id": "0", "Metric": "Average", "Read(MB/s)": "24873.584", "Write(MB/s)": "22010.855"}]；没有候选窗口内带宽。

## main 空间与工作量预算

权重 32.763 GB，可选择块 124976，小参数 1232896 字节；每次扫描约 65.523 GB。三档 Top-K 均需评分全量候选。

完整训练状态对应扫描量约 196.600 GB，仅为预算，未扩展实际检测范围。

完整状态各卡实存大小（字节）：[{"model": 8191660032, "adam_m": 8191660032, "adam_v": 8191660032, "other": 612}, {"model": 8191660032, "adam_m": 8191660032, "adam_v": 8191660032, "other": 612}, {"model": 8191660032, "adam_m": 8191660032, "adam_v": 8191660032, "other": 612}, {"model": 8191660032, "adam_m": 8191660032, "adam_v": 8191660032, "other": 612}]。

完整状态输出近似量（包含常驻副本，元数据待定）：[{"ratio": 0.05, "payload_bytes": 4929053481.6, "metadata_bytes": null, "policy": "Physical state estimate including replicas; all local tensors below 64K elements saved fully. Global TP blocks and selected tails require mapping before extension."}, {"ratio": 0.1, "payload_bytes": 9843309763.2, "metadata_bytes": null, "policy": "Physical state estimate including replicas; all local tensors below 64K elements saved fully. Global TP blocks and selected tails require mapping before extension."}, {"ratio": 0.2, "payload_bytes": 19671822326.4, "metadata_bytes": null, "policy": "Physical state estimate including replicas; all local tensors below 64K elements saved fully. Global TP blocks and selected tails require mapping before extension."}]。

## 图表

![Main timeline](figures/phase-timeline-main.png)

![Main demo](figures/demo-main.png)

![fidelity-curves.png](figures/fidelity-curves.png)

![fidelity-layers-k10.png](figures/fidelity-layers-k10.png)

![fidelity-block-ages.png](figures/fidelity-block-ages.png)

## 已测链路分段

| 组 | 评分/ms | 全局选择/ms | 打包流/ms | D2H 流/ms | payload 校验/ms | 上次提交等待/ms |
|---|---:|---:|---:|---:|---:|---:|
| main:B1 | 0.000 | 0.000 | 12.471 | 441.006 | 5502.367 | 60721.170 |
| main:K10 | 2976.039 | 1548.085 | 未采集 | 未采集 | 547.812 | 7326.757 |
| main:K20 | 2974.633 | 1608.098 | 16.377 | 78.870 | 1095.563 | 13891.212 |
| main:K5 | 2953.015 | 1512.176 | 17.098 | 20.486 | 276.592 | 3309.774 |

数值为 step/rank 观测的中位数，不将其作为独立实验重复。打包与 D2H 使用 ACL 流事件，包含可能的 Host 提交间隙；总捕获时间还包含分配和描述符构建。

## 结论与边界

资源可行性按 3% 训练减速预算判断，另列 1% 和 5% 参考线。当前实现同步完成检测与捕获，只把之后的保存与下一步训练重叠；结果限定于这条实现路径。

恢复 FULL 后首个正式 step 的数据迁移及框架开销保留在全部 20 步总时间中，不从比较结果中扣除。B0 的两次运行用于展示基线变化范围；变化范围不是统计置信区间。

main：已测 Top-K 中，即使用较慢的 B0 作为分母，最小减速仍为 766.92%。这些配置未满足 3% 资源预算。

状态保真度依据实际保存状态的权重误差和固定 batch loss 差异判断。没有预设任务质量预算，因此不把误差曲线标记为质量通过，也不推断恢复续训的收敛性。

## 候选块数定向探针

固定逻辑元素总量，采用实际 64K 块分数；32K 分数由父块等分、128K 分数由相邻块合并。该 CPU 探针测元数据与全局选择成本，不测改变块大小后的真实评分、训练减速或保真度。四卡输入在单个进程内依次编码，不能把此耗时直接等同于分布式训练关键路径。

![Selection block count](figures/selection-block-count.png)

## Host 分数汇总定向探针

使用 rank 0 的真实 TP 几何，共 316940 个片段、1267760 字节固定分数输入。两次逐片段循环汇总中位数 2.9274 秒，向量化汇总 0.0502 秒，完整输出逐项相等且摘要相同。

这解释了实际评分阶段中 Host 汇总耗时的来源。探针不改变正式计时算法，其速度差不能直接当作训练加速；端到端收益仍需后续实现后测量。

# AIC/AIV 训推活跃度实测（2026-11-02）

- [实验报告](REPORT.md)：口径、结果、质量门槛、局限和下一步。
- [如何解释约90%的AIC非忙碌比例](AIC_IDLE_INTERPRETATION.md)：核心容量时间与峰值算力、调度因果关系的区别。
- [核心活跃度主图](aggregate/training-inference-core-device4.png)：10ms sample-only PMU，训练每种长度三次独立运行。
- [带阶段标签的任务时间线](figures/training-inference-task-diagnostic.png)：训练阶段追踪明确标为高开销诊断。
- [联合活跃度分布](aggregate/core-joint-device4.png)、[阈值敏感性](aggregate/core-threshold-sensitivity.csv)。
- [推理独立重复统计](aggregate/inference-independent-runs.json)、[训练独立重复统计](aggregate/training-independent-runs.json)。
- [工具和复现说明](source/tools/aic_activity/README.md)、[训练观察修改补丁](source/training-observation.patch)。

本目录保存完整压缩结果包，包括所有四卡的任务事件、PMU精确计数列、阶段关联证据、窗口、运行配置和源文件快照。`SHA256SUMS.json`是实验结束时的原始文件清单；本次发布新增的README、解释文档和.gitattributes不在该历史清单内。清单内文件保持逐字节一致。

离线工具的独立运行入口：

```sh
cd results/aic-aiv-20261102/source
PYTHONPATH=. python -m tools.aic_activity.render --help
```

绘图需要matplotlib和numpy。`source/python`和`source/measured-snapshots`是实验源码及实际运行文件的归档；分支根目录的训练实现保持原提交版本。不要将归档快照误当作已经合并的训练运行时修改。训练观察补丁及其依赖需在报告记载的实验环境中复现。

原始实验路径保留在证据中用于溯源；其他机器读取本目录相对路径即可查看图表、报告和压缩数据。临时运行输出未使用模型权重文件代替测量证据。

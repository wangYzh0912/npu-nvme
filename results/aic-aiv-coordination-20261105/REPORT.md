# AIC/AIV 协同探索：图与 attention 后端拆解

固定 Qwen3-8B、TP4、BF16、物理卡4–7、1654 KV blocks、greedy seed42、输入128/输出32；每次模型进程为独立重复。
原生 PIECEWISE dense 是受限布局实验；FULL 使用官方 PA 连续缓存。跨图模式差异仍包含融合/发射/图回放变化，不直接归因于单项 CPU 或 AIC/AIV 调度。

初次共享 AOT cache 的运行只保留诊断，不进入后端因果结论；cold 条件按进程使用独立编译缓存。
设备探针四卡每卡1152个 FIA 任务、0个 PA 任务、0个整缓存 Slice，支持 FULL 模式下连续PA cache/FIA路径可执行。其任务追踪约20.5%开销，只用于路径识别。

|配置|负载|独立进程|无profiler中位时延 ms|稳定条件数|
|---|---|---:|---:|---:|
|PIECEWISE / shared KV / FIA|b1-l128-o32|3|1079.331|3/3|
|PIECEWISE / shared KV / FIA|b8-l128-o32|3|1283.187|3/3|
|PIECEWISE / PA cache / FIA|b1-l128-o32|3|1070.030|3/3|
|PIECEWISE / PA cache / FIA|b8-l128-o32|3|1255.146|3/3|
|FULL / PA cache / FIA|b1-l128-o32|3|462.991|3/3|
|FULL / PA cache / FIA|b8-l128-o32|3|683.689|0/3|
|FULL / PA cache / PA|b1-l128-o32|3|451.932|3/3|
|FULL / PA cache / PA|b8-l128-o32|3|623.281|0/3|

三次独立b1重复中，FULL/PA和FULL/FIA输出一致；后端变化远小于PIECEWISE/FULL路径差异。
官方独立K/V分配的PIECEWISE实验用于检查storage owner差异；与FULL/FIA比较仍只隔离到图执行路径及伴随融合行为，尚不能归因为单个CPU发射环节。
b1四卡的32个实际请求step对应block table序列完全一致；终止后的空调度迭代数为2/1/2，完整scheduler行并不完全相同，差异保留在b1-block-table-equivalence.json。

同段系统采样（每条件一次诊断，不冒充三次PMU重复）：

|路径|负载|物理卡|AIC忙碌 %|AIV忙碌 %|HBM读 GiB/调用|HBM写 GiB/调用|
|---|---|---:|---:|---:|---:|---:|
|diagnostic-full-pa|b8-l128-o32|4|25.80|7.98|124.536|3.712|
|diagnostic-full-pa|b1-l128-o32|4|28.55|6.69|113.192|0.487|
|diagnostic-full-fia|b8-l128-o32|4|23.26|7.52|126.161|3.733|
|diagnostic-full-fia|b1-l128-o32|4|28.24|6.27|113.379|0.510|

任务trace的attention是MIX_AIC，未拆解内部AIC/AIV子事件；不把混合算子外层区间算作双引擎同时忙。系统样本与task trace来自不同请求，不作逐点拼接。

表格保留不稳定条件，不将其时延作为性能排名；图中红叉单独标出。输出哈希与全部控制请求见 inference-results.json 和 evidence。

训练既有 trace 审计覆盖 seq128/512 × 四卡；API事件标识不足以恢复生产者/消费者关系，readiness 仍为 unknown。高开销 trace 的算子任务时长不能当作正常训练关键路径。

[后端配对比较](backend-comparisons.json) · [训练依赖审计](training/retained-trace-dependency-audit.json) · [验收](acceptance.json)

未确认真实模型的合法 ready AIC 候选时，不进入真实模型 co-run；未通过候选正确性/稳定性门槛时，不进入多请求优化验收。合成并发既有约4%改善不替代这两项门槛。

## 多提示词严格门槛

原生FULL/PA及实验FULL/FIA均出现同条件重复输出差异；这不单独证明内存损坏，也不归咎于dense布局。原始token和首次差异记录在full-path-diverse-output-analysis.json。
未经过的长输入条件保留未完成，不当作通过。当前只接受固定fixture下的路径和输出配对证据。

## 训练 seq128 新观测

96步执行通过；sample-only采样质量：valid。每条件本轮一个新进程，不冒充三个独立重复。
物理卡4：采样前CV 3.01%，系统采集耗时变化 0.94%，完整step窗口 51。
物理卡4按有效bin时长加权：AIC 2.74%，AIV 43.17%；有效bin 3237/3237。
静态IR及操作数依赖单独归档；静态stream ID不等价于运行时队列就绪，缺少同次设备task映射时readiness仍为unknown。

## 训练 seq512 新观测

96步执行通过；sample-only采样质量：valid。每条件本轮一个新进程，不冒充三个独立重复。
物理卡4：采样前CV 1.44%，系统采集耗时变化 2.55%，完整step窗口 52。
物理卡4按有效bin时长加权：AIC 6.82%，AIV 46.11%；有效bin 3243/3243。
静态IR及操作数依赖单独归档；静态stream ID不等价于运行时队列就绪，缺少同次设备task映射时readiness仍为unknown。

## 后续切入点

优先分解连续KV/FIA下FULL与PIECEWISE的图边界、回放间隙和伴随融合变化；其次调查多提示词greedy输出漂移（固定调度与logit差距）。
训练侧将优化器/张量转换链的融合或分块作为待验证假设；未证明HBM饱和、未证明真实AIC任务已ready，不将这些假设列为已实现加速。

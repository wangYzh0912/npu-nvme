# KV 整理适用范围实验

本轮优先定位触发边界。官方镜像标签与定制镜像是镜像来源，不自动代表两个独立代码版本；实际身份见 stack-identity。
统计单位为一次 generate，b8表示八请求批次。筛查每配置一个模型进程，只作路径证据和诊断时延；独立重复结果另表列出。

整缓存整理具有明确的版本、物理布局和 attention 路径边界；当前证据不支持“大多数 vLLM 推理请求都会遇到”。
同模式布局对照用于定位转换开销；PA 对照同时改变图与 attention 路径，收益单独解释。真实 AIC ready 队列不可见，不能把低活跃度直接归因于调度遗漏。
归档状态：investigation_complete_correctness_unresolved；已完成设备级条件 512，原布局/连续布局输出配对 30。
多提示词批次的严格重复输出门槛未通过，原布局自身也出现差异；连续布局只能作为受控工作负载下的实验对照，尚未通过通用正确性验收。失败输出与TTFT/decode时间戳保留在 failure-evidence 和 diverse-failed-run-analysis.json。

[逐条件表格](RESULTS.md) · [机器可读数据](results.csv) · [验收结果](acceptance.json)

完整容量 6616 blocks；以下数值只展示物理卡4，各物理卡分别留证。

|路径|独立进程|请求 ms|AIC容量忙碌 ms/调用|AIV容量忙碌 ms/调用|HBM读 GiB/调用|HBM写 GiB/调用|
|---|---:|---:|---:|---:|---:|---:|
|native|3|1896.93|201.75|1490.70|1104.017|902.964|
|dense|3|1077.04|137.09|30.34|113.401|0.555|
|pa|3|454.16|132.01|30.30|113.170|0.487|

[配置筛查图](figures/applicability-screen.png)
[布局与 attention 路径重复对照](figures/layout-path-confirmation.png)
[完整容量下的引擎忙碌时间与 HBM 流量](figures/layout-engine-hbm.png)
[长生成与长输入批次扩展](figures/long-workload-extension.png)
[独立矩阵与复制的任务重叠](figures/independent-engine-overlap.png)

## 适用范围的实测证据

官方发布镜像与定制镜像的 vLLM-Ascend attention、cache allocator 和 graph gate 源文件哈希完全一致，因此它们不是两份独立版本证据。PIECEWISE 在两者上均复现整缓存 Slice。
vLLM-Ascend 0.18 使用独立源码和 CANN 8.5.1 的真实模型对照；该版同一 Qwen3-8B ordinary GQA cache 是连续 K/V，实测没有整缓存 Slice。版本结论必须同时注明 Python/PyTorch/CANN/driver 组合，不能声称仅由版本号导致。
v0.30 显式 PIECEWISE 与 NONE 均观察到整缓存 Slice；显式 FULL_DECODE_ONLY 加 pa_shape_list=[1,8] 走连续 PA cache 且没有整缓存 Slice，b1/b8 token 输出一致。显式 FULL_DECODE_ONLY（默认空 pa_shape_list）和 DEFAULT graph path 在当前 driver 上因 FusedInferAttentionScore graph-update 错误退出，不作为无 Slice 的负例。
PA资格会在分配阶段使 requires_contiguous_pa_kv_cache 成立，从而改变缓存布局；某些运行时形状即使仍回到 FIA，也可收到连续输入。因此 PA 条件没有整缓存 Slice 不只是一项算子替换的证据。
官方发布身份已通过 GitHub 标签目标与远端提交文件核对：[v0.30.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc1)；本地关键源文件与该标签提交逐字节一致。镜像本地ID、包版本和远端核验原件保存在 stack-identity。
same-mode dense layout control 使用相同 zero-initialized backing allocation，在 cache bind 前改为 [K][V] 连续 views；实际 cache write/read 仍调用原生路径。block IDs 和 output equivalence 单独验收。
异步批次下，请求进入/退出时机和物理 block 分配可能跨进程不同；b8 block table 不相等不能归因于 layout patch，也不冒充严格相同调度对照。b1 的精确 block table 配对与 b8 的输出等价分别保留。
主矩阵每轮按 native→dense→PA 固定顺序执行，未随机化；长负载扩展和系统采样矩阵轮转顺序。控制CV反映进程内稳定性，不能消除宿主负载或跨进程漂移，也不是显著性检验。
已核对 native/dense 输出哈希对数：30，全部一致：True；实际 block table 对数：14，一致：False。证据见 layout-output-equivalence.json 与 block-table-equivalence.json。

无 profiler、控制稳定且输出一致的布局对照数：30；逐条件延迟变化见 layout-latency-comparisons.json。

Qwen3-4B b1-l128-o1：3 个独立进程配对，连续视图 latency 变化均值 -1.3%，范围 -5.8% 至 1.0%。
Qwen3-4B b1-l128-o32：3 个独立进程配对，连续视图 latency 变化均值 -5.9%，范围 -6.8% 至 -5.1%。
Qwen3-4B b1-l2048-o32：3 个独立进程配对，连续视图 latency 变化均值 -7.0%，范围 -7.4% 至 -6.2%。
Qwen3-4B b8-l128-o32：3 个独立进程配对，连续视图 latency 变化均值 -7.8%，范围 -10.2% 至 -6.5%。
Qwen3-8B b1-l128-o1：3 个独立进程配对，连续视图 latency 变化均值 -2.0%，范围 -9.6% 至 2.7%。
Qwen3-8B b1-l128-o128：3 个独立进程配对，连续视图 latency 变化均值 -4.9%，范围 -6.7% 至 -2.6%。
Qwen3-8B b1-l128-o32：3 个独立进程配对，连续视图 latency 变化均值 -8.3%，范围 -14.6% 至 -5.0%。
Qwen3-8B b1-l2048-o32：3 个独立进程配对，连续视图 latency 变化均值 -8.0%，范围 -13.6% 至 -4.0%。
Qwen3-8B b8-l128-o32：3 个独立进程配对，连续视图 latency 变化均值 -7.5%，范围 -13.6% 至 -4.2%。
Qwen3-8B b8-l2048-o32：3 个独立进程配对，连续视图 latency 变化均值 -5.7%，范围 -6.9% 至 -4.6%。

独立容量条件 PMU/HBM 采样设备行数：36；最小 PMU 有效覆盖率：1.0；最小 HBM 覆盖率：0.9973492101850256。system-summary.json 不与 task trace 跨执行逐点对齐。

same-mode native/dense 的 AIC/AIV 容量忙碌时间与原生 HBM 读写流量见 system-comparisons.json；每项是独立系统采样段中的每次 generate 均值，不能解释为可回收 FLOPs。
独立矩阵/向量复制控制重复数：3；trace flow 归因完整：True。这只是两个合成任务的设备任务 envelope 并发，不证明模型里存在已 ready 的 AIC 工作。

合成分流 co-run / 串行时延比范围：0.960–0.964。三次运行均存在纯AIC/AIV任务envelope重叠，但重叠后的kernel持续时间变长，不能将重叠时长当作直接收益。

八个不同提示词的批次正确性验收：False。异步调度条件中原布局自身曾出现重复token差异，失败原件保留；同步调度诊断与主性能矩阵分别解释。

## 结论约束

Slice是否消失按实际设备任务确认，FIA/PA分支与图捕获任务必须结合元数据及任务清单解释。更换图模式同时可能改变融合、后端和缓存布局，因此跨图模式差值不能归因于单一布局。
未抽样真实部署或请求分布，不计算“大多数vLLM请求”的比例。Qwen3-4B与8B共享同类GQA结构，不能作为跨attention架构的证明。
当前910B3数据不外推CUDA、ROCm或其他NPU；旧版本源码对照与旧版本硬件通过是两类证据。
完整失败日志与命令保留；图模式启动失败、输出变化和未知归因不伪装成无整理的负例。
控制段 CV 超过5%的设备条件数：20，清单见 unstable-controls.json。这些条件可提供路径诊断，不能进入稳定延迟比较；不通过删掉差重复来报告全通过。
真实布局对照在cache绑定前一次建立连续缓存，后续由原生写入路径原地更新；不把长期复用过期副本视为优化。

## 未通过的配置

- correctness-qwen8b-piecewise-dense-r0: strict output hash gate failed within the same layout/condition; original outputs retained；[完整错误与命令](failures/correctness-qwen8b-piecewise-dense-r0.json)。
- correctness-qwen8b-piecewise-native-r0: strict output hash gate failed within the same layout/condition; original outputs retained；[完整错误与命令](failures/correctness-qwen8b-piecewise-native-r0.json)。
- correctness-sync-qwen8b-piecewise-dense-r0: strict output hash gate failed within the same layout/condition; original outputs retained；[完整错误与命令](failures/correctness-sync-qwen8b-piecewise-dense-r0.json)。
- correctness-sync-qwen8b-piecewise-native-r0: strict output hash gate failed within the same layout/condition; original outputs retained；[完整错误与命令](failures/correctness-sync-qwen8b-piecewise-native-r0.json)。
- hybrid-qwen35-27b-piecewise-screen: PyTorch worker initialization abort: Invalid thread pool! INTERNAL ASSERT in ATen/ParallelOpenMP.cpp; model never initialized；[完整错误与命令](failures/hybrid-qwen35-27b-piecewise-screen.json)。
- official-default-screen: driver does not support graph operator update (207019)；[完整错误与命令](failures/official-default-screen.json)。
- official-full_decode_only-screen: driver does not support graph operator update (207019)；[完整错误与命令](failures/official-full_decode_only-screen.json)。

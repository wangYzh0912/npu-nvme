# AIC/AIV 训练与推理活跃度：环境与可行性评估

评估日期：2026-10-30。范围：只读环境检查、现有训练调用链与历史 profiling 数据检查、官方文档核对。本轮没有启动模型训练、推理服务或硬件性能采集。

## 结论

可以开展该实验。建议推理采用 vLLM-Ascend，训练采用已有 MindSpore 2.7.1 + MindFormers 1.7.0 的 Qwen3-8B TP4 基线。训练侧已有真实运行和 profiler 数据基础；推理侧有本地镜像，但尚未验证适合本机旧驱动的运行组合。

最终图建议为**同一张图片、上下两个面板**：上方推理，下方训练；各有 AIC、AIV 两条折线，颜色和纵轴口径一致。面板内时间从各自测量起点开始，避免把两次独立运行画成连续物理时间。背景色带和标注展示阶段，保留真实时间长度。

主要不确定性是能否得到两类核心**同口径、带时间戳的硬件采样**，以及采样分辨率是否足以分辨 sampling 等短阶段。不能先画出经验曲线，再将其作为实测结果。

## 已检查的环境

| 项目 | 现场观察 | 对本任务的影响 |
| --- | --- | --- |
| 主机 | aarch64，约 2 TiB 主存 | 支持现有 Ascend 训练环境与 ARM 容器 |
| NPU | 8 × Ascend 910B3，每卡 65536 MiB HBM；健康状态均为 OK | 有运行 TP4 训练和单卡/TP4 推理的硬件基础 |
| 即时状态 | `npu-smi info` 未列出 NPU 进程，AICore 读数为 0；每卡约 3.4 GiB HBM 已用 | 是检查时的空闲快照，不证明任意时刻可独占设备 |
| 驱动 | 24.1.rc3 | 不能依据容器能启动就认定新 CANN 与旧驱动兼容 |
| 固件 | `npu-smi info -t board -i 0` 的 Firmware Version 返回 NA | 尚未获得可用于兼容矩阵核对的固件版本 |
| 默认 shell | Python 3.13.2；未安装 MindSpore、torch、torch-npu、vLLM | 不可直接用当前 `python` 启动模型任务 |
| 默认 CANN | `/usr/local/Ascend/ascend-toolkit/latest`，版本 8.0.RC3 | 与训练基线不同，必须显式选择环境 |
| 训练独立环境 | Python 3.11.4、MindSpore 2.7.1、MindFormers 1.7.0、Transformers 4.51.3、CANN 8.3.RC1 | 可复用训练基线；不需要为此次评估更换训练框架 |
| 训练 Python | `/models/npu_nvme_exp/user7-stack/conda/bin/python` | 应通过已有隔离环境构造逻辑启动 |
| 训练环境 manifest | `/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json` | 配置明确选择 `candidate`，不依赖 shell 的 `latest` |
| 模型 | `/models/Qwen3-8B`，BF16 原始权重约 16.38 GB（十进制） | 可固定同一模型比较训练和推理 |
| 磁盘 | `/models` 约余 24 GiB，`/home` 约余 21 GiB | 模型转换缓存和 trace 的空间预算是实际约束 |
| 工具 | 两套 CANN 均有 `msprof`；候选环境未安装 matplotlib | 采集与绘图环境可以分离，绘图依赖不必装入已验证训练环境 |
| 粗粒度状态接口 | `npu-smi info -t usages -i 0` 分别返回 Aicore / Aivector Usage Rate | 可作粗粒度交叉检查；尚未验证底层刷新周期，不适合直接归因到短阶段 |

没有创建、修改或停止现有容器。对现有容器只读取了软件元数据、进程名称与 CANN 安装路径。

本地镜像包括 `v0.13.0rc1-openeuler`、`nightly-releases-v0.18.0-openeuler`、`v0.30.0rc1`。运行中的 `wn-vllm-ascend-030` 容器读取结果为：

- Python 3.12.13，torch 2.10.0+cpu，torch-npu 2.10.0.post4。
- vLLM `0.30.0+empty`，vLLM-Ascend `0.30.0rc1+multigraph.732514948`，Transformers 5.14.1。
- CANN 路径为 `/usr/local/Ascend/cann-9.1.0`，不是宿主训练用的 8.3.RC1。

这些是包元数据，不是 NPU 推理通过证据。该容器有定制版本标识和其他工作目录，正式实验应创建本任务自己的隔离环境并记录镜像 digest、代码提交及实际加载库。

官方当前安装页列出的验证组合包含 HDK 26.0.RC1、CANN 9.1.0、Python 3.12，与本机驱动不同。官方历史兼容矩阵列出 vLLM-Ascend **0.13.0rc1 对应 CANN 8.3.RC2**，也不能把现有训练的 **8.3.RC1** 直接视为匹配。现有旧镜像可作为兼容性冒烟候选，本轮未确定最终推理版本，也未证明必须升级驱动。

## 训练入口和可复用证据

当前调用链：`train.py` → `npu_nvme.cli.qwen_training.main/run_fit` → 隔离环境启动 `mindspore.parallel.cluster.run` → `npu_nvme.workloads.qwen` → MindFormers `Trainer`。这是本轮阅读的实际实现，不以入口文件大小推断能力。

当前配置是 Qwen3-8B、TP4/DP1/PP1、batch size 1、sequence length 128、FP32 参数与 BF16 计算。比较图必须记录这些条件；训练和推理的批大小、序列、KV cache 和并行方式不同，其曲线不能解读为框架效率排名。

`python/npu_nvme/workloads/qwen.py` 已有在研究配置预热结束后开启 `ms.Profiler` 的逻辑，采用 `Level1` + `PipeUtilization`。但开关归属于 `incremental_experiment`；普通 `train.py fit --method none` **不会自动采集所需数据**。后续需要独立的 profiling 配置/入口，复用训练调用链，不为采集而启动增量扫描或 Top-K 辅助负载。

历史证据根目录：

```text
/models/npu_nvme_exp/user7-stack/incremental-phase1-20260916-004/
  runs/main-b0-rep0-profile/rank_0/profiler/
```

本轮实际读取了其中 `op_summary_*.csv` 和 `kernel_details.csv`，各有 247080 行，包括：

- `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC`、`MIX_AIV`、`COMMUNICATION` 等类型。
- `Task Start Time(us)`、`Task Duration(us)`、`Block Dim`、`Mix Block Dim`。
- `aic_total_cycles`、`aiv_total_cycles`、`aic_mac_ratio`、`aiv_vec_ratio` 等 PMU 字段。
- 前向网络名称、`Gradients/` 反向名称和 `optimizer-AdamW` 名称。

对应旧运行 rank 0 的校验封装 training.json 记载 `status=pass`、`method=none`、`final_step=24`。设备元数据记载该次运行 20 个 AIC、40 个 AIV、两者频率 1800 MHz；新采集仍须读取本次设备元数据，不直接硬编码这些数值。

这证明该机器曾成功导出两类核心的**算子级**数据和训练阶段名称；没有证明已经得到本次目标的 AIC/AIV **硬件采样时间序列**。已有 `experiments/timeline.py` 明确输出 `task_activity_overlap`，并将 `core_occupancy` 留空。旧脚本不能直接作为硬件活跃率出图工具。

## 图表应测什么

优先定义纵轴为 **AIC / AIV 核心忙碌时间比例（%）**：同一时间窗内，全体对应核心执行任务的核心时间，除以该类核心总数乘以窗口长度。两条曲线分别归一化，不能用 AIC 的核心数归一化 AIV，也不能把二者相加理解为总利用率。

如果使用采样 PMU，可按以下概念式计算：

```text
U_engine(window) = 100 × sum(per-core busy time in window)
                         / (engine core count × window duration)
```

busy cycles 转时间需要该次采集的频率和计数器定义。需验证 `task_cyc` 是窗口计数还是累积计数、首个采样点的时间跨度、计数器复位、缺失核心、缺失样本和时钟映射；不能一律差分、补零或把异常值裁剪后继续宣称结果有效。忙碌包括任务内部的等待，不等于 MAC/Vector 算术吞吐达到峰值。

| 数据 | 可以回答的问题 | 不能直接声称的结论 |
| --- | --- | --- |
| sample-based、逐核心、带时间戳 PMU | 在已验证采样窗口中的 AIC/AIV 忙碌比例 | 比采样周期更细的阶段内波动 |
| 算子 `aic_total_cycles/aiv_total_cycles` | 每个算子的两类核心周期消耗及聚合量 | 周期在整个算子区间内均匀分布 |
| `aic_mac_ratio/aiv_vec_ratio` | 该算子/采样指标定义下的管线指令周期比例 | 整卡忙碌率，或二者相减得到可用算力 |
| 算子任务区间并集 | 时间窗内至少一个该类任务存在的比例 | 有多少核心忙碌、混合算子内部何时同时执行 |
| `Block Dim/Mix Block Dim` | 任务调度几何信息 | 各 block 精确开始/结束时间或实时占用核心数 |
| `npu-smi` 两类 usages | 较长时间尺度上的状态检查 | 微秒/毫秒级采样阶段归因 |

任务区间并集可以制作另外一种明确命名为“任务活跃时间比例”的图。若硬件采样不满足要求，应公开改变图表口径并说明限制，不能静默改用这一指标并保留“核心忙碌率”的标题。

## 采集能力与时间分辨率

候选 CANN 8.3.RC1 的 `msprof --help` 已核实支持：`--ai-core`、`--aic-mode=sample-based`、`--aic-freq=1..100`、`--aic-metrics=PipeUtilization`、`--task-time`、`--msproftx`；也列出 910B 的 `--instr-profiling`。这些是接口存在证据，不是采集成功证据。

100 Hz 对应约 **10 ms** 的采样周期。单次 sampling 或短 decode 可能小于该周期；把采样数据插值成 0.1 ms 折线并不能增加真实分辨率。细阶段应结合任务 trace 展示，必要时另行验证指令级采集能提供的粒度与开销。

不能仅查找两个名为 utilization 的 CSV：官方说明 `ai_vector_core_utilization_*.csv` 是汇总结果，不含时间线。本机 8.3.RC1 的实际工具源码也确认：

- `msinterface/msprof_export_data.py` 的 AIC 分支支持 timeline；AIV 分支调用汇总输出。
- `viewer/ai_core_report.py` 从 `MetricSummary` 输出逐核心及 Average 汇总。
- `msparser/aic_sample/ai_core_sample_parser.py` 有 AIC/AIV sample-based 解析器。
- `msmodel/aic/ai_core_sample_model.py` 在 `AICoreOriginalData` 保留 `timestamp`、`coreid`、`task_cyc` 和事件计数。

因此，**从本机版本的采样原始数据/分析数据库读取 AIV 时间序列是有源码依据的候选路线**，但必须先在实际 910B3 上采到数据再确认。不能仅凭源码分支存在宣称驱动采集链路已打通。解析器应固定到采集工具版本。

系统 PMU 采样和框架 trace 应尽量在同一实验中取得，并验证能否共存。不能未经验证同时启动两个争用同一 profiling 通道的采集器；若只能分开重放，只能声明是相同工作负载的两次运行，不能逐点声称属于同一次设备执行。

## 阶段标注

| 场景 | 目标标签 | 归因方法 |
| --- | --- | --- |
| 推理 | prefill 前向、首次 sampling、decode 前向、后续 sampling、调度/通信/空隙 | 固定实际版本的 vLLM worker/model runner/sampler 调用链；用 profiler 关联的设备任务和 MSTX/框架 range 匹配 |
| 训练 | 输入准备、前向、loss、反向、梯度通信、optimizer update | MindSpore 图/算子 scope、kernel details、必要的 MSTX 打点；核对融合、重计算和通信重叠 |

sampling 应使用实际随机采样配置（固定 seed，记录 temperature/top-p 等），而不是把 greedy argmax 任意称作随机采样。它可能包含 CPU 工作、NPU 向量算子和融合操作，以实际执行 trace 为准。

Python 函数返回常常只代表异步下发完成。Host MSTX 区间不能直接涂成 Device 执行阶段，不能在每个阶段加全设备同步来强制制造边界。使用 profiler 的关联关系及其时钟换算，将阶段对齐到设备任务；不直接相减 `time.monotonic_ns()` 与导出 wall-clock 时间戳。

融合算子、重计算和跨 stream 重叠应保留证据。主标签使用独立阶段色带，通信等可并列显示；无法归类的事件保留 unknown，不用关键词兜底成“前向”。

## 建议实施顺序

1. **验证测量口径。** 使用本任务的独立输出目录与仓库共享硬件锁，在空闲的单张卡上做 idle、长矩阵计算、长向量计算、混合计算短探针。使用同一 CANN 环境，验证 AIC/AIV 非零样本、核心数、窗口含义、时间戳和落盘空间。预设小采集时长和空间上限；若限制触发或有数据丢失，结果不作为正式证据。此阶段无需加载 8B 模型。
2. **验证推理环境。** 以本地镜像为候选、固定版本；先做设备初始化与一次最小真实推理，再确定驱动/CANN/PyTorch/插件组合。若现有版本不能工作，再明确具体兼容阻断项，不先修改宿主驱动。
3. **训练采集。** 复用 Qwen3-8B TP4，`method=none`，关闭 checkpoint 和研究辅助负载；编译和预热后只采集少量稳态完整 step。记录 loss 与 optimizer step 完成证据。普通 fit 需接入独立 profiler 开关。
4. **推理采集。** 同模型先做固定长度单请求，建议初始输入 128 token、输出 32 token、关闭提前 EOS（参数按选定版本核对），固定随机 seed。单卡可以先验证；若用于与 TP4 训练对照的正式图，应优先使用相同物理卡组和 TP4，并在图注列出所有不同条件。预热后采集一次 prefill 与多个 decode/sampling 循环。
5. **标注与出图。** 选择完整训练 step 和完整推理片段；每张卡分别保存原始结果，主图选明确的一张卡，检查其他 rank 是否存在异常差异。不要把未经对齐的 rank 曲线直接平均。输出 PNG/PDF 或 SVG、时间序列 CSV、阶段区间 CSV、环境与运行元数据。另做未开启 profiler 的小样本运行量化采集扰动，性能比较不混用带 profiler 和无 profiler 结果。

训练基线会转换模型格式，当前又使用 FP32 参数，转换缓存可能达到数十 GiB；24 GiB 余量不应直接视为足够。复用缓存必须核对权重和转换身份，或先安排足够空间的独立输出位置；本轮没有删除已有数据、拉取大镜像或复制模型。

时间序列至少包含：`run_id, scenario, physical_device_id, rank, window_start_us, window_end_us, aic_activity_pct, aiv_activity_pct, metric_kind, valid`。阶段区间单独包含：`stage, device_start_us, device_end_us, evidence_source`，允许重叠。图注必须标出采样周期、设备、模型、并行、batch/序列长度、精度、测量口径与未知区间。

## 本轮交付与未验证边界

本轮已交付本评估及同目录的环境快照：`environment.json`。已有工作区包含其他未提交变更，本轮只新增这两个文件。

已经确认硬件与工具存在、训练环境包版本、实际训练/profiler 调用链、历史数据字段、推理容器包版本和磁盘余量。尚未验证本次硬件采样、vLLM 模型执行、双采集器共存、AIV 数据库采样内容及设备阶段归因，因此本轮不交付或宣称已有完整实测双折线图。

## 官方依据

以下页面于本轮在线核对。`latest` 是滚动文档，不能代替安装版本说明；8.3 文档路径为 alpha003，也需与本机 8.3.RC1 帮助及源码交叉核对。

- [vLLM-Ascend 当前安装要求](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html)：当前验证组合及镜像安装边界。
- [vLLM-Ascend 0.13 兼容矩阵](https://docs.vllm.ai/projects/ascend/en/v0.13.0/community/versioning_policy.html)：区分 rc1 的 CANN 8.3.RC2 与后续版本。
- [vLLM-Ascend Service Profiling Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)：当前 `--profiler-config`、start/stop profile 和服务层采集。旧版本参数需另核对。
- [CANN 8.3 系统数据采集](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/Profiling/atlasprofiling_16_0012.html)：sample-based 系统采集及框架数据边界。
- [AI Core utilization 定义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0069.html)：采样周期内任务执行占比与管线指令比例的区别。
- [AI Vector Core utilization 定义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0070.html)：AIV 汇总 CSV 不包含 timeline。
- [MindSpore 2.7.1 ExperimentalConfig](https://www.mindspore.cn/docs/zh-CN/r2.7.1/api_python/mindspore/mindspore.profiler._ExperimentalConfig.html)：NPU activities、PipeUtilization、mstx 开关。
- [MindSpore 2.7.1 Ascend 性能调优](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/debug/profiler.html)：训练性能数据、MSTX 和导出文件。

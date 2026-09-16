# 6 当前问题及下一阶段计划

## 6.1 当前存在的主要问题

### 6.1.1 数据通路的正确性与工程健壮性仍需收敛

目前基于 ACL 与 SPDK 的全量检查点正常读写路径已经打通，但项目仍处于“可运行原型”向“可稳定复现实验系统”过渡的阶段。首先，当前读取状态机中 Host 目标和 NPU HBM 目标的地址类型、拷贝方向与尾块处理需要重新梳理，并在目标环境中分别执行逐字节回读和设备参数恢复验证。现有结果能够支持全量写入性能结论，但尚不足以给出统一的读取带宽、恢复时间和恢复后模型输出一致性。

其次，底层错误状态尚未完整贯通。SPDK completion 失败、ACL 异步拷贝失败和请求环提交失败需要经由有限状态机、C ABI 和 Python 层逐级返回；同步接口应设置超时，Fire-and-Forget 接口则需要通过明确状态区分“正在写入、数据写入完成、元数据已提交、检查点可恢复”和“持久化失败”。否则，当设备异常或请求未取得进展时，上层可能出现无限等待，或者过早把尚未提交元数据的数据视为有效检查点。

再次，初始化和清理阶段的资源所有权仍需进一步统一。Reactor 线程、数据 qpair、meta qpair、DMA 缓冲区、请求环和 ACL context 之间存在明确依赖关系，任何中间步骤失败都应按相反顺序回收已经创建的资源。当前需要重点检查初始化失败回滚、meta qpair 重复释放风险以及 cleanup 期间在途请求的排空或取消。该问题通常不会影响正常性能测试，却直接决定系统在重复运行、异常退出和自动化回归中的稳定性，应在正式实验前优先解决。

### 6.1.2 裸盘空间与多版本元数据尚未形成统一闭环

当前 FULL 检查点区域主要由 Python 层根据设备容量、rank 数量和保留版本数进行规划，Delta ring 则由 C 层独立从设备尾部或指定区域分配。两套布局尚未共享统一空间分配信息，在边界条件下存在区域重叠的风险。下一阶段需要在超级块或独立布局描述中固定 metadata、FULL slots、Delta ring 和保留空间的起止偏移，并在格式化时完成容量检查，使所有读写任务在提交前都能进行越界验证。

双元数据槽已经能够支持活动版本切换，但多 rank 场景下“各 rank 数据完成”和“全局检查点提交”之间的关系尚未定义完整。若不同 rank 在不同时间更新活动元数据，故障恢复时可能得到由不同训练步组成的混合版本。后续需要引入全局 step、rank 完成位图或协调提交者，明确只有当所需 rank 的数据和局部元数据全部完成后，才能切换全局有效版本。与此同时，还需为元数据加入长度、格式版本和校验字段，并保留旧版本以支持提交中断后的回退。

### 6.1.3 Reactor 异常路径和并发压力尚未充分验证

单所有者 Reactor 已经解决 qpair 被多个线程直接操作的问题，但目前验证主要集中在正常的大块顺序写路径。读、写、元数据和 step poller 同时存在时的公平性、长尾延迟与队列背压仍缺少系统测试。请求环满时应明确选择阻塞、返回忙或丢弃非关键触发；当上一 Fire-and-Forget 任务尚未完成时，新的检查点触发需要选择合并、跳过或排队，不能只依赖隐含状态。

此外，Reactor 的 CPU 占用需要用统一工具和采样口径重新测量。阶段记录中的空闲 CPU 占用低于 1%，但缺少核心绑定、采样区间和活跃/空闲状态的完整日志。后续实验应区分空闲轮询、全量写入、全量读取、元数据提交和 Delta 写入，记录 Reactor 所在核心利用率、系统总 CPU 利用率、轮询次数和 I/O 完成延迟，从而说明单线程控制平面在低负载和高负载下的资源代价。

### 6.1.4 NPU 增量压缩尚未完成端到端恢复协议

第三条研究主线已经完成图内 Delta、块范数、Top-K 和 INT8 量化的关键可行性验证，但离完整检查点仍存在若干缺口。首先，`P_old` 使用低精度表示时需要为每个块保存正确的 scale，并保证参数更新、反量化和下一步差分使用一致版本；较小参数也需要接入独立打包路径。其次，图内输出的 block index 必须稳定映射到参数编号和参数内偏移，否则持久化后的 Delta 无法在恢复时定位目标数据。

目前 FaF 任务注册使用的原始缓冲格式、Delta ring 中的帧布局以及 Python 恢复协议尚未完全统一。对于大于单个 chunk 的量化缓冲，需要拆分为多个任务，但同一 Delta 帧的 header、index、scale 和 qdata 又必须作为一个逻辑版本提交。下一阶段需要定义自描述帧头、有效长度、基线 FULL 标识、step、块大小、有效块数、各数据段偏移和校验值，并规定 ring slot 推进和覆盖条件。

现有长期实验还表明，固定 Top-10% 会持续遗漏小幅变化的数据块，导致 NRMSE 随恢复链增长而显著累积。该问题不能仅通过提高 NVMe 带宽解决，需要在算法层引入残差反馈、自适应 Top-K 或阈值选择，并通过周期性 FULL 限制链长。图模式兼容问题导致的 Host 侧临时处理和较高附加开销也需要消除。只有完成“图内生成—异步写入—版本提交—FULL+Delta 重放—恢复后训练”的闭环，才能评价第三项方案的实际有效性。

### 6.1.5 正式实验和复现材料仍不完整

当前项目保存了若干 JSON 结果和阶段文档，但实验产生于不同版本，部分计时边界、单位和参数配置并不一致。现有约 56 GB/s 的异常流水线带宽、无有效量化输出时计算得到的开销以及与 JSON 不一致的旧 NRMSE 图片均不应进入正式结果。读取性能、完整恢复时间、当前版本 Fire-and-Forget 训练干扰和 Reactor CPU 占用也需要补测。

下一阶段需要统一记录代码提交、硬件配置、软件版本、命令行参数、原始日志和结构化结果；图表应只从结构化结果自动生成，并在标题或表注中注明样本数、均值、标准差和计时范围。对于 MindSpore 原生检查点、CheckFreq/PCcheck 类两阶段方案和本项目 SPDK 后端，应明确比较对象是否包含序列化、设备到主机复制、数据持久化和元数据提交，避免不同口径的数字直接横向比较。

## 6.2 下一阶段研究任务

### 6.2.1 建立正确性和资源安全基线

第一项任务是修复读取目标、错误传播、等待超时、初始化回滚、清理顺序和 FULL/Delta 区域冲突，建立可重复执行的正确性回归。测试应覆盖 Host 与 NPU 两类缓冲区、整块与非整块长度、多个参数跨块、不同 pipeline depth、设备边界、元数据槽切换、重复初始化—清理以及模拟 I/O 失败。每次保存后执行回读比较，并检查旧活动元数据在新版本提交失败时仍然可用。

该阶段还将整理无硬件环境可运行的单元测试，把分块映射、对齐计算、磁盘布局、元数据编码、版本选择和 Delta 帧解析从硬件 I/O 中分离出来。目标环境恢复后，再运行 ACL/SPDK 集成测试。只有正确性和资源生命周期回归通过，后续性能结果才进入正式图表。

### 6.2.2 完成全量检查点正式实验

第二项任务是在固定代码版本上测量 FULL 保存、加载和恢复。实验将扫描 1、2、4、8、16 MiB chunk size，多种 pipeline depth、模型状态规模和检查点间隔，分别记录 Python API 可见时间、C 层批量 I/O 时间、元数据提交时间、有效带宽、CPU 占用和长尾。每组配置至少重复多次，报告均值和离散程度，并统一使用 GB/GB/s 或 GiB/GiB/s 口径。

训练实验将比较纯训练、同步保存、仅 step poller、后台全量写入和完整 Fire-and-Forget。若实验条件允许，将增加 MindSpore 原生保存和已有两阶段异步基线，以训练停顿、平均 step time、吞吐下降、Host 内存占用和 CPU 占用作为指标。读取侧记录从调用加载到参数可以继续执行的端到端时间，并通过固定输入下的模型输出或继续训练损失验证恢复正确性。

### 6.2.3 完成 FULL + Delta 端到端闭环

第三项任务是统一 NPU 图内输出与裸盘 Delta 帧协议。项目将补齐 per-block scale、小参数和稳定 block mapping，把 qdata、scale、index 和 header 按 chunk 注册到 FaF 写入路径，并实现 Delta ring 的 slot 选择、版本提交和覆盖保护。恢复侧从最近有效 FULL 开始，按照 step 顺序读取并应用 Delta，校验每一帧的基线版本、长度和校验值。

在正确恢复基础上，实验将扫描 block size、Top-K 比例、FULL 间隔和 Delta 链长度，比较固定 Top-K、残差反馈和自适应策略。评价指标包括 Delta 数据量、图内附加时间、持久化时间、单步与多步 NRMSE、恢复时间以及恢复后训练损失/精度。目标不是追求单次最高压缩倍数，而是在可接受训练开销和恢复误差下得到稳定的数据缩减效果。

### 6.2.4 完成系统对比、论文图表和复现整理

第四项任务是在三条主线收敛后固化最终实验提交，统一生成架构图、状态机图、时间线和性能图表。所有图表使用同一结果目录和生成脚本，图注说明环境、样本数、误差棒、单位和计时范围。对于负面结果，如固定 Top-K 的误差累积，也应保留并解释其对后续设计的推动作用。

代码整理包括公共 API 文档、依赖版本、设备格式化警告、运行命令、结果目录规范和最小复现实验。论文撰写将同步更新系统设计、实现、实验方法和局限性，避免报告内容与最终代码版本脱节。

## 6.3 下一阶段进度安排

| 时间 | 主要任务 | 预期输出与验收条件 |
|---|---|---|
| 2026 年 8 月—9 月 | 修复读取、错误传播、超时、初始化/清理和磁盘区域冲突；建立单元测试与硬件回归 | Host/NPU 写入—读取一致；重复初始化—清理通过；异常请求能够返回明确错误；FULL 与 Delta 区域无重叠 |
| 2026 年 9 月—10 月 | 完成全量保存、加载、chunk size、pipeline depth、FaF 和 CPU 占用正式实验 | 当前版本结构化日志齐全；形成 FULL 延迟/带宽、读取/恢复和训练干扰正式图表 |
| 2026 年 10 月—12 月 | 统一 Delta 帧协议，补齐 scale、小参数、残差和 FULL+Delta 恢复；开展参数扫描 | 完成多步保存与恢复；给出数据比例、图内开销、NMRSE、恢复时间和训练质量权衡 |
| 2027 年 1 月—2 月 | 固化系统版本，补充对比实验与消融实验，完成论文主体 | 架构、实现和实验章节与代码一致；全部图表能够由脚本重现 |
| 2027 年 2 月—3 月 | 论文修改、材料整理和答辩准备 | 完成论文定稿、代码与实验说明、答辩演示及问题清单 |

## 6.4 预期成果与验收指标

下一阶段预期形成一套可稳定运行的 Ascend NPU—NVMe 高频检查点原型。系统层面应实现全量检查点的可靠保存与加载、统一 Reactor 异步控制、训练步触发和持久化完成通知；增量层面应实现图内 Delta 生成、自描述帧持久化以及 FULL+Delta 多步恢复。工程层面应具备错误返回、超时、资源清理、磁盘边界检查和重复回归能力。

正式实验至少应回答五个问题：第一，ACL+SPDK 通路相对于框架原生路径能够达到怎样的保存和加载带宽；第二，单所有者 Reactor 是否在保证 qpair 安全的同时保持数据面性能；第三，Fire-and-Forget 对训练 step time 和 CPU 的实际影响是多少；第四，NPU 图内压缩在扣除计算开销、索引、scale 和周期性 FULL 后能够减少多少写入量；第五，不同 Delta 链长度下的恢复误差、恢复时间和训练质量如何变化。

阶段验收不预先写死必须达到某个不具备充分实验依据的性能倍数，而采用可复现、可比较的指标判断。全量路径要求数据和模型状态恢复正确，性能结果具有稳定重复性；Reactor 路径要求不存在 qpair 越权访问和资源泄漏，异常能够返回；增量路径要求帧可独立解析、恢复链可重放，并给出压缩率—开销—精度之间的实测边界。

## 6.5 风险与应对措施

若目标 NPU/SPDK 环境长期不可用，项目将优先完成磁盘布局、元数据、请求环状态转换和 Delta 协议的无硬件测试，保留硬件相关接口 mock，并提前整理一键回归脚本，以缩短环境恢复后的测试时间。对于无法获得完全一致实现的外部检查点基线，将采用明确计时边界的框架原生保存和两阶段异步原型作为对比，不以不公平口径追求倍率。

若图内 Top-K 或旧状态更新在 MindSpore 2.5 GRAPH_MODE 中持续受算子支持限制，将分别评估可替代的图内算子组合、定制算子或分阶段执行方案，同时保留“图内完成大部分筛选、少量 Host 控制”的降级路径，并量化其中额外数据搬运和训练开销。若长期误差无法在固定 Top-K 下控制，则以残差反馈、动态比例和缩短 FULL 间隔作为主要策略，将恢复精度置于压缩倍数之前。

若优化器状态使全量检查点远大于当前参数检查点，系统将复用现有批量 API 和裸盘布局扩展状态类别，在元数据中区分参数、优化器和训练控制状态；实验则分别报告仅模型参数与完整可恢复状态，避免把较小口径结果误认为完整训练检查点性能。

# 7 主要参考文献

[1] GRATTAFIORI A, DUBEY A, JAUHRI A, et al. The Llama 3 Herd of Models[EB/OL]. arXiv:2407.21783, 2024[2026-07-30]. https://arxiv.org/abs/2407.21783.

[2] SPDK PROJECT. SPDK NVMe Driver[EB/OL]. [2026-07-30]. https://spdk.io/doc/nvme.html.

[3] SPDK PROJECT. Message Passing and Concurrency[EB/OL]. [2026-07-30]. https://spdk.io/doc/concurrency.html.

[4] NVIDIA CORPORATION. GPUDirect Storage Overview Guide[EB/OL]. [2026-07-30]. https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html.

[5] QURESHI Z, MAILTHODY V S, GELADO I, et al. GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture[C]//Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2023: 325-339. DOI: 10.1145/3575693.3575748.

[6] CHEN Y, LIU Z, REN B, et al. On Efficient Constructions of Checkpoints[C]//Proceedings of the 37th International Conference on Machine Learning. 2020: 1627-1636.

[7] EISENMAN A, MATAM K K, INGRAM S, et al. Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models[C]//19th USENIX Symposium on Networked Systems Design and Implementation. 2022: 929-943.

[8] MOHAN J, PHANISHAYEE A, CHIDAMBARAM V. CheckFreq: Frequent, Fine-Grained DNN Checkpointing[C]//19th USENIX Conference on File and Storage Technologies. 2021: 203-216.

[9] NICOLAE B, LI J, WOZNIAK J M, et al. DeepFreeze: Towards Scalable Asynchronous Checkpointing of Deep Learning Models[C]//20th IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing. 2020. DOI: 10.1109/CCGrid49817.2020.00-76.

[10] MAURYA A, UNDERWOOD R, RAFIQUE M M, et al. DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models[C]//33rd International Symposium on High-Performance Parallel and Distributed Computing. 2024. DOI: 10.1145/3625549.3658685.

[11] WANG G, RUWASE O, XIE B, et al. FastPersist: Accelerating Model Checkpointing in Deep Learning[R]. Microsoft Research Technical Report MSR-TR-2024-27, 2024.

[12] STRATI F, FRIEDMAN M, KLIMOVIC A. PCcheck: Persistent Concurrent Checkpointing for ML[C]//Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2025: 811-827. DOI: 10.1145/3669940.3707255.

[13] GUPTA T, KRISHNAN S, KUMAR R, et al. Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures[C]//Proceedings of the Nineteenth European Conference on Computer Systems. 2024: 1110-1125. DOI: 10.1145/3627703.3650085.

[14] CHEN M, HUA Y, BAI R, et al. A Cost-Efficient Failure-Tolerant Scheme for Distributed DNN Training[C]//2023 IEEE 41st International Conference on Computer Design. 2023: 150-157. DOI: 10.1109/ICCD58817.2023.00031.

[15] LIU H, LUO S, LI K, et al. CheckFlow: Low-Overhead Checkpointing for Deep Learning Training[J]. IEEE Computer Architecture Letters, 2025, 24(2): 281-284. DOI: 10.1109/LCA.2025.3596616.

[16] WANG Z, JIA Z, ZHANG S, et al. GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints[C]//Proceedings of the 29th Symposium on Operating Systems Principles. 2023: 364-381. DOI: 10.1145/3600006.3613145.

[17] WAN B, HAN M, SHENG Y, et al. ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development[C]//22nd USENIX Symposium on Networked Systems Design and Implementation. 2025: 559-578.

[18] LIAN X, JACOBS S A, KURILENKO L, et al. Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism[C]//2025 USENIX Annual Technical Conference. 2025: 1519-1534.

[19] WANG S, CAO Q, ZHOU K, et al. ParaCkpt: Heterogeneous Multi-Path Checkpointing Mechanism for Training Deep Learning Models[C]//2024 IEEE 42nd International Conference on Computer Design. 2024: 183-190. DOI: 10.1109/ICCD63220.2024.00036.

[20] LI Y, WU T, LI G, et al. Portus: Efficient DNN Checkpointing to Persistent Memory with Zero-Copy[C]//2024 IEEE 44th International Conference on Distributed Computing Systems. 2024: 59-70. DOI: 10.1109/ICDCS60910.2024.00015.

[21] PANDEY S, KAMATH A K, BASU A. GPM: Leveraging Persistent Memory from a GPU[C]//Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2022: 142-156. DOI: 10.1145/3503222.3507758.

[22] SUN J, SUN M, ZHANG Z, et al. Hyperion: Co-Optimizing SSD Access and GPU Computation for Cost-Efficient GNN Training[C]//2025 IEEE International Conference on Data Engineering. 2025: 321-335. DOI: 10.1109/ICDE65448.2025.00031.

[23] DIDONA D, PFEFFERLE J, IOANNOU N, et al. Understanding Modern Storage APIs: A Systematic Study of libaio, SPDK, and io_uring[C]//Proceedings of the 15th ACM International Conference on Systems and Storage. 2022: 120-127. DOI: 10.1145/3534056.3534945.

[24] AGRAWAL A, REDDY S, BHATTAMISHRA S, et al. Inshrinkerator: Compressing Deep Learning Training Checkpoints via Dynamic Quantization[C]//Proceedings of the ACM Symposium on Cloud Computing. 2024: 1012-1031. DOI: 10.1145/3698038.3698553.

[25] KIM Y, BELYAEV E. An Efficient Compression of Deep Neural Network Checkpoints Based on Prediction and Context Modeling[EB/OL]. arXiv:2506.12000, 2025[2026-07-30]. https://arxiv.org/abs/2506.12000.

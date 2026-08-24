# 中期报告参考文献与国内外研究现状调研

> 调研基准：工作区《开题报告.docx》及当前项目技术路线  
> 检索时间：2026 年 7 月 30 日  
> 当前路线：SPDK 用户态检查点数据通路、单所有者 Reactor 控制平面、NPU 图内增量检查点生成与压缩  
> 使用原则：优先引用正式发表论文、会议官方页面、作者公开论文和官方技术文档；预印本只作为前沿补充

## 1. 调研结论

开题报告原有的“异步检查点—分布式检查点—GPU 直通 SSD”框架仍可保留，但需要补充“增量与压缩检查点”，并将 GPU 直通存储一节调整为“加速器直通存储与用户态 NVMe 数据路径”。推荐形成以下叙事：

1. 异步与高频检查点研究解决“如何隐藏检查点开销”；
2. 分布式检查点研究解决“如何组织、分片、重分片和恢复大规模训练状态”；
3. 加速器直通存储和 SPDK 研究解决“如何缩短数据传输与 I/O 软件路径”；
4. 增量与压缩检查点研究解决“如何减少需要持久化的数据量”；
5. 现有研究尚未同时覆盖 Ascend NPU、用户态裸盘路径、SPDK 队列所有权控制，以及 NPU 图内增量数据生成，由此引出本课题。

中期报告中的“国内外研究现状”应按技术问题组织，而不是先机械地分为“国外研究”和“国内研究”两大段。每个技术方向中分别介绍国内外代表工作，最后再进行总体比较，更容易形成完整的问题链。

## 2. 可直接用于中期报告的研究现状内容

### 2.1 深度学习训练检查点及其性能问题

大规模深度学习训练通常需要周期性保存模型参数、优化器状态、学习率调度状态、随机数状态和数据加载进度，以便在设备故障、进程异常或资源调整后恢复训练。随着模型规模和训练并行度提高，检查点由单机文件保存演变为多节点、多分片并发持久化过程。传统同步检查点会暂停训练并依次完成快照、序列化和存储写入，造成加速器空闲；直接提高保存频率虽然能够减少故障后的重算量，却会进一步放大训练停顿、主机内存占用和存储带宽压力。

因此，现有研究主要从四个层面优化检查点：利用异步流水线隐藏保存时间，利用分布式格式和分层存储提升可扩展性，利用加速器直通或用户态 I/O 缩短数据路径，以及利用差分、稀疏化和量化减少实际写入量。

### 2.2 异步与高频检查点

国外研究较早从训练计算与检查点 I/O 重叠的角度降低停顿。CheckFreq 将检查点拆分为快照和持久化两个阶段，并根据在线分析结果动态调整检查点频率；其报告的运行开销控制在 3.5% 以内，同时将故障后丢失的训练进度由小时级降低到秒级。DeepFreeze 进一步将序列化和 I/O 分散到参与训练的进程中，以异步方式隐藏检查点开销。DataStates-LLM 利用模型参数在前向和反向计算阶段保持不变的特点，预分配并复用锁页主机缓冲区，实施惰性、非阻塞快照和多级流式持久化。FastPersist 则从 NVMe 写入、多 SSD 并行和计算重叠三个方面优化持久化过程。PCcheck 允许多个检查点并发处于复制和持久化阶段，通过分块以及 GPU—CPU、CPU—存储流水线支持高频保存。

国内团队也围绕细粒度异步保存展开了系统研究。华中科技大学提出的 LightCheck 根据训练过程中的层级数据依赖执行细粒度异步检查点，并借助持久内存管理器降低 GPU 到持久介质的传输开销。CheckFlow 将快照创建、驻留和卸载动作纳入计算图调度约束，在控制峰值显存占用的同时安排检查点操作。上述工作说明，异步流水线和训练语义感知调度已经成为降低检查点停顿的主要方法。

但现有异步方案通常仍采用 GPU/NPU 到主机内存的快照，再经文件系统或常规存储接口持久化；同时，多线程复制、序列化和 I/O 容易引入额外的锁竞争和资源所有权问题。因而，仅将保存动作放入后台并不能完整解决底层数据路径和控制路径的开销。

### 2.3 分布式检查点格式、存储与恢复

大模型训练中的检查点通常分散在数据并行、张量并行和流水并行的不同 rank 上，保存格式还可能与训练时的硬件规模和并行策略绑定。Gemini 使用本机和远端 CPU 内存构成高带宽检查点层，并设计放置与通信调度策略，使故障恢复优先从更快的内存层完成。ByteCheckpoint 采用与并行策略无关的表示，在加载阶段执行重分片，并以通用保存/加载流程适配不同训练框架和存储后端。Universal Checkpointing 进一步将检查点结构与并行策略、硬件配置解耦，使训练可以在资源变化后采用不同的并行配置恢复。

国内产业界和高校在该方向已经形成较强的系统成果。ByteCheckpoint 由字节跳动与香港大学合作完成，面向大模型完整生命周期提供并行无关表示、存储后端适配和全栈 I/O 优化。华中科技大学与中国移动提出 ParaCkpt，面向异构存储路径并行组织检查点传输。上海科技大学提出 Portus，使用 GPU 到持久内存的 RDMA 数据路径和三级索引降低序列化、内存复制与文件系统开销。

这类研究重点解决跨节点分片组织、重分片、恢复位置和存储扩展性问题，而本课题更关注单节点内 NPU 到本地 NVMe 的持久化路径及其控制平面。两类工作处于不同层次，可以在报告中表述为互补关系，而不应将本项目包装成分布式检查点格式创新。

### 2.4 加速器直通存储与用户态 NVMe 数据路径

NVIDIA GPUDirect Storage（GDS）为 GPU 显存与本地或远端存储提供直接 DMA 数据路径，避免经由 CPU 内存中的中转缓冲区，从而降低 CPU 占用和额外拷贝。BaM 将存储访问的发起进一步下沉到 GPU，通过 GPU 线程按需访问 NVMe SSD；GPM 则研究 GPU 对持久内存的细粒度访问和持久化语义。这些工作证明了缩短加速器—存储路径对于数据密集型负载的重要性，但其硬件与软件生态主要面向 NVIDIA GPU。

SPDK 从另一条路线优化存储访问：它将 NVMe 驱动移到用户态，使用异步队列和轮询方式绕过传统内核块层与文件系统。SPDK 官方编程模型强调，一个 NVMe queue pair 应由固定线程独占，跨线程操作应通过消息传递交给所有者线程执行，以避免 I/O 路径加锁。对 libaio、io_uring 和 SPDK 的系统比较也表明，轮询策略、CPU 核分配和多设备扩展方式会显著影响用户态存储栈性能。

本课题与 GDS、BaM 的区别必须明确：当前实现通过 ACL 在 NPU HBM 与 SPDK DMA 缓冲区之间搬运数据，再由 SPDK 将缓冲区写入 NVMe，属于文件系统旁路的用户态分块流水线，并非硬件 PCIe P2P，也不是完全零拷贝。其研究价值在于面向 Ascend NPU 检查点场景，将 ACL 数据搬运、SPDK 裸盘布局、异步请求和恢复语义组合为完整系统。

### 2.5 增量与压缩检查点

除隐藏 I/O 时间外，减少写入数据量是提高检查点频率的另一条重要路线。LC-Checkpoint 使用量化、重要信息优先保存和 Huffman 编码构造有损检查点，在其评测中获得最高 28 倍压缩率和 5.77 倍恢复加速。Check-N-Run 针对推荐模型嵌入表更新稀疏的特点，仅记录发生变化的模型部分，并使用量化减少检查点大小；其报告的写带宽需求降低 6～17 倍、容量需求降低 2.5～8 倍。Inshrinkerator 根据训练阶段和不同权重对压缩误差的敏感性动态选择量化精度，并结合面向差分数据的重排与压缩；其评测报告最高 39 倍压缩率，并验证了多次恢复条件下的精度影响。

现有增量方法存在两类局限。第一，Check-N-Run 的显著收益依赖推荐模型嵌入表的稀疏更新特性，不能直接等同于稠密 Transformer 参数的块级变化。第二，量化或稀疏化检查点需要在压缩率、生成开销、恢复时延和多次恢复后的误差积累之间权衡。仅验证单步量化误差不足以证明方案可用于长期训练，还需要周期性全量检查点、Delta 链长度限制、自适应块选择及恢复误差反馈。

本课题第三条路线的合适定位是：借鉴差分、Top-K 和量化思想，将块级变化检测和 INT8 数据生成放入 NPU 训练图中，利用闲置的 Vector 计算资源在数据离开 HBM 前完成筛选与压缩，并与后端 Reactor 和 SPDK 路径衔接。创新点不是单独提出 Top-K 或 INT8，而是压缩计算的放置位置、与 NPU 训练图的融合，以及 FULL + Delta 持久化与恢复协议。

### 2.6 国内外研究现状总结与本课题切入点

总体而言，国外研究在高频异步检查点、分层内存恢复、并行无关检查点格式、GPU 主导存储访问和动态量化压缩方面形成了较完整的技术谱系；国内高校和企业则在细粒度异步检查点、异构多路径、GPU—持久内存直通以及大模型工业级检查点系统方面取得了代表性成果。

现有研究仍呈现分层割裂：异步方案重点隐藏时间，分布式方案重点组织状态，直通存储方案主要面向 GPU 生态，压缩方案则多在 CPU 侧或针对稀疏模型。针对 Ascend NPU 训练，仍缺少一套同时考虑用户态 NVMe 数据通路、SPDK queue pair 单所有者控制、训练步级后台触发和 NPU 图内增量数据生成的检查点系统。本课题据此从“快传、稳控、少写”三个层面展开研究：

| 现有研究层面 | 已解决的主要问题 | 尚存不足 | 本课题对应工作 |
|---|---|---|---|
| 异步与高频检查点 | 计算、快照和持久化重叠 | 主机快照、文件系统和多线程控制开销仍存在 | 单所有者 Reactor、请求环和 FSM |
| 分布式检查点 | 分片、重分片、跨配置恢复 | 不直接解决节点内 NPU—NVMe 路径 | 提供可作为下层后端的本地持久化通路 |
| GPU 直通存储 | 减少 CPU 中转与内核路径 | 依赖 GPU/P2P 生态，缺少 NPU 检查点语义 | ACL + SPDK 用户态分块流水线 |
| 增量与压缩检查点 | 减少写带宽和存储容量 | 稀疏模型依赖、CPU 压缩或误差累积 | NPU 图内 Delta、Top-K、INT8 与 FULL + Delta |

## 3. 文献使用建议

### 3.1 建议作为正文核心引用

| 文献 | 放置位置 | 主要用途 |
|---|---|---|
| CheckFreq | 2.2 异步与高频检查点 | 两阶段检查点、频率调节和训练停顿问题 |
| DataStates-LLM | 2.2 异步与高频检查点 | LLM 语义感知的惰性异步快照 |
| FastPersist | 2.2、2.4 | NVMe 优化、多 SSD 并行和计算重叠 |
| PCcheck | 2.2 | 多检查点并发、分块流水线 |
| LightCheck | 2.2 国内研究 | 华中科技大学代表工作；层级流水线与持久内存 |
| Gemini | 2.3 分布式恢复 | 分层内存检查点与快速恢复 |
| ByteCheckpoint | 2.3 国内产业研究 | 并行无关格式、重分片与工业级全栈优化 |
| Universal Checkpointing | 2.3 | 检查点格式与并行配置解耦 |
| Portus | 2.3、2.4 国内研究 | GPU—持久内存直通、RDMA 和零拷贝 |
| SPDK 官方文档 | 2.4 | 用户态 NVMe、轮询、qpair 单线程所有权 |
| GDS | 2.4 | 硬件直接 DMA 路线及与本项目的边界 |
| BaM | 2.4 | GPU 主导 NVMe 访问 |
| Check-N-Run | 2.5 | 差分检查点和量化 |
| LC-Checkpoint | 2.5 | 检查点压缩的基础研究 |
| Inshrinkerator | 2.5 | 动态量化、差分压缩和恢复误差权衡 |

### 3.2 建议作为补充引用

| 文献 | 使用方式 |
|---|---|
| DeepFreeze | 说明异步序列化和分布式 I/O 的早期代表方案 |
| Just-In-Time Checkpointing | 说明“故障发生时再构造恢复状态”的另一类思路，不作为本项目直接基线 |
| CheckFlow | 说明检查点操作与训练计算图联合调度的近期国内研究 |
| ParaCkpt | 说明国内异构多路径检查点研究，并与开题阶段多路径设想衔接 |
| GPM | 说明 GPU 访问持久内存的持久化语义，不与 NVMe 路径混为一谈 |
| Hyperion | 说明 GPU 发起异步 SSD 访问与计算流水线协同，应用场景是 GNN 而非训练检查点 |

### 3.3 建议降级或删除

1. 只讨论一般 NVMe、传统文件系统或通用压缩算法，但没有连接到训练检查点、用户态 I/O 或加速器数据路径的文献，可从“研究现状”正文移到背景或删除。
2. 开题报告中的 2025 年预测与上下文建模检查点压缩预印本可作为前沿补充，不宜替代 LC-Checkpoint、Check-N-Run 和 Inshrinkerator 三篇正式发表工作。
3. GDS、BaM 和 GPM 应保留为技术背景，但不能被用来证明当前项目已经实现 NPU—NVMe P2P 或硬件零拷贝。
4. Universal Checkpointing 和 ByteCheckpoint 应保留，但主要支撑“分布式格式与重分片”现状，不能直接支撑 SPDK 数据通路创新。
5. Just-In-Time Checkpointing 的故障模型和触发机制与周期性高频持久化不同，宜作为替代范式简述，不必大篇幅展开。

## 4. 参考文献清单

以下条目可作为中期报告参考文献表的初稿。正式提交前应按学院模板统一作者大小写、会议名称、页码、DOI、访问日期和文献类型标识。

### 4.1 异步与高频检查点

[1] MOHAN J, PHANISHAYEE A, CHIDAMBARAM V. CheckFreq: Frequent, Fine-Grained DNN Checkpointing[C]//19th USENIX Conference on File and Storage Technologies. 2021: 203-216. [论文与 BibTeX](https://www.usenix.org/conference/fast21/presentation/mohan).

[2] NICOLAE B, LI J, WOZNIAK J M, et al. DeepFreeze: Towards Scalable Asynchronous Checkpointing of Deep Learning Models[C]//20th IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing. 2020. [作者机构论文页](https://icl.utk.edu/node/1594).

[3] MAURYA A, UNDERWOOD R, RAFIQUE M M, et al. DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models[C]//33rd International Symposium on High-Performance Parallel and Distributed Computing. 2024. DOI: 10.1145/3625549.3658685. [论文](https://arxiv.org/abs/2406.10707).

[4] WANG G, RUWASE O, XIE B, et al. FastPersist: Accelerating Model Checkpointing in Deep Learning[R]. Microsoft Research Technical Report MSR-TR-2024-27, 2024. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/fastpersist-accelerating-model-checkpointing-in-deep-learning/).

[5] STRATI F, FRIEDMAN M, KLIMOVIC A. PCcheck: Persistent Concurrent Checkpointing for ML[C]//Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2025: 811-827. DOI: 10.1145/3669940.3707255. [ACM 论文页](https://doi.org/10.1145/3669940.3707255).

[6] GUPTA T, KRISHNAN S, KUMAR R, et al. Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures[C]//Proceedings of the Nineteenth European Conference on Computer Systems. 2024: 1110-1125. DOI: 10.1145/3627703.3650085. [EuroSys 2024](https://2024.eurosys.org/program.html).

[7] CHEN M, HUA Y, BAI R, et al. A Cost-Efficient Failure-Tolerant Scheme for Distributed DNN Training[C]//2023 IEEE 41st International Conference on Computer Design. 2023. DOI: 10.1109/ICCD58817.2023.00031. [作者公开论文](https://light-chenml.github.io/resources/2023-ICCD-LightCheck.pdf).

[8] LIU H, LUO S, LI K, et al. CheckFlow: Low-Overhead Checkpointing for Deep Learning Training[J]. IEEE Computer Architecture Letters, 2025, 24(2): 281-284. DOI: 10.1109/LCA.2025.3596616. [作者公开论文](https://shouxi.name/publications/cal25-checkflow.pdf).

### 4.2 分布式检查点与恢复

[9] WANG Z, JIA Z, ZHENG S, et al. GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints[C]//Proceedings of the 29th Symposium on Operating Systems Principles. 2023: 364-381. DOI: 10.1145/3600006.3613145. [Amazon Science](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints).

[10] WAN B, HAN M, SHENG Y, et al. ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development[C]//22nd USENIX Symposium on Networked Systems Design and Implementation. 2025: 559-578. [USENIX](https://www.usenix.org/conference/nsdi25/presentation/wan-borui).

[11] LIAN X, JACOBS S A, KURILENKO L, et al. Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism[C]//2025 USENIX Annual Technical Conference. 2025: 1519-1534. [USENIX](https://www.usenix.org/conference/atc25/presentation/lian).

[12] WANG S, CAO Q, ZHOU K, et al. ParaCkpt: Heterogeneous Multi-Path Checkpointing Mechanism for Training Deep Learning Models[C]//2024 IEEE 42nd International Conference on Computer Design. 2024: 183-190. DOI: 10.1109/ICCD63220.2024.00036. [会议目录](https://www.proceedings.com/content/078/078266webtoc.pdf).

[13] LI Y, WU T, LI G, et al. Portus: Efficient DNN Checkpointing to Persistent Memory with Zero-Copy[C]//2024 IEEE 44th International Conference on Distributed Computing Systems. 2024: 59-70. DOI: 10.1109/ICDCS60910.2024.00015. [作者公开论文](https://www.tianyuanwu.com/files/portus.pdf).

### 4.3 用户态与加速器存储访问

[14] SPDK PROJECT. SPDK NVMe Driver[EB/OL]. [SPDK 官方文档](https://spdk.io/doc/nvme.html).

[15] SPDK PROJECT. Message Passing and Concurrency[EB/OL]. [SPDK 官方文档](https://spdk.io/doc/concurrency.html).

[16] DIDONA D, PFEFFERLE J, IOANNOU N, et al. Understanding Modern Storage APIs: A Systematic Study of libaio, SPDK, and io_uring[C]//ACM SYSTOR. 2022: 120-127. [IBM Research](https://research.ibm.com/publications/understanding-modern-storage-apis-a-systematic-study-of-libaio-spdk-and-io-uring).

[17] NVIDIA. GPUDirect Storage Overview Guide[EB/OL]. [NVIDIA 官方文档](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).

[18] QURESHI Z, MAILTHODY V S, GELADO I, et al. GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture[C]//Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2023: 325-339. DOI: 10.1145/3575693.3575748. [论文](https://arxiv.org/abs/2203.04910).

[19] PANDEY S, KAMATH A K, BASU A. GPM: Leveraging Persistent Memory from a GPU[C]//Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2022: 142-156. DOI: 10.1145/3503222.3507758. [作者公开论文](https://www.csa.iisc.ac.in/~arkapravab/papers/ASPLOS22_GPM.pdf).

[20] SUN J, SUN M, ZHANG Z, et al. Hyperion: Co-Optimizing SSD Access and GPU Computation for Cost-Efficient GNN Training[C]//2025 IEEE International Conference on Data Engineering. 2025: 321-335. DOI: 10.1109/ICDE65448.2025.00031. [作者公开论文](https://wangzeke.github.io/doc/Hyperion-ICDE25.pdf).

### 4.4 增量与压缩检查点

[21] EISENMAN A, MATAM K K, INGRAM S, et al. Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models[C]//19th USENIX Symposium on Networked Systems Design and Implementation. 2022: 929-943. [USENIX](https://www.usenix.org/conference/nsdi22/presentation/eisenman).

[22] CHEN Y, LIU Z, REN B, et al. On Efficient Constructions of Checkpoints[C]//Proceedings of the 37th International Conference on Machine Learning. 2020: 1627-1636. [PMLR](https://proceedings.mlr.press/v119/chen20m.html).

[23] AGRAWAL A, REDDY S, BHATTAMISHRA S, et al. Inshrinkerator: Compressing Deep Learning Training Checkpoints via Dynamic Quantization[C]//ACM Symposium on Cloud Computing. 2024: 1012-1031. DOI: 10.1145/3698038.3698553. [作者公开论文](https://kexinrong.github.io/papers/inshrink-socc24.pdf).

[24] An Efficient Compression of Deep Neural Network Checkpoints Based on Prediction and Context Modeling[EB/OL]. arXiv:2506.12000, 2025. [预印本](https://arxiv.org/abs/2506.12000).  
注：该文可用作前沿补充，但不建议作为压缩方向的唯一或首要依据。

## 5. 引用与写作注意事项

1. “检查点”与“梯度检查点/激活重计算”属于不同研究问题。检索和写作时应排除以降低训练显存为目的的 activation checkpointing 文献。
2. 引用论文报告的倍数和百分比时，应明确“在该论文实验环境中”，不要把不同论文的结果直接横向比较。
3. “直接存储”“零拷贝”“用户态旁路”和“GPU/NPU 主导 I/O”含义不同，应分别使用。当前项目最准确的描述是“文件系统旁路的用户态 NPU—NVMe 分块流水化数据通路”。
4. SPDK 的 qpair 单线程所有权和消息传递是官方推荐编程模型。本项目的创新应落在其面向 NPU 检查点的具体控制面设计、请求/FSM 组织和训练触发机制，而不是宣称发明 Reactor 或单线程轮询。
5. Top-K、INT8 和差分检查点都有既有研究基础。本项目第三点应突出 NPU 图内执行、减少 HBM 外传数据、FULL + Delta 协议和误差控制。
6. 中期报告可以说明第三方向已经完成方案和单步可行性验证，但长期恢复误差仍是待解决问题；这一表述与现有压缩研究强调的“压缩率—精度—恢复次数”权衡一致。

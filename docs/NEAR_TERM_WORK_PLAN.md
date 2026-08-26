# 近期研发与实验计划：I/O 路径、设计消融与增量检查点

> 制定日期：2026-08-25
>
> 初版规划基线：`6e8d57c`；本次增量方案复核基线：`190e3c0`
>
> 适用范围：当前 NPU-NVMe 主线实现及硕士论文后续实验
>
> 建议周期：6 周；以正确性门禁和证据完整性为阶段出口，不以单一性能数字作为完成标志

## 1. 近期目标与交付物

近期工作围绕三个问题展开：

1. **为什么需要面向异构存储层次优化检查点 I/O 路径？** 量化 HBM、Host DRAM、
   内核文件系统和用户态裸 NVMe 之间各层软件栈的时间、搬运与 CPU 开销。
2. **当前设计中的每个技术点分别贡献了什么？** 通过一次只改变一个因素的消融实验，
   区分文件系统旁路、批量提交、分块流水、单所有者 Reactor、FaF 触发等设计收益。
3. **增量检查点应沿哪条路线继续实现？** 先观察参数更新的稀疏性、时间局部性、量化
   误差和长期累积，再从无损基线、自适应选择、量化状态和残差反馈方案中作出选择。

本阶段应形成以下可验收交付物：

- 一套统一的 I/O 路径分层计时与环境采集工具；
- 一组通过数据一致性检查的 FULL 保存、加载和训练干扰结果；
- 一张“软件栈分层开销”图和一张“端到端路径对比”图；
- 一组设计点消融表，能够把收益归因到具体机制；
- 一份增量检查点观察实验数据集和路线选择结论；
- 至少一条能够完成 `FULL + N 个 Delta → 恢复 → 数值校验` 的正确性实现；
- 可复现的命令、配置、原始日志、结果 JSON 和绘图输入，不再依赖手工摘录数字。

## 2. 所有实验的共同前置条件

### 2.1 正确性门禁

性能实验开始前必须完成：

1. 修复 FULL 区域与 Delta ring 的物理地址重叠，形成唯一分区表；
2. 在目标服务器完成 Host/NPU × 读/写四条路径的数据回环测试；
3. 完成 FULL 保存、进程重启、FULL 加载和逐参数校验；
4. 明确 `save()` 的“已派发”和“已持久化”两个完成时刻，后台错误必须使实验失败；
5. 固定目标 NVMe、NPU、NUMA 节点和设备绑定状态，禁止不同物理盘的数据直接横比；
6. Delta 实验必须先通过帧校验、边界检查和恢复校验，失败样本不能进入性能均值。

### 2.2 统一实验协议

每个配置至少预热 5 次，正式重复不少于 10 次；超长训练实验可使用 3 次独立重复，
但必须报告置信区间。所有比较固定：模型、参数精度、数据集、训练步数、检查点间隔、
NVMe 设备、CPU 亲和性、NPU/NUMA 拓扑和后台负载。

每次运行至少保存：

| 类别 | 必须记录的内容 |
|---|---|
| 版本 | Git commit、分支、dirty 状态、子模块 commit |
| 硬件 | CPU/NUMA、NPU、NVMe 型号与固件、PCIe 拓扑、容量与 LBA 大小 |
| 软件 | 内核、CANN、驱动、MindSpore、SPDK、DPDK、Python、编译器版本 |
| 配置 | 路径类型、chunk size、pipeline depth、batch 大小、检查点间隔、模型规模 |
| 性能 | 端到端时间、训练阻塞、各层时间、带宽、CPU 利用率、上下文切换、尾延迟 |
| 正确性 | 写入字节、读取字节、校验和、逐参数误差、恢复后 loss、异常与退出码 |

原始结果写入 `experiments/output/`；进入论文或答辩前，再将通过门禁的结果连同环境
清单固化到带日期和 commit 的结果目录。图表只能从结构化结果自动生成。

## 3. 工作包一：异构存储必要性与 I/O 软件栈开销

### 3.1 需要回答的研究问题

- 通用检查点路径的主要开销来自框架序列化、D2H 快照、文件系统，还是设备写入？
- 绕过文件系统后，节省的是 CPU 时间、数据拷贝、尾延迟，还是端到端带宽？
- HBM、Host DRAM 和 NVMe 的带宽/容量差异如何转化为训练停顿？
- 当模型规模、参数对象数量和检查点频率变化时，直通路径的收益是否仍然成立？
- 写入带宽提高后，新的瓶颈位于 ACL 搬运、任务编组、Reactor 还是元数据提交？

### 3.2 对照路径

| 编号 | 路径 | 目的 | 当前入口或实现基础 |
|---|---|---|---|
| P0 | 纯训练，不保存 | 获得训练 step 基线 | `ProbeTrainOneStepCell(enable_probe=False)` |
| P1 | `ms.save_checkpoint` → 文件系统 → NVMe | 通用框架基线 | `experiments/baselines/ms_save_bench.py` |
| P2 | HBM → Host DRAM 快照 → 内核文件系统 → NVMe | 分离快照与持久化开销 | `checkfreq_kernel_fs.py`、`pccheck_kernel_fs.py` |
| P3 | Host buffer → SPDK → 裸 NVMe | 隔离文件系统和 NPU 搬运 | Host 批量 C API 与 C smoke test |
| P4 | HBM → DMA buffer → SPDK → 裸 NVMe | 当前全量直通路径 | `DirectCheckpoint.save()` |
| P5 | 训练步触发 → 后台 P4 | 测量可隐藏时间与实际训练干扰 | FaF/Reactor 路径 |
| P6 | 裸 NVMe → DMA buffer → HBM/Host | 补齐读取与恢复路径 | `DirectCheckpoint.load()` 与读 API |

P1、P2 与 P4 必须使用同一块 NVMe 或性能等价的独占分区；如果文件系统盘与 SPDK
裸盘不是同一设备，应先分别测得设备上限，并将“设备差异”和“软件路径差异”分开报告。

### 3.3 分层计时点

为一次检查点分配统一的 `run_id / checkpoint_id / request_id`，在以下边界记录单调时钟：

1. 训练 step 结束与检查点触发；
2. NPU 同步开始/结束；
3. 参数遍历、地址获取、布局生成和 chunk 数组构造；
4. Python → ctypes → C API 进入与返回；
5. 请求环入队、Reactor 取出和首次提交；
6. 每个 chunk 的 HBM→DMA buffer 搬运；
7. SPDK write/read submit 与 completion；
8. 全部数据完成、元数据写入、active slot 切换；
9. 后台 I/O 完成，以及训练真正等待后台 I/O 的时长。

派生指标统一定义为：

- **端到端延迟**：触发至持久化完成；
- **训练阻塞**：训练线程因快照、同步或等待空闲槽而停止的时间；
- **有效带宽**：有效检查点字节数 ÷ 端到端持久化时间；
- **介质带宽**：NVMe 实际提交字节数 ÷ 首次 submit 至最后 completion 时间；
- **软件栈开销**：端到端时间减去不可重叠的设备传输时间；
- **隐藏比例**：后台 I/O 时间中与后续训练重叠的比例；
- **写放大**：NVMe 实际写入字节数 ÷ 有效检查点字节数。

各层计时存在重叠，不能把所有阶段耗时直接相加；报告应同时给出关键路径时间线和
分层累计 CPU 时间。

### 3.4 必要实验

#### E1：设备与内存层次上限

- 顺序读写裸 NVMe，扫描 4 KiB～16 MiB 请求大小和队列深度；
- 测量 Host DRAM memcpy、HBM↔Host ACL memcpy、HBM↔HBM D2D 带宽；
- 记录跨 NUMA 与本地 NUMA 的差异；
- 输出带宽—块大小曲线，为后续结果提供物理上限。

#### E2：路径端到端对比

- 比较 P1～P5 的保存延迟、训练阻塞、有效带宽、CPU 占用和 P95/P99；
- 至少覆盖 1 GiB 合成缓冲、GPT-2 XL 和一个更大模型配置；
- 对同步保存和异步保存分别报告“API 返回时间”和“真正持久化时间”。

#### E3：软件栈分解

- 在固定 4 MiB chunk、固定队列深度下采集 3.3 节全部时间戳；
- 输出瀑布图或关键路径时间线；
- 分别计算框架/编组、同步/快照、Reactor、ACL、SPDK、元数据的占比；
- 使用 CPU profile、上下文切换和系统调用计数验证文件系统路径的控制开销。

#### E4：对象粒度与模型规模敏感性

- 保持总字节数不变，改变参数对象数量，观察逐参数调用和任务编组开销；
- 扫描模型大小、chunk size、pipeline depth、检查点间隔；
- 判断直通收益是否只在大对象顺序写下成立。

#### E5：读取与恢复路径

- 对 P1/P2/P4 对应的读取路径测量加载时间、首参数可用时间和全模型可用时间；
- 覆盖 Host 参数、NPU 参数和混合参数；
- 逐参数比较 dtype、shape、字节内容，并报告恢复后的首步 loss。

### 3.5 完成判据

- 能用同一套数据解释端到端时间与各层关键路径，时间差无法解释的比例低于 5%；
- 所有性能点均通过数据回读或逐参数恢复校验；
- 至少在两种模型规模下说明文件系统旁路和异构内存/存储协同的收益来源；
- 不把不同 SSD、不同计时边界或同步/异步语义的数据画在同一对比柱状图中。

## 4. 工作包二：各设计点消融实验

### 4.1 消融原则

以当前完整 P5 路径为实验组，每次只关闭或替换一个设计点；不能把不同版本、不同
模型或不同磁盘的结果称为消融。所有配置同时报告正确性、性能和资源开销。

### 4.2 消融矩阵

| 编号 | 设计点 | 对照配置 | 实验配置 | 核心指标 | 希望回答的问题 |
|---|---|---|---|---|---|
| A1 | 文件系统旁路 | 内核 FS | SPDK 裸盘 | 延迟、CPU、syscall、P99 | 旁路本身贡献多少 |
| A2 | Host 中转消除 | HBM→Host→NVMe | HBM→DMA→NVMe | 拷贝量、阻塞、带宽 | 少一次中转的真实收益 |
| A3 | 批量任务提交 | 每 chunk 一次 API | 地址/长度/offset 数组批量提交 | Python/C 时间、CPU | 跨语言和提交摊销收益 |
| A4 | 分块流水 | pipeline depth=1 | depth=2/4/8/16 | 带宽、槽等待、P99 | ACL 与 NVMe 能否重叠 |
| A5 | chunk size | 64 KiB～16 MiB | 当前 4 MiB | 带宽、尾块、内存 | 最优粒度及稳定区间 |
| A6 | 单所有者 Reactor | 同步提交或受控旧实现 | Reactor + request ring/FSM | CPU、锁等待、吞吐 | 控制平面是否减少竞争 |
| A7 | FaF 步触发 | Python 同步调用 | 图内计数 + Reactor poller | step time、触发抖动 | 触发卸载贡献多少 |
| A8 | 元数据双槽 | 每次同步单槽写 | A/B 槽 + 原子切换 | 提交时间、故障恢复 | 一致性成本是多少 |
| A9 | 并发快照槽 | 1 个 in-flight | 2/3/4 个槽 | 阻塞、内存、吞吐 | CheckFreq/PCcheck 式并发边界 |
| A10 | NUMA/CPU 绑定 | 不绑定或远端 NUMA | Reactor/内存本地绑定 | CPU、P99、带宽 | 拓扑感知是否必要 |

其中 A6 不直接恢复已归档且存在正确性风险的旧线程实现；应使用受控同步版本或只关闭
请求环重叠的安全配置，避免为了消融重新引入死锁和资源所有权问题。

### 4.3 增量路径内部消融

在增量正确性闭环后增加：

| 编号 | 因素 | 取值 |
|---|---|---|
| D0 | 恢复语义 | S1 相邻步、S2 已持久化状态替换块、S3 残差加法 |
| D1 | 变化块选择 | 全量 Delta、固定 Top-K、自适应 Top-K、阈值选择 |
| D2 | 块大小 | 16K、64K、256K、512K、1M elements（按参数边界切分） |
| D3 | 编码 | FP16、INT8 per-tensor scale、INT8 per-block scale |
| D4 | 参考状态 | 精确 FP32/FP16 `P_persisted`、量化状态+scale、周期性重建 |
| D5 | 误差反馈 | 无残差、Host residual、NPU 图内 residual |
| D6 | FULL 间隔 | 10、25、50、100 steps 或按误差自适应触发 |
| D7 | 持久化方式 | Host 组帧、NPU 原始缓冲、NPU 自描述帧 |

主要输出包括写入比例、图内计算时间、HBM 额外占用、Delta 帧大小、恢复时间、链长、
逐参数 NRMSE、恢复后 loss 偏差和训练吞吐。

## 5. 工作包三：增量检查点探索路径

### 5.1 先做观察实验，不预设 Top-K 一定成立

#### O1：参数变化的空间稀疏性

同时对两个参照量做统计：相邻训练步变化 `P_t-P_{t-1}`，以及相对最近已持久化版本的
累计变化 `P_t-P_persisted`。逐层、逐参数和逐块记录 L1/L2/L∞ norm、非零比例和累计
变化能量—块比例曲线，分别回答 Top-1%、5%、10%、20% 能覆盖多少变化，以及不同层
是否应使用相同阈值。只统计相邻步变化会掩盖长期未写块的累计漂移，不能直接用于选择
恢复协议。

#### O2：时间局部性与“长期饿死”

记录相邻 step 的选中块 Jaccard、块的连续命中长度、距上次持久化的年龄和未保存残差。
重点观察持续小幅变化但长期进不了 Top-K 的块，这类块是历史 100 步误差累积的主要
候选原因。

#### O3：块粒度

扫描块大小，比较选择元数据、padding、并行度、稀疏性和恢复误差。块不能跨参数语义
边界；小参数应单独打包或直接纳入每次 Delta，不能静默忽略。

#### O4：量化与历史状态误差

分别使用精确 FP32/FP16 `P_persisted`、INT8 `P_persisted` 无 scale、INT8
`P_persisted` + per-block scale，区分“写出值量化误差”“选择参考状态失真”和
“未选择块陈旧误差”。测试单步、10/50/100 步链长，而不是只报告一次量化误差；
`INT8` 无 scale 仅作为预期失败的负对照，不能作为候选实现。

#### O5：训练阶段差异

在训练早期、中期和收敛阶段重复 O1～O4，观察更新分布是否随 learning rate 和
optimizer 状态变化。至少包含权重与 optimizer state，避免只凭模型参数推断总检查点。

#### O6：可利用的 NPU 计算窗口

测量 Cube/Vector 利用率和 step 内空闲窗口，分别运行 norm、Top-K、quant、scatter
微基准。判断这些计算能否隐藏在训练空隙中，还是会延长关键路径。

#### O7：端到端成本模型

对每种策略计算：

`总成本 = 图内检测时间 + 持久化时间 + 周期性 FULL 摊销 + 期望恢复时间`

同时报告 HBM 影子状态占用。只减少写入字节但显著增加训练时间或恢复时间的方案不能
视为有效优化。

### 5.2 必须先锁定的增量语义

当前代码和历史实验混用了三种不同含义的 `P_old`，后续必须先固定语义：

| 语义 | 比较对象 | 选择后如何更新参考状态 | 恢复操作 | 主要风险 |
|---|---|---|---|---|
| S1 相邻步 Delta | `P_t-P_{t-1}` | 每步更新所有块 | 加法或替换 | 未选中的小变化不会累计，容易长期丢失 |
| S2 已持久化状态替换块 | `P_t-P_persisted` | 仅在该块持久化 ACK 后更新 | 用 `P_t` 替换对应块 | 需要精确 ACK、双缓冲和 per-block 版本 |
| S3 残差反馈 Delta | `P_t-P_{t-1}+residual` | 写出后扣除对应 residual | 累加量化 Delta | 重复/漏帧会直接破坏结果 |

R0 固定采用 **S2 的无损替换块语义**：以最近 FULL 初始化 `P_persisted`，选择相对
已落盘版本变化最大的块，帧内保存该块在 step `t` 的完整 FP16 值；只有收到持久化 ACK
后才推进对应参考块。S1 只作负对照，S3 在 S2 闭环后再验证。

必须保持以下不变量：

1. FULL 完成时，参考状态、恢复基线和 `base_generation` 指向同一 step；
2. 块不得跨参数边界，block ID 必须唯一映射到参数名、offset、有效元素数和 dtype；
3. frame 明确记录“当前块值”或“加法 Delta”，恢复端不得猜测；
4. READY 缓冲在 PERSISTED 前不可覆写，失败帧不得推进 `P_persisted`；
5. 小参数、optimizer state、loss-scale、RNG 和 data-loader 位置必须明确归属；
6. 恢复只接受同一 FULL generation 根下连续且 checksum 正确的链。

### 5.3 可探索的实现方案

#### 方案 R0：无损块级 Delta 基线

- 采用 S2 替换块语义，保存所有与 `P_persisted` 不同的块；稠密优化器下集合可能接近
  全量，这是基线结果，不人为声称压缩收益；
- 采用 FP16 数据、自描述 Delta frame 和精确 FP16/FP32 `P_persisted`；
- 实现稳定的全局块 ID、参数名、参数内 offset 和 dtype/shape 映射；
- 目标是先闭合 FULL + Delta 恢复语义，为有损方案提供 oracle。

优点是正确性清晰；缺点是 HBM 影子状态和写入缩减可能有限。该方案应最先完成。

#### 方案 R1：自适应 Top-K/阈值 + per-block INT8

- 每层或每参数根据变化分布选择阈值，而非全模型固定比例；
- 写出值和量化参考状态都保存 per-block scale；先保持精确 `P_persisted`，再单独消融
  量化参考状态，禁止同时改变选择策略和参考精度；
- 将小参数、scale、index 和版本信息纳入统一帧；
- 使用字节预算或误差预算控制每步写入量。

这是当前 Top-K 原型最直接的收敛方向，但必须先修复量化历史状态语义。

#### 方案 R2：残差反馈

- 未持久化变化累积到 residual；
- 选择依据使用“本步变化 + residual”；
- 块达到阈值或年龄上限时强制写出，写出后清零对应 residual；
- 对比 Host residual 与 NPU 图内 residual 的性能和 HBM 成本。

该方案针对持续小变化块的长期误差，预期比单纯固定 Top-K 更稳健。

#### 方案 R3：分层热度与周期性 FULL

- 高频变化块高频 Delta，低频变化块降低写入频率；
- 设置最大 Delta 链长、最大块年龄和误差阈值；
- 任一条件达到上限时触发 FULL；
- 恢复时加载最近 FULL，再按序应用带 checksum 和 step ID 的 Delta frame。

该方案利用更新时间局部性，但调度和元数据更复杂，应在 O2 证明存在稳定热度后再做。

#### 方案 R4：HBM→DRAM→NVMe 的分级持久化

- 训练关键路径只完成 HBM 快照或压缩缓冲转移；
- Host DRAM 作为短期暂存，后台合并小 Delta 后写 NVMe；
- 设置 DRAM 容量上限、背压、故障窗口和强制落盘策略。

该方案体现异构存储层次协同，但 DRAM 中尚未落盘的数据不能算可靠检查点。只有在
明确故障模型并量化可靠性窗口后，才可作为正式方案。

#### 方案 R5：两级粗筛与分层块

- 以较大 super-block 计算低成本摘要，只对超过阈值的区域做细粒度选择；
- 第一层必须维护累计摘要、最大年龄或定期全扫描，不能直接丢弃更新；
- 与全模型 norm + 全局 Top-K 比较 HBM 扫描量、漏选率、元数据和图编译复杂度。

仅当 O1 显示空间聚集性稳定且 5.9.3 证明 norm/Top-K 是主要瓶颈时开展。

#### 方案 R6：按训练状态分治的混合检查点

- 权重、Adam 一阶矩、二阶矩、小参数和控制状态分别选择 FULL、FP16/INT8 全块、
  稀疏替换块或低频保存；
- 对敏感状态保持无损或更高精度，对大且不敏感的状态使用有损编码；
- 以完整训练状态的总物理写量和恢复质量评价，不能只报告权重压缩率。

若 O5 证明权重和 optimizer state 差异显著，R6 优先于全局统一 Top-K。

### 5.4 推荐实现顺序与决策门槛

1. **先做 R0。** 若不能完成 100 步 `FULL + Delta` 字节级或数值级恢复，不进入有损优化。
2. **并行完成 O1～O5。** 用观察数据决定固定 Top-K 是否放弃，以及阈值是否分层。
3. **优先比较 R1 与 R2。** R1 验证量化数据缩减，R2解决长期误差；两者可组合。
4. **有明显时间局部性再做 R3/R5。** 若块热度或空间聚集不稳定，不增加调度层次。
5. **状态差异明显时做 R6。** 先证明完整状态上的收益，不以权重结果代替总检查点。
6. **R4 作为扩展路线。** 先完成本地 NVMe 可靠落盘闭环，再研究 DRAM 分级缓存。

增量方案进入正式性能实验的暂定门槛：

- 帧校验、step 连续性、slot 回绕和故障注入全部通过；
- 单步恢复逐参数 NRMSE 中位数不高于 5e-3；
- 100 步链恢复 NRMSE 中位数不高于 1e-2，且无长期未持久化块；
- 恢复后固定批次 loss 相对偏差不高于 1%；
- 平均实际写入量低于 FULL 的 20%；
- 图内增量处理带来的稳定训练 step 开销不高于 10%。

这些是路线筛选门槛，不是论文预设结论；若观察到 loss 对参数误差更敏感，应收紧阈值。

### 5.5 当前实现与历史结果的证据审计

截至复核基线 `190e3c0`，必须区分“研究方向可行”和“当前实现已经可用”：

| 能力 | 当前等级 | 允许的表述 |
|---|---|---|
| NPU 图内差值、归约、Top-K、量化 | 黄 | 算子组合具有原型可执行性 |
| 压缩 HBM 缓冲直写 NVMe | 黄 | 数据通路可承载压缩缓冲 |
| 固定 Top-10% 减少逻辑字节 | 黄 | 存在数据缩减潜力，尚不等于物理收益 |
| 图内结果与 CPU oracle 一致 | 红 | 当前不能宣称检测正确 |
| `FULL + Delta` 跨进程恢复 | 红 | 当前不能宣称端到端闭环 |
| 长链误差可控 | 红 | 历史 CPU 模拟数值不能外推到当前实现 |
| 训练开销可接受 | 红/黄 | 旧结果只作为排障线索，需按 I0～I7 重测 |
| 崩溃一致性与并发安全 | 红 | ACK 生命周期、ring 回绕和提交协议仍需门禁 |

历史 100-step 高误差暴露了参考状态初始化、累计变化语义和长期未选块问题，但由于
模拟器与当前图实现不是同一算法，数值不能直接判定 Top-K 成败。当前代码必须先显式
验证：逐参数 padding、`P_persisted` 初始化、INT8 scale、替换/加法 frame 语义、小参数、
ACK 回滚、输出缓冲生命周期以及 FULL/Delta 分区不重叠。

### 5.6 分层实验流水线

任何策略按 I0～I7 顺序晋级；上层失败时停止性能优化：

| 阶段 | 实验内容 | 通过条件 |
|---|---|---|
| I0 | CPU-only 语义单测，覆盖 S1/S2/S3 与人工更新 | 每 step 选块、参考状态、恢复结果符合 oracle |
| I1 | 真实训练轨迹 CPU 回放，比较 R0～R3 | R0 精确，其他策略得到稳定误差—字节曲线 |
| I2 | CPU/NPU 图算子等价 | norm、索引、值、scale、有效长度逐项一致 |
| I3 | 图输出冻结、组帧、缓冲生命周期 | generation 对齐，无撕裂和未 ACK 覆盖 |
| I4 | 普通文件跨进程 frame 恢复 | R0 逐参数精确，有损误差与 I1 一致 |
| I5 | Host-SPDK/NPU-SPDK 回环 | frame 字节与 checksum 完全一致 |
| I6 | 裸盘重启、回绕和故障恢复 | 只恢复完整链，损坏时明确回退或硬失败 |
| I7 | 长训练、后台 I/O、背压和故障注入 | 正确性门禁后再比较吞吐、写量和 RTO |

I0/I1 必须独立维护 `oracle_current`、`persisted_reference` 和 `recovered_state`；每个
frame 可单独重放验证幂等性，S3 重复帧必须被 generation 拒绝而非重复累加。

### 5.7 详细增量实验矩阵

#### 5.7.1 人工更新与边界用例

至少覆盖：Z0 无变化、Z1 单块突变、Z2 持续小变、Z3 均匀稠密、Z4 冷热分层、Z5 热块
轮转、Z6 突发变化、Z7 正负抵消、Z8 动态范围、Z9 NaN/Inf。边界覆盖小于块、恰好一块、
`block_size+1`、尾块、零值参数、LayerNorm/bias、小参数、FP16/FP32/BF16、长参数名、
4 KiB 对齐差 1 element；链长使用 `0/1/2/10/50/100`，ring 覆盖
`slots-1/slots/slots+1/2*slots+3`。

#### 5.7.2 真实训练轨迹

覆盖训练早期、中期和收敛阶段；权重、Adam 一阶矩、二阶矩、step、loss-scale，条件允许
时加入 RNG/data position；块大小 16K/64K/256K/512K/1M elements；选择预算
1%/5%/10%/20%/50%/100%；编码包含 FP16 oracle、INT8 per-tensor 和 per-block；FULL
间隔包含 10/25/50/100/500 steps。先用 CPU 回放筛选 Pareto 前沿，再在小模型和 GPT-2 XL
上进行配对 NPU 实验。

#### 5.7.3 故障、并发与持久化

覆盖输出生成时进入下一步、NVMe 慢于检查点间隔、payload/header/manifest 分阶段中断、
checksum/step/base generation 损坏、Delta 重复/缺失/乱序、ring 回绕覆盖、FULL 与 Delta
并发以及单 rank 失败。正确行为必须是回退、阻塞或明确失败，禁止静默覆盖和静默丢失。

### 5.8 指标与统计边界

每次至少报告全局/逐参数 relative L2、update-relative error、RMSE、最大误差、P50/P95/P99、
恢复后 loss/logits、后续 1/10/100 step loss、optimizer/RNG/data state 误差；同时报告
逻辑写入比例、物理写入比例、FULL 摊销写入比例、变化能量覆盖率、块陈旧度、写放大、
训练 step 开销、RTO、额外 HBM/DRAM、CPU 核时和 NVMe 容量。必须给出误差—训练开销—物理
写量—恢复时间—额外内存的 Pareto 曲线，不能只报告压缩率。

### 5.9 观察结果驱动的决策

- 若 R0 在 I0 失败，缩到两参数三块手算并修复语义，不讨论硬件收益。
- 若 CPU 正确而 NPU 不一致，优先检查 padding、dtype、Top-K tie、scale 和零块。
- 若单步误差低但链长漂移，分离量化误差、未选块陈旧和参考状态失真，比较 R1/R2。
- 若权重稀疏而 optimizer state 稠密，转向 R6 分状态策略，不能只引用权重压缩率。
- 若热块稳定再做 R3；若空间聚集稳定且 norm 成本高再做 R5。
- 若图内扫描成本高，评估 Host 处理或两级粗筛；若所有增量方案成本高于 FULL，转向
  Pivot C，保留 FULL I/O、Reactor 和 FaF 的正式结论。

### 5.10 Go / Pivot / Stop 判据

| 决策 | 判据 | 后续动作 |
|---|---|---|
| Go | I0～I6 全过，至少一方案物理写入 <20%、step 开销 <10%、100-step 误差/loss 达标 | 扩大模型、训练阶段、多 rank 和故障实验 |
| Pivot A | 更新稠密或 K≥50% 才达误差，但 INT8 全块可接受 | 采用全块量化 Delta + 周期 FULL |
| Pivot B | NPU 检测开销高，Host 冻结快照可充分重叠 | 比较 HBM→DRAM 后处理和故障窗口 |
| Pivot C | 增量不满足端到端收益 | 聚焦 FULL I/O、Reactor/FaF 和异步流水 |
| Stop | R0 无法闭环，或所有方案长期成本显著高于 FULL | 固化负结果和适用边界 |

证据等级统一为 L0 静态推断、L1 单元/模拟、L2 单次目标机观察、L3 可重复正确性、L4
跨配置端到端结果；论文主结论至少达到 L3，性能泛化结论达到 L4。

### 5.11 增量工作包交付物

- S1/S2/S3 语义与 ACK 不变量文档；
- CPU 轨迹采集/回放工具及 Z0～Z9 测试集；
- 稳定 block manifest、自描述 frame 和三条 I/O 路径一致性测试；
- 跨进程 `FULL + Delta` 恢复 runner、回绕和故障注入矩阵；
- 更新能量、Jaccard、age、误差—写量—开销 Pareto 图；
- 历史结果复核表以及最终 Go/Pivot/Stop 决策记录。

## 6. 六周推进安排

| 周次 | 主要任务 | 阶段出口 |
|---|---|---|
| 第 1 周 | 目标机环境固化、一致性快照、FULL/Delta 分区修复、四路径回环、统一 trace schema | 正确性门禁通过，环境清单可自动生成 |
| 第 2 周 | E1～E3：设备上限、端到端路径、软件栈分解 | 获得可解释的路径时间线与第一版开销图 |
| 第 3 周 | E4～E5 与 A1～A10 主要消融 | 形成 FULL 路径消融表，确定默认 chunk/depth/绑定 |
| 第 4 周 | O1～O7 观察实验、R0 自描述帧闭环 | 完成无损 FULL + Delta 恢复和更新分布分析 |
| 第 5 周 | 实现并比较 R1、R2；扫描 D1～D7 | 确定增量主路线和 FULL 间隔策略 |
| 第 6 周 | 端到端训练、故障恢复、结果复核与图表生成 | 固化正式结果、复现说明和论文实验素材 |

若目标硬件时间受限，优先级顺序为：正确性门禁 → E1/E2/E3 → A1/A2/A4/A7 →
O1/O2/O4 → R0 → R1/R2。R3、R4 和完整参数扫描可以后移。

## 7. 近期代码任务清单

### P0：实验前必须完成

- [ ] 为异步 FULL 建立分代快照和双缓冲生命周期，禁止后台线程读取训练中的活动参数；
- [ ] 为 Delta 输出建立双缓冲或环形缓冲，并由持久化 ACK 控制缓冲复用；
- [ ] 统一 FULL/Delta 磁盘分区分配器并增加容量/重叠断言；
- [ ] 为元数据增加校验、代际、持久化屏障和双副本回退；
- [ ] 为所有异步请求增加超时、失败传播、取消和有界清理；
- [ ] 补齐目标机四路径 C smoke test 和 FULL 重启恢复测试；
- [ ] 增加统一环境采集脚本、运行 ID、配置快照和失败退出码；
- [ ] 为 Python、C、ACL、SPDK 各层加入可关联的时间戳；
- [ ] 统一同步/异步持久化完成语义；
- [ ] 建立可自动生成图表的结构化结果 schema。

### P1：FULL 路径实验

- [ ] 实现 P0～P6 统一 runner；
- [ ] 完成 E1～E5；
- [ ] 完成 A1～A10 中硬件条件允许的配置；
- [ ] 输出路径时间线、带宽/延迟图、CPU 开销图和消融表。

### P1：增量路线

- [ ] 为参数/块建立稳定 ID 和更新统计采集器；
- [ ] 完成 O1～O7；
- [ ] 以自描述帧打通 R0；
- [ ] 修复 `P_old + scale`，处理小参数、零块和 slot 回绕；
- [ ] 实现 R1/R2，完成 D1～D7 主要扫描；
- [ ] 完成恢复后 loss、长链误差和故障注入验证。

### P2：结果与论文材料

- [ ] 每组正式结果关联 commit、配置和原始日志；
- [ ] 自动生成误差条、置信区间和关键路径图；
- [ ] 对异常值保留原因，不手工删除不利数据；
- [ ] 将“阶段观察”“正式结论”“尚未验证”分开表述。

## 8. 近期需要作出的四项技术决策

1. **使用哪一种增量/恢复语义？** 建议 R0 固定为 S2“相对已持久化状态选择、保存当前
   块值、ACK 后推进参考状态”，S1 只作负对照，S3 在残差反馈阶段再引入。
2. **Delta 统一采用自描述 frame，还是继续直接持久化三个原始 HBM 缓冲？** 由于
   当前 `recover()` 只理解 frame，建议以自描述 frame 为唯一磁盘协议。
3. **`P_persisted` 保留精确 FP16/FP32，还是使用带 scale 的 INT8？** 建议 R0 先用精确
   状态建立 oracle，再根据 O4 的 HBM 与误差数据决定是否切换 INT8。
4. **固定 Top-K 是否继续作为主方案？** 在 O1/O2/O5 完成前不做结论；若长期饿死
   和阶段差异明显，主路线改为自适应阈值 + 残差反馈 + 最大块年龄。

完成上述三项决策后，项目的技术叙事可以稳定为：先通过文件系统旁路和分块流水缩短
异构存储 I/O 路径，再通过 Reactor/FaF 降低控制与训练阻塞，最后根据参数更新规律
减少实际写入量，并以周期性 FULL 和可验证 Delta 链保证恢复语义。

## 9. 代码审查整改计划

### 9.1 当前结论与整改边界

当前实现已经具备 Python 后台提交、Reactor 分块状态机、FaF 触发和 Delta 原型，能够
支持路径可行性验证；但在以下门禁完成前，不应把实验结果表述为“训练一致的异步
检查点”“可重启恢复的增量检查点”或“完整的多 rank 检查点”：

1. FULL 与 Delta 都必须冻结某一步的稳定数据，后台 I/O 不得继续读取训练正在修改的
   参数或复用尚未持久化的输出缓冲；
2. FULL、Delta、元数据必须共享唯一磁盘布局，并具有明确的持久化与恢复协议；
3. 后台错误、超时和丢失触发必须传回训练侧，不能以“请求已派发”代替“已持久化”；
4. 正确性测试必须跨进程重启并比较参数、校验和与恢复后 loss，而不是只检查计数器。

整改遵循“先正确、再可恢复、后优化”的顺序。R0 增量路线先使用 FP16 `P_old`
建立无损语义基线；INT8 历史状态、Top-K 和残差反馈只在基线通过后比较。

### 9.2 工作包与验收标准

| 编号 | 优先级 | 问题与修改范围 | 主要交付物 | 验收标准 |
|---|---|---|---|---|
| F1 | P0 | FULL 后台线程读取活动 HBM 参数，训练恢复后可能写入混合 step 数据；涉及 `python/direct_checkpoint.py`、训练保存调用点 | 分代快照；双缓冲或等价 D2D 快照；`READY → WRITING → PERSISTED → REUSABLE` 生命周期 | I/O 延迟大于训练 step 时，落盘哈希仍与触发 step 的冻结快照一致；缓冲未收到 ACK 前不可复用 |
| D1 | P0 | Delta 单组输出缓冲会被下一步覆盖；涉及 `python/delta_cell.py`、`src/npu_nvme.c` | Delta 双缓冲/环形缓冲；generation ID；C 侧持久化 ACK | 人为降低 NVMe 速度并连续触发，所有已确认 generation 均可恢复，且无静默覆盖 |
| L1 | P0 | FULL 从盘尾向前分配、Delta 也从盘尾建立 ring，存在重叠风险 | 唯一分区分配器；superblock 中的分区表；4 KiB 对齐与容量断言 | 初始化时证明各区间不重叠；容量不足直接失败；旧格式要求显式重建或迁移 |
| M1 | P0 | A/B 元数据缺少完整校验、代际选择和持久化屏障 | 元数据 CRC、generation、FUA/flush 顺序、双副本回退 | 注入写中断和单副本损坏后，总能选择最新有效副本或明确失败，不能静默初始化空状态 |
| D2 | P0 | Delta 的 `P_old` INT8 scale 未参与恢复，零 scale、小参数和跨参数分块语义不完整 | FP16 `P_old` 基线；逐参数稳定 ID；逐参数 padding；零块/小参数处理 | 单步 Delta 与 CPU oracle 逐参数一致；小参数非零更新不丢失；零块不产生 NaN/Inf |
| D3 | P0 | FaF 原始缓冲、Host frame 和 `recover()` 使用的格式未闭环；slot 重启后从 0 开始且回绕元数据可能陈旧 | 唯一自描述 frame；step/generation/checksum；持久化 head/tail；启动扫描与索引重建 | 保存 100 步、发生至少两次回绕、重启后仍能定位正确链；step 不匹配必须硬失败 |
| C1 | P0 | 等待循环、批处理和清理缺少超时/取消，设备异常可能永久阻塞 | 请求 deadline；错误上下文；取消/排空；有界 `close()` | 注入 ACL/SPDK 错误和永不完成请求时，调用在设定时间内失败并释放所有等待者 |
| R1 | P0 | 多 rank 仅 rank 0 提交元数据，缺少全局完成协议；当前未覆盖优化器、RNG 和数据位置 | 每 rank manifest/checksum；两阶段 prepare/commit；完整训练状态 | 任一 rank 失败时不发布全局检查点；成功恢复后下一 step 的 loss/数据顺序与连续训练一致 |
| C2 | P1 | 精确 modulo 轮询在 Reactor 忙时可能漏触发，训练 step 与优化器完成缺少显式先后关系 | 优化器完成后的 READY 发布；pending 队列或明确 latest-wins；漏触发计数 | 快训练、慢 I/O 压测下无静默丢失；每次丢弃都有策略和指标，落盘 step 可追溯 |
| H1 | P1 | Host 参数在注册阶段被过滤，Host 保存分支实际不可达 | Device/Host 参数分类；Host 自有快照；placement 元数据 | CPU、NPU、混合参数三组 round-trip 测试逐字节或逐参数通过 |
| O1 | P1 | task 重注册与多 context 共享全局 Reactor 上下文可能导致悬空引用或串扰 | task generation/refcount；进程级运行时管理器；context 生命周期状态机 | I/O 进行中重注册、双 context 并发和重复关闭均通过内存/并发回归测试 |
| P1 | P1 | 当前是 Python 后台 + Reactor 分块重叠，但 ACL 搬运仍同步，尚不是完整的 NPU-copy/NVMe-write 异步流水 | 先准确标注“两级分块流水”；再评估异步 ACL copy、event query 和多 copy slot | 时间线证明 copy 与 write 的实际重叠；升级后端到端收益显著，否则保留较简单实现 |
| T1 | P0 | 现有 Delta E2E 主要检查计数器或缓冲和值，可能出现假阳性 | 跨进程保存/恢复测试；参数、校验和、loss 校验；负向与故障注入测试 | 测试必须证明数据已持久化；破坏 frame、step 或 metadata 时必须失败；C NPU batch 回环不再留作 TBD |
| I1 | P1 | 计时使用非单调时钟、CSV 易覆盖，缺少排队/同步/Host 路径数据；异步 `save()` 返回值易被误认为成功 | 单调时钟；run/request ID；统一 trace；`DISPATCHED/PERSISTED/FAILED` handle | 任意一次请求可跨 Python/C/ACL/SPDK 关联；结果文件不覆盖；失败能由训练侧观测 |

### 9.3 实施顺序与依赖关系

#### 阶段 A：建立安全基线

1. 先完成 T1 的最小测试骨架，固定当前应通过与应失败的行为；
2. 完成 L1，统一磁盘布局，阻止 FULL/Delta 地址重叠；
3. 完成 F1 和 D1，使 FULL 与 Delta 都具有 step 一致的冻结数据；
4. 完成 C1，使后续测试不会因设备或后台线程异常永久挂起。

阶段出口：单卡 FULL 能在训练继续运行时保存，跨进程重启后与触发 step 的冻结副本一致；
超时和后台错误会使测试失败。

#### 阶段 B：形成可恢复协议

1. 完成 M1，确定 superblock、元数据 A/B 槽的提交与回退顺序；
2. 完成 D2，以 FP16 `P_old` 建立 Delta CPU oracle；
3. 完成 D3，统一唯一 on-disk frame，补齐 ring head/tail/generation；
4. 将 FULL 与 Delta 的状态提交都接入统一持久化完成语义。

阶段出口：`FULL + N 个 Delta → 关闭进程 → 重新初始化 → 恢复` 全链路通过，slot
回绕、帧损坏和元数据单副本损坏均有确定行为。

#### 阶段 C：控制面与工程可靠性

1. 完成 C2，明确 FaF 的触发顺序、积压策略和漏触发指标；
2. 完成 H1，恢复 Host/混合参数路径；
3. 完成 O1，解决 task/context 所有权和关闭期间并发问题；
4. 完成 I1，统一请求状态、日志和时间线。

阶段出口：慢盘、连续触发、重复初始化/关闭和混合参数压测均无死锁、悬空引用或
静默丢失。

#### 阶段 D：分布式完整性与性能优化

1. 完成 R1，多 rank 两阶段提交并保存完整训练状态；
2. 以 P1 的时间线判断是否实现真正的异步 ACL copy 与多槽流水；
3. 在 R0 正确性基线上实现 INT8 `P_old`、残差反馈、自适应阈值和最大块年龄；
4. 最后执行 I/O 路径分析、设计点消融和增量写入量实验。

阶段出口：多 rank 故障恢复可继续训练；所有正式性能结果均建立在对应正确性门禁
已经通过的 commit 上。

### 9.4 正确性与实验门禁

| 门禁 | 必须通过的验证 | 未通过时的限制 |
|---|---|---|
| G0 基础 I/O | Host/NPU × 读/写回环；批量边界、非对齐、容量不足；超时和错误注入 | 不进行任何性能横比 |
| G1 FULL 一致性 | 并发训练下冻结快照哈希一致；跨进程重启逐参数校验；后台失败可见 | 不宣称异步 FULL 正确，不开展 FULL 消融 |
| G2 持久化元数据 | A/B 槽、CRC、generation、flush/FUA、写中断恢复 | 不开展掉电/故障恢复实验 |
| G3 Delta 链 | CPU oracle；小参数/零块；100 步；两次以上回绕；重启与损坏注入 | 不报告实际写入量收益，不比较压缩策略 |
| G4 多 rank | 全 rank prepare/commit；单 rank 失败；优化器/RNG/data state 连续性 | 结论仅限单卡原型，不扩展为分布式检查点 |
| G5 性能证据 | 统一 run ID、环境快照、分层时间线、重复实验与置信区间 | 数据只能作为调试观察，不能作为正式结论 |

每个门禁都应留下：运行命令、配置、commit、设备信息、原始日志、结构化结果和失败
原因。性能优化若改变持久化格式或并发语义，必须重新执行受影响的前置门禁。

### 9.5 建议的提交拆分

为便于回归和消融，整改不合并为一个大提交，建议依赖顺序如下：

1. `测试：建立跨进程恢复与故障注入基线`
2. `磁盘布局：统一 FULL 与 Delta 分区并增加边界校验`
3. `检查点一致性：引入分代快照与双缓冲生命周期`
4. `控制面：增加请求超时、失败传播与有界清理`
5. `元数据：补齐校验、代际与持久化屏障`
6. `增量语义：建立 FP16 P_old 无损恢复基线`
7. `增量协议：统一自描述帧与环形槽元数据`
8. `控制面：补齐 FaF 积压策略与上下文所有权`
9. `分布式恢复：增加多 rank 两阶段提交与完整训练状态`
10. `性能观测：统一分层时间线、请求状态与运行标识`

每个提交至少包含对应单元/集成测试；涉及磁盘格式的提交必须说明是否兼容旧盘，若
不兼容则提供明确的格式版本检查和重新初始化提示。

### 9.6 下一迭代的具体安排

下一迭代只瞄准 G0、G1 和 G2，不同时展开增量压缩策略：

1. **第 1～2 天：**补齐跨进程 FULL 测试、错误注入入口和当前行为基线；
2. **第 3～4 天：**实现统一分区表、容量/重叠断言与格式版本；
3. **第 5～7 天：**实现 FULL 分代快照、双缓冲和持久化 ACK；
4. **第 8 天：**补齐请求超时、错误传播和有界关闭；
5. **第 9～10 天：**完成元数据 CRC/generation/flush 与损坏回退测试；
6. **迭代复盘：**只在 G0～G2 全部通过后，开始 D2/D3 的 R0 Delta 闭环。

该安排的首个可展示成果不是带宽峰值，而是“训练继续执行时生成的 FULL 检查点，
在进程重启后仍精确对应触发 step，并且任一后台故障都可被可靠发现”。它将成为后续
异步流水、消融实验和减少实际写入量论证的共同可信基线。

### 9.7 本次 G0～G5 实施记录（2026-08-25）

本轮在目标服务器的 `0000:83:00.0`（V2 格式、独占 SPDK）上完成了正确性与证据门禁：

| 门禁 | 结果 | 结构化证据 |
|---|---|---|
| G0 | PASS：Host/NPU 读写回环、批量边界、容量检查、超时/恢复 | `experiments/output/gates/g0_*`、`tests/hardware/g0_roundtrip.py` |
| G1 | PASS：GPT-2 XL 772 参数 FULL、DISPATCHED/PERSISTED、跨进程逐参数摘要校验 | `experiments/output/gates/g1_*/g1_manifest.json` |
| G2 | PASS：metadata A/B generation/CRC 回退，superblock 损坏硬失败并修复 | `tests/hardware/g2_metadata.py` |
| G3 | PASS：300 个 FP16 无损 Delta frame、超过两次 slot 回绕、重启恢复和 CRC 注入 | `experiments/output/gates/g3/g3_manifest.json` |
| G4 | PASS：NPU1/NPU2 两 rank prepare/commit，协调器独占 NPU7/83.0.0，rank1 故障不发布 | `experiments/output/gates/g4/g4_manifest.json` |
| G5 | PASS：5 次预热、10 次正式回环、统一 run/request ID、分层时间戳和 95% CI | `experiments/output/gates/g5_20260825_host/` |

G5 当前是固定设备上的 Host→SPDK→裸 NVMe 证据样本，不是跨 SSD 横比，也不替代后续
E1～E5、A1～A10 和完整训练阻塞测量。G4 已验证完整训练状态字段（权重、优化器、RNG、
data cursor）的两阶段提交与摘要恢复；跨进程继续训练后的 loss/数据顺序仍列为后续
分布式长程实验，不能仅凭本轮小状态回环宣称已完成。

### 9.8 工作包一实施记录（2026-08-25）

本轮在临时分支 `exp/wp1-io-overhead` 完成了工作包一的工具和可执行证据补充：

| 项目 | 结果 | 证据 |
|---|---|---|
| E1 裸 NVMe | PASS：6 个请求大小 × 4 个 pipeline depth，正式样本全部通过回读 | `experiments/output/wp1/e1_root/` |
| E1 内存层次 | PASS：Host memcpy、ACL H2D/D2H/D2D，4 KiB～16 MiB | `experiments/output/wp1/e1_memory/` |
| E2 | 既有 1 GiB 合成缓冲、GPT-2 XL P0～P5 结果已生成只读 manifest | `experiments/output/wp1/manifests/e2/manifest.json` |
| E3 正式路径 | PASS：GPT-2 XL P4，4 MiB chunk、depth=4、3 次，回读校验通过 | `experiments/output/wp1/e3_p4/` |
| E3 外部剖析 | PASS：`perf stat`、`perf record`、`strace -f -c` 均返回 0 | `experiments/output/wp1/profiles/e3_p4/` |
| E4/E5 | 既有正式结果已生成只读 manifest | `experiments/output/wp1/manifests/e4/`、`e5/` |
| 时间线完整性 | PASS：WP1 新结果 53 个，0 个非单调或负时长样本 | `experiments/output/wp1/summary.json` |

外部剖析运行会重新触发 MindSpore 首次图编译，单次耗时约 124～158 秒；因此 profile
只用于 CPU、系统调用和调用栈归因，不与正式延迟均值混合。`strace` 汇总显示该运行有
约 265 万次系统调用，说明文件系统/运行时控制面开销需要与设备传输时间分开分析。

更大模型门禁当前未通过：本机 `/models/Qwen3-8B` 是 `model_type=qwen3` 的 HuggingFace
格式，而 `mindformers==1.3.2` 不支持该架构；`llama2_7b` 配置可以识别，但单卡完整模型
构建在限定时间内未完成。因此当前不能宣称 E2/E4 已覆盖更大真实模型；后续需先升级或
适配模型运行时，再补做大模型正式路径实验。

### 9.9 详细 checkpoint I/O 拆解结果（2026-08-25）

本轮完成了 83.0.0 同盘 ext4/SPDK 对照、模型规模矩阵、P1/P2/P4/P5 路径和外部系统调用
采样。83.0.0 最终已恢复为 `uio_pci_generic` + V2 SPDK，84.0.0 始终保持 XFS `/models`，
所有样本均通过回读或逐参数 hash 校验。

#### 同盘固定字节流对照

256 MiB、4 MiB chunk、10 次正式样本；write 指标包含 fdatasync 或 SPDK completion，
read 指标仅表示读回，hash 校验单独计时：

| 后端 | 写持久化均值 | 读均值 | 端到端均值 | 相对 SPDK 写 |
|---|---:|---:|---:|---:|
| ext4 buffered + fdatasync | 279.0 ms | 239.8 ms | 944.0 ms | 4.48× |
| ext4 O_DIRECT | 66.0 ms | 183.6 ms | 701.0 ms | 1.06× |
| Host buffer → SPDK | 62.3 ms | 176.9 ms | 688.6 ms | 1.00× |

该结果支持一个更精确的结论：**buffered 文件系统路径中的 page-cache/writeback/fsync 以及
相关内核控制面开销占比很大；O_DIRECT 后文件系统路径已接近 SPDK，不能把全部差异归因于
“内核代码本身”。** 因此论文中应将“通用框架 buffered FS 路径”与“direct-I/O FS 路径”
分开描述。

#### MindFormers checkpoint-only 模型矩阵

模型使用真实 MindFormers 网络结构和随机初始化参数，不包含 Adam 状态；编译/预热时间不在
checkpoint 计时区间内。P4 为 HBM→SPDK→HBM，4 MiB chunk、depth=4、3 次正式样本：

| 模型 | 参数字节 | 参数对象数 | 写均值 | 读均值 | 结果 |
|---|---:|---:|---:|---:|---|
| GPT-2 124M | 0.50 GB | 196 | 125.1 ms | 103.5 ms | PASS |
| GPT-2 XL | 3.27 GB | 772 | 726.6 ms | 636.8 ms | PASS |
| Llama2 7B | 13.48 GB | 291 | 3104.1 ms | 2098.6 ms | PASS |
| GLM4 9B | 18.80 GB | 283 | 4358.1 ms | 2891.0 ms | PASS* |
| GPT-2 13B | 26.20 GB | 644 | 6083.6 ms | 4096.1 ms | PASS |

GLM4 使用 `use_past=False` 绕过当前 CANN 缺少的 `ReshapeAndCache` adapter；原生 KV-cache
路径的两个失败门禁也已保留，GLM4 结果只代表参数 I/O 和普通前向分配路径，不代表完整
推理性能。模型结果和原始事件位于
`experiments/output/wp1/current/checkpoint_matrix_summary.json`。

#### 文件系统、原生框架与异步路径

- GPT-2 XL P1：`ms.save_checkpoint` save 均值 2946.4 ms，restore 均值 5085.7 ms。
- GPT-2 XL P2：HBM→Host snapshot 568.4 ms，84.0.0 XFS persist 3630.0 ms，restore
  331.8 ms。
- GPT-2 13B P2：snapshot 4002.6 ms，84.0.0 XFS persist 25794.5 ms，restore 2087.2 ms。
- GPT-2 XL P5：4 次异步 checkpoint 的 snapshot 726.4–727.3 ms，后台 persist
  706.0–816.1 ms，触发线程等待仅 0.02–0.06 ms；训练 step 的统计包含首次图编译异常值，
  不用于宣称训练吞吐提升。

P2/P4 使用 84.0.0 与 83.0.0 两块不同设备，只用于端到端路径观察；文件系统软件栈的强归因
使用前述同盘固定字节流实验。外部 `strace -f -c` 控制样本记录 32 次 pwrite、32 次 pread
和 2 次 fdatasync，共 2218 次系统调用，证据文件位于
`experiments/output/wp1/current/external_profile/`。

### 9.10 工作包二消融设计与执行协议（2026-08-25）

工作包二固定两个模型规模：GPT-2 XL（约 3.27 GB 参数）和 GPT-2 13B（约 26.20 GB
参数）。A1～A6、A8～A10 使用真实 MindFormers 网络的随机初始化参数，不包含 optimizer
状态；A7 使用真实训练 cell 和图内 step counter。所有正式 I/O 对照固定在
`0000:83:00.0`，`0000:84:00.0` 只保留作普通文件系统环境，不参与同盘结论。

正式样本协议为短实验 5 次预热 + 10 次正式重复，模型/训练实验 2 次预热 + 3 次独立
正式重复；每个样本必须通过逐参数 hash 或字节回读，失败样本不得进入统计。每组结果
记录控制变量、唯一变化因素、NPU/NUMA/PCIe 拓扑、触发/快照/提交/完成时间、CPU 和
上下文切换，并写入独立的 A 编号结果目录。

| 消融 | 控制与实验配置 | 固定变量与关键计时 | 执行状态 |
|---|---|---|---|
| A1 | 同一块 83.0.0：ext4 Host-FS vs Host-SPDK 裸盘 | 模型、4 MiB、depth=4；FS 切换前后均记录设备身份 | WP1 字节流已通过，模型版待执行 |
| A2 | P3 HBM→Host→SPDK vs P4 HBM→SPDK | 同一 raw 83、同一模型；D2H、SPDK、H2D/验证和阻塞 | runner 已有 P3/P4，待同盘正式运行 |
| A3 | P4 batch 数组 vs 每 chunk 一次 API | 同一模型、chunk/depth 固定；Python/C 提交时间和 CPU | 已补 `--submit-mode scalar` |
| A4 | depth=1 vs 2/4/8/16 | 同一模型和 chunk；ACL/NVMe 重叠、槽等待、P99 | 复用 P4 runner |
| A5 | chunk=64 KiB/256 KiB/1 MiB/4 MiB/16 MiB | depth=4；尾块、内存和有效/介质带宽 | 复用 P4 runner |
| A6 | Reactor request-ring/FSM vs 受控同步提交 | 同一 context 和设备；锁/等待/CPU/吞吐 | 需先加入独立 sync API，禁止恢复旧死锁实现 |
| A7 | Python 同步/线程触发 vs 图内计数 + Reactor poller | 同一训练步和间隔；step time、触发抖动、训练阻塞 | 需使用 `setup_faf_checkpointing` 正式路径 |
| A8 | 单槽同步元数据 vs A/B generation/CRC/active 切换 | 安全 metadata 区域；提交时间和故障回退 | 现有脚本仅 preliminary，需补 active 切换门禁 |
| A9 | 1/2/3/4 个并发快照槽 | 单 Reactor owner；槽等待、内存、实际重叠和吞吐 | 需禁止把 pipeline depth 代替 slot count |
| A10 | CPU/内存 node=2（NPU7 本地）vs node=4（SSD 本地） | 同一模型、设备和参数；CPU、P99、带宽 | 通过 `numactl` 外部固定绑定 |

A1 的模型版必须在 83.0.0 文件系统挂载阶段使用 `model_paths.py --fs-root`
写入该挂载点，然后卸载、恢复 `uio_pci_generic` 并重新执行 P4；不得拿当前 84.0.0
P2 结果直接充当 A1。A3 的 scalar 模式允许逐 chunk 调用同一个安全 C API，不能调用
已归档的旧线程实现。A6、A7、A9 在对应安全实现和正确性门禁完成前，只能报告为设计
准备或 preliminary，不能填入正式 FULL 消融表。

首轮执行顺序为 A1/A2/A4/A7，再执行 A3/A5/A6/A8/A9/A10；每完成一个设计点立即生成
结构化汇总和 hash 清单。最终表格只比较同一模型、同一设备、同一计时边界的控制/实验
对，另将 13B 的完整矩阵作为规模敏感性复核，不用跨设备结果替代消融。

### 9.11 工作包二首轮执行记录（2026-08-25）

首轮结果写入 `experiments/output/wp2/`，并使用 GPT-2 XL 的真实 MindFormers 网络
参数完成了关键路径验证。模型实验均为随机初始化参数的 checkpoint-only 测量；除 A7
外不包含 optimizer 状态，且所有样本均通过逐参数摘要或字节回读校验。

| 消融 | 首轮结果 | 状态与解释 |
|---|---|---|
| A1 | 同一块 83.0.0、GPT-2 XL：P2 ext4 persist 4275.3 ms；P4 SPDK write 764.1 ms | PASS；模型版同盘对照完成 |
| A2 | GPT-2 XL：P3 snapshot/write/read = 574.9/713.6/857.4 ms；P4 write/read = 764.1/641.2 ms | PASS；P3 的 Host 中转和 P4 的 HBM 直达路径已分开记录 |
| A3 | 合成回环 batch 的有效带宽随 items 1/4/16/64 为 186.8/364.9/431.6/465.8 MiB/s；GPT-2 XL scalar write 2303.3 ms，明显高于既有 batch 约 726.6 ms | PASS；说明逐 chunk Python/C 提交开销不可忽略 |
| A4 | 合成 depth 1/2/4/8/16 端到端均值约 10.66～11.29 ms | preliminary；模型版 depth 矩阵待补，不能据此宣称模型吞吐无收益 |
| A5 | 合成 chunk 64 KiB～16 MiB 有效带宽 26.1～485.5 MiB/s | preliminary；模型版 chunk 矩阵待补 |
| A6 | Reactor depth proxy：depth 1/4 为 374.6/360.3 MiB/s | preliminary；尚未实现独立安全同步 API，不纳入正式消融表 |
| A7 | GPT-2 XL 真实训练 cell：20 steps、4 个图内计数触发点全部完成；step 均值 427.0 ms，边界等待均值 435.0 ms | PASS；使用图内 counter + Reactor poller，未替换已编译 device Parameter |
| A8 | metadata 单写 1.104 ms；A/B 写后双读协议 5.490 ms | preliminary；现有脚本尚未完成 active-slot 切换与故障注入门禁 |
| A9 | 1/2/4 worker proxy 有效带宽 392.3/291.0/235.0 MiB/s | preliminary；当前不是严格的 snapshot-slot 生命周期重叠实验 |
| A10 | 合成 node2/node4 有效带宽 355.3/315.2 MiB/s | preliminary；模型版 NUMA 对照待补 |

首轮执行还暴露出一个环境要求：root 下运行 MindFormers 模型实验时，`sudo -E`
不足以保留 CANN 环境，必须在 root shell 中显式执行
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。后续所有模型命令沿用这一启动
方式。A4/A5/A10 的合成结果仅用于先验证开关和测量链路；GPT-2 13B 的完整规模敏感性
矩阵与尚未正式化的 A6/A8/A9 将继续单独记录，不能与本节 PASS 项混合。

### 9.12 工作包二 GPT-2 13B 规模复核记录（2026-08-25）

GPT-2 13B 使用 644 个真实 MindFormers 参数对象、26,204,712,960 bytes，4 MiB
chunk；每个正式样本均通过 644 项参数摘要校验。除特别说明外，模型路径均为
checkpoint-only、2 次预热 + 3 次正式重复、同一块 83.0.0。

| 消融 | 配置 | 正式均值（ms） | 结果 |
|---|---|---:|---|
| A1 | P2 ext4 Host-FS | snapshot 3922.1；persist 33425.1；restore 2072.0 | PASS |
| A1/A2 control | P4 HBM→SPDK→HBM，depth=4 | write 6051.3；read 4137.6 | PASS |
| A2 | P3 HBM→Host→SPDK→Host | snapshot 4158.0；write 6065.5；read 7099.3 | PASS |
| A3 | P4 scalar，每 chunk 一次 API | write 13758.3；read 13822.9 | PASS；相对 batch 分别为 2.27×、3.34× |
| A4 | P4 batch，depth=1/2/4/8/16 | write 8851.0/6127.4/6051.3/6081.7/6018.4；read 9607.9/5679.7/4137.6/3889.4/3807.3 | PASS；depth=4 复用 A2 控制样本 |

A4 depth=16 的首轮运行曾因读 FSM 将 transient `-ENOMEM/-EAGAIN` 错误地视为永久
失败而返回 `raw SPDK read returned -1`；修复读提交重试后重新执行 3 次正式样本全部
通过。失败目录保留在 `experiments/output/wp2/model_gpt2_13b_a4_d16/`，修复后的正式
目录为 `model_gpt2_13b_a4_d16_v2/`，该缺陷修复提交为 `b6ccb8a`。

这些结果支持两个规模敏感性结论：13B 的 P2 文件系统持久化均值约为 P4 raw write
的 5.5 倍；在 P4 内，pipeline depth 从 1 增加到 8/16 后读均值从约 9.61 s 降至
约 3.89/3.81 s，但 write 在 depth=2 以后已基本进入 6.0～6.1 s 平台。A5 的 13B
chunk-size 矩阵和 A10 的 13B NUMA 矩阵仍待执行；A6/A8/A9 仍按 9.11 的 preliminary
规则处理。

### 9.13 工作包二 A5/A8/A10 补充记录（2026-08-25）

继续执行后，13B 模型版补齐了 A5 的 1/4/16 MiB 三个中间与端点配置，以及 A10
的两个 NUMA 绑定；A8 已升级为独立安全区中的真实 generation/CRC/active-slot
协议测试。

| 消融 | 配置 | 正式结果 | 状态 |
|---|---|---:|---|
| A5 | P4、chunk=1 MiB | write 6399.7 ms；read 6556.8 ms | PASS |
| A5 | P4、chunk=4 MiB | write 6051.3 ms；read 4137.6 ms | PASS，复用 A2 控制 |
| A5 | P4、chunk=16 MiB | write 6101.7 ms；read 4040.4 ms | PASS |
| A8 | A/B generation/CRC/active，10 次正式 | 单槽 write/read 1.108/1.708 ms；A/B commit 7.040 ms；损坏回退 6.619 ms | PASS；128 GiB 安全区，未改 live metadata |
| A10 | node2（NPU 本地） | write/read 6088.3/4223.1 ms | PASS |
| A10 | node4（SSD 本地） | write/read 6035.0/4234.9 ms | PASS |

A5 合成实验仍覆盖 64 KiB、256 KiB、1 MiB、4 MiB、16 MiB；13B 的 64/256 KiB
模型版因分别产生约 400k/100k chunks，已补入正式模型表，不能用中间尺寸结果
替代这两个极端。为使极小 chunk 与 depth=16 可执行，本轮还修复了 `npu-smi`
进程表误判（`b36ea16`）、高 pipeline 深度读提交重试（`b6ccb8a`）和读写 FSM
重复扫描导致的 O(N²) 开销（`d2fb8fc`）。修复后的 13B depth=16 结果已通过，
其首轮失败目录仅作为缺陷复现证据保留。

### 9.18 工作包二 A5 极小 chunk 模型版收口与 WP3 I0 基线（2026-08-26）

在临时实验分支 `exp/wp1-wp2-closeout` 上，GPT-2 13B 的 P4 HBM→SPDK→HBM
checkpoint-only 极小 chunk 测试已完成。模型包含 644 个真实 MindFormers 参数对象，
每个正式样本均通过逐参数摘要校验；64 KiB 运行使用 `NPU_NVME_IO_TIMEOUT_MS=600000`
以覆盖约 400k chunk 的长请求，不能与默认 60 s 超时失败混为数据失败。

| chunk | chunks（约） | write mean (ms) | read mean (ms) | first-param mean (ms) | 状态 |
|---:|---:|---:|---:|---:|---|
| 256 KiB | 100k | 19227.6 | 19152.4 | 768.2 | PASS |
| 64 KiB | 400k | 67506.4 | 71447.3 | 2660.6 | PASS |

256 KiB 原始运行目录为 `experiments/output/wp2_closeout/E4_20260826_114234_3d6c63ea`，
64 KiB 正式目录为 `experiments/output/wp2_closeout/E4_20260826_120003_101a4f3b`；
64 KiB 首次以默认 60 s 超时运行的失败目录
`E4_20260826_115342_7a3620dc` 保留为配置边界证据，失败为 `-ETIMEDOUT`，没有进入
样本校验阶段。测试结束后 NPU 7 HBM 使用率回落至 5%，83.0.0 仍由
`uio_pci_generic` 接管，84.0.0 未触碰。

同一分支新增 `python/s2_delta.py` 与 `tests/python/test_s2_delta.py`，建立 S2/R0
CPU oracle 和 v3 replacement frame 基线：块始终在参数内切分，frame 携带稳定 manifest
digest、native dtype、base/generation 和 CRC；`observe()` 只读取最后 ACK 的
`persisted_reference`，只有 `ack()` 推进参考状态，旧 v1/v2 additive frame 保持兼容。
Z0～Z9 及协议 dispatch/CRC 测试当前 18 项通过。这是 I0 的第一版 L3 单机 CPU 门禁，
尚不等同于 I1 真实轨迹、I2 NPU 图、I3 缓冲生命周期或 I4～I6 存储闭环。

新增 `experiments/benchmarks/s2_oracle_trajectory.py`，提供 100-step 确定性早期稀疏、
中期稠密、后期热块轨迹回放，输出每 step 的 frame bytes、selected block、Jaccard、
generation 和最终恢复校验；当前 smoke 运行 PASS（总 frame 458,668 bytes）。该工具
已具备 I1/O1/O2 的指标接口，但默认轨迹是 CPU 合成轨迹，真实 MindSpore 权重/Adam
轨迹采集仍未完成，不能把该 smoke 作为训练结论。

A7 GPT-2 13B 训练扩展在 2026-08-26 进行了多次隔离重试：数据列映射、短序列裁剪和
attention 配置问题已修复，但 MindFormers 1.3.2 的 GPT-2 实例仍在静态图中保留
1025/128 不一致的内部 `position_ids/seq_length` 常量，最终均在训练图 warmup 前失败，
没有产生训练或 I/O 样本。该结果只定义“本安装版本的 13B 短序列 A7 harness 边界”，
不否定已完成的 GPT-2 XL A7；若需 13B A7，应改用原生 1025 序列或修订 MindFormers
模型实现，并重新评估 64 GiB HBM/Adam 余量。

I4 基础设施已新增 `FileS2Ring`：完整 frame 经 flush+fsync 后原子替换 slot，读取端
重新校验 CRC 和版本；文件 ring 回绕/损坏及独立子进程读取测试已纳入 Python 门禁，当前总计 29 项通过。
这仍是普通文件跨进程恢复基础，不代表 I4 跨进程 replay、I5 NPU-SPDK 或 I6 裸盘故障门禁已通过。

I5 的首轮 Host-SPDK 与 NPU-HBM-SPDK frame loopback 均已通过：分别在 83.0.0 安全区
64 GiB 和 64 GiB+16 MiB 偏移写入 4,159 B frame，均按 8,192 B 对齐传输，读回逐字节
一致，ACK 和独立恢复均到 generation 1；Host C-layer write 为 348 us。未修改 live
metadata，84.0.0 未触碰。结构化摘要位于 `results/wp3-20260826/s2_gate_summary.json`。
I5 的非对齐、尾块和多 segment 矩阵仍待补齐；I2/I3 和 I6 仍开放，I4 仍需补真正跨进程 replay 门禁。

### 9.19 工作包二 A9 真实 HBM slot 收口记录（2026-08-26）

新增 `experiments/benchmarks/a9_hbm_slots.py` 和 `python/hbm_slots.py`，以 GPT-2 XL
真实 MindSpore 训练 cell 验证 `FREE → SNAPSHOT → READY → IO → PERSISTED → FREE`。
每个 slot 使用 3,280,687,104 B 的 D2D HBM shadow buffer，由单一 SPDK owner 后台写入
83.0.0，再通过 Host 读回并与冻结 HBM slot 做 SHA-256 比对。

正常矩阵为 25 steps、5 次触发、2 次 warmup；1/2/4 slot 均 PASS，各有 3 个正式样本。
受控慢盘矩阵在每次写入前增加 5,000 ms 延迟，15 steps、5 次触发；1/2/4 slot 也均 PASS。
慢盘下平均 slot wait 从 1 slot 的 7,038.4 ms 降至 2 slot 的 3,114.4 ms，并在 4 slot
降至 0.05 ms，说明多 slot 能降低冻结缓冲等待，但不能减少物理写入时间。

完整记录位于 `results/wp2-20260826/a9_hbm/`。该结果关闭了 A9 的真实 HBM slot
scope-specific 门禁；多 rank HBM、长训练三 seed 和与 A7 统一训练基线仍未完成。

### 9.14 A6 安全同步 API 对照记录（2026-08-25）

A6 在同一块 83.0.0 的独立安全区使用 400 KiB payload、5 次预热和 10 次正式
重复，对比已有 `npu_nvme_sync_meta_io` 受控同步调用与 request-ring/FSM 的
`write_batch_host/read_batch_host`。结果为：

| 路径 | 写均值 | 读均值 | 状态 |
|---|---:|---:|---|
| sync control | 1.091 ms | 1.385 ms | PASS |
| request-ring/FSM | 1.058 ms | 1.676 ms | PASS |

该实验正式关闭了“同步 API 与 Reactor request-ring 的控制面可运行性”门禁，证据位于
`experiments/output/wp2/a6_formal/`；但由于同步 API 的安全边界是 metadata-size，
本结果只能解释 API/控制面开销，不能替代 A1/A2/A3 的模型 checkpoint 路径比较。

### 9.15 A9 host snapshot slot 生命周期记录（2026-08-25）

A9 使用单一 DirectCheckpoint/Reactor owner，预分配并复用独立 host slot，显式检查
`FREE → SNAPSHOT → READY → IO → FREE` 生命周期；pipeline depth 固定为 4，slot
数量改变为 1/2/4。每组 5 次预热、10 次正式 wave，正式样本数分别为 10/20/40，
所有 slot 在 wave 结束后回到 FREE。

| slot 数 | 端到端均值 | 有效带宽均值 | 状态 |
|---:|---:|---:|---|
| 1 | 6.903 ms | 579.5 MiB/s | PASS |
| 2 | 9.953 ms | 412.2 MiB/s | PASS |
| 4 | 13.845 ms | 309.2 MiB/s | PASS |

该结果正式关闭了 host snapshot slot 生命周期与单 Reactor owner 的门禁；由于
snapshot payload 在 host DRAM 中生成，不能把它等同于真实 MindFormers HBM 快照，
模型 HBM slot 的内存占用与训练重叠仍是后续增强项。正式证据位于
`experiments/output/wp2/a9_formal/`。

### 9.16 工作包二剩余工作与收口计划

1. **A5 极小 chunk 模型版**：已在 83.0.0、P4、depth=4 下补齐 GPT-2 13B 的 256 KiB 和
   64 KiB；保留约 100k/400k chunks 的完整原始结果，记录提交开销、尾块、带宽和失败率。
2. **A7 规模扩展**：在 GPT-2 XL 真实训练 cell 之外，补充 GPT-2 13B 短训练或可行的更大
   MindFormers 模型，比较 Python/线程触发与图内计数 + Reactor poller 的 step、抖动、
   阻塞、积压和失败传播。
3. **A9 HBM slot 增强**：保留 host-slot formal 结果，同时使用真实 MindFormers 训练 cell
   完成 1/2/4 个 HBM snapshot slot；验证 `FREE → SNAPSHOT → READY → IO → FREE`、冻结
   generation、槽等待、HBM 占用和慢盘背压，不能用 host payload 代替模型结论。
4. **WP2 结果收口**：统一生成 A1～A10 矩阵、P95/P99 和置信区间、失败样本及修复前后
   对比；明确 A6 的控制面范围、A8 的安全区范围、A9 的 host/HBM 边界以及未完成项。

在以上工作完成前，WP2 可称为“各设计点已具备正式 scope-specific PASS”，但不能称为
“所有模型规模和所有真实 HBM 并发场景均已完成”。

### 9.17 工作包三完整执行顺序

#### 阶段 A：语义与 CPU oracle

- 固定 S2：相对 `P_persisted` 选择、保存当前块值、ACK 后推进参考状态；S1 作负对照，
  S3 后置；
- 建立稳定 block manifest、参数边界、有效长度、dtype、padding 和 frame version；
- 完成 I0 与 Z0～Z9，维护独立的 oracle、persisted reference、recovered state；
- 从真实训练采集 O1～O7 所需的权重、optimizer state、Jaccard、age、能量和成本数据。

#### 阶段 B：NPU、内存与 frame 闭环

- 完成 I2：CPU/NPU norm、索引、值、scale、有效长度逐项等价；
- 完成 I3：双缓冲/环形缓冲、generation、ACK、未持久化不可覆写；
- 完成 I4：普通文件跨进程 `FULL + Delta` 恢复；
- 完成 I5：Host-SPDK/NPU-SPDK frame 字节、checksum、非对齐和尾块回环。

#### 阶段 C：裸盘恢复与 R0

- 完成 I6：100-step 链、两次以上回绕、payload/header/manifest/checksum/step/base generation
  故障、重复/缺失/乱序 frame；
- 只在 I0～I6 全部通过后实现 R0 FP16 无损替换块；
- 通过后再测恢复后 loss、RTO、物理写入比例和 FULL 摊销。

#### 阶段 D：R1/R2 与候选筛选

- 实现自适应阈值/Top-K、per-block INT8 和误差预算；
- 实现 residual feedback、最大块年龄和写出后清零；
- 扫描 D0～D7，先 CPU Pareto 粗筛，再进行配对 NPU 训练实验；
- 仅在数据支持时引入 R3 热度、R5 粗筛、R6 分状态和 R4 分级持久化。

#### 阶段 E：I7 与路线决策

- 执行长训练、后台 I/O、慢盘背压、多 slot、进程故障和单 rank 故障；
- 与无 checkpoint、FULL-only baseline 对比训练吞吐、物理写量、恢复误差和 RTO；
- 按 Go/Pivot/Stop 判据决定继续图上增量、转向量化/Host 处理或收敛到 FULL I/O 优化；
- 固化原始日志、结构化 JSON、图表输入、失败分析和最终论文结论边界。

### 9.20 工作包三 I1/I2 实施记录（2026-08-26）

本轮在 `exp/wp2-wp3-remaining` 上补齐了真实轨迹采集器的数值门禁和第一版 NPU
算子等价门禁，仍只使用 83.0.0 作为目标裸盘；I2 本身不做盘 I/O。Python 测试套件为
29 passed。

#### I1 真实 MindSpore 轨迹门禁

`experiments/benchmarks/s2_real_trajectory.py` 现在在 warmup 后、训练前先对权重和
Adam `m/v/global_step` 做有限性检查，并写出 `numeric_gate.json`。GPT-2 和 GPT-2 XL
均在该门禁失败：

| 模型 | 状态数组 | 非有限数组 | 失败阶段 | 结论 |
|---|---:|---:|---|---|
| GPT-2 | 589 | 21 | post-warmup initial state | 未开始采样 |
| GPT-2 XL | 772 | 2 | post-warmup initial state | 未开始采样 |

GPT-2 的非有限值主要出现在首层 attention/output 权重以及对应 Adam 状态；GPT-2 XL
出现在 word/position embedding。两次运行均由 `npu-smi info` 确认目标 NPU 空闲，且
结果以 `fail` 固化，未混入任何性能均值。因此，I1 的真实训练轨迹尚不能声称通过，
需要后续先解决 MindFormers 1.3.2/CANN 组合的 warmup 数值初始化问题，或改用已验证
可加载且有限的真实 checkpoint，再进行 100-step、三阶段和 optimizer-state 实验。

#### I2 CPU/NPU 图算子等价

在 NPU 5 上运行 `32 × 257` 的确定性块输入，使用不同有效长度、零变化块和非满尾块，
逐项比较 CPU reference 与 MindSpore Ascend 图输出。norm、Top-K 值、Top-K 索引、选中
块值、per-block scale 和 INT8 量化六项全部 PASS。此次结果关闭的是“算子/有效长度
语义等价”这一 scope-specific 门禁，不代表真实模型 I1 已通过，也不替代 I3 缓冲
生命周期测试。

#### I3/I4 追加结果

新增 `python/frame_lifecycle.py` 及 3 个单测，验证不可变 payload、generation 单调性、
慢写背压、错误 ACK 拒绝和失败释放；A9 已提供真实 HBM snapshot slot 的硬件证据，
但“图输出/真实 HBM frame buffer → writer”整合仍未关闭。另完成 I4 普通文件跨进程
回放：父进程写入 `FULL + 10` 个 S2 frame，子进程从 16-slot ring 读取并恢复到
generation 10，最终状态摘要逐字节一致。期间修复了 S2 v3 负 `layer_id` 的打包 bug，
保持 12 字节 block header 布局不变，并新增 LayerNorm 大参数测试。

I6 第一阶段也已完成：在 83.0.0 安全区写入 100 代 frame，父进程释放 SPDK context，
子进程使用新的 `SPDK_SHM_ID` 重新初始化并恢复到 generation 100，状态逐字节一致；
payload 翻转样本被 CRC 拒绝。该结果关闭“100-step 原始盘重启 + 单帧损坏”的
scope-specific 门禁，ring 回绕、header/manifest/step/base-generation 组合故障和
重复/缺失/乱序矩阵仍未完成。

证据目录：`results/wp3-20260826/`；I1 的两个失败运行、I2/I4/I6 的原始 JSON、环境
快照和时间线均保留。当前下一顺序为：解决/替换 I1 数值输入 → I3 图输出到 HBM frame
buffer 整合 → I6 100-step/回绕/故障矩阵；在 I0～I6 完成前不进入增量方案性能结论。

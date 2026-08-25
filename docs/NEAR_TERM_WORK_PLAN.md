# 近期研发与实验计划：I/O 路径、设计消融与增量检查点

> 制定日期：2026-08-25
>
> 代码审查基线：`6e8d57c`
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
| D1 | 变化块选择 | 全量 Delta、固定 Top-K、自适应 Top-K、阈值选择 |
| D2 | 块大小 | 16K、64K、256K、512K、1M elements（按参数边界切分） |
| D3 | 编码 | FP16、INT8 per-tensor scale、INT8 per-block scale |
| D4 | 历史状态 | 精确 FP16 `P_old`、量化 `P_old+scale`、周期性重建 |
| D5 | 误差反馈 | 无残差、Host residual、NPU 图内 residual |
| D6 | FULL 间隔 | 10、25、50、100 steps 或按误差自适应触发 |
| D7 | 持久化方式 | Host 组帧、NPU 原始缓冲、NPU 自描述帧 |

主要输出包括写入比例、图内计算时间、HBM 额外占用、Delta 帧大小、恢复时间、链长、
逐参数 NRMSE、恢复后 loss 偏差和训练吞吐。

## 5. 工作包三：增量检查点探索路径

### 5.1 先做观察实验，不预设 Top-K 一定成立

#### O1：参数变化的空间稀疏性

逐层、逐参数和逐块记录 `|P_t-P_{t-1}|` 的 norm、最大值和非零比例，绘制累计变化
能量—块比例曲线。分别回答 Top-1%、5%、10%、20% 能覆盖多少变化，以及不同层是否
应使用相同阈值。

#### O2：时间局部性与“长期饿死”

记录相邻 step 的选中块 Jaccard、块的连续命中长度、距上次持久化的年龄和未保存残差。
重点观察持续小幅变化但长期进不了 Top-K 的块，这类块是历史 100 步误差累积的主要
候选原因。

#### O3：块粒度

扫描块大小，比较选择元数据、padding、并行度、稀疏性和恢复误差。块不能跨参数语义
边界；小参数应单独打包或直接纳入每次 Delta，不能静默忽略。

#### O4：量化与历史状态误差

分别使用精确 FP16 `P_old`、INT8 `P_old` 无 scale、INT8 `P_old` + per-block scale，
区分“Delta 量化误差”和“上一版本状态失真”。测试单步、10/50/100 步链长，而不是
只报告一次量化误差。

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

### 5.2 可探索的实现方案

#### 方案 R0：无损块级 Delta 基线

- 保存所有发生变化的块，采用 FP16 数据和自描述 Delta frame；
- 使用精确 FP16 `P_old`；
- 实现稳定的全局块 ID、参数名、参数内 offset 和 dtype/shape 映射；
- 目标是先闭合 FULL + Delta 恢复语义，为有损方案提供 oracle。

优点是正确性清晰；缺点是 HBM 影子状态和写入缩减可能有限。该方案应最先完成。

#### 方案 R1：自适应 Top-K/阈值 + per-block INT8

- 每层或每参数根据变化分布选择阈值，而非全模型固定比例；
- Delta 和 `P_old` 都保存 per-block scale；
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

### 5.3 推荐实现顺序与决策门槛

1. **先做 R0。** 若不能完成 100 步 `FULL + Delta` 字节级或数值级恢复，不进入有损优化。
2. **并行完成 O1～O5。** 用观察数据决定固定 Top-K 是否放弃，以及阈值是否分层。
3. **优先比较 R1 与 R2。** R1 验证量化数据缩减，R2解决长期误差；两者可组合。
4. **有明显时间局部性再做 R3。** 若块热度不稳定，不增加热度调度复杂度。
5. **R4 作为扩展路线。** 先完成本地 NVMe 可靠落盘闭环，再研究 DRAM 分级缓存。

增量方案进入正式性能实验的暂定门槛：

- 帧校验、step 连续性、slot 回绕和故障注入全部通过；
- 单步恢复逐参数 NRMSE 中位数不高于 5e-3；
- 100 步链恢复 NRMSE 中位数不高于 1e-2，且无长期未持久化块；
- 恢复后固定批次 loss 相对偏差不高于 1%；
- 平均实际写入量低于 FULL 的 20%；
- 图内增量处理带来的稳定训练 step 开销不高于 10%。

这些是路线筛选门槛，不是论文预设结论；若观察到 loss 对参数误差更敏感，应收紧阈值。

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

## 8. 近期需要作出的三项技术决策

1. **Delta 统一采用自描述 frame，还是继续直接持久化三个原始 HBM 缓冲？** 由于
   当前 `recover()` 只理解 frame，建议以自描述 frame 为唯一磁盘协议。
2. **`P_old` 保留精确 FP16，还是使用带 scale 的 INT8？** 建议 R0 先用 FP16 建立
   oracle，再根据 O4 的 HBM 与误差数据决定是否切换 INT8。
3. **固定 Top-K 是否继续作为主方案？** 在 O1/O2/O5 完成前不做结论；若长期饿死
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

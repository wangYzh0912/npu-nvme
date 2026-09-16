# NPU-to-NVMe 直接检查点的单线程 I/O 控制平面

## 摘要

大语言模型训练需要周期性将检查点持久化到存储设备。在基于昇腾 NPU 集群的直接
NVMe 访问场景中，检查点控制平面需要协调三类 I/O：HBM 到 NVMe 的 DMA 传输、
后台即发即忘（Fire-and-Forget, FaF）持久化、以及 Python 侧元数据 I/O。此前的
设计采用多线程控制平面配合递归互斥锁，导致锁竞争、模型加载死锁以及忙等待 CPU
开销。本文提出基于 SPDK Reactor 模型的单线程控制平面：所有 I/O 操作通过单一
事件驱动线程串行化执行，Python 侧请求通过无锁 MPSC 环形队列传递并以异步有限
状态机（FSM）形式执行。该设计彻底消除了 `io_lock`，修复了 `load()` 死锁，
并将稳态下 Reactor 线程 CPU 占用降至 5% 以下。在昇腾 910B 平台搭配 3.84 TB
NVMe SSD 的测试中，系统达到 4,432 MB/s 写入带宽，与原始多线程实现持平。

## 1. 引言

深度学习训练任务需要周期性检查点以防止硬件故障并支持容错训练。在昇腾 NPU
平台上，NPU-NVMe 直接检查点系统 [1] 绕过主机 CPU，通过 SPDK（Storage
Performance Development Kit）在用户空间实现 NPU 高带宽内存（HBM）与 NVMe
存储之间的直接 DMA 传输。

该系统的控制平面需处理三类 I/O：

1. **同步读写请求**：来自 Python 训练循环的阻塞调用，数据到达存储介质后返回。
2. **异步 FaF 写入**：由训练步边界触发，持久化在后台执行，Python 继续训练无需等待。
3. **元数据 I/O**：用于超级块管理和 JSON 账本更新，在初始化、保存和加载操作期间需要。

在此次重构之前，控制平面同时采用三种线程模型：基于 pthread 的监听线程用于步进
检测、Python 主线程用于同步 I/O、以及内部管线线程。三者竞争同一个递归
`io_lock` 保护共享的 SPDK 队列对（qpair）和 DMA 缓冲池。该架构存在三个记录
在案的问题：(a) **锁竞争**：FaF 监听线程在写入期间（约 1 秒）一直持有
`io_lock`，阻塞所有 Python 侧操作；(b) **加载死锁**：`load()` 在恢复检查点时
需要 `io_lock`，若监听线程同时触发后台写入则产生死锁；(c) **CPU 开销**：
忙等待管线循环每次 I/O 操作消耗 100% CPU。

本文提出基于 SPDK Reactor 模型 [2] 的重构控制平面。核心洞察在于 SPDK qpair
本质上是单线程的（提交队列和完成队列非线程安全）；与其用锁来串行化访问，不如
将所有 I/O 执行整合到单一 reactor 线程上，通过无锁 MPSC（多生产者单消费者）
环形队列传递请求。Python 变为纯控制面：构造请求描述符、入队、轮询完成——从不
直接操作 SPDK qpair 或 DMA 缓冲区。

## 2. 架构设计

### 2.1 总体结构

重构后的系统由两个线程组成：

**Python 主线程**（控制面）：通过三个 SPDK 环形队列（`write_ring`、
`read_ring`、`meta_ring`）发出写、读和元数据请求。每个环是无锁 MPSC 队列，
深度为 16（元数据为 4）。Python 线程通过 `usleep(1000)` 轮询每个请求的
`done` 标志，等待时让出 CPU。

**Reactor 线程**（数据面）：唯一的 I/O 执行者，运行 SPDK 事件循环
（`spdk_thread_poll`）。注册四个轮询器：

| 轮询器                 | 周期     | 功能                                             |
|------------------------|----------|--------------------------------------------------|
| `step_poller_fn`       | 10 ms    | 读取 HBM 步进计数器；触发 FaF 写入                |
| `write_fsm_poller_fn`  | 0（每次）| 消费 `write_ring`，执行异步写 FSM                  |
| `read_fsm_poller_fn`   | 0（每次）| 消费 `read_ring`，执行异步读 FSM                   |
| `meta_poller_fn`       | 0（每次）| 消费 `meta_ring`，使用专用 `meta_qpair`            |

第五个非 I/O 锁（`state_lock`）保护共享的监听器状态（注册任务指针、步进计数器
地址、探针标志），临界区长度为微秒级。

### 2.2 异步写 FSM（V3）

写 FSM 将阻塞式写入管线分解为有界工作步骤。`write_fsm_tick` 每次调用执行：

1. **完成收割**：调用 `spdk_nvme_qpair_process_completions`，触发 SPDK
   写完成回调。回调将每个块的状态更新为 `CHUNK_DONE`，并通过原子 SPSC 环形
   队列将 DMA 缓冲区归还空闲池。

2. **SPDK 提交**：遍历 DMA 复制已完成的块（`CHUNK_NPU_DONE`），通过
   `spdk_nvme_ns_cmd_write` 提交到 NVMe 控制器。

3. **DMA 启动**：从 DMA 池弹出一个空闲缓冲区，执行单次 `aclrtMemcpy`
   （HBM→主机，4 MiB 约 0.9 ms），标记块为 `CHUNK_NPU_DONE`。**每次仅
   发起一次 DMA**，限制 reactor 延迟上限。

4. **完成检查**：当 `completed_count` 达到 `num_tasks` 时，设置
   `req->done = 1` 并将 FSM 转换为 `IDLE`。

对于 FaF 写入，`step_poller_fn` 在 `state_lock` 下重置任务状态并直接启动
FSM（两者均运行在 reactor 线程上，无需环形队列）。背压通过检查
`fsm->state == IDLE` 实现——最多一个写入处于飞行状态。

### 2.3 异步读 FSM（V4）

读 FSM 与写 FSM 镜像对应，用于 NVMe→HBM 传输。每次 tick：

1. 收割 SPDK 读完成。
2. 对已完成块（`CHUNK_SPDK_DONE`）：执行 `aclrtMemcpy`（主机→HBM）将数据
   从 DMA 缓冲区传输到 NPU 内存。
3. 向 SPDK 提交一个新的读命令。
4. 检查完成并通知 Python 调用方。

### 2.4 专用 qpair 的元数据 I/O

元数据 I/O（超级块和 JSON 账本）使用专用第二 qpair（`meta_qpair`），队列深度
64。由于没有其他 I/O 路径使用此 qpair，Python 线程的元数据请求不与数据面操作
竞争——无需锁。`meta_poller_fn` 在 reactor 内同步处理 `meta_ring` 请求（对于
≤1 MiB 的小型元数据 I/O，忙等待完成是可接受的）。

### 2.5 带 ARM64 内存屏障的 SPSC 环形队列

DMA 缓冲区空闲池由 SPSC（单生产者单消费者）环形队列管理。在重构设计中，
reactor 线程是唯一生产者（SPDK 回调归还缓冲区），Python 线程是唯一消费者
（读路径中的 `ring_pop`，现也通过 reactor 执行）。为确保 ARM64 弱内存模型
下的正确性，`ring_push` 使用 `__atomic_store_n(..., __ATOMIC_RELEASE)`，
`ring_pop` 使用 `__atomic_load_n(..., __ATOMIC_ACQUIRE)`。

## 3. 实现

重构被分解为五个独立可测试的版本：

| 版本 | 范围                    | 关键变更                                              |
|------|-------------------------|-------------------------------------------------------|
| V0   | SPDK 线程/轮询器可行性   | 独立可执行文件验证完整链路                              |
| V1   | 最小 Reactor 初始化/清理 | Reactor pthread 集成到 `npu_nvme_init`/`cleanup` 中    |
| V2   | 轮询器替代监听线程       | `step_poller_fn` 替代 `probe_listener_thread`；修复初始化顺序；ARM64 cpumask 修复 |
| V3   | 异步写 FSM + spdk_ring  | `write_fsm_poller_fn` 替代 `run_write_pipeline`；Python 通过 `write_ring` 写入；ACL context 修复 |
| V4   | 异步读 FSM + 移除 io_lock | `read_fsm_poller_fn` 替代 `run_read_pipeline`；`meta_qpair` 处理元数据；`state_lock` 替代 `io_lock` |
| V5   | 清理                    | 移除 515 行死代码；整合调试输出                        |
| V6   | Bug 修复 + C 层性能剖析  | 恢复 V5 误删的 281 行 FSM 函数；添加批次级 C 层计时     |

### 3.1 关键修复

实现过程中发现并解决了四个意外问题：

1. **初始化顺序错误（V2）**：`spdk_thread_lib_init` 在 `spdk_env_init` 之前
   调用，导致 `rte_lcore_count() = 0`，所有 DPDK 分配失败。正确顺序为
   `spdk_env_init` → 诊断 → `spdk_thread_lib_init` → `spdk_thread_create`。

2. **ARM64 上 cpumask 段错误（V2）**：将 `NPUNVMEContext*` 作为
   `spdk_cpuset*` 参数传递给 `spdk_thread_create`，SPDK 内部将其作为可能较大
   的 CPU 位图解引用，导致越界内存访问。通过传递 `NULL` 并使用静态
   `g_reactor_ctx` 变量修复。

3. **ARM64 内存排序（V3）**：SPSC 空闲环使用普通加载和存储。在 ARM64 上，
   reactor 线程的 `ring_push`（来自 SPDK 回调）对 Python 线程的 `ring_pop`
   不可见，导致读管线在看似空的环上无限自旋。通过添加 `__atomic` 获取/释放
   屏障修复。

4. **FSM 中缺少 ACL 上下文（V3）**：Reactor 线程的 FSM 调用 `aclrtMemcpy`
   而未先通过 `aclrtSetDevice`/`aclrtSetCurrentContext` 绑定 ACL 上下文。
   调用静默失败（返回错误），FSM 无限重试。通过在 `write_fsm_tick` 开头调用
   `ensure_acl_context` 修复。

### 3.2 C 层批次性能剖析

为将纯数据路径延迟与 Python 开销分离，我们在 C 层 FSM 中添加了批次级时间戳。
在 `initiate_write_fsm` 和 `initiate_read_fsm` 中，`ts_batch_start` 在首次
DMA 提交开始时记录。在 FSM 完成路径（`completed_count` 达到 `num_tasks` 时），
记录 `ts_batch_end` 并将差值存入 `ctx->last_write_io_us` 或
`ctx->last_read_io_us`。公共 API（`npu_nvme_get_last_io_us`）将此值暴露给
Python。

剖析数据通过两种方式输出：
- **批次级计时**：通过 `npu_nvme_get_last_io_us()` 返回值直接传给 Python 调用
  方，由 Python 决定如何展示（打印到终端、写入 JSON 等）。
- **逐块明细**：通过 `enable_profiling=True` 启用，写入
  `{profiling_dir}/time_write.csv` 和 `time_read.csv` 文件，包含每个块的
  NPU DMA 时间、SPDK NVMe 时间和端到端总时间。

对 1 GB 主机写入（256 块 × 4 MiB）的测试表明，C 层延迟为 259.2 ms
（4,143 MB/s），而 Python 层测量（包含 numpy 分配和 ctypes 编组）为
260.5 ms（4,122 MB/s）。**Python 开销仅 0.5%**，确认命令卸载架构引入的
软件开销可忽略不计——数据路径由 DMA 和 NVMe 传输时间主导。

## 4. 评估

### 4.1 实验环境

| 组件          | 规格                                       |
|---------------|--------------------------------------------|
| NPU           | 昇腾 910B，64 GB HBM                        |
| NVMe SSD      | 3.84 TB 三星 PM9A3（PCIe Gen4 ×4）          |
| CPU           | ARM64（鲲鹏 920），96 核                     |
| 操作系统      | openEuler 22.03 LTS（Linux 5.10）           |
| SPDK          | v26.01-pre（DPDK 25.07）                    |
| 测试模型      | GPT-2 XL，3.28 GB FP16 参数                  |

### 4.2 带宽

使用 4 MiB 块和管线深度 8 测量的顺序写入带宽：

| 指标               | V2（忙等待）   | V6（异步 FSM）  |
|--------------------|---------------|-----------------|
| 最佳单次           | 4,121 MB/s    | 4,432 MB/s      |
| 三次平均           | 3,707 MB/s    | 4,300+ MB/s     |
| C 层（1 GB 主机写）| 不适用         | 4,143 MB/s      |
| Python 开销        | 不适用         | 0.5%            |
| 管线深度扫描        | 全部通过       | 全部通过         |
| 块大小扫描          | 最佳 4 MB     | 最佳 4 MB       |

异步 FSM 未引入带宽退化；轻微提升归因于锁竞争减少。C 层剖析确认 Python 编组
开销可忽略不计（0.5%）。

### 4.3 Reactor CPU 占用

稳态（无活跃 I/O）下，reactor 线程的 CPU 使用主要由 `spdk_thread_poll`
迭代之间的 100 μs 睡眠决定。四个注册轮询器在空闲时均立即返回，实测 CPU
< 1%——远低于 5% 目标。

### 4.4 正确性

- **数据完整性**：1 MB 主机写入/回读产生逐位精确匹配
- **多次初始化/清理循环**：raw_bw 测试中 15+ 次循环，全部通过
- **冒烟测试**：初始化 → 写入 → 读取 → 验证 → 清理，无错误
- **锁移除验证**：`grep -n 'io_lock' src/npu_nvme.c` 返回零匹配；
  `state_lock` 引用限于 32 处，全部在监听器状态保护中

## 5. 相关工作

SPDK Reactor 模型广泛用于存储目标（NVMe-oF、vhost），但此前未被应用于
NPU-to-NVMe 检查点控制平面。CheckFreq [3] 使用至多一个飞行中写入的两阶段
管线，PCcheck [4] 将其扩展为 N 槽位并发管线。两者都将控制平面与数据路径紧密
耦合。我们的工作证明通过无锁环形队列的命令卸载在保持吞吐量的同时消除了线程
安全隐患。

## 6. 结论

本文提出了面向 NPU-to-NVMe 直接检查点的单线程事件驱动控制平面。通过将所有
I/O 执行整合到单一 SPDK reactor 线程上，并通过无锁 MPSC 环形队列传递请求，
我们消除了递归 `io_lock`，解决了 `load()` 死锁，并将 CPU 开销降至可忽略
水平。五版本重构方法论确保每个变更独立可测、可提交。在昇腾 910B 上的实现
达到 4,432 MB/s 写入带宽，与原始多线程性能持平，但无同步风险。

## 参考文献

[1] NPU-NVMe 直接检查点系统。`npu-nvme` 仓库。
[2] SPDK: Storage Performance Development Kit。https://spdk.io
[3] CheckFreq: Frequent, Fine-Grained DNN Checkpointing。FAST 2021。
[4] PCcheck: Persistent Checkpointing for Large-Scale DNN Training。ASPLOS 2025。

## 一、I1: 基于 SPDK 的用户态零拷贝 I/O 通路

### 1.1 核心问题

传统深度学习框架的检查点持久化路径为：**NPU HBM → Host DRAM（`aclrtMemcpy D2H`）→ 内核文件系统（`write()` syscall）→ NVMe 驱动 → SSD**。这条路径存在三个痛点：

| 痛点 | 传统路径 | 后果 |
|------|------|------|
| **多次数据搬运** | HBM→CPU→Page Cache→NVMe | 3 次拷贝，浪费带宽 |
| **内核态开销** | `write()` → VFS → block layer → NVMe driver | 系统调用 + 中断 + 上下文切换 |
| **CPU 占用** | 内核 I/O 线程竞争训练线程的 CPU | 干扰训练 |

**I1 的核心思路**：使用 SPDK（Storage Performance Development Kit）将 NVMe 驱动从内核态移至用户态，结合 NPU 的 DMA 引擎，实现 **HBM → SPDK DMA buffer → NVMe SSD** 的一次拷贝直通路径，全程不经 CPU 和内核。

### 1.2 模块设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                     I1 系统架构                                      │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │ M1: SPDK 环境层      │    │ M3: DMA 缓冲池                    │   │
│  │                      │    │                                  │   │
│  │ · DPDK 大页内存      │    │ · SPSC Ring (pipeline_depth)     │   │
│  │ · spdk_env_init      │    │ · spdk_zmalloc (2MB对齐)         │   │
│  │ · NUMA 感知          │◄───│ · spdk_vtophys (物理地址映射)     │   │
│  │ · 大页自动扩展       │    │ · 槽位状态机 (IDLE→COPY→WRITE)   │   │
│  │   ensure_hugepages() │    └──────────────────────────────────┘   │
│  │ · 多进程 SHM ID      │                                          │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│  ┌──────────▼───────────┐    ┌──────────────────────────────────┐   │
│  │ M2: NVMe 传输层      │    │ M4: 异步传输流水线                │   │
│  │                      │    │                                  │   │
│  │ · PCIe probe/attach  │    │ 写流水线:                        │   │
│  │ · IO Queue Pair      │    │   NPU→DMA (aclrtMemcpy)          │   │
│  │   (depth=512)        │    │   → SPDK write (ns_cmd_write)    │   │
│  │ · Block/NS 发现      │    │   → 完成回调 + 槽位回收          │   │
│  │ · MDTS 限制检测      │    │                                  │   │
│  │                      │    │ 读流水线:                        │   │
│  └──────────────────────┘    │   SPDK read (ns_cmd_read)         │   │
│                              │   → DMA→NPU (aclrtMemcpyAsync)   │   │
│  ┌──────────────────────┐    │   → Event 等待 + 槽位回收        │   │
│  │ M5: 元数据 I/O       │    │                                  │   │
│  │                      │    │ 反压与异常处理:                   │   │
│  │ · 独立 DMA buffer    │    │   · Ring Buffer 满 → 等待        │   │
│  │   (64MB)             │    │   · 50ms 无进展 → Poke 机制      │   │
│  │ · sync_meta_io       │    │   · 3s 死锁 → Watchdog 诊断      │   │
│  │ · Superblock (4KB)   │    └──────────────────────────────────┘   │
│  │ · A/B 双槽元数据日志 │                                          │
│  └──────────────────────┘                                          │
│                                                                     │
│  ┌──────────────────────┐                                          │
│  │ M6: Python 绑定层    │                                          │
│  │                      │                                          │
│  │ · ctypes 接口封装    │                                          │
│  │ · build_chunks()     │                                          │
│  │   参数→4K对齐块切分   │                                          │
│  │ · rebuild_chunks()   │                                          │
│  │   元数据→块列表重建   │                                          │
│  │ · Profiling CSV 导出 │                                          │
│  └──────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### M1: SPDK 环境层

**职责**：初始化 SPDK/DPDK 运行时环境，管理大页内存

**关键实现**：
- `spdk_env_init()`: 初始化 DPDK EAL，配置大页内存、NUMA 亲和
- `ensure_hugepages()` (`src/npu_nvme.c:68`): 自动检测并扩展大页池。NPU 驱动预留了全部启动时大页（8544 页 = 17GB），DPDK 需要额外空闲大页。自动追加 512 页（1GB）
- 多进程支持: 通过 `SPDK_SHM_ID` 环境变量，多卡 Rank 共享同一 SPDK 主进程的 NVMe 硬件队列

**目标效果**：
- SPDK 初始化成功率 > 99%（排除人为配置错误）
- 多 Rank 共享 NVMe 无竞态

#### M2: NVMe 传输层

**职责**：发现、挂载 NVMe 设备，管理 I/O 队列

**关键实现**：
- `probe_cb()`: PCIe 地址精确匹配，防止误接系统盘
- `attach_cb()`: 发现活跃 Namespace，获取 block_size 和 total_blocks
- IO Queue Pair: depth=512，支持深度并发
- MDTS 检测: 从 controller data 计算单次最大传输，硬上限 4MB

**目标效果**：
- 设备发现延迟 < 100ms
- 支持任意 PCIe Gen3/Gen4 NVMe SSD

#### M3: DMA 缓冲池

**职责**：管理 pipeline_depth 个 SPDK DMA buffer 的生命周期

**关键实现**：
- SPSC Ring Buffer（`ring_t`, `src/npu_nvme.c:126-158`）：无锁单生产者单消费者
- `spdk_zmalloc(size, 2MB, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA)`: 分配 2MB 对齐的物理连续大页
- `spdk_vtophys()`: 获取物理地址（SPDK 直接使用物理地址写 NVMe）
- 状态机: `IDLE → NPU_COPYING → NPU_DONE → SPDK_WRITING → DONE`

**目标效果**：
- pipeline_depth=8, chunk_size=4MB → 32MB DMA 池
- 内存利用率 > 95%（实际使用/分配总量）

#### M4: 异步传输流水线

**职责**：驱动 NPU DMA 和 SPDK NVMe 之间的 Zero-Bubble 数据传输

**写流水线** (`process_write_pipeline`, `src/npu_nvme.c:1051`):
```
while not all done:
  Engine 1: Submit NPU→DMA copies (fill Ring Buffer)
  Engine 2: Poll NPU Events → Submit SPDK writes (drain to NVMe)
  Engine 3: Poll SPDK completions → reclaim buffer slots
  No progress > 50ms → Poke: force-sync stalled NPU event
  No progress > 3s → Watchdog: dump state + reset timer
```

**读流水线** (`process_read_pipeline`, `src/npu_nvme.c:1254`):
```
while not all done:
  Engine 1: Submit SPDK reads (fill Ring Buffer)
  Engine 2: Poll SPDK completions → Submit NPU DMA copies
  Engine 3: Poll NPU Events → reclaim buffer slots
```

**目标效果**：
- 写入 BW > 4000 MB/s（pipeline_depth=8, chunk=4MB）
- 读取 BW > 6000 MB/s（NVMe 读比写快）
- Pipeline 利用率 > 80%（槽位同时被占用的时间占比）

#### M5: 元数据 I/O

**职责**：管理裸盘上的 Superblock 和 JSON 元数据日志

**关键实现**：
- 独立 64MB DMA buffer（`meta_dma_buf`），避免与数据通道竞争
- `sync_meta_io()`: 同步读/写，阻塞等待 SPDK 完成
- Superblock (4KB, offset 0): magic + active slot + stack bounds
- A/B 双槽 (offset 4KB/404KB): 轮流写入，crash-safe

**目标效果**：
- 元数据读写延迟 < 2ms（64MB buffer 内）
- A/B 双槽确保写一半崩溃时至少有一个完整副本

#### M6: Python 绑定层

**职责**：为 Python 训练脚本提供 ctypes 接口

**关键实现**：
- `direct_checkpoint.py:142-165`: `build_chunks()` — 按 4K 对齐切分参数
- `direct_checkpoint.py:167-213`: `rebuild_chunks_from_meta()` — 从元数据重建块列表
- `direct_checkpoint.py:836-989`: `save()` — FULL checkpoint 端到端流程（prep → layout → SPDK → meta）

**目标效果**：
- Python 层开销 < 50ms（prep + layout，不含 SPDK 写盘）

### 1.3 Motivation 论证结构（论文 §3.2 建议）

```
§3.2.1 传统检查点 I/O 路径分析
  ① NPU→CPU→Kernel→NVMe 的三次拷贝路径
  ② 内核文件系统开销（系统调用、中断、页缓存）
  ③ CPU 占用对训练线程的干扰

§3.2.2 SPDK 用户态 NVMe 驱动适配
  ① DPDK 大页内存管理
  ② 用户态 NVMe 命令提交（无系统调用）
  ③ HBM→DMA buffer→NVMe 直通路径

§3.2.3 双引擎异步流水线设计
  ① NPU DMA 与 SPDK I/O 的并行调度
  ② SPSC Ring Buffer 的零锁设计
  ③ 反压机制与异常恢复（Poke + Watchdog）

§3.2.4 裸盘布局与元数据管理
  ① Superblock + A/B 双槽设计
  ② 检查点链（FULL + delta slots）
  ③ Crash-safe 保证
```

### 1.4 I1 Evaluation 实验设计

| 编号 | 实验 | 验证什么 | 状态 |
|:---:|------|------|:---:|
| **E1.1** | Raw SPDK 读写带宽 | HBM↔NVMe 直通的理论上限 | ⚠️ 有历史数据，需重采 |
| **E1.2** | Pipeline Depth 扩展性 | 并发度对带宽的影响 | 🔲 |
| **E1.3** | Chunk Size 扩展性 | 块大小对带宽/延迟的影响 | 🔲 |
| **E1.4** | 内核 NVMe 对比 | SPDK 相比内核路径的带宽/CPU 优势 | 🔲 |
| **E1.5** | 多 Rank 扩展性 | 多卡共享 NVMe 的带宽分配 | 🔲 |
| **E1.6** | FULL Checkpoint E2E | GPT-2 XL 完整保存/加载性能 | ⚠️ 有历史数据，需重采 |
| **E1.7** | 数据完整性 | 写入→读回→逐字节校验 | ⚠️ C 层 test 已有 |

---

### E1.1: Raw SPDK 读写带宽

**问题**：SPDK 用户态 NVMe 直通路径的读写带宽上限是多少？

**方法**：
```
配置: pipeline_depth=8, chunk_size=4MB
写: npu_nvme_write_batch() — HBM→DMA→NVMe
读: npu_nvme_read_batch() — NVMe→DMA→HBM
数据大小: 3.1GB（GPT-2 XL 全量，覆盖多 chunk 场景）

测量:
  - Wall-clock 总时间
  - 有效 BW (total_bytes / wall_time)
  - Pipeline DMA BW (所有 chunk 的 DMA 时间求和 / chunk 数)
  - 微结构: npu_copy_us, spdk_nvme_us, total_e2e_us (profiling CSV)

历史参考: 
  write BW = 4412 MB/s, pipeline DMA BW = 52,000 MB/s (Step 1c)
  read BW 待采集
```

**论文数据需求**：
- 表格：读写 BW (MB/s)，含 wall BW 和 pipeline DMA BW
- profiling 时间分解（npu_copy vs spdk_nvme 占比）

---

### E1.2: Pipeline Depth 扩展性

**问题**：pipeline_depth 如何影响带宽？最优值是多少？

**方法**：
```
固定 chunk_size=4MB, 扫描 pipeline_depth: 1, 2, 4, 8, 16
各写 3.1GB, 测量 BW
```

**预期**：
- depth=1: 最低 BW（NPU DMA 和 SPDK 写完全串行）
- depth=4-8: BW 快速上升（硬件流水线充分填充）
- depth=16: BW 趋于平稳或略降（Ring Buffer 管理开销）

**论文数据需求**：
- 曲线图：pipeline_depth vs BW
- 标注最优 depth

---

### E1.3: Chunk Size 扩展性

**问题**：chunk_size 如何影响带宽？

**方法**：
```
固定 pipeline_depth=8, 扫描 chunk_size: 64KB, 256KB, 1MB, 4MB
各写 3.1GB, 测量 BW
```

**预期**：
- 小块：BW 低（NVMe 命令提交频率高，每个 chunk 一次命令）
- 大块：BW 高（单次命令传输更多数据），但粒度粗

**论文数据需求**：
- 曲线图：chunk_size vs BW

---

### E1.4: 内核 NVMe 对比

**问题**：SPDK 相比传统内核 NVMe 路径优势有多大？

**方法**：
```
条件 A (SPDK):  npu_nvme_write_batch() (HBM→NVMe 直通)
条件 B (内核):    aclrtMemcpy(HBM→Host) + os.write() (标准路径)

相同数据 (3.1GB), 相同硬件, 测量:
  - 端到端延迟
  - 有效 BW
  - CPU 利用率 (通过 top/mpstat 采集)
```

**论文数据需求**：
- 对比柱状图：SPDK vs 内核的 BW + CPU 利用率

---

### E1.5: 多 Rank 扩展性

**问题**：多张 NPU 卡共享同一块 NVMe SSD 时，带宽如何分配？

**方法**：
```
配置: pipeline_depth=8, chunk=4MB, SPDK_SHM_ID 区分 Rank
条件: 1 Rank / 2 Rank / 4 Rank 同时写 (各写 3.1GB)

测量:
  - 单 Rank BW
  - 聚合 BW
  - 带宽争抢程度 (单 Rank BW / 总 Rank 数 vs 实际单 Rank BW)
```

---

### E1.6: FULL Checkpoint E2E 性能

**问题**：GPT-2 XL 全量检查点的端到端性能？

**方法**：
```
DirectCheckpoint.save() 完整流程分析:
  Phase 1 (T_Prep): 参数遍历、指针获取
  Phase 2 (T_Layout): 物理布局计算、4K 对齐、ctypes 数组组装
  Phase 3 (T_SPDK): HBM→NVMe DMA 写盘
  Phase 4 (T_Meta): Superblock + JSON 元数据落盘

历史参考: 2.90GB in 674ms → 4412 MB/s (Step 1c)
```

**论文数据需求**：
- 四阶段时间分解饼图
- 与 `torch.save()` 的延迟对比

---

### E1.7: 数据完整性

**问题**：SPDK 路径写盘后读回的数据是否与原始数据完全一致？

**方法**：
```
写: 填充已知 pattern 的 NPU buffer → write_batch → NVMe
读: read_batch → 逐字节比较
Mode: 1 (0x11), 2 (0x22), 3 (0x33) 不同区域填不同 pattern
```

**已有**：`src/test_npu_nvme.c` 中实现了此逻辑，需在标准配置下重跑

**论文数据需求**：
- "3.1GB 数据逐字节校验通过，0 错误"

---


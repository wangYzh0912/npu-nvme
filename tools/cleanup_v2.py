#!/usr/bin/env python3
"""Remove dev-phase labels, emotional markers, and section numbers."""
with open('src/npu_nvme.c', 'r', encoding='utf-8') as f:
    c = f.read()

reps = [
    # --- section headers: remove numbering ---
    ('Section 1 - SPDK Probe', 'SPDK Probe'),
    ('Section 2 - Init (SPDK', 'Init (SPDK'),
    ('Section 3 - Cleanup', 'Cleanup'),
    ('Section 5 - Public API', 'Public API'),
    ('Section 6 - Synchronous Metadata', 'Synchronous Metadata'),
    ('Section 7 - Delta Frame I/O', 'Delta Frame I/O'),

    # --- file header: remove I1/I2/I3 ---
    (' *   I1  SPDK user-space NVMe driver', ' *   - SPDK user-space NVMe driver'),
    (' *   I2  FaF device-memory-polling listener', ' *   - device-memory-polling listener'),
    (' *   I3  Delta-frame I/O', ' *   - delta-frame I/O'),

    # --- Phase/E development labels ---
    ('[Phase 5 E11] Delta', 'delta'),
    ('P2-1: WaitProbe', 'DEPRECATED: WaitProbe'),
    ('P2-1 note:', ''),

    # --- emotional markers ---
    ('【非常关键】：', ''),
    ('【修复 1】：', ''),
    ('【新增】：', ''),
    ('【核心修改】：', ''),
    ('【核心修复】：', ''),
    ('!!', ''),
    ('！！', ''),

    # --- emotional phrases to standard English ---
    ('必须在当前线程重新绑定 NPU Device，否则后续 Destroy 会直接引发段错误',
     'Re-bind NPU device on the current thread before releasing ACL resources'),
    ('Host 内存直接 memcpy 到锁页大页内存，瞬间完成，跳过 NPU 异步流',
     'Host buffer: plain memcpy to hugepage-backed DMA buffer'),
    ('抛弃 Async 和 Event，直接使用同步拷贝',
     'Use synchronous aclrtMemcpy; sub-millisecond blocking is harmless in background thread'),
    ('因为我们在后台线程，阻塞这零点几毫秒对外界毫无影响',
     ''),
    ('致命读错误', 'fatal SPDK read error'),
    ('C 层自动根据底层真实的 block_size 换算 LBA 和物理块数',
     'Convert byte offset to LBA using the hardware block size'),
    ('Python 彻底解脱', ''),
    ('如果队列拒收，直接返回报错，绝对不能进入下面的 while 循环',
     'If the submission queue rejects the command, fail immediately'),
    ('导出极度硬核的微观 Profiling 数据', 'export per-chunk profiling data to CSV'),
    ('基础防御性编程', 'argument validation'),
    ('0 代表一切顺利', 'success'),
    ('清理状态机内存，完美退出', 'free task state machine'),
    ('后台监听与 SPDK 轮询线程 (FaF: step_counter polling)',
     'Background listener thread -- polls a device-side step counter via'),
    ('看门狗计时器', 'stall-detection timer'),
    ('状态卡死监控与主动唤醒逻辑', 'stall detection and recovery'),
    ('核心上下文', 'NPU-NVMe runtime context'),
    ('元数据专属的简化回调', 'simplified SPDK completion callback for metadata I/O'),
    ('用于 probe 时过滤设备', 'PCIe BDF address for probe filtering'),
    ('保存命名空间指针', 'active namespace'),
    ('Ring Buffer 深度', 'DMA pipeline depth'),
    ('DRAM 大页内存池', 'DMA buffer pool (hugepage-backed)'),
    ('空闲槽位队列', 'free-slot SPSC ring'),
    ('[全异步核心组件]', 'asynchronous I/O core'),
    ('NPU 专属异步数据流', 'ACL stream for NPU-DMA copies'),
    ('与 Ring Buffer 槽位一一对应的硬件事件', 'one hardware event per ring-buffer slot'),
    ('专用于元数据读写的大页内存', 'dedicated DMA buffer for metadata I/O'),
    ('[新增] 探针后台持久化任务表', 'registered task table for background persistence'),
    ('[新增] 后台监听线程控制', 'listener thread control'),
    ('[新增] NPU 侧探针 flag 设备指针', 'probe-flag device pointer'),
    ('[新增] FaF step_counter polling', 'step-counter polling'),
    ('Delta ring 起始字节偏移', 'byte offset of delta ring on NVMe'),
    ('每槽位字节数', 'bytes per delta slot'),
    ('槽位总数', 'total number of delta slots'),
    ('最后写入的槽位 (用于恢复遍历)', 'index of last committed slot'),
    ('初始状态：空闲', 'initial state: idle'),
    ('尚未分配 Ring Buffer 槽位', 'no ring-buffer slot assigned yet'),
    ('Ring Buffer 满', 'ring buffer full'),
    ('NPU 显存：发起真正的异步 DMA 搬运', 'NPU device memory: launch DMA copy'),
    ('瞬间拷贝完成，直接进入下一步', 'synchronous copy complete; proceed to SPDK'),
    ('归还槽位', 'return slot to free pool'),
    ('引擎 1：发射异步任务', 'engine 1: submit DMA / SPDK commands'),
    ('引擎 2：监控 NPU Event 并下发 SPDK', 'engine 2: poll NPU events, submit ready chunks'),
    ('引擎 3：轮询 SPDK 完成队列', 'engine 3: reap SPDK completion queue'),
    ('有进展则重置时间', 'reset stall timer on progress'),
    ('打印表头', 'CSV header'),
    ('计算 NPU 纯异步搬运耗时', 'NPU copy time'),
    ('计算 SPDK 队列等待 + NVMe 物理落盘耗时', 'SPDK queue + NVMe write time'),
    ('单个 Chunk 从开始下发到彻底落盘的端到端耗时', 'end-to-end chunk latency'),
    ('状态翻转：硬盘数据已就绪', 'data is now in the DMA buffer'),
    ('引擎 1: 发射 NVMe 读取请求', 'engine 1: submit NVMe read commands'),
    ('引擎 2: DRAM -> NPU', 'engine 2: DMA buffer to NPU'),
    ('注意这里的字段顺序与 write 是反的', 'note: field order differs from write_batch'),
    ('1. 先算硬盘拉取时间 (SSD -> DRAM)', '1. NVMe read time (SSD to DMA buffer)'),
    ('2. 再算 NPU 异步搬运时间 (DRAM -> NPU)', '2. DMA to NPU copy time'),
    ('3. 总耗时', '3. end-to-end latency'),
    ('逻辑与 write_batch 类似，生成 time_read.csv', ''),
    ('如果重复注册，清理旧的内存', 'free previous registration if present'),
    ('使用原有的辅助函数，预分配并填充好物理内存布局',
     'populate the I/O task table from raw pointer arrays'),
    ('找到第一个活跃的命名空间就退出', 'use the first active namespace'),
    ('1. 分配上下文', '1. allocate context'),
    ('2. 初始化 SPDK 环境 (只在首次调用时有效', '2. initialise SPDK environment (once per process'),
    ('3. 探测并挂载 NVMe 硬盘', '3. probe and attach NVMe device'),
    ('4. 创建 SPDK I/O Queue Pair', '4. allocate SPDK I/O queue pair'),
    ('为了应对极致的异步并发，把队列深度开大', 'deep queue for high-throughput asynchronous I/O'),
    ('5. 初始化 NPU 环境 (绑定设备、创建异步流)',
     '5. initialise NPU environment (bind device, create ACL stream)'),
    ('6. 分配 Ring Buffer 及关联的 NPU Hardware Event',
     '6. allocate DMA buffer pool and associated NPU events'),
    ('使用 SPDK 申请物理连续、锁页的大页内存，彻底消除 TLB Miss',
     'allocate physically contiguous, page-locked DMA memory via SPDK'),
    ('创建 NPU 事件', 'create NPU event for this slot'),
    ('默认放行一次以避免训练首步被意外阻塞',
     'initialise probe_flags to unblock the first training step'),
    ('后续由 SPDK 写入负责实际放行', ''),
    ('启动后台监听线程', 'launch the background listener thread'),
    ('先停止后台线程，防止 SPDK 资源在使用时被销毁',
     'stop the background thread before tearing down SPDK resources'),
    ('唤醒等待中的后台线程', 'wake the listener thread if it is sleeping'),
    ('安全销毁事件', 'destroy NPU events'),
    ('安全销毁流', 'destroy ACL stream'),
    ('释放内存与 SPDK 资源', 'release memory and SPDK resources'),
    ('1. 初始化全异步状态机任务队列 (调用 Phase 2 的函数)',
     '1. build the I/O task state machine'),
    ('2. 启动 Zero-Bubble 双轨异步流水线 (调用 Phase 3 的函数)',
     '2. run the dual-polling pipeline to completion'),
    ('这个函数会接管 CPU，不休眠地驱动 NPU 和 NVMe，直到所有任务落盘完毕',
     ''),

    # --- cleanup remaining ! in comments ---
    (' // 如果绑定失败（例如框架已经强行接管），为了防止暴毙，宁可泄漏也不要强行 Destroy',
     ' // If ACL device binding fails (e.g. framework has claimed the device),'),
]

miss = 0
for old, new in reps:
    if old in c:
        c = c.replace(old, new)
    else:
        miss += 1
        # silently skip -- many patterns are context-dependent

with open('src/npu_nvme.c', 'w', encoding='utf-8') as f:
    f.write(c)
print(f'Done. {len(reps)-miss}/{len(reps)} patterns applied, {miss} skipped.')

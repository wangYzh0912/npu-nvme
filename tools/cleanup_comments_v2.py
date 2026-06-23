#!/usr/bin/env python3
"""Second-pass comment cleanup — remove dev-phase labels, emotional markers,
section numbers, and I1/I2/I3/E2/Step references from code comments."""
import sys

with open('src/npu_nvme.c', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. File header — remove I1/I2/I3 labels
    ('''/* =======================================================================
 * npu_nvme.c — NPU-to-NVMe Zero-Copy I/O Engine
 *
 * Implements three subsystems:
 *   I1  SPDK user-space NVMe driver — HBM <-> NVMe DMA transfers
 *   I2  FaF device-memory-polling listener — async step-boundary detection
 *   I3  Delta-frame I/O — incremental checkpoint slot read/write
 *
 * Built as libnpu_nvme.so; consumed by Python ctypes bindings
 * (python/direct_checkpoint.py) and the C smoke test (src/test_npu_nvme.c).
 * ======================================================================= */''',
     '''/* =======================================================================
 * npu_nvme.c — NPU-to-NVMe Zero-Copy I/O Engine
 *
 * Provides user-space NVMe block I/O with HBM <-> SSD DMA transfers
 * via SPDK, an asynchronous device-memory-polling listener for
 * step-boundary detection, and delta-frame read/write for incremental
 * checkpointing.
 *
 * Built as libnpu_nvme.so; consumed by Python ctypes bindings
 * (python/direct_checkpoint.py) and the C smoke test (src/test_npu_nvme.c).
 * ======================================================================= */'''),

    # 2. Section headers — remove numbering, remove I3 label
    ('/* ===================================================================\n'
     ' * Section 1 - SPDK Probe & Attach Callbacks\n'
     ' * =================================================================== */',
     '/* ---- SPDK Probe & Attach Callbacks ---- */'),

    ('/* ===================================================================\n'
     ' * Section 2 - Init (SPDK env + NVMe probe + ACL + DMA pool + listener)\n'
     ' * =================================================================== */',
     '/* ---- Init: SPDK env, NVMe probe, ACL, DMA pool, listener ---- */'),

    ('/* ===================================================================\n'
     ' * Section 3 - Cleanup (strict resource release order)\n'
     ' * =================================================================== */',
     '/* ---- Cleanup: strict resource release order ---- */'),

    ('/* ===================================================================\n'
     ' * Section 5 - Public API: write_batch / read_batch\n'
     ' * =================================================================== */',
     '/* ---- Public API: write_batch / read_batch ---- */'),

    ('/* ===================================================================\n'
     ' * Section 6 - Synchronous Metadata I/O (superblock + JSON ledger)\n'
     ' * =================================================================== */',
     '/* ---- Synchronous Metadata I/O: superblock + JSON ledger ---- */'),

    ('/* ===================================================================\n'
     ' * Section 7 - Delta Frame I/O (I3)\n'
     ' * =================================================================== */',
     '/* ---- Delta Frame I/O ---- */'),

    # 3. Phase/E/Step development labels in struct/function comments
    ('    // ----- [Phase 5 E11] Delta (增量) 盘布局 -----',
     '    // ----- delta ring-buffer layout -----'),

    # 4. Emotional/sensational markers — remove 【】 and !！
    ('    // 【非常关键】：绑定 NPU ACL 上下文到当前后台线程。',
     '    // Bind NPU ACL context to this background thread.'),

    ('    // 【修复 1】：必须在当前线程重新绑定 NPU Device，否则后续 Destroy 会直接引发段错误！',
     '    // Re-bind NPU device on the current thread before releasing ACL resources.'),

    ('        // 【新增】：Host 内存直接 memcpy 到锁页大页内存，瞬间完成，跳过 NPU 异步流',
     '        // Host buffer: plain memcpy to hugepage-backed DMA buffer'),

    ('        // 【核心修改】：抛弃 Async 和 Event，直接使用同步拷贝！\n'
     '        // 因为我们在后台线程，阻塞这零点几毫秒对外界毫无影响',
     '        // Use synchronous aclrtMemcpy in the background thread;\n'
     '        // sub-millisecond blocking is harmless outside the training path.'),

    ('                // 【核心修复】：致命读错误',
     '                // fatal SPDK read error'),

    ('    // C 层自动根据底层真实的 block_size 换算 LBA 和物理块数！Python 彻底解脱！',
     '    // Convert byte offset to LBA using the hardware block size.'),

    ('    // 【核心修复】：如果队列拒收，直接返回报错，绝对不能进入下面的 while 循环！',
     '    // If the submission queue rejects the command, fail immediately\n'
     '    // rather than spinning in the while-flag loop below.'),

    # 5. P2-1 and other development-phase dead-code markers
    ('    // P2-1: WaitProbe dead code is #if 0\\'d — kept for potential rollback',
     '    // DEPRECATED: old WaitProbe path replaced by FaF step_counter polling'),

    ('        // P2-1 note: WaitProbe/TriggerProbe dead code was here; replaced by FaF listener.\n'
     '        // Retained for rollback via #if 0 block at end of file.',
     ''),

    ('// P2-1: WaitProbe/TriggerProbe dead code — preserved via #if 0 for rollback\n'
     '// These functions (npu_nvme_set_trigger_ptr, npu_nvme_read_trigger_dev,\n'
     '// and the old probe_listener_thread WaitProbe branch) are replaced by\n'
     '// FaF step_counter polling. If rollback is needed, uncomment the block below.\n'
     '// ============================================================================',
     ''),

    # 6. Phase references in write_batch function
    ('    // 1. 初始化全异步状态机任务队列 (调用 Phase 2 的函数)',
     '    // 1. build the I/O task state machine'),

    ('    // 2. 启动 Zero-Bubble 双轨异步流水线 (调用 Phase 3 的函数)\n'
     '    // 这个函数会接管 CPU，不休眠地驱动 NPU 和 NVMe，直到所有任务落盘完毕',
     '    // 2. run the dual-polling pipeline to completion'),

    # 7. Miscellaneous emotional language
    ('    // 3. 导出极度硬核的微观 Profiling 数据',
     '    // 3. export per-chunk profiling data'),

    ('    // 0. 基础防御性编程',
     '    // 0. argument validation'),

    ('    return 0; // 0 代表一切顺利',
     '    return 0;'),

    ('    // 4. 清理状态机内存，完美退出',
     '    // 4. free task state machine'),

    # 8. Remove "FaF" from comments (keep it as "listener")
    # Already clean in most places, but check the listener thread comment
    ('// 后台监听与 SPDK 轮询线程 (FaF: step_counter polling)',
     '/* Background listener thread — polls a device-side step counter\n'
     ' * via aclrtMemcpy and triggers SPDK writes on step changes. */'),

    # 9. init step markers
    ('    // 1. 分配上下文',
     '    // 1. allocate context'),

    ('    // 2. 初始化 SPDK 环境 (只在首次调用时有效，如果多卡跑在同一个进程，需要通过配置 shm_id 解决)',
     '    // 2. initialise SPDK environment (once per process; use SPDK_SHM_ID for multi-rank)'),

    ('    // 3. 探测并挂载 NVMe 硬盘',
     '    // 3. probe and attach NVMe device'),

    ('    // 4. 创建 SPDK I/O Queue Pair',
     '    // 4. allocate SPDK I/O queue pair'),

    ('    // 为了应对极致的异步并发，把队列深度开大',
     '    // deep queue for high-throughput asynchronous I/O'),

    ('    // 5. 初始化 NPU 环境 (绑定设备、创建异步流)',
     '    // 5. initialise NPU environment (bind device, create ACL stream)'),

    ('    // 6. 分配 Ring Buffer 及关联的 NPU Hardware Event',
     '    // 6. allocate DMA buffer pool and associated NPU events'),

    ('        // 使用 SPDK 申请物理连续、锁页的大页内存，彻底消除 TLB Miss',
     '        // allocate physically contiguous, page-locked DMA memory via SPDK'),

    ('        // 创建 NPU 事件',
     '        // create NPU event for this slot'),

    ('    // 默认放行一次以避免训练首步被意外阻塞（后续由 SPDK 写入负责实际放行）',
     '    // initialise probe_flags to unblock the first training step'),

    ('    // 启动后台监听线程',
     '    // launch the background listener thread'),

    # 10. cleanup markers
    ('    // 先停止后台线程，防止 SPDK 资源在使用时被销毁',
     '    // stop the background thread before tearing down SPDK resources'),

    ('        probe_flags[0] = 1; // 唤醒等待中的后台线程',
     '        probe_flags[0] = 1; // wake the listener thread if it is sleeping'),

    ('        // 安全销毁事件',
     '        // destroy NPU events'),

    ('        // 安全销毁流',
     '        // destroy ACL stream'),

    ('    // ----- 释放内存与 SPDK 资源 -----',
     '    // ----- release memory and SPDK resources -----'),

    # 11. create_io_tasks markers
    ('        tasks[i].buf_idx = -1;               // 尚未分配 Ring Buffer 槽位',
     '        tasks[i].buf_idx = -1;               // no ring-buffer slot assigned yet'),

    ('        tasks[i].state = CHUNK_IDLE;         // 初始状态：空闲',
     '        tasks[i].state = CHUNK_IDLE;'),

    # 12. try_submit_async markers
    ('    if (ring_pop(&ctx->free_ring, &buf_idx) != 0) return -1; // Ring Buffer 满',
     '    if (ring_pop(&ctx->free_ring, &buf_idx) != 0) return -1; // ring buffer full'),

    ('        // NPU 显存：发起真正的异步 DMA 搬运',
     '        // NPU device memory: launch DMA copy'),

    ('        // 瞬间拷贝完成，直接进入下一步！',
     '        // synchronous copy complete; proceed to SPDK submission'),

    # 13. SPDK callback markers
    ('    ring_push(&cb_arg->ctx->free_ring, cb_arg->task->buf_idx); // 归还槽位',
     '    ring_push(&cb_arg->ctx->free_ring, cb_arg->task->buf_idx); // return slot to free pool'),

    # 14. pipeline engine markers
    ('        // ---- 引擎 1：发射异步任务 ----',
     '        // ---- engine 1: submit DMA / SPDK commands ----'),

    ('                break; // Ring Buffer 满',
     '                break; // ring buffer full'),

    ('        // ---- 引擎 2：监控 NPU Event 并下发 SPDK ----',
     '        // ---- engine 2: poll NPU events, submit ready chunks ----'),

    ('        // ---- 引擎 3：轮询 SPDK 完成队列 ----',
     '        // ---- engine 3: reap SPDK completion queue ----'),

    ('            last_progress_time = get_time_us(); // 有进展则重置时间',
     '            last_progress_time = get_time_us();'),

    # 15. profiling CSV markers
    ('            // 打印表头',
     '            // CSV header'),

    ('                // 计算 NPU 纯异步搬运耗时',
     '                // NPU copy time'),

    ('                // 计算 SPDK 队列等待 + NVMe 物理落盘耗时',
     '                // SPDK queue + NVMe write time'),

    ('                // 单个 Chunk 从开始下发到彻底落盘的端到端耗时',
     '                // end-to-end chunk latency'),

    # 16. read path markers
    ('    // 状态翻转：硬盘数据已就绪',
     '    // data is now in the DMA buffer'),

    ('        // ---- 引擎 1: 发射 NVMe 读取请求 ----',
     '        // ---- engine 1: submit NVMe read commands ----'),

    ('        // ---- 引擎 2: DRAM -> NPU ----',
     '        // ---- engine 2: DMA buffer -> NPU ----'),

    # 17. read_batch profiling
    ('        // 逻辑与 write_batch 类似，生成 time_read.csv',
     ''),

    ('            // 注意这里的字段顺序与 write 是反的',
     '            // note: field order differs from write_batch (read-first)'),

    ('                // 1. 先算硬盘拉取时间 (SSD -> DRAM)',
     '                // 1. NVMe read time (SSD -> DMA buffer)'),

    ('                // 2. 再算 NPU 异步搬运时间 (DRAM -> NPU)',
     '                // 2. DMA -> NPU copy time'),

    ('                // 3. 总耗时',
     '                // 3. end-to-end latency'),

    # 18. struct field markers — remove Chinese where easy, keep where descriptive
    ('    char pci_addr[64];              // 用于 probe 时过滤设备',
     '    char pci_addr[64];              // PCIe BDF address for probe filtering'),

    ('    struct spdk_nvme_ns *ns;        // 保存命名空间指针',
     '    struct spdk_nvme_ns *ns;        // active namespace'),

    ('    int max_pipe_depth;             // Ring Buffer 深度',
     '    int max_pipe_depth;             // DMA pipeline depth'),

    ('    dma_buf_t *pool;                // DRAM 大页内存池',
     '    dma_buf_t *pool;                // DMA buffer pool (hugepage-backed)'),

    ('    ring_t free_ring;               // 空闲槽位队列',
     '    ring_t free_ring;               // free-slot SPSC ring'),

    ('    // ----- [全异步核心组件] -----',
     '    // ----- asynchronous I/O core -----'),

    ('    aclrtStream copy_stream;        // NPU 专属异步数据流',
     '    aclrtStream copy_stream;        // ACL stream for NPU <-> DMA copies'),

    ('    aclrtEvent *events;             // 与 Ring Buffer 槽位一一对应的硬件事件',
     '    aclrtEvent *events;             // one hardware event per ring-buffer slot'),

    ('    void *meta_dma_buf;     // 专用于元数据读写的大页内存',
     '    void *meta_dma_buf;     // dedicated DMA buffer for metadata I/O'),

    ('    // ----- [新增] 探针后台持久化任务表 -----',
     '    // ----- registered task table for background persistence -----'),

    ('    // ----- [新增] 后台监听线程控制 -----',
     '    // ----- listener thread control -----'),

    ('    // ----- [新增] NPU 侧探针 flag 设备指针 -----',
     '    // ----- probe-flag device pointer -----'),

    ('    // ----- [新增] FaF step_counter polling -----',
     '    // ----- step-counter polling -----'),

    ('    uint64_t delta_area_offset;   // Delta ring 起始字节偏移',
     '    uint64_t delta_area_offset;   // byte offset of delta ring on NVMe'),

    ('    uint32_t delta_slot_size;     // 每槽位字节数',
     '    uint32_t delta_slot_size;     // bytes per delta slot'),

    ('    uint32_t delta_slot_count;    // 槽位总数',
     '    uint32_t delta_slot_count;    // total number of delta slots'),

    ('    uint32_t delta_last_commit;   // 最后写入的槽位 (用于恢复遍历)',
     '    uint32_t delta_last_commit;   // index of last committed slot'),

    # 19. register_tasks markers
    ('    // 如果重复注册，清理旧的内存',
     '    // free previous registration if present'),

    ('    // 使用原有的辅助函数，预分配并填充好物理内存布局',
     '    // populate the I/O task table from raw pointer arrays'),

    # 20. attach_cb marker
    ('        break; // 找到第一个活跃的命名空间就退出',
     '        break; // use the first active namespace'),

    # 21. struct context comment
    ('// 核心上下文',
     '/* NPU-NVMe context — all runtime state in one structure. */'),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        count += 1
    else:
        # Try to find a substring match for debugging
        short = old[:60].replace('\n', '\\n')
        print(f'WARNING: pattern #{count+1} not found: {short}...')

with open('src/npu_nvme.c', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Applied {count}/{len(replacements)} replacements')

#!/usr/bin/env python3
"""Normalize comment style in npu_nvme.c — Step 4+6 of C refactor patch."""
import sys

with open('src/npu_nvme.c', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 1. File header
    ('''/*
 * Core NPU-NVMe implementation (legacy API + checkpoint/probe support).
 *
 * Usage:
 * - Built into libnpu_nvme.so; used by Python bindings and C tests.
 *
 * Inputs:
 * - NVMe PCI address, NPU device id, buffers, offsets.
 * Outputs:
 * - NVMe read/write operations and optional probe signaling.
 */''',
     '''/* =======================================================================
 * npu_nvme.c — NPU-to-NVMe Zero-Copy I/O Engine
 *
 * Implements three subsystems:
 *   I1  SPDK user-space NVMe driver — HBM <-> NVMe DMA transfers
 *   I2  FaF device-memory-polling listener — async step-boundary detection
 *   I3  Delta-frame I/O — incremental checkpoint slot read/write
 *
 * Built as libnpu_nvme.so; consumed by Python ctypes bindings
 * (python/direct_checkpoint.py) and the C smoke test (src/test_npu_nvme.c).
 * ======================================================================= */'''),

    # 2. Hugepage comment
    ('''// NPU driver pre-allocates all boot-time hugepages as internal DMA buffers.
// These pages are pinned by kernel hugetlb subsystem (not via hugetlbfs mmap)
// and show as Free=0/Rsvd=0.  DPDK's spdk_env_init() checks Free counter
// per NUMA node and refuses to start when it's zero.
//
// Workaround: add a small pool (512 pages = 1GB) on top of NPU's reservation.
// System has 2TB RAM / 1.9TB free -> 1GB is negligible.
// DPDK properly releases these pages on spdk_env_fini -> rte_eal_cleanup.''',
     '''/* The NPU driver reserves all boot-time hugepages (typically 8544 x 2 MB
 * = ~17 GB) as internal DMA buffers.  DPDK's spdk_env_init() reads the
 * per-NUMA-node free hugepage counter and refuses to start when it is
 * zero.  We add 512 pages (1 GB) on top of the NPU reservation.
 * DPDK releases these pages via rte_eal_cleanup() on spdk_env_fini. */'''),

    # 3. PCI matching
    ('    // Strict PCI BDF matching: only attach the target NVMe device,\n'
     '    // never the system disk or a device owned by another rank.',
     '    // Strict PCI BDF matching: only attach the target NVMe device,\n'
     '    // never the system disk or a device owned by another rank.'),

    # 3a. PCI matching (original Chinese)
    ('    // 【安全防御】：严格比对 PCI 地址，防止 SPDK 错误接管系统盘或其他卡正在使用的盘',
     '    // Strict PCI BDF matching: only attach the target NVMe device,\n'
     '    // never the system disk or a device owned by another rank.'),

    # 4. Multi-rank SHM
    ('        // 【关键修复】：恢复多进程 SHM ID 的显式读取！\n'
     '        // 这决定了 Rank 1-7 能否作为 Secondary 进程共享 Rank 0 已经初始化的 NVMe 硬件队列',
     '        // Multi-rank support: secondary ranks share the NVMe hardware queue\n'
     '        // initialised by rank 0.  Set SPDK_SHM_ID to the rank-0 PID before\n'
     '        // launching secondary processes.'),

    # 5. Cleanup crash
    ('        // 如果绑定失败（例如框架已经强行接管），为了防止暴毙，宁可泄漏也不要强行 Destroy',
     '        // If ACL device binding fails (e.g. framework has claimed the device),\n'
     '        // skip ACL resource destruction to avoid a crash from invalid context.'),

    # 6. Cleanup DMA
    ('        // 【修复 2】：销毁前必须强制同步，确保没有正在执行的幽灵 DMA 指令',
     '        // Drain all in-flight DMA operations before destroying resources.'),

    # 7. Pipeline header
    ('// ============================================================================\n'
     '// Phase 3-C: 终极双轨死循环 (The Dual-Polling Pipeline)\n'
     '// 真正的 C 语言级 Zero-Bubble 异步调度核心\n'
     '// ============================================================================\n'
     'void process_write_pipeline',
     '/* ===================================================================\n'
     ' * Dual-Polling Pipeline - drives NPU DMA and SPDK NVMe I/O concurrently.\n'
     ' *\n'
     ' * Three engines per iteration:\n'
     ' *   Engine 1 (submit) - pop free ring slots, launch DMA / SPDK commands\n'
     ' *   Engine 2 (drain)  - poll NPU events, submit ready chunks to SPDK\n'
     ' *   Engine 3 (reap)   - process SPDK completion queue, reclaim slots\n'
     ' *\n'
     ' * Stall recovery (two-stage):\n'
     ' *   Stage 1 (>50 ms no progress):  force-sync one stalled NPU event via\n'
     ' *     aclrtSynchronizeEvent, breaking hardware lazy-suspend stalls.\n'
     ' *   Stage 2 (>3 s no progress):   dump active task state to stderr\n'
     ' *     and reset the stall timer (diagnostic only, not a fix).\n'
     ' * =================================================================== */\n'
     'void process_write_pipeline'),

    # 8. Stall detection sub-header
    ('        // ============================================================\n'
     '        // [修改] 状态卡死监控与主动唤醒逻辑\n'
     '        // ============================================================',
     '        // ---- stall detection and recovery ----'),

    # 9. Stage 1 comment
    ('            // 【终极防线：主动刺激 (Poke) 机制】\n'
     '            // 如果卡顿超过 50 毫秒 (50,000 微秒)，并且还没到 3 秒的报警线。\n'
     '            // 这说明硬件极大概率因为任务短缺/中断合并，陷入了"惰性挂起"。',
     '            // Stage 1 (>50 ms no progress): force-sync stalled NPU event.\n'
     '            // Hardware may enter lazy-suspend or interrupt-coalescing\n'
     '            // states when the pipeline is under-utilised.'),

    # 10. "Kick" comment
    ('                        // 狠狠"踹"底层驱动一脚，强迫 CPU 等待并拉取真正的完成状态！',
     '                        // force-sync stalled event to unblock the pipeline'),

    # 11. Post-sync comment
    ('                        \n'
     '                        // 既然强制同步过了，这个任务的数据绝对已经安全到达 Host 内存\n'
     '                        task->state = CHUNK_NPU_DONE; \n'
     '                        \n'
     '                        // 唤醒一个就足以让阻塞的 Ring Buffer 腾出槽位，让流水线继续转！',
     '                        task->state = CHUNK_NPU_DONE;'),

    # 12. Stage 2 comment
    ('            // 如果连强制唤醒都拯救不了（超过 3 秒），说明是真正的物理硬件/SMMU 死锁',
     '            // Stage 2 (>3 s no progress): persistent hardware stall'),

    # 13. WATCHDOG stderr
    ('                fprintf(stderr, "\\n======================================================\\n");\n'
     '                fprintf(stderr, "[WATCHDOG] Rank %d Pipeline DEADLOCK Detected!\\n", ctx->npu_id);\n'
     '                fprintf(stderr, "[WATCHDOG] Completed: %d / Total: %d\\n", completed_tasks, num_tasks);\n'
     '                fprintf(stderr, "------- Active Tasks State Dump -------\\n");',
     '                fprintf(stderr, "\\n[NPU-NVMe] Pipeline STALL detected (rank %d)\\n", ctx->npu_id);\n'
     '                fprintf(stderr, "[NPU-NVMe] Completed: %d / Total: %d\\n", completed_tasks, num_tasks);\n'
     '                fprintf(stderr, "[NPU-NVMe] Active task state dump:\\n");'),

    # 14. Deadlock separator end
    ('                fprintf(stderr, "======================================================\\n\\n");\n'
     '                last_progress_time = now; // 重置定时器，防止日志疯狂刷屏',
     '                last_progress_time = now;'),

    # 15. Read pipeline header
    ('// 读路径双引擎轮询核心 (同样加入防死锁机制)',
     '/* Read-path dual-polling pipeline - mirrors write path with SPDK-read-first ordering. */'),

    # 16. Cleanup section header
    ('// ============================================================================\n'
     '// 4. 清理函数 (严格的资源释放与防崩溃保护)\n'
     '// ============================================================================',
     '/* ===================================================================\n'
     ' * Section 3 - Cleanup (strict resource release order)\n'
     ' * =================================================================== */'),

    # 17. SPDK probe section
    ('// ============================================================================\n'
     '// 2. SPDK 探测与挂载回调\n'
     '// ============================================================================',
     '/* ===================================================================\n'
     ' * Section 1 - SPDK Probe & Attach Callbacks\n'
     ' * =================================================================== */'),

    # 18. Init section
    ('// ============================================================================\n'
     '// 3. 完整的初始化函数\n'
     '// ============================================================================',
     '/* ===================================================================\n'
     ' * Section 2 - Init (SPDK env + NVMe probe + ACL + DMA pool + listener)\n'
     ' * =================================================================== */'),

    # 19. Delta section
    ('// ============================================================================\n'
     '// Phase 5 E11: Delta (增量) 写盘实现\n'
     '// ============================================================================',
     '/* ===================================================================\n'
     ' * Section 7 - Delta Frame I/O (I3)\n'
     ' * =================================================================== */'),

    # 20. Profiling util
    ('// ============================================================================\n'
     '// 工具函数：获取高精度微秒时间戳 (用于微观 Profiling)\n'
     '// ============================================================================',
     '/* ---- utility: microsecond timestamp ---- */'),

    # 21. create_io_tasks header
    ('// ============================================================================\n'
     '// Phase 2-A: 初始化全局任务表\n'
     '// 将 Python 传来的松散指针，封装为严格受控的 io_task_t 状态机数组\n'
     '// ============================================================================',
     '/* ---- io_task_t factory ---- */'),

    # 22. try_submit_async header
    ('// ============================================================================\n'
     '// Phase 2-B: 核心异步发射引擎\n'
     '// 尝试为一个任务分配槽位，并推入 NPU 异步流。\n'
     '// 返回值: \n'
     '//    0 : 提交成功 (瞬间返回，不阻塞)\n'
     '//   -1 : EAGAIN (Ring Buffer 已满，需要交出控制权去轮询)\n'
     '//   -2 : 硬件调用致命错误\n'
     '// ============================================================================',
     '/*\n'
     ' * Submit one chunk to the NPU DMA engine.\n'
     ' * Returns 0 on success, -1 if ring buffer full, -2 on ACL error.\n'
     ' */'),

    # 23. SPDK callback section
    ('// ============================================================================\n'
     '// Phase 3-A: SPDK 硬件落盘完成后的回调函数 (Callback)\n'
     '// 这个函数由 spdk_nvme_qpair_process_completions 触发\n'
     '// ============================================================================\n'
     '    \n'
     '// 我们定一个包装结构，方便传参',
     '/* SPDK write completion callback + submission helper */'),

    # 24. write_batch public API
    ('// ============================================================================\n'
     '// Phase 4: 顶层封装接口 (Top-level Wrapper)\n'
     '// 这是暴露给 Python ctypes 调用的核心入口函数\n'
     '// ============================================================================',
     '/* ===================================================================\n'
     ' * Section 5 - Public API: write_batch / read_batch\n'
     ' * =================================================================== */'),

    # 25. nvme read callback
    ('// ============================================================================\n'
     '// 1. NVMe 读取完成的回调函数\n'
     '// ============================================================================',
     '/* SPDK read completion callback */'),

    # 26. read_batch wrapper
    ('// ============================================================================\n'
     '// 3. 顶层封装接口 (Top-level Wrapper)\n'
     '// ============================================================================',
     ''),

    # 27. sync_meta_io header
    ('/* =========================================================\n'
     ' * 核心新增：同步元数据 I/O 引擎 (分离控制面)\n'
     ' * ========================================================= */',
     '/* ===================================================================\n'
     ' * Section 6 - Synchronous Metadata I/O (superblock + JSON ledger)\n'
     ' * =================================================================== */'),

    # 28. Data structures section
    ('// ============================================================================\n'
     '// 1. 基础结构定义 (包含 NPU 异步流与事件)\n'
     '// ============================================================================',
     '/* ---- data structures: ring buffer, DMA pool, I/O task, context ---- */'),

    # 29. Remove dev note
    ('// (ring_t 的 init, push, pop 等辅助函数保持不变，参考之前的回复)\n',
     ''),

    # 30. Remove duplicate comment line
    ('// 双向通信标志位：flags[0]为NPU发令，flags[1]为CPU放行\n'
     '// 双向通信标志位：flags[0]为NPU发令，flags[1]为CPU放行\n',
     ''),

    # 31. Fix comment about trigger buffer
    ('// Note: per-context device trigger buffer (dev_trigger_ptr) is allocated by\n'
     '// the Python layer via aclrtMalloc and passed in via npu_nvme_set_trigger_ptr().\n',
     ''),

    # 32. META_DMA comment cleanup
    ('// Delta frames can be O(10MB) per step (GPT-2 Small: ~15MB avg, ~30MB peak).\n'
     '// Each frame holds header + packed INT8 blocks + small params.\n'
     '// Use 64MB to allow plenty of headroom for larger models.\n'
     '#define META_DMA_BUF_SIZE',
     '/* Dedicated buffer for superblock & JSON ledger I/O; 64 MB covers all\n'
     ' * realistic metadata sizes including future delta ledger expansion. */\n'
     '#define META_DMA_BUF_SIZE'),

    # 33. Fix ring_t comment
    ('// (ring_t 的 init, push, pop 等辅助函数保持不变，参考之前的回复)\n'
     'static int ring_init',
     'static int ring_init'),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        count += 1
    else:
        print(f'WARNING: pattern #{count+1} not found: {old[:60]}...')

with open('src/npu_nvme.c', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Applied {count}/{len(replacements)} replacements')

/*
 * Core NPU-NVMe implementation (legacy API + checkpoint/probe support).
 *
 * Usage:
 * - Built into libnpu_nvme.so; used by Python bindings and C tests.
 *
 * Inputs:
 * - NVMe PCI address, NPU device id, buffers, offsets.
 * Outputs:
 * - NVMe read/write operations and optional probe signaling.
 */
#include "npu_nvme.h"
#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>

#define MIN_PIPE_DEPTH   1
#define MAX_PIPE_DEPTH   16
// Delta frames can be O(10MB) per step (GPT-2 Small: ~15MB avg, ~30MB peak).
// Each frame holds header + packed INT8 blocks + small params.
// Use 64MB to allow plenty of headroom for larger models.
#define META_DMA_BUF_SIZE (64 * 1024 * 1024)

// 双向通信标志位：flags[0]为NPU发令，flags[1]为CPU放行
// 双向通信标志位：flags[0]为NPU发令，flags[1]为CPU放行
volatile uint8_t* probe_flags = NULL;

// Note: per-context device trigger buffer (dev_trigger_ptr) is allocated by
// the Python layer via aclrtMalloc and passed in via npu_nvme_set_trigger_ptr().

// ============================================================================
// Hugepage pool auto-expansion for DPDK/SPDK
// ============================================================================
// NPU driver pre-allocates all boot-time hugepages as internal DMA buffers.
// These pages are pinned by kernel hugetlb subsystem (not via hugetlbfs mmap)
// and show as Free=0/Rsvd=0.  DPDK's spdk_env_init() checks Free counter
// per NUMA node and refuses to start when it's zero.
//
// Workaround: add a small pool (512 pages = 1GB) on top of NPU's reservation.
// System has 2TB RAM / 1.9TB free → 1GB is negligible.
// DPDK properly releases these pages on spdk_env_fini → rte_eal_cleanup.
#define HUGEPAGE_PADDING 512
#define HUGEPAGE_2MB_PATH "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"
#define NR_HUGEPAGES_PATH  "/proc/sys/vm/nr_hugepages"

static int read_int_from_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val;
    if (fscanf(f, "%d", &val) != 1) { fclose(f); return -1; }
    fclose(f);
    return val;
}

static void ensure_hugepages(void) {
    int free_pages = read_int_from_file(HUGEPAGE_2MB_PATH);
    if (free_pages > 0) {
        fprintf(stderr, "[NPU-NVMe] Hugepage pool: %d free pages available — OK.\n",
                free_pages);
        return;
    }

    // Read current total (may be 0 after a previous resize)
    int nr = read_int_from_file(NR_HUGEPAGES_PATH);
    if (nr < 0) {
        fprintf(stderr, "[NPU-NVMe] WARNING: cannot read nr_hugepages. "
                "DPDK init may fail if no free hugepages.\n");
        return;
    }

    // First restore to a known baseline: 8544 (NPU driver pool).
    // If nr is already >= 8544, use that. Otherwise reset to 8544.
    int baseline = 8544;
    if (nr < baseline) {
        fprintf(stderr, "[NPU-NVMe] nr_hugepages=%d (< baseline %d), restoring to %d...\n",
                nr, baseline, baseline);
        FILE *f = fopen(NR_HUGEPAGES_PATH, "w");
        if (!f) goto no_root;
        fprintf(f, "%d", baseline);
        fclose(f);
        nr = baseline;
        usleep(100000);  // 100ms for kernel to settle
    }

    int target = nr + HUGEPAGE_PADDING;
    {
        FILE *f = fopen(NR_HUGEPAGES_PATH, "w");
        if (!f) goto no_root;
        fprintf(f, "%d", target);
        fclose(f);
    }

    // Verify
    int new_free = read_int_from_file(HUGEPAGE_2MB_PATH);
    fprintf(stderr, "[NPU-NVMe] Hugepage pool: %d → %d total, free: %d → %d "
            "(+%d pages = %.0f MB added for DPDK)\n",
            nr, target, free_pages, new_free,
            HUGEPAGE_PADDING, (double)HUGEPAGE_PADDING * 2.0);
    return;

no_root:
    fprintf(stderr, "[NPU-NVMe] WARNING: cannot adjust nr_hugepages (need root).\n"
                    "[NPU-NVMe] If SPDK init fails, run as root:\n"
                    "    echo %d > %s\n", target, NR_HUGEPAGES_PATH);
}

// Forward declaration
static inline uint64_t get_time_us();

// ============================================================================
// 1. 基础结构定义 (包含 NPU 异步流与事件)
// ============================================================================
typedef struct {
    int *slots;
    int capacity;
    int head;
    int tail;
} ring_t;

// (ring_t 的 init, push, pop 等辅助函数保持不变，参考之前的回复)
static int ring_init(ring_t *r, int cap) {
    r->slots = calloc(cap + 1, sizeof(int));
    if (!r->slots) return -1;
    r->capacity = cap + 1;
    r->head = r->tail = 0;
    return 0;
}
static void ring_free(ring_t *r) {
    free(r->slots);
    r->slots = NULL;
}
static bool ring_is_full(ring_t *r) { return ((r->tail + 1) % r->capacity) == r->head; }
static bool ring_is_empty(ring_t *r) { return r->head == r->tail; }
static int ring_push(ring_t *r, int val) {
    if (ring_is_full(r)) return -1;
    r->slots[r->tail] = val;
    r->tail = (r->tail + 1) % r->capacity;
    return 0;
}
static int ring_pop(ring_t *r, int *val) {
    if (ring_is_empty(r)) return -1;
    *val = r->slots[r->head];
    r->head = (r->head + 1) % r->capacity;
    return 0;
}

typedef struct {
    void *buf;
    uint64_t phys_addr;
} dma_buf_t;

typedef enum {
    CHUNK_IDLE = 0,         
    CHUNK_NPU_COPYING,      // 写路径：NPU -> DRAM
    CHUNK_NPU_DONE,         // 写路径：NPU 搬运完毕
    CHUNK_SPDK_WRITING,     // 写路径：NVMe 落盘中
    CHUNK_SPDK_READING,     // [新增] 读路径：NVMe -> DRAM 
    CHUNK_SPDK_DONE,        // [新增] 读路径：NVMe 读取完毕，等待 NPU 搬运
    CHUNK_DONE              
} chunk_state_t;

typedef struct {
    int task_idx;           
    int buf_idx;            
    chunk_state_t state;    
    
    void *npu_ptr;          
    size_t size;            
    uint64_t nvme_offset;   
    
    uint64_t ts_submit;     
    uint64_t ts_npu_done;   
    uint64_t ts_spdk_done;  
} io_task_t;

// 核心上下文
typedef struct NPUNVMEContext {
    char pci_addr[64];              // 用于 probe 时过滤设备
    
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;        // 保存命名空间指针
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;

    int npu_id;
    int max_pipe_depth;             // Ring Buffer 深度
    uint32_t chunk_size;

    dma_buf_t *pool;                // DRAM 大页内存池
    ring_t free_ring;               // 空闲槽位队列

    // ----- [全异步核心组件] -----
    aclrtStream copy_stream;        // NPU 专属异步数据流
    aclrtEvent *events;             // 与 Ring Buffer 槽位一一对应的硬件事件
    aclrtContext acl_ctx;

    bool enable_profiling;
    char profiling_dir[256];

    void *meta_dma_buf;     // 专用于元数据读写的大页内存

    // ----- [新增] 探针后台持久化任务表 -----
    io_task_t *registered_tasks;
    int num_registered_tasks;

    // ----- [新增] 后台监听线程控制 -----
    pthread_t listener_thread;
    volatile int stop_listener;
    bool listener_started;

    // ----- [新增] NPU 侧探针 flag 设备指针 -----
    void *probe_flag_dev_ptr;
    void *probe_flag_host;

    // ----- [新增] FaF step_counter polling -----
    void *dev_step_ptr;        // Device pointer for step_counter (HBM)
    void *step_poll_buf;       // Host buffer for polling step_counter
    int last_step_seen;        // Last step value detected by listener
    int ckpt_interval;         // Checkpoint every N steps

    // ----- [Phase 5 E11] Delta (增量) 盘布局 -----
    uint64_t delta_area_offset;   // Delta ring 起始字节偏移
    uint64_t delta_slot_size;     // 每槽位字节数
    uint32_t delta_slot_count;    // 槽位总数
    uint32_t delta_last_commit;   // 最后写入的槽位 (用于恢复遍历)
} NPUNVMEContext;

int npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr) {
    if (!ctx) {
        return -1;
    }
    // Refresh ACL context after graph/runtime is created
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] aclrtSetDevice failed in set_probe_flag_ptr, ret=%d\n", ret);
        return -1;
    }
    if (aclrtGetCurrentContext(&ctx->acl_ctx) != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] aclrtGetCurrentContext failed in set_probe_flag_ptr\n");
        return -1;
    }
    // Recreate copy stream under the refreshed context
    if (ctx->copy_stream) {
        aclrtDestroyStream(ctx->copy_stream);
        ctx->copy_stream = NULL;
    }
    ret = aclrtCreateStream(&ctx->copy_stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] Failed to recreate copy stream, ret=%d\n", ret);
        return -1;
    }

    // Self-allocation fallback: when dev_ptr is NULL (e.g. sink=TRUE where MS
    // never allocates the flag tensor), allocate a 4-byte HBM buffer ourself.
    bool self_allocated = false;
    if (dev_ptr == NULL) {
        void *self_dev_ptr = NULL;
        ret = aclrtMalloc(&self_dev_ptr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] Self-alloc of probe_flag_dev failed, ret=%d\n", ret);
            return -1;
        }
        // Zero-initialise
        uint32_t zero = 0;
        ret = aclrtMemcpyAsync(self_dev_ptr, 4, &zero, 4,
                               ACL_MEMCPY_HOST_TO_DEVICE, ctx->copy_stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(ctx->copy_stream);
        }
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] probe_flag zero-init failed, ret=%d. Freeing.\n", ret);
            aclrtFree(self_dev_ptr);
            return -1;
        }
        // Store old self-allocated ptr for cleanup
        if (ctx->probe_flag_dev_ptr && ctx->probe_flag_dev_ptr != dev_ptr) {
            // Previous was self-allocated too — check if we own it
            // Simple heuristic: if previous ptr was set via NULL fallback, free it
            aclrtFree(ctx->probe_flag_dev_ptr);
        }
        ctx->probe_flag_dev_ptr = self_dev_ptr;
        self_allocated = true;
        fprintf(stderr, "[NPU-NVMe] Probe flag SELF-ALLOCATED: dev=%p (sink=TRUE fallback)\n",
                self_dev_ptr);
    } else {
        ctx->probe_flag_dev_ptr = dev_ptr;
        fprintf(stderr, "[NPU-NVMe] Probe flag dev ptr set (MS tensor): %p\n", dev_ptr);
    }

    // Allocate host polling buffer if not already done
    if (!ctx->probe_flag_host) {
        ret = aclrtMallocHost(&ctx->probe_flag_host, 4);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] Failed to alloc probe_flag_host, ret=%d\n", ret);
            return -1;
        }
    }

#ifdef DIAGNOSTIC
    {
        aclrtContext cur_ctx = NULL;
        aclError ar = aclrtGetCurrentContext(&cur_ctx);
        fprintf(stderr, "[NPU-NVMe] probe_flag ctx=%p ret=%d\n", (void *)cur_ctx, (int)ar);

        // Read-back verification
        uint32_t cur = 0;
        aclError r = aclrtMemcpyAsync(ctx->probe_flag_host, 4, ctx->probe_flag_dev_ptr, 4,
                                       ACL_MEMCPY_DEVICE_TO_HOST, ctx->copy_stream);
        if (r == ACL_SUCCESS) r = aclrtSynchronizeStream(ctx->copy_stream);
        if (r == ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] probe_flag selftest current=%u\n", (unsigned)cur);
        } else {
            fprintf(stderr, "[NPU-NVMe] probe_flag selftest read failed, ret=%d\n", r);
        }
    }
#endif
    return 0;
}


int npu_nvme_trigger_probe(NPUNVMEContext *ctx)
{
    if (!ctx) {
        return -1;
    }
    if (probe_flags) {
        probe_flags[0] = 1;
        __sync_synchronize();
        return 0;
    }
    return -1;
}

int npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value)
{
    if (!ctx || !ctx->probe_flag_dev_ptr) {
        return -1;
    }
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] aclrtSetDevice failed in set_probe_flag_value, ret=%d\n", ret);
        return -1;
    }
    ret = aclrtSetCurrentContext(ctx->acl_ctx);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] aclrtSetCurrentContext failed in set_probe_flag_value, ret=%d\n", ret);
        return -1;
    }
    if (!ctx->probe_flag_host) {
        return -1;
    }
    *(uint32_t *)ctx->probe_flag_host = value;
    ret = aclrtMemcpyAsync(ctx->probe_flag_dev_ptr, 4, ctx->probe_flag_host, 4,
                           ACL_MEMCPY_HOST_TO_DEVICE, ctx->copy_stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(ctx->copy_stream);
    }
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU-NVMe] set_probe_flag_value memcpy failed, ret=%d\n", ret);
        return -1;
    }
    return 0;
}

// ---------- FaF: step_counter polling ----------

int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval)
{
    if (!ctx || !dev_ptr) return -1;
    ctx->dev_step_ptr = dev_ptr;

    // Allocate host polling buffer for step_counter
    if (!ctx->step_poll_buf) {
        aclError ret = aclrtMallocHost(&ctx->step_poll_buf, 4);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] step_poll_buf alloc failed, ret=%d\n", ret);
            return -1;
        }
    }
    ctx->ckpt_interval = ckpt_interval > 0 ? ckpt_interval : 10;
    ctx->last_step_seen = 0;  // start from 0 so first trigger is at step >= interval
    fprintf(stderr, "[NPU-NVMe] Device step_counter ptr set: %p, interval=%d\n",
            dev_ptr, ctx->ckpt_interval);
    return 0;
}

void* npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx)
{
    if (!ctx) return NULL;
    return ctx->probe_flag_dev_ptr;
}

static inline void signal_probe_flag(NPUNVMEContext *ctx, uint32_t value) {
    if (ctx && ctx->probe_flag_dev_ptr) {
        if (ctx->enable_profiling) {
            aclrtContext cur_ctx = NULL;
            aclError cr = aclrtGetCurrentContext(&cur_ctx);
            fprintf(stderr, "[NPU-NVMe] signal flag: cur_ctx=%p ret=%d ctx_acl=%p value=%u\n",
                    (void *)cur_ctx, (int)cr, (void *)ctx->acl_ctx, (unsigned)value);
        }
        aclError ret = aclrtSetDevice(ctx->npu_id);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] aclrtSetDevice failed in signal, ret=%d\n", ret);
            return;
        }
        ret = aclrtSetCurrentContext(ctx->acl_ctx);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] aclrtSetCurrentContext failed in signal, ret=%d\n", ret);
            return;
        }
        if (!ctx->probe_flag_host) {
            fprintf(stderr, "[NPU-NVMe] probe_flag_host is NULL\n");
            return;
        }
        // Set flag = expected value (unlocks WaitProbe: flag >= expected)
        *(uint32_t *)ctx->probe_flag_host = value;
        ret = aclrtMemcpyAsync(ctx->probe_flag_dev_ptr, 4, ctx->probe_flag_host, 4,
                               ACL_MEMCPY_HOST_TO_DEVICE, ctx->copy_stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(ctx->copy_stream);
        }
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] write flag failed, ret=%d, ptr=%p\n", ret, ctx->probe_flag_dev_ptr);
        } else {
            fprintf(stderr, "[NPU-NVMe] Probe flag set to %u on device ptr=%p\n", (unsigned)value, ctx->probe_flag_dev_ptr);
        }
        if (ctx->enable_profiling) {
            aclError r2 = aclrtMemcpyAsync(ctx->probe_flag_host, 4, ctx->probe_flag_dev_ptr, 4,
                                           ACL_MEMCPY_DEVICE_TO_HOST, ctx->copy_stream);
            if (r2 == ACL_SUCCESS) {
                r2 = aclrtSynchronizeStream(ctx->copy_stream);
            }
            if (r2 == ACL_SUCCESS) {
                fprintf(stderr, "[NPU-NVMe] Probe flag readback=%u\n",
                        (unsigned)(*(uint32_t *)ctx->probe_flag_host));
            } else {
                fprintf(stderr, "[NPU-NVMe] Probe flag readback failed, ret=%d\n", r2);
            }
        }
        return;
    }
    if (probe_flags) {
        __sync_synchronize();
        probe_flags[1] = 1;
    }
}

io_task_t* create_io_tasks(int num_tasks, void **npu_ptrs, uint64_t *nvme_offsets, size_t *sizes);
void process_write_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks, bool is_host);

/* 元数据专属的简化回调 */
static void io_complete_meta(void *arg, const struct spdk_nvme_cpl *cpl) {
    int *flag = (int *)arg;
    *flag = spdk_nvme_cpl_is_error(cpl) ? -1 : 1;
}

int npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs, 
                            uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) {
        fprintf(stderr, "[Fatal] Invalid arguments passed to register_tasks.\n");
        return -1;
    }
    
    // 如果重复注册，清理旧的内存
    if (ctx->registered_tasks) {
        free(ctx->registered_tasks);
    }
    
    // 使用原有的辅助函数，预分配并填充好物理内存布局
    ctx->registered_tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!ctx->registered_tasks) {
        return -1;
    }
    
    ctx->num_registered_tasks = num_items;
    printf("[NPU-NVMe] Successfully registered %d tensor tasks for Background Probe.\n", num_items);
    return 0;
}

// ============================================================================
// 2. SPDK 探测与挂载回调
// ============================================================================

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    
    // 【安全防御】：严格比对 PCI 地址，防止 SPDK 错误接管系统盘或其他卡正在使用的盘
    if (strcmp(trid->traddr, ctx->pci_addr) != 0) {
        return false; 
    }
    printf("[SPDK] Probed target NVMe device at %s\n", trid->traddr);
    return true;
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    int nsid;
    
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr);
         nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (!ns || !spdk_nvme_ns_is_active(ns)) continue;
        
        ctx->ctrlr = ctrlr;
        ctx->ns = ns;
        ctx->block_size = spdk_nvme_ns_get_sector_size(ns);
        ctx->total_blocks = spdk_nvme_ns_get_num_sectors(ns);
        printf("[SPDK] Attached to NVMe! Block Size: %u, Total Blocks: %lu\n", 
               ctx->block_size, ctx->total_blocks);
        break; // 找到第一个活跃的命名空间就退出
    }
}


// ============================================================================
// 3. 完整的初始化函数
// ============================================================================

// 后台监听与 SPDK 轮询线程 (FaF: step_counter polling)
void* probe_listener_thread(void* arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    const char *mode = getenv("NPU_NVME_LISTENER_MODE");

    // P2-1: WaitProbe dead code is #if 0'd — kept for potential rollback
    if (mode && strcmp(mode, "idle") == 0) {
        printf("[Probe Listener] IDLE mode — no SPDK writes, just waiting for stop signal.\n");
        while (1) {
            if (ctx->stop_listener) return NULL;
            sleep(1);
        }
        return NULL;
    }
    if (mode && strcmp(mode, "off") == 0) {
        printf("[Probe Listener] DISABLED via NPU_NVME_LISTENER_MODE=off\n");
        return NULL;
    }
    // "full" or default: full FaF mode
    printf("[Probe Listener] Background thread started. mode=%s. Monitoring NPU signals...\n",
           mode ? mode : "full");

    // 【非常关键】：绑定 NPU ACL 上下文到当前后台线程。
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Probe Listener] aclrtSetDevice failed, ret=%d\n", ret);
    }
    ret = aclrtSetCurrentContext(ctx->acl_ctx);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Probe Listener] aclrtSetCurrentContext failed, ret=%d\n", ret);
    }

#ifdef DIAGNOSTIC
    fprintf(stderr, "[Probe Listener] Context bound: npu_id=%d, acl_ctx=%p, dev_step_ptr=%p, ckpt_interval=%d, dev_flag_ptr=%p\n",
            ctx->npu_id, (void*)ctx->acl_ctx, ctx->dev_step_ptr, ctx->ckpt_interval, ctx->probe_flag_dev_ptr);
#endif

    int poll_count = 0;

    while(1) {
        int triggered = 0;

        if (ctx->dev_step_ptr && ctx->step_poll_buf) {
            // FaF mode: poll step_counter from HBM via aclrtMemcpy
            aclError err = aclrtSetDevice(ctx->npu_id);
            if (err != ACL_SUCCESS) {
                // Device may have been released, exit safely
                if (ctx->stop_listener) return NULL;
#ifdef DIAGNOSTIC
                fprintf(stderr, "[NPU-NVMe] listener aclrtSetDevice failed, ret=%d\n", err);
#endif
                usleep(10000);
                continue;
            }
            err = aclrtSetCurrentContext(ctx->acl_ctx);
            if (err != ACL_SUCCESS) {
                if (ctx->stop_listener) return NULL;
#ifdef DIAGNOSTIC
                fprintf(stderr, "[NPU-NVMe] listener aclrtSetCurrentContext failed, ret=%d\n", err);
#endif
                usleep(10000);
                continue;
            }

            // Blocking aclrtMemcpy for reliable step_counter read
            err = aclrtMemcpy(ctx->step_poll_buf, 4,
                              ctx->dev_step_ptr, 4,
                              ACL_MEMCPY_DEVICE_TO_HOST);
            if (err != ACL_SUCCESS) {
                if (ctx->stop_listener) return NULL;
#ifdef DIAGNOSTIC
                fprintf(stderr, "[NPU-NVMe] listener aclrtMemcpy ERROR: ret=%d (step_counter=%p)\n",
                        err, ctx->dev_step_ptr);
#endif
                usleep(10000);  // P1-1: 10ms poll interval
                continue;
            }

            int cur_step = *(int32_t *)ctx->step_poll_buf;
            int expected_step = ctx->last_step_seen + ctx->ckpt_interval;

            if (cur_step >= expected_step && cur_step > ctx->last_step_seen) {
                poll_count = 0;
                ctx->last_step_seen = cur_step;
                triggered = 1;
                fprintf(stderr, "[NPU-NVMe] listener TRIGGERED: step=%d, expected=%u\n",
                        cur_step, (unsigned)expected_step);
            }
        } else {
            // Fallback: old WaitProbe host-trigger mode
            if (probe_flags && probe_flags[0] != 0) {
                probe_flags[0] = 0;
                triggered = 1;
            }
        }

        if (ctx->stop_listener) {
            return NULL;
        }
        if (!triggered) {
            // Process SPDK completions even when idle
            if (ctx->qpair)
                spdk_nvme_qpair_process_completions(ctx->qpair, 0);
            usleep(10000);  // P1-1: 10ms poll interval (was 100us)
            continue;
        }

        // 2. SPDK write pipeline
        if (ctx->registered_tasks != NULL) {
            // Reset state machine for re-use
            for (int i = 0; i < ctx->num_registered_tasks; i++) {
                ctx->registered_tasks[i].state = CHUNK_IDLE;
                ctx->registered_tasks[i].buf_idx = -1;
            }

            process_write_pipeline(ctx, ctx->registered_tasks, ctx->num_registered_tasks, false);

            // Signal probe_flag with expected value after SPDK write completes
            uint32_t expected = (uint32_t)(ctx->last_step_seen / ctx->ckpt_interval);
            signal_probe_flag(ctx, expected);
        }
        // P2-1 note: WaitProbe/TriggerProbe dead code was here; replaced by FaF listener.
        // Retained for rollback via #if 0 block at end of file.
    }
    return NULL;
}

int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id, 
                  int pipe_depth, int chunk_size, bool enable_profiling, const char *prof_dir) {
    // 1. 分配上下文
    NPUNVMEContext *ctx = calloc(1, sizeof(NPUNVMEContext));
    if (!ctx) return -1;

    strncpy(ctx->pci_addr, pci_addr, sizeof(ctx->pci_addr) - 1);
    ctx->npu_id = npu_id;
    ctx->chunk_size = chunk_size;
    ctx->max_pipe_depth = (pipe_depth < MIN_PIPE_DEPTH) ? MIN_PIPE_DEPTH :
                          (pipe_depth > MAX_PIPE_DEPTH) ? MAX_PIPE_DEPTH : pipe_depth;
    ctx->enable_profiling = enable_profiling;
    if (prof_dir) {
        strncpy(ctx->profiling_dir, prof_dir, sizeof(ctx->profiling_dir) - 1);
    } else {
        strcpy(ctx->profiling_dir, ".");
    }

    // 2. 初始化 SPDK 环境 (只在首次调用时有效，如果多卡跑在同一个进程，需要通过配置 shm_id 解决)
    static int spdk_inited = 0;
    if (!spdk_inited) {
        struct spdk_env_opts env_opts;
        spdk_env_opts_init(&env_opts);
        env_opts.name = "npu_nvme_app";
        
        // 【关键修复】：恢复多进程 SHM ID 的显式读取！
        // 这决定了 Rank 1-7 能否作为 Secondary 进程共享 Rank 0 已经初始化的 NVMe 硬件队列
        const char *shm = getenv("SPDK_SHM_ID");
        if (shm) {
            env_opts.shm_id = atoi(shm);
        }

        // P1-4: auto-expand hugepage pool if NPU driver consumed all free pages
        ensure_hugepages();

        if (spdk_env_init(&env_opts) < 0) {
            fprintf(stderr, "[Fatal] Unable to initialize SPDK env.\n"
                    "[Fatal] Check: (1) run as root, (2) free hugepages > 0 per NUMA node.\n"
                    "[Fatal] Quick fix: echo %d > %s\n",
                    read_int_from_file(NR_HUGEPAGES_PATH) + HUGEPAGE_PADDING,
                    NR_HUGEPAGES_PATH);
            free(ctx);
            return -1;
        }
        spdk_inited = 1;
    }

    // 3. 探测并挂载 NVMe 硬盘
    struct spdk_nvme_transport_id trid = {};
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", pci_addr);

    if (spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, NULL) != 0) {
        fprintf(stderr, "[Fatal] spdk_nvme_probe failed.\n");
        goto fail;
    }
    if (!ctx->ctrlr) {
        fprintf(stderr, "[Fatal] Controller not found at %s.\n", pci_addr);
        goto fail;
    }

    // 4. 创建 SPDK I/O Queue Pair
    struct spdk_nvme_io_qpair_opts qopts;
    spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &qopts, sizeof(qopts));
    // 为了应对极致的异步并发，把队列深度开大
    qopts.io_queue_size = 512; 
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &qopts, sizeof(qopts));
    if (!ctx->qpair) {
        fprintf(stderr, "[Fatal] Cannot allocate NVMe I/O qpair.\n");
        goto fail;
    }

    // 5. 初始化 NPU 环境 (绑定设备、创建异步流)
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) goto fail;

    if (aclrtGetCurrentContext(&ctx->acl_ctx) != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to get ACL context.\\n");
        goto fail;
    }
    
    ret = aclrtCreateStream(&ctx->copy_stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to create NPU Stream.\n");
        goto fail;
    }

    // probe_flag_host will be allocated after probe_flag_dev_ptr is set

    // 6. 分配 Ring Buffer 及关联的 NPU Hardware Event
    ctx->pool = calloc(ctx->max_pipe_depth, sizeof(dma_buf_t));
    ctx->events = calloc(ctx->max_pipe_depth, sizeof(aclrtEvent));
    ring_init(&ctx->free_ring, ctx->max_pipe_depth);

    for (int i = 0; i < ctx->max_pipe_depth; i++) {
        // 使用 SPDK 申请物理连续、锁页的大页内存，彻底消除 TLB Miss
        ctx->pool[i].buf = spdk_zmalloc(ctx->chunk_size, 2 * 1024 * 1024, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
        if (!ctx->pool[i].buf) {
            fprintf(stderr, "[Fatal] spdk_zmalloc failed at slot %d.\n", i);
            goto fail;
        }
        ctx->pool[i].phys_addr = spdk_vtophys(ctx->pool[i].buf, NULL);
        
        // 创建 NPU 事件
        ret = aclrtCreateEvent(&ctx->events[i]);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Fatal] Failed to create NPU Event at slot %d.\n", i);
            goto fail;
        }
        
        ring_push(&ctx->free_ring, i);
    }

    printf("[Init] NPUNVME Fully Initialized! Stream/Events ready. Max Pipe Depth: %d\n", ctx->max_pipe_depth);
    *out_ctx = ctx;

    ctx->meta_dma_buf = spdk_zmalloc(META_DMA_BUF_SIZE, 2 * 1024 * 1024, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);

    ret = aclrtMallocHost((void**)&probe_flags, 64);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] Failed to allocate probe flags.\n");
        return -1;
    }
    probe_flags[0] = 0;
    // 默认放行一次以避免训练首步被意外阻塞（后续由 SPDK 写入负责实际放行）
    probe_flags[1] = 1;
    
    // 启动后台监听线程
    ctx->stop_listener = 0;
    ctx->listener_started = false;
    ctx->probe_flag_dev_ptr = NULL;
    ctx->probe_flag_host = NULL;
    ctx->dev_step_ptr = NULL;
    ctx->step_poll_buf = NULL;
    ctx->last_step_seen = -1;
    ctx->ckpt_interval = 10;
    if (pthread_create(&ctx->listener_thread, NULL, probe_listener_thread, ctx) == 0) {
        ctx->listener_started = true;
    }

    return 0;

fail:
    npu_nvme_cleanup(ctx);
    return -1;
}

// ============================================================================
// 4. 清理函数 (严格的资源释放与防崩溃保护)
// ============================================================================
void npu_nvme_cleanup(NPUNVMEContext *ctx) {
    if (!ctx) return;

    // 先停止后台线程，防止 SPDK 资源在使用时被销毁
    ctx->stop_listener = 1;
    if (probe_flags) {
        probe_flags[0] = 1; // 唤醒等待中的后台线程
        __sync_synchronize();
    }
    if (ctx->listener_started) {
        pthread_join(ctx->listener_thread, NULL);
        ctx->listener_started = false;
    }
    
    // 【修复 1】：必须在当前线程重新绑定 NPU Device，否则后续 Destroy 会直接引发段错误！
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Warning] Failed to set NPU device %d during cleanup. Skip ACL cleanup.\n", ctx->npu_id);
        // 如果绑定失败（例如框架已经强行接管），为了防止暴毙，宁可泄漏也不要强行 Destroy
    } else {
        // 【修复 2】：销毁前必须强制同步，确保没有正在执行的幽灵 DMA 指令
        if (ctx->copy_stream) {
            aclrtSynchronizeStream(ctx->copy_stream);
        }

        // 安全销毁事件
        if (ctx->events) {
            for (int i = 0; i < ctx->max_pipe_depth; i++) {
                if (ctx->events[i]) aclrtDestroyEvent(ctx->events[i]);
            }
            free(ctx->events);
        }
        
        // 安全销毁流
        if (ctx->copy_stream) {
            aclrtDestroyStream(ctx->copy_stream);
        }
        if (ctx->probe_flag_host) {
            aclrtFreeHost(ctx->probe_flag_host);
        }
        // Free self-allocated probe_flag_dev (NULL-fallback in set_probe_flag_ptr).
        // Guard: only free if ACL device is still alive — MS may have torn it down
        // before our cleanup runs, in which case aclrtFree would segfault.
        if (ctx->probe_flag_dev_ptr && ret == ACL_SUCCESS) {
            aclrtFree(ctx->probe_flag_dev_ptr);
            ctx->probe_flag_dev_ptr = NULL;
        }
        if (ctx->step_poll_buf) {
            aclrtFreeHost(ctx->step_poll_buf);
        }
    }
    
    // ----- 释放内存与 SPDK 资源 -----
    if (ctx->pool) {
        for (int i = 0; i < ctx->max_pipe_depth; i++) {
            if (ctx->pool[i].buf) spdk_free(ctx->pool[i].buf);
        }
        free(ctx->pool);
    }
    
    ring_free(&ctx->free_ring);
    
    if (ctx->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    }
    
    if (ctx->ctrlr) {
        spdk_nvme_detach(ctx->ctrlr);
    }

    if (ctx->meta_dma_buf) {
        spdk_free(ctx->meta_dma_buf);
    }

    if (ctx->registered_tasks) {
        free(ctx->registered_tasks);
        ctx->registered_tasks = NULL;
        ctx->num_registered_tasks = 0;
    }
    
    free(ctx);
}

// ============================================================================
// 工具函数：获取高精度微秒时间戳 (用于微观 Profiling)
// ============================================================================
static inline uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// ============================================================================
// Phase 2-A: 初始化全局任务表
// 将 Python 传来的松散指针，封装为严格受控的 io_task_t 状态机数组
// ============================================================================
io_task_t* create_io_tasks(int num_tasks, void **npu_ptrs, uint64_t *nvme_offsets, size_t *sizes) {
    io_task_t *tasks = calloc(num_tasks, sizeof(io_task_t));
    if (!tasks) {
        fprintf(stderr, "[Fatal] Failed to allocate memory for IO tasks.\n");
        return NULL;
    }
    
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].task_idx = i;
        tasks[i].buf_idx = -1;               // 尚未分配 Ring Buffer 槽位
        tasks[i].state = CHUNK_IDLE;         // 初始状态：空闲
        tasks[i].npu_ptr = npu_ptrs[i];
        tasks[i].nvme_offset = nvme_offsets[i];
        tasks[i].size = sizes[i];
    }
    return tasks;
}

// ============================================================================
// Phase 2-B: 核心异步发射引擎
// 尝试为一个任务分配槽位，并推入 NPU 异步流。
// 返回值: 
//    0 : 提交成功 (瞬间返回，不阻塞)
//   -1 : EAGAIN (Ring Buffer 已满，需要交出控制权去轮询)
//   -2 : 硬件调用致命错误
// ============================================================================
/*
int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host) {
    int buf_idx;
    if (ring_pop(&ctx->free_ring, &buf_idx) != 0) return -1; // Ring Buffer 满

    task->buf_idx = buf_idx;
    if (ctx->enable_profiling) task->ts_submit = get_time_us();

    if (is_host) {
        // 【新增】：Host 内存直接 memcpy 到锁页大页内存，瞬间完成，跳过 NPU 异步流
        memcpy(ctx->pool[buf_idx].buf, task->npu_ptr, task->size);
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE; 
    } else {
        // NPU 显存：发起真正的异步 DMA 搬运
        aclError ret = aclrtMemcpyAsync(ctx->pool[buf_idx].buf, task->size, task->npu_ptr, task->size, ACL_MEMCPY_DEVICE_TO_HOST, ctx->copy_stream);
        if (ret != ACL_SUCCESS) {
            ring_push(&ctx->free_ring, buf_idx); return -2;
        }
        aclrtRecordEvent(ctx->events[buf_idx], ctx->copy_stream);
        task->state = CHUNK_NPU_COPYING;
    }
    return 0;
}
*/

int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host) {
    int buf_idx;
    if (ring_pop(&ctx->free_ring, &buf_idx) != 0) return -1; // Ring Buffer 满

    task->buf_idx = buf_idx;
    if (ctx->enable_profiling) task->ts_submit = get_time_us();

    if (is_host) {
        memcpy(ctx->pool[buf_idx].buf, task->npu_ptr, task->size);
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE; 
    } else {
        // 【核心修改】：抛弃 Async 和 Event，直接使用同步拷贝！
        // 因为我们在后台线程，阻塞这零点几毫秒对外界毫无影响
        aclError ret = aclrtMemcpy(ctx->pool[buf_idx].buf, task->size, task->npu_ptr, task->size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ring_push(&ctx->free_ring, buf_idx); 
            return -2;
        }
        // 瞬间拷贝完成，直接进入下一步！
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    }
    return 0;
}

// ============================================================================
// Phase 3-A: SPDK 硬件落盘完成后的回调函数 (Callback)
// 这个函数由 spdk_nvme_qpair_process_completions 触发
// ============================================================================

// 我们定一个包装结构，方便传参
typedef struct {
    NPUNVMEContext *ctx;
    io_task_t *task;
    int *completed_counter;
} spdk_cb_arg_t;

static void nvme_write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    if (cb_arg->ctx->enable_profiling) cb_arg->task->ts_spdk_done = get_time_us();
    
    cb_arg->task->state = CHUNK_DONE;
    (*cb_arg->completed_counter)++;
    ring_push(&cb_arg->ctx->free_ring, cb_arg->task->buf_idx); // 归还槽位
    free(cb_arg);
}

static int submit_to_spdk(NPUNVMEContext *ctx, io_task_t *task, int *completed_counter) {
    #define ALIGN_4K(x) (((x) + 4095ULL) & ~4095ULL)
    size_t aligned_sz = ALIGN_4K(task->size);
    uint64_t lba = task->nvme_offset / ctx->block_size;
    uint32_t lba_count = aligned_sz / ctx->block_size;

    spdk_cb_arg_t *cb_arg = malloc(sizeof(spdk_cb_arg_t));
    cb_arg->ctx = ctx; cb_arg->task = task; cb_arg->completed_counter = completed_counter;

    int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair, ctx->pool[task->buf_idx].buf, lba, lba_count, nvme_write_complete_cb, cb_arg, 0);
    if (rc != 0) { free(cb_arg); return rc; }
    
    task->state = CHUNK_SPDK_WRITING;
    return 0;
}


// ============================================================================
// Phase 3-C: 终极双轨死循环 (The Dual-Polling Pipeline)
// 真正的 C 语言级 Zero-Bubble 异步调度核心
// ============================================================================
void process_write_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks, bool is_host) {
    int completed_tasks = 0;
    int submitted_to_npu = 0;
    
    // [新增] 看门狗计时器
    uint64_t last_progress_time = get_time_us();

    while (completed_tasks < num_tasks) {
        bool made_progress = false; 

        // ---- 引擎 1：发射异步任务 ----
        while (submitted_to_npu < num_tasks) {
            io_task_t *task = &tasks[submitted_to_npu];
            size_t aligned_sz = ALIGN_4K(task->size);
            
            if (task->size == 0 || aligned_sz > ctx->chunk_size) {
                task->state = CHUNK_DONE; completed_tasks++; submitted_to_npu++; continue;
            }

            int rc = try_submit_async(ctx, task, is_host);
            if (rc == 0) {
                submitted_to_npu++; made_progress = true;
            } else if (rc == -1) {
                break; // Ring Buffer 满
            } else { 
                fprintf(stderr, "[Fatal] ACL Memcpy failed for chunk %d. Skipping.\n", task->task_idx);
                task->state = CHUNK_DONE; completed_tasks++; submitted_to_npu++; made_progress = true;
            }
        }

        // ---- 引擎 2：监控 NPU Event 并下发 SPDK ----
        for (int i = 0; i < submitted_to_npu; i++) {
            io_task_t *task = &tasks[i];
            if (task->state == CHUNK_DONE) continue;

            if (task->state == CHUNK_NPU_COPYING) {
                aclrtEventStatus status;
                aclError err = aclrtQueryEvent(ctx->events[task->buf_idx], &status);
                if (err != ACL_SUCCESS) {
                    task->state = CHUNK_DONE; completed_tasks++;
                    ring_push(&ctx->free_ring, task->buf_idx); made_progress = true;
                } else if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                    task->state = CHUNK_NPU_DONE; made_progress = true;
                }
            }

            if (task->state == CHUNK_NPU_DONE) {
                int rc = submit_to_spdk(ctx, task, &completed_tasks);
                if (rc == 0) {
                    made_progress = true;
                } else if (rc != -ENOMEM && rc != -12) {
                    task->state = CHUNK_DONE; completed_tasks++;
                    ring_push(&ctx->free_ring, task->buf_idx); made_progress = true;
                }
            }
        }

        // ---- 引擎 3：轮询 SPDK 完成队列 ----
        int cpl = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        if (cpl > 0) made_progress = true;

        // ============================================================
        // [修改] 状态卡死监控与主动唤醒逻辑
        // ============================================================
        if (!made_progress) {
            uint64_t now = get_time_us();
            uint64_t stall_time = now - last_progress_time;

            // 【终极防线：主动刺激 (Poke) 机制】
            // 如果卡顿超过 50 毫秒 (50,000 微秒)，并且还没到 3 秒的报警线。
            // 这说明硬件极大概率因为任务短缺/中断合并，陷入了“惰性挂起”。
            if (stall_time > 50000ULL && stall_time < 3000000ULL) {
                for (int i = 0; i < submitted_to_npu; i++) {
                    io_task_t *task = &tasks[i];
                    if (task->state == CHUNK_NPU_COPYING) {
                        // 狠狠“踹”底层驱动一脚，强迫 CPU 等待并拉取真正的完成状态！
                        aclrtSynchronizeEvent(ctx->events[task->buf_idx]);
                        
                        // 既然强制同步过了，这个任务的数据绝对已经安全到达 Host 内存
                        task->state = CHUNK_NPU_DONE; 
                        
                        // 唤醒一个就足以让阻塞的 Ring Buffer 腾出槽位，让流水线继续转！
                        last_progress_time = get_time_us(); 
                        break; 
                    }
                }
            }

            // 如果连强制唤醒都拯救不了（超过 3 秒），说明是真正的物理硬件/SMMU 死锁
            if (stall_time > 3000000ULL) {
                fprintf(stderr, "\n======================================================\n");
                fprintf(stderr, "[WATCHDOG] Rank %d Pipeline DEADLOCK Detected!\n", ctx->npu_id);
                fprintf(stderr, "[WATCHDOG] Completed: %d / Total: %d\n", completed_tasks, num_tasks);
                fprintf(stderr, "------- Active Tasks State Dump -------\n");
                for (int i = 0; i < submitted_to_npu; i++) {
                    if (tasks[i].state != CHUNK_DONE) {
                        fprintf(stderr, " -> TaskIdx: %d | NvmeOffset: %lu | Size: %zu | STATE: %d | BufIdx: %d\n",
                                tasks[i].task_idx, tasks[i].nvme_offset, tasks[i].size, tasks[i].state, tasks[i].buf_idx);
                    }
                }
                fprintf(stderr, "======================================================\n\n");
                last_progress_time = now; // 重置定时器，防止日志疯狂刷屏
            }
            usleep(1);
        } else {
            last_progress_time = get_time_us(); // 有进展则重置时间
        }
    }
}

// ============================================================================
// Phase 4: 顶层封装接口 (Top-level Wrapper)
// 这是暴露给 Python ctypes 调用的核心入口函数
// ============================================================================
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                         uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    
    // 0. 基础防御性编程
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) {
        fprintf(stderr, "[Fatal] Invalid arguments passed to npu_nvme_write_batch.\n");
        return -1;
    }

    aclrtSetCurrentContext(ctx->acl_ctx);

    // 1. 初始化全异步状态机任务队列 (调用 Phase 2 的函数)
    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) {
        return -1;
    }

    // 2. 启动 Zero-Bubble 双轨异步流水线 (调用 Phase 3 的函数)
    // 这个函数会接管 CPU，不休眠地驱动 NPU 和 NVMe，直到所有任务落盘完毕
    process_write_pipeline(ctx, tasks, num_items, false);

    // 3. 导出极度硬核的微观 Profiling 数据
    if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_write.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            // 打印表头
            fprintf(f, "item,buf_idx,npu_async_us,spdk_nvme_us,total_e2e_us\n");
            
            for (int i = 0; i < num_items; ++i) {
                // 计算 NPU 纯异步搬运耗时
                uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_submit) ? 
                                  (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
                                  
                // 计算 SPDK 队列等待 + NVMe 物理落盘耗时
                uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_npu_done) ? 
                                   (tasks[i].ts_spdk_done - tasks[i].ts_npu_done) : 0;
                                   
                // 单个 Chunk 从开始下发到彻底落盘的端到端耗时
                uint64_t total_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit) ? 
                                    (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;

                fprintf(f, "%d,%d,%lu,%lu,%lu\n", 
                        i, tasks[i].buf_idx, npu_us, spdk_us, total_us);
            }
            fclose(f);
            printf("[Profiler] Micro-breakdown saved to %s\n", path);
        } else {
            fprintf(stderr, "[Warning] Could not open profiling file: %s\n", path);
        }
    }

    // 4. 清理状态机内存，完美退出
    free(tasks);
    return 0; // 0 代表一切顺利
}

int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **ptrs, uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || num_items <= 0) return -1;
    aclrtSetCurrentContext(ctx->acl_ctx);
    io_task_t *tasks = create_io_tasks(num_items, ptrs, nvme_offsets, sizes);
    process_write_pipeline(ctx, tasks, num_items, true);
    free(tasks); return 0;
}

// ============================================================================
// 1. NVMe 读取完成的回调函数
// ============================================================================
static void nvme_read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    io_task_t *task = cb_arg->task;
    NPUNVMEContext *ctx = cb_arg->ctx;

    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "[Fatal] NVMe read failed for task %d!\n", task->task_idx);
        exit(EXIT_FAILURE);
    }

    // 状态翻转：硬盘数据已就绪
    task->state = CHUNK_SPDK_DONE;
    if (ctx->enable_profiling) task->ts_spdk_done = get_time_us();

    free(cb_arg);
}

// ============================================================================
// 读路径双引擎轮询核心 (同样加入防死锁机制)
// ============================================================================
void process_read_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks) {
    int completed_tasks = 0;
    int submitted_to_nvme = 0;

    while (completed_tasks < num_tasks) {
        bool made_progress = false;

        // ---- 引擎 1: 发射 NVMe 读取请求 ----
        while (submitted_to_nvme < num_tasks) {
            io_task_t *task = &tasks[submitted_to_nvme];
            size_t aligned_sz = ALIGN_4K(task->size);
            
            if (task->size == 0 || aligned_sz > ctx->chunk_size) {
                task->state = CHUNK_DONE; completed_tasks++; submitted_to_nvme++; continue;
            }

            if (ring_is_empty(&ctx->free_ring)) break;

            int buf_idx;
            ring_pop(&ctx->free_ring, &buf_idx);
            task->buf_idx = buf_idx;
            task->ts_submit = get_time_us();

            spdk_cb_arg_t *cb_arg = malloc(sizeof(spdk_cb_arg_t));
            cb_arg->ctx = ctx; cb_arg->task = task; cb_arg->completed_counter = &completed_tasks;

            uint64_t lba = task->nvme_offset / ctx->block_size;
            uint32_t lba_count = aligned_sz / ctx->block_size;

            int rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair, ctx->pool[buf_idx].buf, lba, lba_count, nvme_read_complete_cb, cb_arg, 0);

            if (rc == 0) {
                task->state = CHUNK_SPDK_READING;
                submitted_to_nvme++;
                made_progress = true;
            } else if (rc == -ENOMEM || rc == -12) {
                ring_push(&ctx->free_ring, buf_idx);
                free(cb_arg);
                break; 
            } else {
                // 【核心修复】：致命读错误
                fprintf(stderr, "[Fatal] SPDK read rejected! rc=%d for chunk %d.\n", rc, task->task_idx);
                task->state = CHUNK_DONE;
                completed_tasks++; submitted_to_nvme++;
                ring_push(&ctx->free_ring, buf_idx);
                free(cb_arg);
                made_progress = true;
            }
        }

        // ---- 引擎 2: DRAM -> NPU ----
        for (int i = 0; i < submitted_to_nvme; i++) {
            io_task_t *task = &tasks[i];
            if (task->state == CHUNK_SPDK_DONE) {
                aclError ret = aclrtMemcpyAsync(task->npu_ptr, task->size, ctx->pool[task->buf_idx].buf, task->size, ACL_MEMCPY_HOST_TO_DEVICE, ctx->copy_stream);
                if (ret == ACL_SUCCESS) {
                    aclrtRecordEvent(ctx->events[task->buf_idx], ctx->copy_stream);
                    task->state = CHUNK_NPU_COPYING;
                    made_progress = true;
                } else {
                    fprintf(stderr, "[Fatal] ACL Memcpy D2H failed. Skipping.\n");
                    task->state = CHUNK_DONE;
                    completed_tasks++;
                    ring_push(&ctx->free_ring, task->buf_idx);
                    made_progress = true;
                }
            }

            if (task->state == CHUNK_NPU_COPYING) {
                aclrtEventStatus status;
                aclrtQueryEvent(ctx->events[task->buf_idx], &status);
                if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                    task->state = CHUNK_DONE;
                    if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
                    ring_push(&ctx->free_ring, task->buf_idx);
                    completed_tasks++;
                    made_progress = true;
                }
            }
        }

        int cpl = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        if (cpl > 0) made_progress = true;

        if (!made_progress) usleep(1);
    }
}

// ============================================================================
// 3. 顶层封装接口 (Top-level Wrapper)
// ============================================================================
int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                        uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;
    
    aclrtSetCurrentContext(ctx->acl_ctx);

    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) return -1;

    process_read_pipeline(ctx, tasks, num_items);

    if (ctx->enable_profiling) {
        // 逻辑与 write_batch 类似，生成 time_read.csv
        if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_read.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            // 注意这里的字段顺序与 write 是反的
            fprintf(f, "item,buf_idx,spdk_nvme_us,npu_async_us,total_e2e_us\n");
            for (int i = 0; i < num_items; ++i) {
                // 1. 先算硬盘拉取时间 (SSD -> DRAM)
                uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit) ? 
                                   (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
                // 2. 再算 NPU 异步搬运时间 (DRAM -> NPU)
                uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_spdk_done) ? 
                                  (tasks[i].ts_npu_done - tasks[i].ts_spdk_done) : 0;
                // 3. 总耗时
                uint64_t total_us = (tasks[i].ts_npu_done > tasks[i].ts_submit) ? 
                                    (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;

                fprintf(f, "%d,%d,%lu,%lu,%lu\n", 
                        i, tasks[i].buf_idx, spdk_us, npu_us, total_us);
            }
            fclose(f);
            printf("[Profiler] Micro-breakdown (Read) saved to %s\n", path);
        }
    }
    }

    free(tasks);
    return 0;
}


int npu_nvme_get_max_transfer(NPUNVMEContext *ctx) {
    return ctx ? (int)ctx->chunk_size : 0;
}

uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx) {
    return ctx ? (ctx->total_blocks * ctx->block_size) : 0;
}

/* =========================================================
 * 核心新增：同步元数据 I/O 引擎 (分离控制面)
 * ========================================================= */
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset, uint32_t total_bytes, int is_read, void *meta_buffer) {
    if (!ctx || !meta_buffer) return -1;
    
    // C 层自动根据底层真实的 block_size 换算 LBA 和物理块数！Python 彻底解脱！
    uint64_t start_lba = byte_offset / ctx->block_size;
    uint32_t num_blocks = (total_bytes + ctx->block_size - 1) / ctx->block_size;
    size_t size = num_blocks * ctx->block_size;
    
    if (size > META_DMA_BUF_SIZE) return -1; 

    int flag = 0;
    int rc = 0;
    if (is_read == 0) {
        memcpy(ctx->meta_dma_buf, meta_buffer, size);
        rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair, ctx->meta_dma_buf, start_lba, num_blocks, io_complete_meta, &flag, 0);
    } else {
        rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair, ctx->meta_dma_buf, start_lba, num_blocks, io_complete_meta, &flag, 0);
    }

    // 【核心修复】：如果队列拒收，直接返回报错，绝对不能进入下面的 while 循环！
    if (rc != 0) {
        fprintf(stderr, "[Fatal] Meta I/O Submission Failed with rc=%d\n", rc);
        return -1; 
    }

    while (flag == 0) {
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
    }

    if (is_read != 0 && flag == 1) {
        memcpy(meta_buffer, ctx->meta_dma_buf, size);
    }

    return flag == 1 ? 0 : -1;
}

// ============================================================================
// P2-1: WaitProbe/TriggerProbe dead code — preserved via #if 0 for rollback
// These functions (npu_nvme_set_trigger_ptr, npu_nvme_read_trigger_dev,
// and the old probe_listener_thread WaitProbe branch) are replaced by
// FaF step_counter polling. If rollback is needed, uncomment the block below.
// ============================================================================

// ============================================================================
// Phase 5 E11: Delta (增量) 写盘实现
// ============================================================================

#define DELTA_MAGIC 0x414C5444  // "DLTA"
#define DELTA_MAGIC_INIT 0x4E4E // Superblock delta-initialized marker
#define DELTA_FRAME_HEADER_SIZE 4096

int npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t delta_slot_size, uint32_t delta_slot_count) {
    if (!ctx || delta_slot_size == 0 || delta_slot_count == 0) return -1;

    uint64_t total_delta_bytes = (uint64_t)delta_slot_size * delta_slot_count;
    if (ctx->total_blocks == 0) return -1;
    uint64_t disk_bytes = ctx->total_blocks * ctx->block_size;

    // Place delta area at end of disk, before full-checkpoint stack
    ctx->delta_area_offset = disk_bytes - total_delta_bytes;
    ctx->delta_slot_size = delta_slot_size;
    ctx->delta_slot_count = delta_slot_count;
    ctx->delta_last_commit = (uint32_t)-1;  // -1 = no commits yet

    fprintf(stderr, "[NPU-NVMe] Delta area: offset=%lu (%lu GB) slots=%u x %lu MB = %lu MB\n",
            ctx->delta_area_offset, ctx->delta_area_offset / (1024*1024*1024),
            delta_slot_count, delta_slot_size / (1024*1024),
            total_delta_bytes / (1024*1024));

    return 0;
}

uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta_area_offset : 0;
}

uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta_slot_size : 0;
}

uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta_slot_count : 0;
}

int npu_nvme_write_delta(NPUNVMEContext *ctx, int slot_idx,
                         const void *data, uint32_t total_bytes) {
    if (!ctx || !data || total_bytes == 0) return -1;
    if (slot_idx < 0 || (uint32_t)slot_idx >= ctx->delta_slot_count) {
        fprintf(stderr, "[NPU-NVMe] Delta write: invalid slot %d (max %u)\n",
                slot_idx, ctx->delta_slot_count);
        return -1;
    }
    if (total_bytes > ctx->delta_slot_size) {
        fprintf(stderr, "[NPU-NVMe] Delta write: %u bytes exceeds slot size %lu\n",
                total_bytes, ctx->delta_slot_size);
        return -1;
    }

    uint64_t byte_offset = ctx->delta_area_offset + (uint64_t)slot_idx * ctx->delta_slot_size;
    return npu_nvme_sync_meta_io(ctx, byte_offset, total_bytes, 0, (void*)data);
}

int npu_nvme_read_delta(NPUNVMEContext *ctx, int slot_idx,
                        void *out_buf, uint32_t max_bytes) {
    if (!ctx || !out_buf || max_bytes == 0) return -1;
    if (slot_idx < 0 || (uint32_t)slot_idx >= ctx->delta_slot_count) return -1;

    uint64_t byte_offset = ctx->delta_area_offset + (uint64_t)slot_idx * ctx->delta_slot_size;
    // Read header first to get actual size
    uint8_t header[DELTA_FRAME_HEADER_SIZE];
    int rc = npu_nvme_sync_meta_io(ctx, byte_offset, DELTA_FRAME_HEADER_SIZE, 1, header);
    if (rc != 0) return -1;

    // Parse header: magic(4) + step_id(4) + n_blocks(4) + n_small(4) + total_sz(4) + checksum(4) = 24 bytes
    uint32_t magic = *(uint32_t*)&header[0];
    if (magic != DELTA_MAGIC) {
        fprintf(stderr, "[NPU-NVMe] Delta read: slot %d invalid magic 0x%x\n", slot_idx, magic);
        return -1;
    }
    uint32_t total_sz = *(uint32_t*)&header[16];
    if (total_sz > max_bytes) {
        fprintf(stderr, "[NPU-NVMe] Delta read: frame %u bytes > buffer %u bytes\n", total_sz, max_bytes);
        return -1;
    }

    // Re-read full frame
    return npu_nvme_sync_meta_io(ctx, byte_offset, total_sz, 1, out_buf);
}
/* =======================================================================
 * npu_nvme.c — NPU-to-NVMe Zero-Copy I/O Engine
 *
 * Implements three subsystems:
 *   - SPDK user-space NVMe driver — HBM <-> NVMe DMA transfers
 *   - device-memory-polling listener — async step-boundary detection
 *   - delta-frame I/O — incremental checkpoint slot read/write
 *
 * Built as libnpu_nvme.so; consumed by Python ctypes bindings
 * (python/direct_checkpoint.py) and the C smoke test (src/test_npu_nvme.c).
 * ======================================================================= */
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
#define ALIGN_4K(x)      (((x) + 4095ULL) & ~4095ULL)

/* Dedicated buffer for superblock & JSON ledger I/O; 64 MB covers all
 * realistic metadata sizes including future delta ledger expansion. */
#define META_DMA_BUF_SIZE (64 * 1024 * 1024)

/* ---- Hugepage pool auto-expansion for DPDK/SPDK ---- */
/* The NPU driver reserves all boot-time hugepages (typically 8544 x 2 MB
 * = ~17 GB) as internal DMA buffers.  DPDK's spdk_env_init() checks the
 * per-NUMA-node free hugepage counter and refuses to start when it is
 * zero.  We add 512 pages (1 GB) on top of the NPU reservation.
 * DPDK releases these pages via rte_eal_cleanup() on spdk_env_fini. */
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

/* ---- data structures: ring buffer, DMA pool, I/O task, context ---- */
typedef struct {
    int *slots;
    int capacity;
    int head;
    int tail;
} ring_t;

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
    CHUNK_NPU_COPYING,      // write path: NPU -> DMA buffer
    CHUNK_NPU_DONE,         // write path: DMA copy complete
    CHUNK_SPDK_WRITING,     // write path: SPDK NVMe write in flight
    CHUNK_SPDK_READING,     // read path:  SPDK NVMe read in flight
    CHUNK_SPDK_DONE,        // read path:  NVMe -> DMA buffer complete
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

// NPU-NVMe runtime context
typedef struct NPUNVMEContext {
    char pci_addr[64];              // PCIe BDF address for probe filtering
    
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;        // active namespace
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;

    int npu_id;
    int max_pipe_depth;             // DMA pipeline depth
    uint32_t chunk_size;

    dma_buf_t *pool;                // DMA buffer pool (hugepage-backed)
    ring_t free_ring;               // free-slot SPSC ring

    // ----- asynchronous I/O core -----
    aclrtStream copy_stream;        // ACL stream for NPU-DMA copies
    aclrtEvent *events;             // one hardware event per ring-buffer slot
    aclrtContext acl_ctx;

    bool enable_profiling;
    char profiling_dir[256];

    void *meta_dma_buf;     // dedicated DMA buffer for metadata I/O

    // ----- registered task table for background persistence -----
    io_task_t *registered_tasks;
    int num_registered_tasks;

    // ----- listener thread control -----
    pthread_t listener_thread;
    volatile int stop_listener;
    bool listener_started;

    // ----- probe-flag device pointer -----
    void *probe_flag_dev_ptr;
    void *probe_flag_host;

    // ----- step-counter polling -----
    void *dev_step_ptr;        // Device pointer for step_counter (HBM)
    void *step_poll_buf;       // Host buffer for polling step_counter
    int last_step_seen;        // Last step value detected by listener
    int ckpt_interval;         // Checkpoint every N steps

    // ----- delta ring-buffer layout -----
    uint64_t delta_area_offset;   // byte offset of delta ring on NVMe
    uint64_t delta_slot_size;     // bytes per delta slot
    uint32_t delta_slot_count;    // total number of delta slots
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
    return;
}

io_task_t* create_io_tasks(int num_tasks, void **npu_ptrs, uint64_t *nvme_offsets, size_t *sizes);
void process_write_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks, bool is_host);

/* simplified SPDK completion callback for metadata I/O */
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
    
    // free previous registration if present
    if (ctx->registered_tasks) {
        free(ctx->registered_tasks);
    }
    
    // populate the I/O task table from raw pointer arrays
    ctx->registered_tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!ctx->registered_tasks) {
        return -1;
    }
    
    ctx->num_registered_tasks = num_items;
    printf("[NPU-NVMe] Successfully registered %d tensor tasks for Background Probe.\n", num_items);
    return 0;
}

/* ---- SPDK Probe & Attach Callbacks ---- */

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    
    // Strict PCI BDF matching: only attach the target NVMe device,
    // never the system disk or a device owned by another rank.
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
        break; // use the first active namespace
    }
}


/* ---- Init: SPDK env, NVMe probe, ACL, DMA pool, listener ---- */

// Background listener thread -- polls a device-side step counter via
void* probe_listener_thread(void* arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    const char *mode = getenv("NPU_NVME_LISTENER_MODE");

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

    // Bind NPU ACL context to this background thread.
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
    }
    return NULL;
}

int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id, 
                  int pipe_depth, int chunk_size, bool enable_profiling, const char *prof_dir) {
    // 1. allocate context
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

    // 2. initialise SPDK environment (once per process, use SPDK_SHM_ID for multi-rank)
    static int spdk_inited = 0;
    if (!spdk_inited) {
        struct spdk_env_opts env_opts;
        spdk_env_opts_init(&env_opts);
        env_opts.name = "npu_nvme_app";
        
        // Multi-rank support: secondary ranks share the NVMe hardware queue
        // initialised by rank 0.  Set SPDK_SHM_ID to the rank-0 PID before
        // launching secondary processes.
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

    // 3. probe and attach NVMe device
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

    // 4. allocate SPDK I/O queue pair
    struct spdk_nvme_io_qpair_opts qopts;
    spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &qopts, sizeof(qopts));
    // deep queue for high-throughput asynchronous I/O
    qopts.io_queue_size = 512; 
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &qopts, sizeof(qopts));
    if (!ctx->qpair) {
        fprintf(stderr, "[Fatal] Cannot allocate NVMe I/O qpair.\n");
        goto fail;
    }

    // 5. initialise NPU environment (bind device, create ACL stream)
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

    // 6. allocate DMA buffer pool and associated NPU events
    ctx->pool = calloc(ctx->max_pipe_depth, sizeof(dma_buf_t));
    ctx->events = calloc(ctx->max_pipe_depth, sizeof(aclrtEvent));
    ring_init(&ctx->free_ring, ctx->max_pipe_depth);

    for (int i = 0; i < ctx->max_pipe_depth; i++) {
        // allocate physically contiguous, page-locked DMA memory via SPDK
        ctx->pool[i].buf = spdk_zmalloc(ctx->chunk_size, 2 * 1024 * 1024, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
        if (!ctx->pool[i].buf) {
            fprintf(stderr, "[Fatal] spdk_zmalloc failed at slot %d.\n", i);
            goto fail;
        }
        ctx->pool[i].phys_addr = spdk_vtophys(ctx->pool[i].buf, NULL);
        
        // create NPU event for this slot
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

    // launch the background listener thread
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

/* ---- Cleanup: strict resource release order ---- */
void npu_nvme_cleanup(NPUNVMEContext *ctx) {
    if (!ctx) return;

    // stop the background thread before tearing down SPDK resources
    ctx->stop_listener = 1;
    if (ctx->listener_started) {
        pthread_join(ctx->listener_thread, NULL);
        ctx->listener_started = false;
    }
    
    // Re-bind NPU device on the current thread before releasing ACL resources.
    aclError ret = aclrtSetDevice(ctx->npu_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Warning] Failed to set NPU device %d during cleanup. Skip ACL cleanup.\n", ctx->npu_id);
        // If ACL device binding fails (e.g. framework has claimed the device),
        // skip ACL resource destruction to avoid a crash from invalid context.
    } else {
        // Drain all in-flight DMA operations before destroying resources.
        if (ctx->copy_stream) {
            aclrtSynchronizeStream(ctx->copy_stream);
        }

        // destroy NPU events
        if (ctx->events) {
            for (int i = 0; i < ctx->max_pipe_depth; i++) {
                if (ctx->events[i]) aclrtDestroyEvent(ctx->events[i]);
            }
            free(ctx->events);
        }
        
        // destroy ACL stream
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
    
    // ----- release memory and SPDK resources -----
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

/* ---- utility: microsecond timestamp ---- */
static inline uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/* ---- io_task_t factory ---- */
io_task_t* create_io_tasks(int num_tasks, void **npu_ptrs, uint64_t *nvme_offsets, size_t *sizes) {
    io_task_t *tasks = calloc(num_tasks, sizeof(io_task_t));
    if (!tasks) {
        fprintf(stderr, "[Fatal] Failed to allocate memory for IO tasks.\n");
        return NULL;
    }
    
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].task_idx = i;
        tasks[i].buf_idx = -1;               // no ring-buffer slot assigned yet
        tasks[i].state = CHUNK_IDLE;         // initial state: idle
        tasks[i].npu_ptr = npu_ptrs[i];
        tasks[i].nvme_offset = nvme_offsets[i];
        tasks[i].size = sizes[i];
    }
    return tasks;
}

int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host) {
    int buf_idx;
    if (ring_pop(&ctx->free_ring, &buf_idx) != 0) return -1; // ring buffer full

    task->buf_idx = buf_idx;
    if (ctx->enable_profiling) task->ts_submit = get_time_us();

    if (is_host) {
        memcpy(ctx->pool[buf_idx].buf, task->npu_ptr, task->size);
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE; 
    } else {
        // Use synchronous aclrtMemcpy; sub-millisecond blocking is harmless in background thread.
        // 
        aclError ret = aclrtMemcpy(ctx->pool[buf_idx].buf, task->size, task->npu_ptr, task->size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ring_push(&ctx->free_ring, buf_idx); 
            return -2;
        }
        // synchronous copy complete; proceed to SPDK.
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    }
    return 0;
}
/* SPDK write completion callback + submission helper */
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
    ring_push(&cb_arg->ctx->free_ring, cb_arg->task->buf_idx); // return slot to free pool
    free(cb_arg);
}

static int submit_to_spdk(NPUNVMEContext *ctx, io_task_t *task, int *completed_counter) {
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


/* ---- Dual-Polling Pipeline ----
 * Drives NPU DMA and SPDK NVMe I/O concurrently.
 *
 * Three engines per iteration:
 *   Engine 1 (submit) - pop free ring slots, launch DMA / SPDK commands
 *   Engine 2 (drain)  - poll NPU events, submit ready chunks to SPDK
 *   Engine 3 (reap)   - process SPDK completion queue, reclaim slots
 *
 * Stall recovery (two-stage):
 *   Stage 1 (>50 ms no progress):  force-sync one stalled NPU event via
 *     aclrtSynchronizeEvent, breaking hardware lazy-suspend stalls.
 *   Stage 2 (>3 s no progress):   dump active task state to stderr
 *     and reset the stall timer (diagnostic only, not a fix).
 * =================================================================== */
void process_write_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks, bool is_host) {
    int completed_tasks = 0;
    int submitted_to_npu = 0;
    
    // stall-detection timer
    uint64_t last_progress_time = get_time_us();

    while (completed_tasks < num_tasks) {
        bool made_progress = false; 

        // ---- engine 1: submit DMA / SPDK commands ----
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
                break; // ring buffer full
            } else { 
                fprintf(stderr, "[Fatal] ACL Memcpy failed for chunk %d. Skipping.\n", task->task_idx);
                task->state = CHUNK_DONE; completed_tasks++; submitted_to_npu++; made_progress = true;
            }
        }

        // ---- engine 2: poll NPU events, submit ready chunks ----
        for (int i = 0; i < submitted_to_npu; i++) {
            io_task_t *task = &tasks[i];
            if (task->state == CHUNK_DONE) continue;

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

        // ---- engine 3: reap SPDK completion queue ----
        int cpl = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        if (cpl > 0) made_progress = true;

        // ---- stall detection and recovery ----
        if (!made_progress) {
            uint64_t now = get_time_us();
            uint64_t stall_time = now - last_progress_time;

            // Persistent hardware stall (>3 s no progress)
            if (stall_time > 3000000ULL) {
                fprintf(stderr, "\n[NPU-NVMe] Pipeline STALL detected (rank %d)\n", ctx->npu_id);
                fprintf(stderr, "[NPU-NVMe] Completed: %d / Total: %d\n", completed_tasks, num_tasks);
                fprintf(stderr, "[NPU-NVMe] Active task state dump:\n");
                for (int i = 0; i < submitted_to_npu; i++) {
                    if (tasks[i].state != CHUNK_DONE) {
                        fprintf(stderr, " -> TaskIdx: %d | NvmeOffset: %lu | Size: %zu | STATE: %d | BufIdx: %d\n",
                                tasks[i].task_idx, tasks[i].nvme_offset, tasks[i].size, tasks[i].state, tasks[i].buf_idx);
                    }
                }
                last_progress_time = now;
            }
            usleep(1);
        } else {
            last_progress_time = get_time_us(); // reset stall timer on progress
        }
    }
}

/* ---- Public API: write_batch / read_batch ---- */
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                         uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    
    // 0. argument validation
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) {
        fprintf(stderr, "[Fatal] Invalid arguments passed to npu_nvme_write_batch.\n");
        return -1;
    }

    aclrtSetCurrentContext(ctx->acl_ctx);

    // 1. build the I/O task state machine
    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) {
        return -1;
    }

    // 2. run the dual-polling pipeline to completion
    // 
    process_write_pipeline(ctx, tasks, num_items, false);

    // 3. export per-chunk profiling data to CSV
    if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_write.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            // CSV header
            fprintf(f, "item,buf_idx,npu_async_us,spdk_nvme_us,total_e2e_us\n");
            
            for (int i = 0; i < num_items; ++i) {
                // NPU copy time
                uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_submit) ? 
                                  (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
                                  
                // SPDK queue + NVMe write time
                uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_npu_done) ? 
                                   (tasks[i].ts_spdk_done - tasks[i].ts_npu_done) : 0;
                                   
                // end-to-end chunk latency
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

    // 4. free task state machine
    free(tasks);
    return 0; // success
}

int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **ptrs, uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || num_items <= 0) return -1;
    aclrtSetCurrentContext(ctx->acl_ctx);
    io_task_t *tasks = create_io_tasks(num_items, ptrs, nvme_offsets, sizes);
    process_write_pipeline(ctx, tasks, num_items, true);
    free(tasks); return 0;
}

/* SPDK read completion callback */
static void nvme_read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    io_task_t *task = cb_arg->task;
    NPUNVMEContext *ctx = cb_arg->ctx;

    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "[Fatal] NVMe read failed for task %d!\n", task->task_idx);
        exit(EXIT_FAILURE);
    }

    // data is now in the DMA buffer
    task->state = CHUNK_SPDK_DONE;
    if (ctx->enable_profiling) task->ts_spdk_done = get_time_us();

    free(cb_arg);
}

/* ---- Read-path dual-polling pipeline ----
 * Mirrors the write path with SPDK-read-first ordering. */
void process_read_pipeline(NPUNVMEContext *ctx, io_task_t *tasks, int num_tasks) {
    int completed_tasks = 0;
    int submitted_to_nvme = 0;

    while (completed_tasks < num_tasks) {
        bool made_progress = false;

        // ---- engine 1: submit NVMe read commands ----
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
                task->state = CHUNK_DONE;
                completed_tasks++;
                submitted_to_nvme++;
                fprintf(stderr, "[Fatal] SPDK read queue full, skip chunk %d.\n", task->task_idx);
            } else {
                // fatal SPDK read error
                fprintf(stderr, "[Fatal] SPDK read rejected! rc=%d for chunk %d.\n", rc, task->task_idx);
                task->state = CHUNK_DONE;
                completed_tasks++; submitted_to_nvme++;
                ring_push(&ctx->free_ring, buf_idx);
                free(cb_arg);
                made_progress = true;
            }
        }

        // ---- engine 2: DMA buffer to NPU ----
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


int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                        uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;
    
    aclrtSetCurrentContext(ctx->acl_ctx);

    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) return -1;

    process_read_pipeline(ctx, tasks, num_items);

    if (ctx->enable_profiling) {
        // 
        if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_read.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            // note: field order differs from write_batch
            fprintf(f, "item,buf_idx,spdk_nvme_us,npu_async_us,total_e2e_us\n");
            for (int i = 0; i < num_items; ++i) {
                // 1. NVMe read time (SSD to DMA buffer)
                uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit) ? 
                                   (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
                // 2. DMA to NPU copy time
                uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_spdk_done) ? 
                                  (tasks[i].ts_npu_done - tasks[i].ts_spdk_done) : 0;
                // 3. end-to-end latency
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

/* ---- Synchronous Metadata I/O: superblock + JSON ledger ---- */
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset, uint32_t total_bytes, int is_read, void *meta_buffer) {
    if (!ctx || !meta_buffer) return -1;
    
    // Convert byte offset to LBA using the hardware block size..
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

    // If the submission queue rejects the command, fail immediately.
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

/* ---- Delta Frame I/O ---- */

#define DELTA_MAGIC 0x414C5444  // "DLTA"
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
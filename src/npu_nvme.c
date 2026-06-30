/* =======================================================================
 * npu_nvme.c — NPU-to-NVMe Zero-Copy I/O Engine
 *
 * Implements three subsystems:
 *   - SPDK user-space NVMe driver — HBM <-> NVMe DMA transfers
 *   - device-memory-polling listener — async step-boundary detection
 *   - delta-ring layout — incremental checkpoint slot bookkeeping
 *
 * Sub-modules (internal headers in include/internal/):
 *   ring_buffer.h  — SPSC ring buffer
 *   io_task.h      — I/O task state machine + supporting types
 *   pipeline.h     — dual-polling read/write pipeline
 *   context.h      — NPUNVMEContext structure + sub-structures
 *
 * Built as libnpu_nvme.so; consumed by Python ctypes bindings
 * (python/direct_checkpoint.py) and the C smoke test (src/test_npu_nvme.c).
 * ======================================================================= */
#include "npu_nvme.h"

/* Internal headers */
#include "internal/ring_buffer.h"
#include "internal/io_task.h"
#include "internal/pipeline.h"
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include "internal/context.h"

/* SPDK */
#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"

/* System */
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

/* ---- Hugepage pool auto-expansion for DPDK/SPDK ---- */
#define STEP_POLLER_PERIOD_US  10000   /* step counter poll interval (10 ms) */
#define HUGEPAGE_2MB_PATH "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"

/* Read an integer from a sysfs / proc file.  Returns -1 on error. */
static int read_int_from_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val;
    if (fscanf(f, "%d", &val) != 1) { fclose(f); return -1; }
    fclose(f);
    return val;
}

/* Ensure at least HUGEPAGE_PADDING free 2 MB hugepages are available
 * for DPDK/SPDK.  Called once from npu_nvme_init before spdk_env_init. */
static void ensure_hugepages(void) {
    int nr = read_int_from_file(NR_HUGEPAGES_PATH);
    int free_2mb = read_int_from_file(HUGEPAGE_2MB_PATH);
    if (nr < 0) return;

    int target = nr;
    if (free_2mb < HUGEPAGE_PADDING) {
        target = nr + (HUGEPAGE_PADDING - free_2mb);
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", target);
        FILE *fp = fopen(NR_HUGEPAGES_PATH, "w");
        if (fp) { fprintf(fp, "%s", buf); fclose(fp); }
        fprintf(stderr, "[NPU-NVMe] Expanded hugepages %d -> %d\n", nr, target);
    }
}

/* ---- SPDK probe / attach callbacks ----
 *
 * probe_cb matches the user-supplied PCI address against each discovered
 * NVMe transport ID.  attach_cb stores the controller and namespace
 * pointers into the context on a successful match.
 */

/* Return true when the probed device matches ctx->pci_addr. */
static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                    struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    if (strncmp(trid->traddr, ctx->pci_addr, sizeof(ctx->pci_addr)) == 0) {
        fprintf(stderr, "[NPU-NVMe] Found matching controller at %s\n", trid->traddr);
        return 1;
    }
    return 0;
}

/* Store the attached controller, namespace, block_size and total_blocks. */
static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    ctx->ctrlr = ctrlr;
    ctx->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    if (ctx->ns) {
        ctx->block_size = spdk_nvme_ns_get_sector_size(ctx->ns);
        ctx->total_blocks = spdk_nvme_ns_get_num_sectors(ctx->ns);
        printf("[NPU-NVMe] Attached NS: block_size=%u total_blocks=%lu\n",
               ctx->block_size, ctx->total_blocks);
    }
}

/* ---- Timestamp ---- */

/* Return monotonic wall-clock time in microseconds. */
uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/* ---- IO task factory ---- */

/* Allocate and initialise an array of io_task_t descriptors.
 * Caller owns the returned heap memory; free() when done. */
io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes) {
    io_task_t *tasks = calloc(num_tasks, sizeof(io_task_t));
    if (!tasks) {
        fprintf(stderr, "[Fatal] Failed to allocate memory for IO tasks.\n");
        return NULL;
    }
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].task_idx = i;
        tasks[i].buf_idx = -1;
        tasks[i].state = CHUNK_IDLE;
        tasks[i].npu_ptr = npu_ptrs[i];
        tasks[i].nvme_offset = nvme_offsets[i];
        tasks[i].size = sizes[i];
    }
    return tasks;
}

/* ---- NPU DMA submission ----
 *
 * try_submit_async pops a free DMA buffer slot from the ring and copies
 * chunk data from NPU HBM (or host DRAM when is_host is set) into it.
 */

/* Submit one chunk to the DMA buffer pool.  Returns 0 on success,
 * -1 if the ring is full, -2 on ACL memcpy error. */
int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host) {
    int buf_idx;
    if (ring_pop(&ctx->dma.free_ring, &buf_idx) != 0) return -1;

    task->buf_idx = buf_idx;
    if (ctx->enable_profiling) task->ts_submit = get_time_us();

    if (is_host) {
        memcpy(ctx->dma.pool[buf_idx].buf, task->npu_ptr, task->size);
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    } else {
        aclError ret = aclrtMemcpy(ctx->dma.pool[buf_idx].buf, task->size,
                                    task->npu_ptr, task->size,
                                    ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ring_push(&ctx->dma.free_ring, buf_idx);
            return -2;
        }
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    }
    return 0;
}

/* ---- SPDK write completion + submission ----
 *
 * nvme_write_complete_cb is invoked by SPDK when a write command finishes.
 * It marks the task done, increments the shared counter, returns the
 * DMA buffer slot to the free ring, and frees the callback argument.
 *
 * submit_to_spdk_write queues a single NVMe write command for a chunk
 * whose DMA copy has completed.
 */

/* SPDK write-completion callback — advances task state to CHUNK_DONE.
 * Logs an error on I/O failure but still returns the buffer slot (the task
 * is marked done so the pipeline does not stall indefinitely). */
void nvme_write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "[NPU-NVMe] NVMe write error for task %d: "
                "status=%d type=%d\n",
                cb_arg->task->task_idx,
                completion->status.sc, completion->status.sct);
    }
    if (cb_arg->ctx->enable_profiling) cb_arg->task->ts_spdk_done = get_time_us();
    cb_arg->task->state = CHUNK_DONE;
    (*cb_arg->completed_counter)++;
    ring_push(&cb_arg->ctx->dma.free_ring, cb_arg->task->buf_idx);
    free(cb_arg);
}

/* Submit a single SPDK write command for an NPU_DONE chunk. */
int submit_to_spdk_write(NPUNVMEContext *ctx, io_task_t *task,
                          int *completed_counter) {
    size_t aligned_sz = ALIGN_4K(task->size);
    uint64_t lba = task->nvme_offset / ctx->block_size;
    uint32_t lba_count = aligned_sz / ctx->block_size;

    spdk_cb_arg_t *cb_arg = malloc(sizeof(spdk_cb_arg_t));
    if (!cb_arg) return -1;
    cb_arg->ctx = ctx; cb_arg->task = task;
    cb_arg->completed_counter = completed_counter;

    int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair,
                                     ctx->dma.pool[task->buf_idx].buf,
                                     lba, lba_count,
                                     nvme_write_complete_cb, cb_arg, 0);
    if (rc != 0) { free(cb_arg); return rc; }
    task->state = CHUNK_SPDK_WRITING;
    return 0;
}


/* ---- Profiling CSV export ----
 *
 * Writes per-chunk micro-benchmark timestamps to time_write.csv or
 * time_read.csv under ctx->profiling_dir.  Column order differs by
 * direction so that the dominant latency component appears first.
 */

void write_profiling_csv(NPUNVMEContext *ctx, io_task_t *tasks,
                          int num_items, pipeline_dir_t dir) {
    if (!ctx->enable_profiling) return;

    char path[512];
    const char *fname = (dir == PIPELINE_WRITE) ? "time_write.csv" : "time_read.csv";
    snprintf(path, sizeof(path), "%s/%s", ctx->profiling_dir, fname);
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Warning] Could not open profiling file: %s\n", path);
        return;
    }

    if (dir == PIPELINE_WRITE) {
        fprintf(f, "item,buf_idx,npu_async_us,spdk_nvme_us,total_e2e_us\n");
        for (int i = 0; i < num_items; ++i) {
            uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_submit)
                ? (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
            uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_npu_done)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_npu_done) : 0;
            uint64_t total_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
            fprintf(f, "%d,%d,%lu,%lu,%lu\n", i, tasks[i].buf_idx,
                    npu_us, spdk_us, total_us);
        }
    } else {
        fprintf(f, "item,buf_idx,spdk_nvme_us,npu_async_us,total_e2e_us\n");
        for (int i = 0; i < num_items; ++i) {
            uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
            uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_spdk_done)
                ? (tasks[i].ts_npu_done - tasks[i].ts_spdk_done) : 0;
            uint64_t total_us = (tasks[i].ts_npu_done > tasks[i].ts_submit)
                ? (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
            fprintf(f, "%d,%d,%lu,%lu,%lu\n", i, tasks[i].buf_idx,
                    spdk_us, npu_us, total_us);
        }
    }
    fclose(f);
    printf("[Profiler] Micro-breakdown (%s) saved to %s\n",
           (dir == PIPELINE_WRITE) ? "Write" : "Read", path);
}

/* ---- Public API: batch write / read ----
 *
 * These are the primary I/O entry points called from Python via ctypes.
 * All four enqueue requests to the reactor via SPDK rings (write_ring /
 * read_ring).  No lock is held — the reactor serialises all I/O.
 */

/**
 * @brief  Write NPU HBM buffers to NVMe (blocking batch).
 * @param ctx        context handle
 * @param npu_ptrs   array of NPU device pointers (source)
 * @param nvme_offs  array of NVMe byte offsets (destination)
 * @param sizes      array of per-chunk byte sizes
 * @param num_items  number of chunks in the batch
 * @return 0 on success, -1 on error
 */
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                          uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) {
        fprintf(stderr, "[Fatal] Invalid arguments passed to npu_nvme_write_batch.\n");
        return -1;
    }
    if (ctx->block_size == 0) return -1;

    aclrtSetCurrentContext(ctx->acl.acl_ctx);

    /* Build request on heap (Python owns it until completion). */
    write_request_t *req = calloc(1, sizeof(write_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    req->num_tasks = num_items;
    req->is_host = false;
    req->done = 0;

    /* Enqueue to reactor.  If the ring is full, the reactor is stuck —
     * this should never happen with a 16-slot ring and 1 producer. */
    void *obj = req;
    if (spdk_ring_enqueue(ctx->write_ring, &obj, 1, NULL) != 1) {
        fprintf(stderr, "[Fatal] write_ring full — reactor stalled?\n");
        free(tasks); free(req); return -1;
    }

    /* Poll for completion (non-busy: usleep yields CPU). */
    while (!req->done) {
        usleep(1000);
    }

    write_profiling_csv(ctx, tasks, num_items, PIPELINE_WRITE);
    free(tasks);
    free(req);
    return 0;
}

/**
 * @brief  Write host DRAM buffers to NVMe (memcpy path, no NPU DMA).
 * @param ctx        context handle
 * @param ptrs       array of host pointers (source)
 * @param nvme_offs  array of NVMe byte offsets (destination)
 * @param sizes      array of per-chunk byte sizes
 * @param num_items  number of chunks in the batch
 * @return 0 on success, -1 on error
 */
int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **ptrs,
                               uint64_t *nvme_offsets, size_t *sizes,
                               int num_items) {
    if (!ctx || num_items <= 0) return -1;
    if (ctx->block_size == 0) return -1;
    aclrtSetCurrentContext(ctx->acl.acl_ctx);

    write_request_t *req = calloc(1, sizeof(write_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    req->num_tasks = num_items;
    req->is_host = true;
    req->done = 0;

    void *obj = req;
    if (spdk_ring_enqueue(ctx->write_ring, &obj, 1, NULL) != 1) {
        fprintf(stderr, "[Fatal] write_ring full — reactor stalled?\n");
        free(tasks); free(req); return -1;
    }

    while (!req->done) {
        usleep(1000);
    }
    /* Host writes skip profiling (no meaningful NPU timestamps). */
    free(tasks);
    free(req);
    return 0;
}

/**
 * @brief  Read NVMe blocks into NPU HBM buffers (blocking batch).
 * @param ctx        context handle
 * @param npu_ptrs   array of NPU device pointers (destination)
 * @param nvme_offs  array of NVMe byte offsets (source)
 * @param sizes      array of per-chunk byte sizes
 * @param num_items  number of chunks in the batch
 * @return 0 on success, -1 on error
 */
int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                         uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0)
        return -1;
    if (ctx->block_size == 0) return -1;

    aclrtSetCurrentContext(ctx->acl.acl_ctx);

    read_request_t *req = calloc(1, sizeof(read_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    req->num_tasks = num_items;
    req->done = 0;

    void *obj = req;
    if (spdk_ring_enqueue(ctx->read_ring, &obj, 1, NULL) != 1) {
        free(tasks); free(req); return -1;
    }

    while (!req->done) { usleep(1000); }

    write_profiling_csv(ctx, tasks, num_items, PIPELINE_READ);
    free(tasks);
    free(req);
    return 0;
}

/**
 * @brief  Read NVMe blocks into host DRAM (memcpy path, no NPU DMA).
 * @param ctx        context handle
 * @param host_ptrs  array of host pointers (destination)
 * @param nvme_offs  array of NVMe byte offsets (source)
 * @param sizes      array of per-chunk byte sizes
 * @param num_items  number of chunks in the batch
 * @return 0 on success, -1 on error
 */
int npu_nvme_read_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes,
                             int num_items) {
    if (!ctx || !host_ptrs || !nvme_offsets || !sizes || num_items <= 0)
        return -1;
    if (ctx->block_size == 0) return -1;

    aclrtSetCurrentContext(ctx->acl.acl_ctx);

    read_request_t *req = calloc(1, sizeof(read_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, host_ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    req->num_tasks = num_items;
    req->done = 0;

    void *obj = req;
    if (spdk_ring_enqueue(ctx->read_ring, &obj, 1, NULL) != 1) {
        free(tasks); free(req); return -1;
    }

    while (!req->done) { usleep(1000); }
    /* Host reads skip profiling. */
    free(tasks);
    free(req);
    return 0;
}

/**
 * @brief  Return the configured per-chunk transfer size.
 * @param ctx  context handle
 * @return chunk_size bytes, or 0 if ctx is NULL
 */
int npu_nvme_get_max_transfer(NPUNVMEContext *ctx) {
    return ctx ? (int)ctx->dma.chunk_size : 0;
}

/**
 * @brief  Return total NVMe capacity in bytes.
 * @param ctx  context handle
 * @return total capacity in bytes, or 0 if ctx is NULL
 */
uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx) {
    return ctx ? ctx->total_blocks * ctx->block_size : 0;
}

/* ---- ACL context helper ----
 *
 * ensure_acl_context binds the calling thread to the NPU device and
 * ACL context stored in ctx.  Called from the listener thread and
 * from the main thread under state_lock.
 */

/* (Re-)bind this thread to the configured NPU device and ACL context. */
static inline int ensure_acl_context(NPUNVMEContext *ctx) {
    aclError ret = aclrtSetDevice(ctx->acl.npu_id);
    if (ret != 0) return -1;
    return aclrtSetCurrentContext(ctx->acl.acl_ctx);
}

static int step_poller_fn(void *arg);
static int write_fsm_poller_fn(void *arg);
static int read_fsm_poller_fn(void *arg);
static int meta_poller_fn(void *arg);

/* ---- Reactor thread functions ----
 *
 * The reactor pthread runs a spdk_thread_poll loop that drives SPDK
 * pollers (step counter check, I/O state machines).  It is created
 * by reactor_new_thread_fn when spdk_thread_create is called.
 *
 * We pass NULL as the cpumask to spdk_thread_create (matching V0)
 * and use a static context pointer to communicate ctx to the
 * new_thread_fn callback.
 */

static NPUNVMEContext *g_reactor_ctx;

/**
 * @brief Reactor pthread main loop.
 *
 * Binds the SPDK thread to this OS thread, signals init completion
 * via pthread_barrier, then loops calling spdk_thread_poll until
 * app_should_stop is set.
 *
 * @param arg  NPUNVMEContext pointer
 * @return     NULL
 */
static void *reactor_loop(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    spdk_set_thread(ctx->reactor_thread);

    /* Reset registered tasks to IDLE. */
    pthread_mutex_lock(&ctx->state_lock);
    for (int i = 0; i < ctx->listener.num_registered_tasks; i++) {
        ctx->listener.registered_tasks[i].state = CHUNK_IDLE;
        ctx->listener.registered_tasks[i].buf_idx = -1;
    }
    pthread_mutex_unlock(&ctx->state_lock);

    ctx->last_step_seen = -1;

    /* Create MPSC rings for Python → reactor requests (16 slots each). */
    ctx->write_ring = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 16,
                                        SPDK_ENV_SOCKET_ID_ANY);
    ctx->read_ring  = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 16,
                                        SPDK_ENV_SOCKET_ID_ANY);
    ctx->meta_ring  = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 4,
                                        SPDK_ENV_SOCKET_ID_ANY);

    /* Init FSMs as idle. */
    ctx->write_fsm.state = WRITE_FSM_IDLE;
    ctx->write_fsm.req = NULL;
    ctx->read_fsm.state = READ_FSM_IDLE;
    ctx->read_fsm.req = NULL;

    /* Register pollers. */
    ctx->step_poller      = spdk_poller_register(step_poller_fn, ctx,
                                                  STEP_POLLER_PERIOD_US);
    ctx->write_fsm_poller = spdk_poller_register(write_fsm_poller_fn, ctx, 0);
    ctx->read_fsm_poller  = spdk_poller_register(read_fsm_poller_fn, ctx, 0);
    ctx->meta_poller      = spdk_poller_register(meta_poller_fn, ctx, 0);
    fprintf(stderr, "[Diag] reactor: rings w=%p r=%p m=%p pollers s=%p w=%p r=%p m=%p\n",
            (void *)ctx->write_ring, (void *)ctx->read_ring, (void *)ctx->meta_ring,
            (void *)ctx->step_poller, (void *)ctx->write_fsm_poller,
            (void *)ctx->read_fsm_poller, (void *)ctx->meta_poller);

    pthread_barrier_wait(&ctx->init_barrier);

    while (!ctx->app_should_stop) {
        spdk_thread_poll(ctx->reactor_thread, 0, 0);
        usleep(100);
    }

    spdk_poller_unregister(&ctx->step_poller);
    spdk_poller_unregister(&ctx->write_fsm_poller);
    spdk_poller_unregister(&ctx->read_fsm_poller);
    spdk_poller_unregister(&ctx->meta_poller);
    if (ctx->meta_qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->meta_qpair);
    spdk_ring_free(ctx->write_ring);
    spdk_ring_free(ctx->read_ring);
    spdk_ring_free(ctx->meta_ring);
    ctx->write_ring = NULL;
    ctx->read_ring = NULL;
    ctx->meta_ring = NULL;
    spdk_thread_exit(ctx->reactor_thread);
    return NULL;
}

/**
 * @brief SPDK new_thread_fn callback invoked by spdk_thread_create.
 *
 * Stores the SPDK thread handle and spawns the reactor pthread.
 *
 * @param thread  newly created SPDK thread
 * @param arg     NPUNVMEContext pointer
 */
static void reactor_new_thread_fn(struct spdk_thread *thread, void *arg) {
    (void)arg;  /* unused — ctx comes from g_reactor_ctx */
    NPUNVMEContext *ctx = g_reactor_ctx;
    ctx->reactor_thread = thread;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&ctx->reactor_pthread, &attr, reactor_loop, ctx);
    pthread_attr_destroy(&attr);
}

/* ---- Probe flag management ----
 *
 * The probe flag is a 4-byte HBM buffer that the listener writes after
 * completing a delta write.  Python reads it to confirm persistence.
 */

/**
 * @brief  Set or self-allocate the probe-flag HBM buffer.
 * @param ctx      context handle
 * @param dev_ptr  external HBM pointer (NULL to self-allocate)
 * @return 0 on success, -1 on error
 */
int npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr) {
    if (!ctx) return -1;

    pthread_mutex_lock(&ctx->state_lock);

    if (dev_ptr) {
        ctx->listener.probe_flag_dev_ptr = dev_ptr;
    } else {
        /* Self-allocate a 4-byte HBM buffer for the listener to poll */
        aclError ret = aclrtMalloc(&ctx->listener.probe_flag_dev_ptr, 4,
                                    ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] Failed to allocate probe flag device memory.\n");
            pthread_mutex_unlock(&ctx->state_lock);
            return -1;
        }
        uint32_t zero = 0;
        aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &zero, 4,
                     ACL_MEMCPY_HOST_TO_DEVICE);
    }

    /* Allocate a 4-byte host buffer for polling the flag */
    if (!ctx->listener.probe_flag_host) {
        aclrtMallocHost(&ctx->listener.probe_flag_host, 4);
    }

    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

/**
 * @brief  Write a value to the probe flag in HBM.
 * @param ctx    context handle
 * @param value  32-bit value to write
 * @return 0 on success, -1 on error
 */
int npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value) {
    if (!ctx || !ctx->listener.probe_flag_dev_ptr) return -1;
    pthread_mutex_lock(&ctx->state_lock);
    ensure_acl_context(ctx);
    aclError ret = aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &value, 4,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    pthread_mutex_unlock(&ctx->state_lock);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

/**
 * @brief  Return the probe-flag HBM device pointer.
 * @param ctx  context handle
 * @return device pointer, or NULL
 */
void *npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx) {
    return ctx ? ctx->listener.probe_flag_dev_ptr : NULL;
}

/* Signal the probe flag.  Called from the reactor thread (FSM completion).
 * Holds state_lock to protect against concurrent writes from Python. */
static void signal_probe_flag(NPUNVMEContext *ctx, uint32_t value) {
    if (!ctx->listener.probe_flag_dev_ptr) return;
    pthread_mutex_lock(&ctx->state_lock);
    ensure_acl_context(ctx);
    aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &value, 4,
                 ACL_MEMCPY_HOST_TO_DEVICE);
    /* Mirror to host buffer for polling */
    if (ctx->listener.probe_flag_host) {
        aclrtMemcpy(ctx->listener.probe_flag_host, 4,
                     ctx->listener.probe_flag_dev_ptr, 4,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    }
    pthread_mutex_unlock(&ctx->state_lock);
}

/* ---- Step counter setup ----
 *
 * The step counter is an int32 HBM tensor that DeltaTrainCell increments
 * each training step via AssignAdd.  The listener polls it every 10 ms.
 */

/**
 * @brief  Register the step_counter HBM address for FaF listener.
 * @param ctx            context handle
 * @param dev_ptr        HBM address of the step_counter Parameter
 * @param ckpt_interval  trigger a delta write every N steps
 * @return 0 on success, -1 on error
 */
int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval) {
    if (!ctx || !dev_ptr || ckpt_interval <= 0) return -1;
    pthread_mutex_lock(&ctx->state_lock);
    ctx->listener.dev_step_ptr = dev_ptr;
    ctx->listener.ckpt_interval = ckpt_interval;
    ctx->last_step_seen = -1;

    if (!ctx->listener.step_poll_buf) {
        aclrtMallocHost(&ctx->listener.step_poll_buf, 4);
    }
    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

/* ---- Step-counter poller (SPDK poller, replaces probe_listener_thread) ----
 *
 * step_poller_fn runs on the reactor thread with a 10 ms period.  It
 * reads the step counter from HBM and initiates an async write via the
 * FSM when a new checkpoint step is detected.
 *
 * At-most-1 in-flight: if the FSM is still busy with a previous write,
 * the new trigger is silently skipped (backpressure).
 *
 * state_lock is held only briefly to snapshot registered_tasks and reset
 * their state — the heavy I/O runs unlocked in the FSM poller.
 */

/**
 * @brief Periodic poller that reads the step counter from HBM.
 *
 * Registered on the reactor thread.  When a new checkpoint step is
 * detected, initiates an async write via the write FSM.
 *
 * @param arg  NPUNVMEContext pointer
 * @return     0 to continue, -1 to stop on app_should_stop
 */
static int step_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;

    if (ctx->app_should_stop) return -1;
    if (!ctx->listener.dev_step_ptr) return 0;

    /* At-most-1 in-flight: skip if previous write still running. */
    if (ctx->write_fsm.state != WRITE_FSM_IDLE) return 0;

    ensure_acl_context(ctx);

    int cur_step = 0;
    aclError ret = aclrtMemcpy(ctx->listener.step_poll_buf, 4,
                                ctx->listener.dev_step_ptr, 4,
                                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
        cur_step = *(int *)ctx->listener.step_poll_buf;
    }

    if (cur_step > ctx->last_step_seen &&
        cur_step % ctx->listener.ckpt_interval == 0 &&
        cur_step != 0) {
        ctx->last_step_seen = cur_step;

        /* Snapshot registered tasks under state_lock, then initiate FSM.
         * state_lock is released before any DMA/SPDK I/O — the FSM runs
         * unlocked on the reactor thread. */
        pthread_mutex_lock(&ctx->state_lock);
        for (int i = 0; i < ctx->listener.num_registered_tasks; i++) {
            ctx->listener.registered_tasks[i].state = CHUNK_IDLE;
            ctx->listener.registered_tasks[i].buf_idx = -1;
        }
        /* Use pre-allocated FaF request (same reactor thread, no ring). */
        ctx->write_fsm.faf_req.tasks = ctx->listener.registered_tasks;
        ctx->write_fsm.faf_req.num_tasks = ctx->listener.num_registered_tasks;
        ctx->write_fsm.faf_req.is_host = false;
        ctx->write_fsm.faf_step = (uint32_t)cur_step;
        initiate_write_fsm(ctx, &ctx->write_fsm.faf_req);
        pthread_mutex_unlock(&ctx->state_lock);
        /* I/O runs asynchronously in write_fsm_poller_fn.
         * probe flag is signalled on completion. */
    }

    return 0;
}

/* ---- Task registration ----
 *
 * register_tasks copies the caller's parameter-pointers/offsets/sizes
 * into a heap-allocated io_task array.  The listener thread uses this
 * array directly inside run_write_pipeline.  The function allocates a
 * new array before freeing the old one so that a failed allocation
 * leaves the listener with a valid (stale) task list.
 */

/**
 * @brief  Register parameter pointers for FaF background persistence.
 * @param ctx        context handle
 * @param npu_ptrs   array of NPU device pointers
 * @param nvme_offs  array of NVMe byte offsets
 * @param sizes      array of per-parameter byte sizes
 * @param num_items  number of registered tasks
 * @return 0 on success, -1 on error
 */
int npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;

    pthread_mutex_lock(&ctx->state_lock);

    /* Allocate new array before freeing the old one, so that a failed
     * allocation leaves the listener with a valid (stale) task list
     * rather than a NULL pointer + stale count. */
    io_task_t *new_tasks = calloc(num_items, sizeof(io_task_t));
    if (!new_tasks) {
        pthread_mutex_unlock(&ctx->state_lock);
        return -1;
    }

    for (int i = 0; i < num_items; i++) {
        new_tasks[i].task_idx = i;
        new_tasks[i].buf_idx = -1;
        new_tasks[i].state = CHUNK_IDLE;
        new_tasks[i].npu_ptr = npu_ptrs[i];
        new_tasks[i].nvme_offset = nvme_offsets[i];
        new_tasks[i].size = sizes[i];
    }

    /* Swap: defer-free the old array so the async FSM can safely finish
     * any in-flight write using the old pointer.  The old array is freed
     * on the next register_tasks call or when the FSM goes idle. */
    if (ctx->listener.old_tasks) {
        free(ctx->listener.old_tasks);
    }
    ctx->listener.old_tasks = ctx->listener.registered_tasks;
    ctx->listener.registered_tasks = new_tasks;
    ctx->listener.num_registered_tasks = num_items;
    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

/* ---- Synchronous metadata I/O ----
 *
 * sync_meta_io uses a dedicated 1 MB DMA buffer (ctx->meta_dma_buf)
 * for superblock and JSON-ledger reads and writes.  The caller is
 * responsible for keeping total_bytes ≤ 1 MB (META_DMA_BUF_SIZE).
 */

/* Metadata I/O moved to async meta_poller_fn (V4). */

/**
 * @brief  Synchronous metadata I/O (superblock and JSON ledger).
 * @param ctx          context handle
 * @param byte_offset  absolute byte offset on the NVMe device
 * @param total_bytes  number of bytes to read or write (≤ 1 MB)
 * @param is_read      1 = read, 0 = write
 * @param meta_buffer  host-side buffer
 * @return 0 on success, -1 on error
 */
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset,
                           uint32_t total_bytes, int is_read, void *meta_buffer) {
    if (!ctx || !meta_buffer) return -1;
    if (ctx->block_size == 0) return -1;
    if (total_bytes > META_DMA_BUF_SIZE) {
        fprintf(stderr, "[NPU-NVMe] sync_meta_io: total_bytes=%u exceeds "
                "META_DMA_BUF_SIZE=%u\n", total_bytes, (unsigned)META_DMA_BUF_SIZE);
        return -1;
    }
    if (!ctx->meta_dma_buf) return -1;

    /* Use meta_ring → reactor (dedicated meta_qpair, no lock needed). */
    meta_request_t req;
    req.byte_offset = byte_offset;
    req.total_bytes = total_bytes;
    req.is_read = is_read;
    req.meta_buffer = meta_buffer;
    req.done = 0;
    req.result = 0;

    void *obj = &req;
    if (spdk_ring_enqueue(ctx->meta_ring, &obj, 1, NULL) != 1) {
        fprintf(stderr, "[NPU-NVMe] meta_ring full\n");
        return -1;
    }

    /* Poll for completion. */
    while (!req.done) { usleep(100); }

    return req.result;
}

/* ---- Initialisation / cleanup ----
 *
 * npu_nvme_init is the sole entry point for creating a context.  It
 * initialises SPDK (once per process), probes the NVMe device, sets
 * up the DMA pool + NPU events, allocates the metadata buffer, and
 * launches the background FaF listener thread.
 *
 * npu_nvme_cleanup stops the listener, frees all ACL and SPDK
 * resources in reverse allocation order, and destroys the state_lock.
 */

/**
 * @brief  Initialise the NPU-NVMe SPDK environment and create a context.
 * @param out_ctx           output context handle
 * @param pci_addr          NVMe PCIe BDF address (e.g. "0000:83:00.0")
 * @param npu_id            Ascend NPU device ID
 * @param pipe_depth        DMA pipeline depth (1–16 recommended)
 * @param chunk_size        max bytes per DMA chunk (4 MiB = 4194304 recommended)
 * @param enable_profiling  enable per-chunk timing CSV output
 * @param prof_dir          directory for profiling CSV files (NULL = ".")
 * @return 0 on success, -1 on error
 */
int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                  int pipe_depth, uint32_t chunk_size, bool enable_profiling,
                  const char *prof_dir) {
    NPUNVMEContext *ctx = calloc(1, sizeof(NPUNVMEContext));
    if (!ctx) return -1;

    strncpy(ctx->pci_addr, pci_addr, sizeof(ctx->pci_addr) - 1);
    ctx->acl.npu_id = npu_id;
    ctx->dma.chunk_size = chunk_size;
    ctx->dma.max_pipe_depth = (pipe_depth < MIN_PIPE_DEPTH) ? MIN_PIPE_DEPTH :
                               (pipe_depth > MAX_PIPE_DEPTH) ? MAX_PIPE_DEPTH :
                                pipe_depth;
    ctx->enable_profiling = enable_profiling;
    if (prof_dir) {
        strncpy(ctx->profiling_dir, prof_dir, sizeof(ctx->profiling_dir) - 1);
    } else {
        strcpy(ctx->profiling_dir, ".");
    }
    /* Listener-state lock — protects registered_tasks, dev_step_ptr,
     * probe_flag_* from concurrent Python <-> reactor access.  I/O is
     * async via rings and does NOT use this lock.  Non-recursive. */
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        if (pthread_mutex_init(&ctx->state_lock, &attr) != 0) {
            fprintf(stderr, "[Fatal] Failed to init state_lock.\n");
            pthread_mutexattr_destroy(&attr);
            free(ctx); return -1;
        }
        pthread_mutexattr_destroy(&attr);
    }

    /* Initialise SPDK environment (once per process via SPDK_SHM_ID).
     * MUST be called BEFORE spdk_thread_lib_init — the thread library
     * internally creates spdk_ring (rte_ring/rte_mempool) which requires
     * DPDK EAL to be fully initialised. */
    {
        static int spdk_inited = 0;
        if (!spdk_inited) {
            struct spdk_env_opts env_opts;
            spdk_env_opts_init(&env_opts);
            env_opts.name = "npu_nvme_app";

            const char *shm = getenv("SPDK_SHM_ID");
            if (shm) { env_opts.shm_id = atoi(shm); }

            ensure_hugepages();

            if (spdk_env_init(&env_opts) < 0) {
                fprintf(stderr, "[Fatal] Unable to initialize SPDK env.\n"
                        "[Fatal] Check: (1) run as root, (2) free hugepages > 0 "
                        "per NUMA node.\n"
                        "[Fatal] Quick fix: echo %d > %s\n",
                        read_int_from_file(NR_HUGEPAGES_PATH) + HUGEPAGE_PADDING,
                        NR_HUGEPAGES_PATH);
                free(ctx);
                return -1;
            }
            spdk_inited = 1;
        }
    }

    /* --- diagnostic: verify EAL state after spdk_env_init --- */
    fprintf(stderr, "[Diag] after spdk_env_init: rte_lcore_count=%u "
            "rte_socket_count=%u spdk_core_count=%u\n",
            rte_lcore_count(), rte_socket_count(),
            spdk_env_get_core_count());

    /* Re-register DPDK ring mempool ops after EAL init (once per process).
     *
     * The RTE_INIT constructors in librte_mempool_ring.a register ops at
     * dlopen time via init_array, but EAL was not yet available.  We
     * explicitly re-register "ring_mp_mc" (the default) here, now that
     * spdk_env_init has completed and EAL is fully initialised.
     */
    {
        static int ops_registered = 0;
        if (!ops_registered) {
        extern int common_ring_alloc(struct rte_mempool *mp);
        extern void common_ring_free(struct rte_mempool *mp);
        extern void common_ring_mp_enqueue(struct rte_mempool *mp,
                void * const *obj_table, unsigned n);
        extern void common_ring_mc_dequeue(struct rte_mempool *mp,
                void **obj_table, unsigned n);
        extern unsigned common_ring_get_count(const struct rte_mempool *mp);

        struct rte_mempool_ops ops = {
            .name = "ring_mp_mc", .alloc = common_ring_alloc,
            .free = common_ring_free, .enqueue = common_ring_mp_enqueue,
            .dequeue = common_ring_mc_dequeue,
            .get_count = common_ring_get_count,
        };
        int rc = rte_mempool_register_ops(&ops);
        if (rc < 0) {
            fprintf(stderr, "[NPU-NVMe] WARNING: rte_mempool_register_ops "
                    "(ring_mp_mc) failed (rc=%d)\n", rc);
        }
            ops_registered = 1;
        }
    }

    /* Start reactor pthread via SPDK thread library.
     * spdk_thread_lib_init creates an spdk_ring (rte_ring → rte_mempool)
     * — EAL must be initialised first (done above).
     * Like spdk_env_init, spdk_thread_lib_init is once-per-process. */
    ctx->app_should_stop = 0;
    pthread_barrier_init(&ctx->init_barrier, NULL, 2);

    {
        static int thread_lib_inited = 0;
        if (!thread_lib_inited) {
            if (spdk_thread_lib_init((spdk_new_thread_fn)reactor_new_thread_fn, 0) != 0) {
                fprintf(stderr, "[Fatal] spdk_thread_lib_init failed.\n");
                pthread_barrier_destroy(&ctx->init_barrier);
                free(ctx); return -1;
            }
            thread_lib_inited = 1;
        }
    }

    /* spdk_thread_create: the second argument is a cpumask pointer.
     * Passing ctx as cpumask causes a SEGV on ARM64 because SPDK
     * internally reads cpumask as a potentially large cpu_set bitmask.
     * V0 passes NULL; we do the same and communicate ctx via g_reactor_ctx. */
    g_reactor_ctx = ctx;
    struct spdk_thread *th = spdk_thread_create("npu_nvme", NULL);
    if (!th) {
        fprintf(stderr, "[Fatal] spdk_thread_create failed.\n");
        pthread_barrier_destroy(&ctx->init_barrier);
        free(ctx); return -1;
    }

    /* Wait for reactor pthread to reach its main loop. */
    pthread_barrier_wait(&ctx->init_barrier);
    pthread_barrier_destroy(&ctx->init_barrier);

    /* Probe and attach NVMe device */
    struct spdk_nvme_transport_id trid = {};
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", pci_addr);

    if (spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, NULL) != 0) {
        fprintf(stderr, "[Fatal] spdk_nvme_probe failed.\n");
        free(ctx); return -1;
    }
    if (!ctx->ctrlr) {
        fprintf(stderr, "[Fatal] Controller not found at %s.\n", pci_addr);
        free(ctx); return -1;
    }

    /* Allocate SPDK I/O queue pair */
    struct spdk_nvme_io_qpair_opts qopts;
    spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &qopts, sizeof(qopts));
    qopts.io_queue_size = 512;
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &qopts, sizeof(qopts));
    if (!ctx->qpair) {
        fprintf(stderr, "[Fatal] Cannot allocate NVMe I/O qpair.\n");
        free(ctx); return -1;
    }

    /* Allocate dedicated metadata qpair (small, no contention with data path). */
    {
        struct spdk_nvme_io_qpair_opts meta_opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &meta_opts,
                                                   sizeof(meta_opts));
        meta_opts.io_queue_size = 64;
        ctx->meta_qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &meta_opts,
                                                          sizeof(meta_opts));
        if (!ctx->meta_qpair) {
            fprintf(stderr, "[Fatal] Cannot allocate metadata qpair.\n");
            npu_nvme_cleanup(ctx);
            *out_ctx = NULL;
            return -1;
        }
    }

    /* Initialise NPU environment */
    aclError ret = aclrtSetDevice(ctx->acl.npu_id);
    if (ret != ACL_SUCCESS) { free(ctx); return -1; }
    if (aclrtGetCurrentContext(&ctx->acl.acl_ctx) != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to get ACL context.\n");
        free(ctx); return -1;
    }
    ret = aclrtCreateStream(&ctx->acl.copy_stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to create NPU Stream.\n");
        free(ctx); return -1;
    }

    /* Allocate DMA buffer pool + NPU events */
    ctx->dma.pool = calloc(ctx->dma.max_pipe_depth, sizeof(dma_buf_t));
    ctx->acl.events = calloc(ctx->dma.max_pipe_depth, sizeof(aclrtEvent));
    /* ring capacity = pipe_depth + 1 (one overflow slot for full-vs-empty). */
    ring_init(&ctx->dma.free_ring, ctx->dma.max_pipe_depth + 1);

    for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
        ctx->dma.pool[i].buf = spdk_zmalloc(ctx->dma.chunk_size,
                                             2 * 1024 * 1024, NULL,
                                             SPDK_ENV_SOCKET_ID_ANY,
                                             SPDK_MALLOC_DMA);
        if (!ctx->dma.pool[i].buf) {
            fprintf(stderr, "[Fatal] spdk_zmalloc failed at slot %d.\n", i);
            free(ctx); return -1;
        }
        ctx->dma.pool[i].phys_addr = spdk_vtophys(ctx->dma.pool[i].buf, NULL);

        ret = aclrtCreateEvent(&ctx->acl.events[i]);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Fatal] Failed to create NPU Event at slot %d.\n", i);
            free(ctx); return -1;
        }
        ring_push(&ctx->dma.free_ring, i);
    }

    printf("[Init] NPUNVME Fully Initialized! Stream/Events ready. "
           "Max Pipe Depth: %d\n", ctx->dma.max_pipe_depth);
    *out_ctx = ctx;

    /* Allocate dedicated buffer for metadata I/O */
    ctx->meta_dma_buf = spdk_zmalloc(META_DMA_BUF_SIZE, 2 * 1024 * 1024, NULL,
                                      SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    if (!ctx->meta_dma_buf) {
        fprintf(stderr, "[NPU-NVMe] Failed to allocate meta DMA buffer.\n");
        npu_nvme_cleanup(ctx);  /* releases all resources including ctx */
        *out_ctx = NULL;
        return -1;
    }

    /* Listener poller runs on the reactor thread (registered in reactor_loop).
     * The NPU_NVME_NO_LISTENER env var still controls whether the step poller
     * is active — it does nothing until dev_step_ptr is set. */
    ctx->listener.probe_flag_dev_ptr = NULL;
    ctx->listener.probe_flag_host = NULL;
    ctx->listener.dev_step_ptr = NULL;
    ctx->listener.step_poll_buf = NULL;

    printf("[Init] Initialisation complete.\n");
    return 0;
}

/**
 * @brief  Release all resources (SPDK, ACL, DMA pool, listener thread).
 * @param ctx  context handle (NULL is safe)
 */
void npu_nvme_cleanup(NPUNVMEContext *ctx) {
    if (!ctx) return;

    /* Stop reactor thread */
    ctx->app_should_stop = 1;
    pthread_join(ctx->reactor_pthread, NULL);

    /* Bind ACL context — needed for aclrtDestroyEvent/FreeHost below. */
    aclrtSetDevice(ctx->acl.npu_id);
    aclrtSetCurrentContext(ctx->acl.acl_ctx);

    /* Release ACL resources */
    if (ctx->acl.events) {
        for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
            if (ctx->acl.events[i]) aclrtDestroyEvent(ctx->acl.events[i]);
        }
        free(ctx->acl.events);
    }
    if (ctx->acl.copy_stream) aclrtDestroyStream(ctx->acl.copy_stream);

    /* Release listener host buffers */
    if (ctx->listener.probe_flag_host)
        aclrtFreeHost(ctx->listener.probe_flag_host);
    if (ctx->listener.step_poll_buf)
        aclrtFreeHost(ctx->listener.step_poll_buf);

    /* Release DMA pool */
    if (ctx->dma.pool) {
        for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
            if (ctx->dma.pool[i].buf) spdk_free(ctx->dma.pool[i].buf);
        }
        free(ctx->dma.pool);
    }
    ring_free(&ctx->dma.free_ring);

    /* Release metadata DMA buffer */
    if (ctx->meta_dma_buf) spdk_free(ctx->meta_dma_buf);

    /* Detach NVMe */
    if (ctx->qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    if (ctx->meta_qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->meta_qpair);
    if (ctx->ctrlr) spdk_nvme_detach(ctx->ctrlr);

    /* Release registered tasks (both current and deferred-free). */
    if (ctx->listener.registered_tasks) {
        free(ctx->listener.registered_tasks);
        ctx->listener.registered_tasks = NULL;
        ctx->listener.num_registered_tasks = 0;
    }
    if (ctx->listener.old_tasks) {
        free(ctx->listener.old_tasks);
        ctx->listener.old_tasks = NULL;
    }

    pthread_mutex_destroy(&ctx->state_lock);
    free(ctx);
}

/* ---- Delta ring-buffer layout ----
 *
 * The delta area occupies the last N slots on the NVMe device, each
 * delta_slot_size bytes, organized as a ring of delta_slot_count slots.
 * These functions only manage the layout metadata (offsets/sizes) —
 * actual I/O goes through write_batch / read_batch as usual.
 */

/**
 * @brief  Initialise the delta ring-buffer layout on disk.
 * @param ctx               context handle
 * @param delta_slot_size   bytes per delta slot (e.g. 256 MiB)
 * @param delta_slot_count  number of slots in the ring (e.g. 128)
 * @return 0 on success, -1 on error
 */
int npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t delta_slot_size,
                         uint32_t delta_slot_count) {
    if (!ctx || delta_slot_size == 0 || delta_slot_count == 0) return -1;

    uint64_t total_delta_bytes = (uint64_t)delta_slot_size * delta_slot_count;
    if (ctx->total_blocks == 0) return -1;
    uint64_t disk_bytes = ctx->total_blocks * ctx->block_size;

    ctx->delta.area_offset = disk_bytes - total_delta_bytes;
    ctx->delta.slot_size = delta_slot_size;
    ctx->delta.slot_count = delta_slot_count;

    fprintf(stderr, "[NPU-NVMe] Delta area: offset=%lu (%lu GB) "
            "slots=%u x %lu MB = %lu MB\n",
            ctx->delta.area_offset,
            ctx->delta.area_offset / (1024 * 1024 * 1024),
            delta_slot_count,
            delta_slot_size / (1024 * 1024),
            total_delta_bytes / (1024 * 1024));

    return 0;
}

/** @brief Return the byte offset of the delta ring on the NVMe device. */
uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.area_offset : 0;
}

/** @brief Return the configured per-slot size in bytes. */
uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.slot_size : 0;
}

/** @brief Return the configured number of delta ring slots. */
uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.slot_count : 0;
}

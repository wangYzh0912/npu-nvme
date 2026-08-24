/* Internal: I/O task state machine and supporting types.
 *
 * Shared between write and read pipelines.
 */
#ifndef NPU_NVME_IO_TASK_H
#define NPU_NVME_IO_TASK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>

/* --- DMA buffer descriptor --- */
typedef struct {
    void *buf;
    uint64_t phys_addr;
} dma_buf_t;

/* --- Chunk lifecycle state --- */
typedef enum {
    CHUNK_IDLE = 0,
    CHUNK_NPU_COPYING,      /* write path: NPU -> DMA buffer in flight */
    CHUNK_NPU_DONE,         /* write path: DMA copy complete */
    CHUNK_SPDK_WRITING,     /* write path: SPDK NVMe write in flight */
    CHUNK_SPDK_READING,     /* read path:  SPDK NVMe read in flight */
    CHUNK_SPDK_DONE,        /* read path:  NVMe -> DMA buffer complete */
    CHUNK_DONE
} chunk_state_t;

/* --- Per-chunk I/O descriptor --- */
typedef struct {
    int task_idx;
    int buf_idx;            /* ring buffer slot index, -1 = unassigned */
    chunk_state_t state;
    void *npu_ptr;          /* source (write) or dest (read) address */
    size_t size;
    uint64_t nvme_offset;   /* absolute byte offset on NVMe */
    uint64_t ts_submit;     /* profiling: submission timestamp */
    uint64_t ts_npu_done;   /* profiling: NPU copy completion */
    uint64_t ts_spdk_done;  /* profiling: SPDK I/O completion */
} io_task_t;

/* --- SPDK callback context --- */
typedef struct {
    struct NPUNVMEContext *ctx;
    io_task_t *task;
    int *completed_counter;
    int *result;
} spdk_cb_arg_t;

/* --- Pipeline direction --- */
typedef enum { PIPELINE_WRITE, PIPELINE_READ } pipeline_dir_t;

/* --- Shared helpers --- */
#define ALIGN_4K(x)  (((x) + 4095ULL) & ~4095ULL)

io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes);

/* ---- Async write FSM (V3) ----
 *
 * The write FSM replaces the blocking run_write_pipeline with a non-blocking
 * state machine driven by a SPDK poller on the reactor thread.
 *
 * write_request_t encapsulates a single write operation.  For Python-initiated
 * writes, the caller allocates this on the heap, enqueues it in write_ring,
 * and polls ->done.  For FaF (Fire-and-Forget) writes triggered by the
 * step poller, the request is built inline and initiated directly (same
 * reactor thread, no ring needed).
 */

typedef enum {
    WRITE_FSM_IDLE = 0,
    WRITE_FSM_RUNNING,
} write_fsm_state_t;

typedef struct {
    io_task_t *tasks;           /* array of per-chunk descriptors */
    int num_tasks;              /* number of chunks in this write */
    bool is_host;               /* true → memcpy, false → aclrtMemcpy D2H */
    atomic_int done;            /* set to 1 when all chunks complete */
    int result;                 /* 0 = success, -1 = any chunk failed */
    uint64_t ts_batch_start;    /* C-layer: first DMA submit time (us) */
    uint64_t ts_batch_end;      /* C-layer: last SPDK completion time (us) */
} write_request_t;

typedef struct {
    write_fsm_state_t state;
    write_request_t *req;       /* current active request, NULL when idle */
    write_request_t faf_req;    /* pre-allocated FaF request (reused each trigger) */
    uint32_t faf_step;          /* step number that triggered current FaF write */
    int next_submit_idx;        /* next chunk index to DMA-copy */
    int completed_count;        /* number of fully completed chunks */
} write_fsm_ctx_t;

/* ---- Async read FSM (V4) ---- */

typedef enum {
    READ_FSM_IDLE = 0,
    READ_FSM_RUNNING,
} read_fsm_state_t;

typedef struct {
    io_task_t *tasks;           /* array of per-chunk descriptors */
    int num_tasks;              /* number of chunks in this read */
    bool is_host;               /* true → memcpy, false → aclrtMemcpy H2D */
    atomic_int done;            /* set to 1 when all chunks complete */
    int result;                 /* 0 = success, -1 = any chunk failed */
    uint64_t ts_batch_start;    /* C-layer: first SPDK submit time (us) */
    uint64_t ts_batch_end;      /* C-layer: last DMA completion time (us) */
} read_request_t;

typedef struct {
    read_fsm_state_t state;
    read_request_t *req;        /* current active request, NULL when idle */
    int next_submit_idx;        /* next chunk index to submit */
    int completed_count;        /* number of fully completed chunks */
} read_fsm_ctx_t;

/* ---- Metadata I/O request (V4) ---- */

typedef struct {
    uint64_t byte_offset;       /* absolute byte offset on NVMe */
    uint32_t total_bytes;       /* number of bytes to read/write */
    int is_read;                /* 1 = read, 0 = write */
    void *meta_buffer;          /* caller's host buffer */
    atomic_int done;            /* set to 1 when I/O completes */
    int result;                 /* 0 = success, -1 = I/O error */
} meta_request_t;

#endif

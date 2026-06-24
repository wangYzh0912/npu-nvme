/* Internal: I/O task state machine and supporting types.
 *
 * Shared between write and read pipelines.
 */
#ifndef NPU_NVME_IO_TASK_H
#define NPU_NVME_IO_TASK_H

#include <stdint.h>
#include <stddef.h>

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
} spdk_cb_arg_t;

/* --- Pipeline direction --- */
typedef enum { PIPELINE_WRITE, PIPELINE_READ } pipeline_dir_t;

/* --- Shared helpers --- */
#define ALIGN_4K(x)  (((x) + 4095ULL) & ~4095ULL)

io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes);

#endif

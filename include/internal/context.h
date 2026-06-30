/* Internal: NPUNVMEContext structure definition.
 *
 * This header exposes the full struct layout to internal compilation
 * units.  External consumers (Python ctypes) see only the opaque
 * forward declaration in npu_nvme.h.
 */
#ifndef NPU_NVME_CONTEXT_H
#define NPU_NVME_CONTEXT_H

#include "ring_buffer.h"
#include "io_task.h"

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <acl/acl.h>
#include <spdk/thread.h>

/* Dedicated buffer for superblock & JSON metadata I/O. */
#define META_DMA_BUF_SIZE  (1 * 1024 * 1024)

/* Pipeline depth bounds */
#define MIN_PIPE_DEPTH  1
#define MAX_PIPE_DEPTH  16

/* Hugepage pool auto-expansion constants */
#define NR_HUGEPAGES_PATH  "/proc/sys/vm/nr_hugepages"
#define HUGEPAGE_PADDING   512

/* ---- ACL runtime sub-structure ---- */
typedef struct {
    int npu_id;
    aclrtStream copy_stream;
    aclrtEvent *events;        /* one event per ring-buffer slot */
    aclrtContext acl_ctx;
} acl_state_t;

/* ---- DMA buffer pool ---- */
typedef struct {
    dma_buf_t *pool;           /* SPDK DMA buffers, pool[max_pipe_depth] */
    ring_t free_ring;          /* SPSC ring of available slot indices */
    int max_pipe_depth;
    uint32_t chunk_size;
} dma_pool_t;

/* ---- FaF listener / probe-flag state ---- */
typedef struct {
    void *probe_flag_dev_ptr;  /* device (HBM) address of probe flag */
    void *probe_flag_host;     /* host-side mirror for polling */
    void *dev_step_ptr;        /* device (HBM) address of step_counter */
    void *step_poll_buf;       /* host buffer for polling step_counter */
    int ckpt_interval;         /* trigger SPDK write every N steps */
    io_task_t *registered_tasks;
    int num_registered_tasks;
} listener_state_t;

/* ---- Delta ring-buffer layout (bookkeeping only, no I/O) ---- */
typedef struct {
    uint64_t area_offset;      /* byte offset of delta area on NVMe */
    uint64_t slot_size;        /* bytes per slot */
    uint32_t slot_count;       /* total number of slots */
} delta_state_t;

/* ---- Master context ---- */
typedef struct NPUNVMEContext {
    /* SPDK device */
    char pci_addr[64];
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;

    /* ACL + DMA */
    acl_state_t acl;
    dma_pool_t dma;

    /* Metadata I/O */
    void *meta_dma_buf;
    bool enable_profiling;
    char profiling_dir[256];

    /* Fire-and-Forget listener */
    listener_state_t listener;

    /* Delta ring layout */
    delta_state_t delta;

    /* ---- Reactor thread ---- */
    struct spdk_thread *reactor_thread;
    pthread_t reactor_pthread;
    pthread_barrier_t init_barrier;
    volatile int app_should_stop;

    /* ---- Step-counter poller ---- */
    struct spdk_poller *step_poller;
    int last_step_seen;

    /* I/O serialization — the listener thread and the main thread share
     * a single SPDK qpair.  SPDK qpairs (submission + completion queues)
     * are not thread-safe; this mutex serialises all qpair access. */
    pthread_mutex_t io_lock;
} NPUNVMEContext;

#endif

#ifndef NPU_NVME_H
#define NPU_NVME_H

#include <stdint.h>

#define NPU_NVME_ABI_VERSION 2
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t npu_nvme_get_abi_version(void);

/** @brief Opaque context handle.  Python sees this as an opaque pointer. */
typedef struct NPUNVMEContext NPUNVMEContext;
typedef struct NPUNVMERequest NPUNVMERequest;

/** Runtime counters for one context. Values are a point-in-time snapshot. */
typedef struct {
    uint64_t nvme_submit_count;
    uint64_t nvme_complete_count;
    uint32_t nvme_outstanding;
    uint32_t nvme_outstanding_peak;
    uint32_t dma_inflight;
    uint32_t dma_inflight_peak;
    uint32_t request_ring_depth;
    uint32_t request_ring_peak;
    uint64_t async_dma_submit_count;
    uint64_t async_event_query_count;
    uint64_t async_event_query_error_count;
    uint64_t stream_sync_fallback_count;
    uint64_t spdk_retry_count;
    uint64_t completion_error_count;
    uint64_t reactor_cpu_us;
} NPUNVMEStats;

/**
 * @brief Initialise the NPU-NVMe SPDK environment.
 *
 * @param out_ctx           output context handle
 * @param pci_addr          NVMe PCIe BDF address (e.g. "0000:83:00.0")
 * @param npu_id            Ascend NPU device ID
 * @param pipe_depth        DMA pipeline depth (4--16 recommended)
 * @param chunk_size        max bytes per DMA chunk (4 MB = 4194304 recommended)
 * @param enable_profiling  enable per-chunk timing CSV output
 * @param prof_dir          directory for profiling CSV files (NULL = ".")
 * @return 0 on success, -1 on error
 */
int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                  int pipe_depth, uint32_t chunk_size, bool enable_profiling,
                  const char *prof_dir);

/** @brief Release all resources (SPDK, ACL, DMA pool, Reactor thread). */
void npu_nvme_cleanup(NPUNVMEContext *ctx);

/**
 * @brief Submit an HBM-to-NVMe batch without waiting for completion.
 *
 * The source buffers must remain valid until poll/wait reports completion.
 * A timeout does not cancel the request.  Release is valid only after the
 * request reaches a terminal state.
 */
int npu_nvme_submit_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                uint64_t *nvme_offsets, size_t *sizes,
                                int num_items, NPUNVMERequest **out_request);

/**
 * @brief Submit a pinned Host-to-NVMe batch without waiting for completion.
 *
 * The Host buffers remain caller-owned and must stay valid until the request
 * reaches a terminal state.  This is the persistence half of live FULL
 * capture: ACL first fills generation-owned pinned staging, then the Reactor
 * consumes that immutable staging without blocking the training thread.
 */
int npu_nvme_submit_write_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                                     uint64_t *nvme_offsets, size_t *sizes,
                                     int num_items,
                                     NPUNVMERequest **out_request);

/** @brief Poll a submitted request; result is returned once done is true. */
int npu_nvme_poll_request(NPUNVMERequest *request, int *done);

/** @brief Observe for timeout_ms; zero selects the finite request default.
 * Timeout does not cancel I/O or prove source/destination buffers are safe.
 */
int npu_nvme_wait_request(NPUNVMERequest *request, uint32_t timeout_ms);

/** @brief Drop caller's request reference, including before completion.
 * The queue/reactor retains its reference. This never transfers ownership of
 * caller data buffers: retain those until completion or proven quiescence.
 */
void npu_nvme_release_request(NPUNVMERequest *request);

/** @brief Return total NVMe capacity in bytes. */
uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx);

/** @brief Return the configured per-chunk transfer size. */
int npu_nvme_get_max_transfer(NPUNVMEContext *ctx);

/**
 * @brief Synchronous metadata I/O (superblock and JSON ledger).
 *
 * @param ctx          context handle
 * @param byte_offset  absolute byte offset on the NVMe device
 * @param total_bytes  number of bytes to read or write
 * @param is_read      1 = read, 0 = write
 * @param meta_buffer  host-side buffer
 * @return 0 on success, -1 on error
 */
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset,
                          uint32_t total_bytes, int is_read, void *meta_buffer);

/** @brief Submit and wait for an NVMe namespace flush on the metadata qpair. */
int npu_nvme_flush(NPUNVMEContext *ctx);

/**
 * @brief Batch write: NPU HBM -> NVMe (blocking).
 *
 * @param ctx          context handle
 * @param npu_ptrs     array of NPU device pointers (source)
 * @param nvme_offsets array of NVMe byte offsets (destination)
 * @param sizes        array of per-chunk byte sizes
 * @param num_items    number of chunks
 * @return 0 on success, -1 on error
 */
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                         uint64_t *nvme_offsets, size_t *sizes, int num_items);


/**
 * @brief Batch read: NVMe -> NPU HBM (blocking).
 *
 * @param ctx          context handle
 * @param npu_ptrs     array of NPU device pointers (destination)
 * @param nvme_offsets array of NVMe byte offsets (source)
 * @param sizes        array of per-chunk byte sizes
 * @param num_items    number of chunks
 * @return 0 on success, -1 on error
 */
int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                        uint64_t *nvme_offsets, size_t *sizes, int num_items);

/**
 * @brief Batch read: NVMe -> Host DRAM (memcpy, no NPU involvement).
 */
int npu_nvme_read_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                              uint64_t *nvme_offsets, size_t *sizes, int num_items);

/**
 * @brief Batch write: Host DRAM -> NVMe (memcpy, no NPU involvement).
 */
int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **ptrs,
                              uint64_t *nvme_offsets, size_t *sizes, int num_items);

/** @brief Set the bounded timeout used by blocking C API calls. */
int npu_nvme_set_io_timeout_ms(NPUNVMEContext *ctx, uint32_t timeout_ms);

/** @brief Return the configured blocking I/O timeout in milliseconds. */
uint32_t npu_nvme_get_io_timeout_ms(NPUNVMEContext *ctx);

/**
 * @brief Wait until all queued and in-flight Reactor requests are quiescent.
 *
 * This is required after a blocking API returns a timeout and before the
 * caller releases any HBM buffer or ACL context referenced by that request.
 * @return 0 when no request remains, -ETIMEDOUT when the bound expires.
 */
int npu_nvme_wait_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms);

/** Diagnostic ownership snapshot; no entry authorizes freeing a buffer.
 * reason: 0 active, 1 event record stop unproven, 2 event query stop unproven,
 * 3 observation timeout. Free only after request completion/proven quiescence.
 */
typedef struct {
    uint32_t slot;
    uint32_t reason;
    uint64_t request_id;
    uint64_t bytes;
    uint64_t nvme_offset;
} NPUNVMERetainedSlot;
int npu_nvme_get_retained_slots(NPUNVMEContext *ctx, NPUNVMERetainedSlot *slots,
                                 uint32_t capacity, uint32_t *count);

/** Close admission and wait for reactor exit, retaining context on failure.
 *  0 timeout selects the finite default. May be retried; does not free ctx.
 *  -EIO means DMA stop is unproven; buffers and context must remain alive.
 *  After successful close and joined submitters, cleanup releases the owner.
 */
int npu_nvme_close(NPUNVMEContext *ctx, uint32_t timeout_ms);



/**
 * @brief Return the C-layer I/O latency of the most recent batch operation.
 *
 * Measures pure DMA + SPDK time (first DMA start to last SPDK completion),
 * excluding Python marshalling overhead.
 *
 * @param ctx     context handle
 * @param is_read 0 = last write, 1 = last read
 * @return latency in microseconds, or 0 if no I/O has been performed
 */
uint64_t npu_nvme_get_last_io_us(NPUNVMEContext *ctx, int is_read);

/** @brief Copy current runtime counters into caller-owned storage. */
int npu_nvme_get_stats(NPUNVMEContext *ctx, NPUNVMEStats *out_stats);

#ifdef __cplusplus
}
#endif

#endif // NPU_NVME_H

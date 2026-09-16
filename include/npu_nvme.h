#ifndef NPU_NVME_H
#define NPU_NVME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#define NPU_NVME_ROLE_COMBINED 0u
#define NPU_NVME_ROLE_HOST_OWNER 1u
#define NPU_NVME_ROLE_COPY_RANK 2u
typedef struct {
    uint32_t struct_size, version, role, pipe_depth, chunk_size;
    int32_t npu_id;
    uint32_t reserved;
} NPUNVMEInitOptions;
/* Additive ABI2 initialization. Copy ranks never probe/attach PCI devices;
 * Host owners never create an ACL context. One context per process remains. */
int npu_nvme_init_ex(NPUNVMEContext **out_ctx, const char *pci_addr,
                    const NPUNVMEInitOptions *options, const char *prof_dir);

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

/* B2 additive, versioned API. Buffer memory remains caller-owned until
 * completion or an explicit stop proof; descriptors/results are request-owned.
 * Flags describe implementation capabilities, not hardware acceptance. */
#define NPU_NVME_TRANSFER_VERSION 1u
#define NPU_NVME_TRANSFER_WRITE 0u
#define NPU_NVME_TRANSFER_READ 1u
#define NPU_NVME_TRANSFER_META_READ 2u
#define NPU_NVME_TRANSFER_META_WRITE 3u
#define NPU_NVME_TRANSFER_FLUSH 4u
#define NPU_NVME_COPY_D2H 5u
#define NPU_NVME_COPY_H2D 6u
#define NPU_NVME_MEMORY_HBM 0u
#define NPU_NVME_MEMORY_HOST 1u
#define NPU_NVME_CHECK_CRC32 1u
#define NPU_NVME_CHECK_SHA256 2u

/* Effective limits apply per request/tick, never to a whole checkpoint.
 * Scheduler environment overrides are read once at context initialization:
 * NPU_NVME_{COPY_BYTES,CHECKSUM_BYTES,SUBMIT_ITEMS,QUANTUM_ITEMS,MAX_PENDING}.
 * Invalid overrides reject initialization; no silent clamping. */
typedef struct {
    uint32_t struct_size, version;
    uint32_t operation_mask, block_size, chunk_size, pipe_depth;
    uint32_t max_request_items, max_pending_requests;
    uint32_t copy_bytes_per_tick, checksum_bytes_per_tick;
    uint32_t submit_items_per_tick, quantum_items;
    uint64_t max_request_bytes, namespace_bytes, dma_pool_bytes;
} NPUNVMECapabilities;
int npu_nvme_get_capabilities(NPUNVMEContext *ctx, NPUNVMECapabilities *out,
                             uint32_t size);

/* Copy requests use the same reactor, SPDK DMA pool and receipt ownership.
 * This supports bounded capture hashing and restore application without a
 * second ACL executor. Neither copy operation submits an NVMe command. */
typedef struct {
    uint32_t struct_size, version, operation, checksum_flags;
    void *source, *destination;
    uint64_t length;
    uint32_t expected_crc32, reserved;
    uint8_t expected_sha256[32];
} NPUNVMECopySpec;
int npu_nvme_submit_copy(NPUNVMEContext *ctx, const NPUNVMECopySpec *spec,
                          NPUNVMERequest **out_request);

typedef struct {
    void *address;
    uint64_t offset;
    uint64_t length;
    uint32_t expected_crc32;
    uint32_t reserved;
    uint8_t expected_sha256[32];
} NPUNVMETransferItem;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint32_t operation;
    uint32_t memory_kind;
    uint32_t checksum_flags;
    uint32_t item_count;
    const NPUNVMETransferItem *items;
} NPUNVMETransferSpec;

typedef struct {
    uint64_t request_id;
    uint64_t logical_bytes;
    uint32_t item_count;
    uint32_t operation;
    int32_t result;
    uint32_t done;
    uint32_t source_safe;
    uint32_t transport_safe;
    uint32_t data_durable; /* transfer completion alone never sets this */
    uint32_t reserved;
} NPUNVMETransferReceipt;

typedef struct {
    uint32_t crc32;
    uint32_t reserved;
    uint8_t sha256[32];
} NPUNVMETransferDigest;

int npu_nvme_submit_transfer(NPUNVMEContext *ctx,
    const NPUNVMETransferSpec *spec, NPUNVMERequest **out_request);
/* Receipt/digests available only after completion (-EAGAIN while pending).
 * Failed receipt is inspectable; digest output requires successful completion.
 * None of these calls drops the caller's request reference. */
int npu_nvme_get_transfer_receipt(NPUNVMERequest *request,
    NPUNVMETransferReceipt *receipt, uint32_t receipt_size);
int npu_nvme_get_transfer_digests(NPUNVMERequest *request,
    NPUNVMETransferDigest *digests, uint32_t capacity);

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

/** ABI major, checked before loading this runtime. */
uint32_t npu_nvme_abi_version(void);

/**
 * @brief Register parameter pointers for background persistence by the
 *        Reactor step poller.
 */
int npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes, int num_items);

// -- FaF listener control (I2) --

/** @brief Set the NPU-side probe-flag device address. */
int npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr);

int npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value);

/**
 * @brief Register the step_counter device pointer for the Reactor poller.
 *
 * @param ctx           context handle
 * @param dev_ptr       step_counter device (HBM) pointer
 * @param ckpt_interval trigger a write every N steps
 * @return 0 on success, -1 on error
 */
int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval);

/** @brief Return the self-allocated probe-flag device pointer (or NULL). */
void* npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx);

// -- Delta frame I/O (I3) --

/**
 * @brief Initialise the delta ring-buffer layout on disk.
 *
 * @param ctx              context handle
 * @param delta_slot_size  bytes per delta slot (256 MB = 268435456 recommended)
 * @param delta_slot_count number of slots in the ring (128 recommended)
 * @return 0 on success, -1 on error
 */
int npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t area_offset,
                        uint64_t delta_slot_size, uint32_t delta_slot_count);

/** @brief Return the byte offset of the delta ring on the NVMe device. */
uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx);

uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx);
uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx);

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

/* Delta frame I/O: migrated to Python side via build_chunks_host +
 * write_batch_host / read_batch.  The SPSC ring-buffer pipeline handles
 * arbitrary frame sizes without the 64 MB sync_meta_io limitation. */

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

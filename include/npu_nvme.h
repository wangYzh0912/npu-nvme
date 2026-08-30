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

/** @brief Poll a submitted request; result is returned once done is true. */
int npu_nvme_poll_request(NPUNVMERequest *request, int *done);

/** @brief Wait up to timeout_ms (zero means unbounded). */
int npu_nvme_wait_request(NPUNVMERequest *request, uint32_t timeout_ms);

/** @brief Release a terminal request.  Non-terminal requests are retained. */
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

/** @brief HBM write with one CRC32 result per logical (unpadded) chunk. */
int npu_nvme_write_batch_crc(NPUNVMEContext *ctx, void **npu_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes,
                             uint32_t *crc32_out, int num_items);

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

#ifdef __cplusplus
}
#endif

#endif // NPU_NVME_H

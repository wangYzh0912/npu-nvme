#ifndef NPU_NVME_H
#define NPU_NVME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Opaque context handle
     */
    typedef struct npu_nvme_context npu_nvme_context_t;

    /**
     * @brief Initialize NPU-NVMe transfer environment
     *
     * @param ctx Pointer to receive context handle
     * @param nvme_pci_addr NVMe PCI address (e.g., "0000:83:00.0")
     * @return 0 on success, negative on error
     */
    int npu_nvme_init(npu_nvme_context_t **ctx, const char *nvme_pci_addr);

    /**
     * @brief Write data from NPU to NVMe
     *
     * @param ctx Context handle
     * @param npu_buffer NPU buffer address
     * @param nvme_offset Byte offset on NVMe device
     * @param size Transfer size in bytes
     * @return 0 on success, negative on error
     */
    int npu_nvme_write(npu_nvme_context_t *ctx,
                       void *npu_buffer,
                       uint64_t nvme_offset,
                       size_t size,
                       size_t chunk_size_max);

    /**
     * @brief Read data from NVMe to NPU
     *
     * @param ctx Context handle
     * @param npu_buffer NPU buffer address
     * @param nvme_offset Byte offset on NVMe device
     * @param size Transfer size in bytes
     * @return 0 on success, negative on error
     */
    int npu_nvme_read(npu_nvme_context_t *ctx,
                      void *npu_buffer,
                      uint64_t nvme_offset,
                      size_t size,
                      size_t chunk_size_max);

    /**
     * @brief Cleanup and release all resources
     *
     * @param ctx Context handle
     */
    void npu_nvme_cleanup(npu_nvme_context_t *ctx);

    int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                             void **npu_buffers,
                             uint64_t *nvme_offsets,
                             size_t *sizes,
                             int num_items,
                             int pipeline_depth,
                             size_t chunk_size_max);

    int npu_nvme_write_pipeline(npu_nvme_context_t *ctx,
                                void *npu_buffer,
                                uint64_t nvme_offset,
                                size_t size,
                                int pipeline_depth,
                                size_t chunk_size_max); // 并发深度，建议4-8

    int npu_nvme_read_pipeline(npu_nvme_context_t *ctx,
                               void *npu_buffer,
                               uint64_t nvme_offset,
                               size_t size,
                               int pipeline_depth,
                               size_t chunk_size_max);

    int npu_nvme_write_batch_async(npu_nvme_context_t *ctx,
                                   void **npu_buffers,
                                   uint64_t *nvme_offsets,
                                   size_t *sizes,
                                   int num_items,
                                   int pipeline_depth,
                                   size_t chunk_size_max);

#ifdef __cplusplus
}
#endif

#endif // NPU_NVME_H
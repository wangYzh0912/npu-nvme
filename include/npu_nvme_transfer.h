#ifndef NPU_NVME_TRANSFER_H
#define NPU_NVME_TRANSFER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NPUNVMEContext NPUNVMEContext;

int npu_nvme_transfer_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                           int pipe_depth, int chunk_size, bool enable_profiling, const char *prof_dir);

void npu_nvme_transfer_cleanup(NPUNVMEContext *ctx);

uint64_t npu_nvme_transfer_get_total_blocks(NPUNVMEContext *ctx);

int npu_nvme_transfer_get_max_transfer(NPUNVMEContext *ctx);

int npu_nvme_transfer_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset, uint32_t total_bytes,
                                   int is_read, void *meta_buffer);

int npu_nvme_transfer_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                  uint64_t *nvme_offsets, size_t *sizes, int num_items);

int npu_nvme_transfer_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                 uint64_t *nvme_offsets, size_t *sizes, int num_items);

int npu_nvme_transfer_write_batch_host(NPUNVMEContext *ctx, void **ptrs,
                                       uint64_t *nvme_offsets, size_t *sizes, int num_items);

#ifdef __cplusplus
}
#endif

#endif // NPU_NVME_TRANSFER_H

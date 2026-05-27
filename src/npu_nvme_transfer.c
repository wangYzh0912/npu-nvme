#include "npu_nvme_transfer.h"
#include "npu_nvme.h"

int npu_nvme_transfer_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                           int pipe_depth, int chunk_size, bool enable_profiling, const char *prof_dir) {
    return npu_nvme_init(out_ctx, pci_addr, npu_id, pipe_depth, chunk_size, enable_profiling, prof_dir);
}

void npu_nvme_transfer_cleanup(NPUNVMEContext *ctx) {
    npu_nvme_cleanup(ctx);
}

uint64_t npu_nvme_transfer_get_total_blocks(NPUNVMEContext *ctx) {
    return npu_nvme_get_total_blocks(ctx);
}

int npu_nvme_transfer_get_max_transfer(NPUNVMEContext *ctx) {
    return npu_nvme_get_max_transfer(ctx);
}

int npu_nvme_transfer_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset, uint32_t total_bytes,
                                   int is_read, void *meta_buffer) {
    return npu_nvme_sync_meta_io(ctx, byte_offset, total_bytes, is_read, meta_buffer);
}

int npu_nvme_transfer_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                  uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    return npu_nvme_write_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items);
}

int npu_nvme_transfer_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                 uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    return npu_nvme_read_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items);
}

int npu_nvme_transfer_write_batch_host(NPUNVMEContext *ctx, void **ptrs,
                                       uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    return npu_nvme_write_batch_host(ctx, ptrs, nvme_offsets, sizes, num_items);
}

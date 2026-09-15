/* validation: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

bool test_fault(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

int namespace_capacity(NPUNVMEContext *ctx, uint64_t *capacity) {
    if (!ctx || !ctx->block_size || !ctx->total_blocks ||
        ctx->total_blocks > UINT64_MAX / ctx->block_size) return -1;
    *capacity = ctx->total_blocks * (uint64_t)ctx->block_size;
    return 0;
}

int validate_io_batch(NPUNVMEContext *ctx, void **ptrs,
                             uint64_t *nvme_offsets, size_t *sizes,
                             int num_items) {
    if (ctx && atomic_load(&ctx->quarantined)) return -EIO;
    if (ctx && (atomic_load(&ctx->app_should_stop) || atomic_load(&ctx->admission_closed)))
        return -ESHUTDOWN;
    if (!ctx || !ptrs || !nvme_offsets || !sizes || num_items <= 0 ||
        num_items > MAX_BATCH_ITEMS || atomic_load(&ctx->app_should_stop) ||
        atomic_load(&ctx->admission_closed) ||
        atomic_load(&ctx->quarantined) || ctx->block_size == 0 ||
        ctx->dma.chunk_size == 0 || ctx->dma.chunk_size % 4096 != 0) {
        return -1;
    }

    uint64_t capacity, total_bytes = 0;
    if (namespace_capacity(ctx, &capacity) != 0) return -1;
    for (int i = 0; i < num_items; i++) {
        if (!ptrs[i] || sizes[i] == 0 || sizes[i] > ctx->dma.chunk_size) {
            fprintf(stderr,
                    "[NPU-NVMe] Invalid I/O item %d: ptr=%p size=%zu\n",
                    i, ptrs[i], sizes[i]);
            return -1;
        }
        size_t aligned_sz = ALIGN_4K(sizes[i]);
        if (aligned_sz > MAX_BATCH_BYTES - total_bytes) return -1;
        total_bytes += aligned_sz;
        if (aligned_sz > ctx->dma.chunk_size ||
            nvme_offsets[i] % ctx->block_size != 0 ||
            aligned_sz % ctx->block_size != 0 ||
            nvme_offsets[i] > capacity ||
            aligned_sz > capacity - nvme_offsets[i]) {
            fprintf(stderr,
                    "[NPU-NVMe] Invalid I/O item %d: ptr=%p offset=%lu "
                    "size=%zu aligned=%zu\n",
                    i, ptrs[i], nvme_offsets[i], sizes[i], aligned_sz);
            return -1;
        }
    }
    return 0;
}

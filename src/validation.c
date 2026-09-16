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

int npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t area_offset,
                         uint64_t delta_slot_size, uint32_t delta_slot_count) {
    if (!ctx || area_offset % 4096 != 0 || delta_slot_size == 0 ||
        delta_slot_size % 4096 != 0 || delta_slot_count == 0) return -1;

    if (delta_slot_size > UINT64_MAX / delta_slot_count) return -1;
    uint64_t total_delta_bytes = delta_slot_size * delta_slot_count;
    uint64_t disk_bytes;
    if (namespace_capacity(ctx, &disk_bytes) != 0) return -1;
    if (area_offset > disk_bytes || total_delta_bytes > disk_bytes - area_offset)
        return -1;

    ctx->delta.area_offset = area_offset;
    ctx->delta.slot_size = delta_slot_size;
    ctx->delta.slot_count = delta_slot_count;

    fprintf(stderr, "[NPU-NVMe] Delta area: offset=%lu (%lu GB) "
            "slots=%u x %lu MB = %lu MB\n",
            ctx->delta.area_offset,
            ctx->delta.area_offset / (1024 * 1024 * 1024),
            delta_slot_count,
            delta_slot_size / (1024 * 1024),
            total_delta_bytes / (1024 * 1024));

    return 0;
}

uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.area_offset : 0;
}

uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.slot_size : 0;
}

uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx) {
    return ctx ? ctx->delta.slot_count : 0;
}

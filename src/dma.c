/* dma: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void own_dma_slot(NPUNVMEContext *ctx, io_task_t *task, int slot) {
    atomic_store(&ctx->dma.pool[slot].owned_bytes, task->size);
    atomic_store(&ctx->dma.pool[slot].owned_offset, task->nvme_offset);
    atomic_store_explicit(&ctx->dma.pool[slot].owner_request_id, task->request_id,
                          memory_order_release);
}

void release_dma_slot(NPUNVMEContext *ctx, int slot) {
    atomic_store_explicit(&ctx->dma.pool[slot].owner_request_id, 0, memory_order_release);
    ring_push(&ctx->dma.free_ring, slot);
}

uint32_t crc32_buffer(const void *data, size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320U & (-(int)(crc & 1U)));
    }
    return ~crc;
}

int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host,
                     bool async_dma) {
    int buf_idx;
    if (atomic_load(&ctx->quarantined)) return -3;
    if (ctx->enable_profiling && task->ts_slot_wait == 0)
        task->ts_slot_wait = get_time_us();
    if (ring_pop(&ctx->dma.free_ring, &buf_idx) != 0) return -1;

    task->buf_idx = buf_idx;
    own_dma_slot(ctx, task, buf_idx);
    dma_inflight_inc(ctx);
    if (ctx->enable_profiling) {
        task->ts_slot_acquire = get_time_us();
        task->ts_submit = task->ts_slot_acquire;
    }

    /* NVMe writes are block-aligned.  Zero reused-buffer padding so the
     * final partial block never exposes bytes from an earlier request. */
    size_t aligned_sz = ALIGN_4K(task->size);
    if (aligned_sz > task->size) {
        memset((char *)ctx->dma.pool[buf_idx].buf + task->size, 0,
               aligned_sz - task->size);
    }

    if (is_host) {
        memcpy(ctx->dma.pool[buf_idx].buf, task->npu_ptr, task->size);
        task->crc32 = 0;
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    } else if (async_dma) {
        if (test_fault("NPU_NVME_TEST_FAIL_ACL_COPY")) {
            release_dma_slot(ctx, buf_idx);
            task->buf_idx = -1;
            dma_inflight_dec(ctx);
            return -2;
        }
        atomic_fetch_add_explicit(&ctx->async_dma_submit_count, 1,
                                  memory_order_relaxed);
        aclError ret = aclrtMemcpyAsync(ctx->dma.pool[buf_idx].buf, task->size,
                                       task->npu_ptr, task->size,
                                       ACL_MEMCPY_DEVICE_TO_HOST,
                                       ctx->acl.copy_stream);
        if (ret == ACL_SUCCESS) {
            if (test_fault("NPU_NVME_TEST_FAIL_EVENT_RECORD")) {
                ret = ACL_ERROR_FAILURE;
            } else {
                ret = aclrtRecordEvent(ctx->acl.events[buf_idx],
                                       ctx->acl.copy_stream);
            }
            if (ret != ACL_SUCCESS) {
                atomic_fetch_add_explicit(&ctx->stream_sync_fallback_count, 1,
                                          memory_order_relaxed);
                /* The copy was accepted but cannot be polled.  Drain the
                 * stream before making this slot reusable. */
                aclError stop = aclrtSynchronizeStream(ctx->acl.copy_stream);
                if (test_fault("NPU_NVME_TEST_FAIL_STREAM_SYNC")) stop = ACL_ERROR_FAILURE;
                if (stop != ACL_SUCCESS) {
                    task->state = CHUNK_QUARANTINED;
                    atomic_store(&ctx->safety_reason, 1);
                    atomic_store_explicit(&ctx->quarantined, 1, memory_order_release);
                    return -3;
                }
            }
        }
        if (ret != ACL_SUCCESS) {
            release_dma_slot(ctx, buf_idx);
            task->buf_idx = -1;
            dma_inflight_dec(ctx);
            return -2;
        }
        task->state = CHUNK_NPU_COPYING;
    } else {
        aclError ret = aclrtMemcpy(ctx->dma.pool[buf_idx].buf, task->size,
                                   task->npu_ptr, task->size,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            release_dma_slot(ctx, buf_idx);
            task->buf_idx = -1;
            dma_inflight_dec(ctx);
            return -2;
        }
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        task->state = CHUNK_NPU_DONE;
    }
    return 0;
}

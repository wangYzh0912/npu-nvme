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

static uint32_t crc_table[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;
static void init_crc_table(void) {
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (unsigned j = 0; j < 8; ++j)
            c = (c >> 1) ^ (0xEDB88320U & (-(int)(c & 1U)));
        crc_table[i] = c;
    }
}
uint32_t crc32_buffer(const void *data, size_t size) {
    pthread_once(&crc_once, init_crc_table);
    const unsigned char *bytes = data;
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < size; ++i)
        crc = crc_table[(crc ^ bytes[i]) & 255U] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFU;
}

bool checksum_task(NPUNVMEContext *ctx, write_request_t *req, io_task_t *task,
                   size_t *budget) {
    if (task->checksum_done) return true;
    if (!req->compute_crc && !req->compute_sha256) return true;
    if (!*budget) return false;
    const unsigned char *data = ctx->dma.pool[task->buf_idx].buf;
    if (!task->checksum_offset) {
        task->checksum_crc=0xFFFFFFFFU;
        if (req->compute_crc) pthread_once(&crc_once,init_crc_table);
        if (req->compute_sha256) SHA256_Init(&task->checksum_sha);
    }
    size_t count=task->size-task->checksum_offset;
    if (count>*budget) count=*budget;
    data+=task->checksum_offset;
    if (req->compute_crc) {
        uint32_t crc=task->checksum_crc;
        for (size_t i=0;i<count;++i) crc=crc_table[(crc^data[i])&255U]^(crc>>8);
        task->checksum_crc=crc;
    }
    if (req->compute_sha256) SHA256_Update(&task->checksum_sha,data,count);
    task->checksum_offset+=count; *budget-=count;
    if (task->checksum_offset!=task->size) return false;
    if (req->compute_crc) task->crc32=task->checksum_crc^0xFFFFFFFFU;
    if (req->compute_sha256) SHA256_Final(task->sha256,&task->checksum_sha);
    task->checksum_done=true; return true;
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
        task->host_copy_offset = 0;
        task->state = CHUNK_HOST_COPYING;
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

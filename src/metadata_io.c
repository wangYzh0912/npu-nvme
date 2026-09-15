/* metadata_io: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void meta_io_complete_cb(void *arg, const struct spdk_nvme_cpl *cpl) {
    meta_request_t *req = (meta_request_t *)arg;
    req->result = spdk_nvme_cpl_is_error(cpl) ? -1 : 0;
    /* The callback only confirms the device command.  The reactor still has
     * to copy read data and detach ctx->meta_req before the caller may free
     * this request. */
    atomic_store_explicit(&req->io_done, 1, memory_order_release);
}

int wait_meta_request_done(NPUNVMEContext *ctx, meta_request_t *req) {
    uint64_t start = get_time_us();
    for (;;) {
        if (atomic_load_explicit(&req->done, memory_order_acquire))
            return 0;

        if (ctx->io_timeout_ms > 0 &&
            get_time_us() - start >= (uint64_t)ctx->io_timeout_ms * 1000ULL) {
            int expected = META_CALLER_WAITING;
            if (atomic_compare_exchange_strong_explicit(
                    &req->owner_state, &expected, META_CALLER_DETACHED,
                    memory_order_acq_rel, memory_order_acquire))
                return -ETIMEDOUT;
            /* COPYING means the reactor has borrowed meta_buffer.  Wait for
             * its bounded memcpy to finish before returning to the caller. */
        }
        usleep(1000);
    }
}

int meta_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;

    if (!ctx->meta_req) {
        void *obj = NULL;
        if (spdk_ring_dequeue(ctx->meta_ring, &obj, 1) == 1) {
            meta_request_t *req = (meta_request_t *)obj;
            if (atomic_load(&req->owner_state) == META_CALLER_DETACHED) {
                atomic_fetch_sub(&ctx->pending_requests, 1);
                meta_request_put(req);
            } else {
                ctx->meta_req = req;
            }
        }
    }

    meta_request_t *req = ctx->meta_req;
    if (!req) return ctx->app_should_stop ? -1 : 0;

    /* Deliberately non-blocking: a caller timeout must not wedge the reactor.
     * This delay is only used by the C fault-injection gate. */
    if (!req->submitted && req->submit_not_before_us != 0 &&
        get_time_us() < req->submit_not_before_us)
        return 0;

    if (!req->submitted) {
        uint64_t lba = req->byte_offset / ctx->block_size;
        uint32_t nblk = req->total_bytes / ctx->block_size;
        int rc;
        if (req->is_flush) {
            rc = spdk_nvme_ns_cmd_flush(ctx->ns, ctx->meta_qpair,
                                        meta_io_complete_cb, req);
        } else if (req->is_read) {
            rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->meta_qpair,
                                       ctx->meta_dma_buf, lba, nblk,
                                       meta_io_complete_cb, req, 0);
        } else {
            memcpy(ctx->meta_dma_buf, req->owned_buffer, req->total_bytes);
            rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->meta_qpair,
                                        ctx->meta_dma_buf, lba, nblk,
                                        meta_io_complete_cb, req, 0);
        }
        req->submitted = (rc == 0);
        if (rc != 0) {
            req->result = -1;
            atomic_store(&req->io_done, 1);
        }
    }

    if (req->submitted && !atomic_load_explicit(&req->io_done,
                                                memory_order_acquire))
        spdk_nvme_qpair_process_completions(ctx->meta_qpair, 0);

    if (atomic_load_explicit(&req->io_done, memory_order_acquire)) {
        int expected = META_CALLER_WAITING;
        bool caller_attached = atomic_compare_exchange_strong_explicit(
            &req->owner_state, &expected, META_REACTOR_COPYING,
            memory_order_acq_rel, memory_order_acquire);

        if (caller_attached && !req->is_flush && req->is_read &&
            req->result == 0) {
            /* Device -> reactor-owned storage -> caller storage.  The
             * caller is not released until both copies and ctx detachment
             * are complete. */
            memcpy(req->owned_buffer, ctx->meta_dma_buf, req->total_bytes);
            memcpy(req->meta_buffer, req->owned_buffer, req->total_bytes);
        }
        ctx->meta_req = NULL;
        if (caller_attached) {
            atomic_store_explicit(&req->owner_state, META_CALLER_DONE,
                                  memory_order_release);
            atomic_store_explicit(&req->done, 1, memory_order_release);
        }
        atomic_fetch_sub(&ctx->pending_requests, 1);
        meta_request_put(req);
    }

    return 0;
}

int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset,
                           uint32_t total_bytes, int is_read, void *meta_buffer) {
    if (!ctx || !meta_buffer) return -1;
    if (!is_read && test_fault("NPU_NVME_TEST_FAIL_METADATA_WRITE"))
        return -EIO;
    uint64_t capacity;
    if (namespace_capacity(ctx, &capacity) != 0 ||
        byte_offset > capacity || total_bytes > capacity - byte_offset ||
        (is_read != 0 && is_read != 1)) return -1;
    if (byte_offset % ctx->block_size != 0 ||
        total_bytes == 0 || total_bytes % ctx->block_size != 0) return -1;
    if (total_bytes > META_DMA_BUF_SIZE) {
        fprintf(stderr, "[NPU-NVMe] sync_meta_io: total_bytes=%u exceeds "
                "META_DMA_BUF_SIZE=%u\n", total_bytes, (unsigned)META_DMA_BUF_SIZE);
        return -1;
    }
    if (!ctx->meta_dma_buf) return -1;

    /* Use meta_ring → reactor (dedicated meta_qpair, no lock needed). */
    meta_request_t *req = calloc(1, sizeof(*req));
    if (!req) return -1;
    req->byte_offset = byte_offset;
    req->total_bytes = total_bytes;
    req->is_read = is_read;
    req->is_flush = 0;
    req->meta_buffer = meta_buffer;
    req->owned_buffer = malloc(total_bytes);
    if (!req->owned_buffer) { free(req); return -1; }
    if (!is_read) memcpy(req->owned_buffer, meta_buffer, total_bytes);
    atomic_init(&req->done, 0);
    atomic_init(&req->io_done, 0);
    atomic_init(&req->owner_state, META_CALLER_WAITING);
    req->result = 0;
    req->submit_not_before_us = 0;
    req->submitted = 0;
    const char *delay_env = getenv("NPU_NVME_TEST_META_DELAY_MS");
    if (delay_env && delay_env[0] != '\0') {
        char *end = NULL;
        unsigned long delay_ms = strtoul(delay_env, &end, 10);
        if (end != delay_env && *end == '\0')
            req->submit_not_before_us = get_time_us() + delay_ms * 1000ULL;
    }
    req->ctx = ctx;
    atomic_init(&req->refs, 2);
    atomic_fetch_add(&ctx->refs, 1);
    int enqueue_rc = enqueue_request(ctx, 2, req);
    if (enqueue_rc != 0) { meta_request_put(req); meta_request_put(req); return enqueue_rc; }

    /* Poll for completion. */
    if (wait_meta_request_done(ctx, req) != 0) {
        meta_request_put(req);
        return -ETIMEDOUT;
    }

    int result = req->result;
    meta_request_put(req);
    return result;
}

int npu_nvme_flush(NPUNVMEContext *ctx) {
    if (!ctx || !ctx->meta_ring || !ctx->meta_qpair) return -1;
    if (test_fault("NPU_NVME_TEST_FAIL_FLUSH")) return -EIO;
    meta_request_t *req = calloc(1, sizeof(*req));
    if (!req) return -1;
    req->is_flush = 1;
    req->is_read = 0;
    req->total_bytes = 0;
    atomic_init(&req->done, 0);
    atomic_init(&req->io_done, 0);
    atomic_init(&req->owner_state, META_CALLER_WAITING);
    req->result = 0;
    req->submitted = 0;
    req->ctx = ctx;
    atomic_init(&req->refs, 2);
    atomic_fetch_add(&ctx->refs, 1);
    int enqueue_rc = enqueue_request(ctx, 2, req);
    if (enqueue_rc != 0) { meta_request_put(req); meta_request_put(req); return enqueue_rc; }
    if (wait_meta_request_done(ctx, req) != 0) {
        meta_request_put(req);
        return -ETIMEDOUT;
    }
    int result = req->result;
    meta_request_put(req);
    return result;
}

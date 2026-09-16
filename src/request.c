/* request: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void identify_tasks(NPUNVMEContext *ctx, io_task_t *tasks, int count) {
    uint64_t id = atomic_fetch_add(&ctx->request_serial, 1) + 1;
    for (int i = 0; i < count; ++i) tasks[i].request_id = id;
}

void write_request_put(write_request_t *req) {
    if (atomic_fetch_sub_explicit(&req->refs, 1, memory_order_acq_rel) == 1) {
        NPUNVMEContext *ctx = req->ctx;
        free(req->tasks); free(req); context_put(ctx);
    }
}

void read_request_put(read_request_t *req) {
    if (atomic_fetch_sub_explicit(&req->refs, 1, memory_order_acq_rel) == 1) {
        NPUNVMEContext *ctx = req->ctx;
        free(req->tasks); free(req); context_put(ctx);
    }
}

void meta_request_put(meta_request_t *req) {
    if (atomic_fetch_sub_explicit(&req->refs, 1, memory_order_acq_rel) == 1) {
        NPUNVMEContext *ctx = req->ctx;
        free(req->owned_buffer); free(req); context_put(ctx);
    }
}

uint32_t finite_timeout(uint32_t timeout) { return timeout ? timeout : 60000; }

int wait_request_done(NPUNVMEContext *ctx, atomic_int *done) {
    uint32_t timeout = finite_timeout(ctx->io_timeout_ms);
    uint64_t start = get_time_us();
    while (!atomic_load_explicit(done, memory_order_acquire)) {
        if (get_time_us() - start >= (uint64_t)timeout * 1000ULL) return -ETIMEDOUT;
        usleep(1000);
    }
    return 0;
}

int wait_reactor_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms) {
    if (!ctx) return -EINVAL;
    uint64_t start = get_time_us();
    timeout_ms = finite_timeout(timeout_ms);
    for (;;) {
        if (atomic_load(&ctx->quarantined)) return -EIO;
        if (atomic_load(&ctx->pending_requests) == 0 &&
            atomic_load(&ctx->dma_inflight) == 0 &&
            atomic_load(&ctx->nvme_outstanding) == 0) return 0;
        if (get_time_us() - start >= (uint64_t)timeout_ms * 1000ULL) return -ETIMEDOUT;
        usleep(1000);
    }
}

void cancel_queued_requests(NPUNVMEContext *ctx) {
    /* Requests that timed out before the reactor dequeued them never entered
     * an FSM.  Reclaim those queue-owned objects before freeing the rings. */
    if (ctx->write_ring) {
        void *obj = NULL;
        while (spdk_ring_dequeue(ctx->write_ring, &obj, 1) == 1) {
            atomic_fetch_sub(&ctx->queued_writes, 1);
            write_request_t *req = (write_request_t *)obj;
            req->result = -ECANCELED;
            atomic_store_explicit(&req->done, 1, memory_order_release);
            atomic_fetch_sub(&ctx->pending_requests, 1);
            write_request_put(req);
        }
    }
    if (ctx->read_ring) {
        void *obj = NULL;
        while (spdk_ring_dequeue(ctx->read_ring, &obj, 1) == 1) {
            read_request_t *req = (read_request_t *)obj;
            req->result = -ECANCELED;
            atomic_store_explicit(&req->done, 1, memory_order_release);
            atomic_fetch_sub(&ctx->pending_requests, 1);
            read_request_put(req);
        }
    }
    if (ctx->meta_ring) {
        void *obj = NULL;
        while (spdk_ring_dequeue(ctx->meta_ring, &obj, 1) == 1) {
            meta_request_t *req = (meta_request_t *)obj;
            req->result = -ECANCELED;
            atomic_store_explicit(&req->done, 1, memory_order_release);
            atomic_fetch_sub(&ctx->pending_requests, 1);
            meta_request_put(req);
        }
    }
}

io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes) {
    io_task_t *tasks = calloc(num_tasks, sizeof(io_task_t));
    if (!tasks) {
        fprintf(stderr, "[Fatal] Failed to allocate memory for IO tasks.\n");
        return NULL;
    }
    for (int i = 0; i < num_tasks; i++) {
        tasks[i].task_idx = i;
        tasks[i].buf_idx = -1;
        tasks[i].state = CHUNK_IDLE;
        tasks[i].npu_ptr = npu_ptrs[i];
        tasks[i].nvme_offset = nvme_offsets[i];
        tasks[i].size = sizes[i];
    }
    return tasks;
}

int submit_write_common(NPUNVMEContext *ctx, void **ptrs,
                               uint64_t *nvme_offsets, size_t *sizes,
                               int num_items, bool is_host, bool compute_crc,
                               bool async_dma,
                               NPUNVMERequest **out_request) {
    if (!out_request) return -EINVAL;
    *out_request = NULL;
    int validation = validate_io_batch(ctx, ptrs, nvme_offsets, sizes, num_items);
    if (validation != 0) return validation == -1 ? -EINVAL : validation;

    write_request_t *req = calloc(1, sizeof(*req));
    if (!req) return -ENOMEM;
    req->tasks = create_io_tasks(num_items, ptrs, nvme_offsets, sizes);
    if (!req->tasks) {
        free(req);
        return -ENOMEM;
    }
    req->ctx = ctx;
    identify_tasks(ctx, req->tasks, num_items);
    req->num_tasks = num_items;
    req->is_host = is_host;
    req->async_dma = async_dma;
    req->compute_crc = compute_crc;
    atomic_init(&req->done, 0);
    atomic_init(&req->detached, 0);

    uint32_t queued = atomic_load(&ctx->queued_writes);
    for (int i = 0; i < num_items; ++i)
        req->tasks[i].queue_depth = queued + 1;
    atomic_init(&req->refs, 2);
    atomic_fetch_add(&ctx->refs, 1);
    req->timeout_ms = finite_timeout(ctx->io_timeout_ms);
    int enqueue_rc = enqueue_request(ctx, 0, req);
    if (enqueue_rc != 0) {
        write_request_put(req); write_request_put(req);
        return enqueue_rc;
    }
    update_peak_uint(&ctx->request_ring_peak, queued + 1);
    *out_request = req;
    return 0;
}

int npu_nvme_submit_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                uint64_t *nvme_offsets, size_t *sizes,
                                int num_items, NPUNVMERequest **out_request) {
    return submit_write_common(ctx, npu_ptrs, nvme_offsets, sizes, num_items,
                               false, false, true, out_request);
}

int npu_nvme_submit_write_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                                     uint64_t *nvme_offsets, size_t *sizes,
                                     int num_items,
                                     NPUNVMERequest **out_request) {
    return submit_write_common(ctx, host_ptrs, nvme_offsets, sizes, num_items,
                               true, false, false, out_request);
}

int npu_nvme_poll_request(NPUNVMERequest *request, int *done) {
    if (!request || !done) return -EINVAL;
    *done = atomic_load_explicit(&request->done, memory_order_acquire);
    return *done ? request->result : 0;
}

int npu_nvme_wait_request(NPUNVMERequest *request, uint32_t timeout_ms) {
    if (!request) return -EINVAL;
    uint64_t start = get_time_us();
    if (!timeout_ms) timeout_ms = finite_timeout(request->timeout_ms);
    while (!atomic_load_explicit(&request->done, memory_order_acquire)) {
        if (timeout_ms && get_time_us() - start >= timeout_ms * 1000ULL) {
            /* Stop new admission but keep the reactor polling for late completion. */
            atomic_store(&request->ctx->admission_closed, 1);
            atomic_store(&request->ctx->safety_reason, 3);
            return -ETIMEDOUT;
        }
        usleep(1000);
    }
    return request->result;
}

void npu_nvme_release_request(NPUNVMERequest *request) {
    if (request) write_request_put(request);
}

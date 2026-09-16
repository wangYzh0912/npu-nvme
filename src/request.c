/* request: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void identify_tasks(NPUNVMEContext *ctx, io_task_t *tasks, int count) {
    uint64_t id = atomic_fetch_add(&ctx->request_serial, 1) + 1;
    for (int i = 0; i < count; ++i) tasks[i].request_id = id;
}

void write_request_put(write_request_t *req) {
    if (atomic_fetch_sub_explicit(&req->refs, 1, memory_order_acq_rel) == 1) {
        NPUNVMEContext *ctx = req->ctx;
        for (uint32_t i=0; i<req->dependency_count; ++i) write_request_put(req->dependencies[i]);
        free(req->dependencies); free(req->owned_buffer);
        free(req->tasks); free(req); context_put(ctx);
    }
}

void read_request_put(read_request_t *req) { write_request_put(req); }
void meta_request_put(meta_request_t *req) { write_request_put(req); }

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
            forget_accepted_write(ctx, req);
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
    for (int direction=0; direction<2; ++direction) {
        write_request_t *req;
        while ((req=ctx->ready_head[direction])) {
            ctx->ready_head[direction]=req->ready_next;
            req->ready_next=NULL; req->result=-ECANCELED;
            if (!direction) forget_accepted_write(ctx,req);
            atomic_store_explicit(&req->done,1,memory_order_release);
            atomic_fetch_sub(&ctx->pending_requests,1); write_request_put(req);
        }
        ctx->ready_tail[direction]=NULL;
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

static int submit_write_owned(NPUNVMEContext *ctx, void **ptrs,
                               uint64_t *nvme_offsets, size_t *sizes,
                               int num_items, bool is_host, bool compute_crc,
                               bool async_dma, bool compute_sha256,
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
    req->compute_sha256 = compute_sha256;
    req->operation = NPU_NVME_TRANSFER_WRITE;
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

int submit_write_common(NPUNVMEContext *ctx, void **ptrs,
    uint64_t *offsets, size_t *sizes, int count, bool host, bool crc,
    bool async_dma, NPUNVMERequest **out) {
    return submit_write_owned(ctx, ptrs, offsets, sizes, count, host, crc,
                              async_dma, false, out);
}

static int submit_read_owned(NPUNVMEContext *ctx, void **ptrs,
        uint64_t *offsets, size_t *sizes, const NPUNVMETransferSpec *spec,
        NPUNVMERequest **out) {
    int rc = validate_io_batch(ctx, ptrs, offsets, sizes, spec->item_count);
    if (rc) return rc == -1 ? -EINVAL : rc;
    read_request_t *req = calloc(1, sizeof(*req));
    if (!req) return -ENOMEM;
    req->tasks = create_io_tasks(spec->item_count, ptrs, offsets, sizes);
    if (!req->tasks) { free(req); return -ENOMEM; }
    req->ctx = ctx; req->num_tasks = spec->item_count;
    req->is_host = spec->memory_kind == NPU_NVME_MEMORY_HOST;
    req->async_dma = true; req->operation = NPU_NVME_TRANSFER_READ;
    req->compute_crc = spec->checksum_flags & NPU_NVME_CHECK_CRC32;
    req->compute_sha256 = spec->checksum_flags & NPU_NVME_CHECK_SHA256;
    req->timeout_ms = finite_timeout(ctx->io_timeout_ms);
    identify_tasks(ctx, req->tasks, req->num_tasks);
    for (int i = 0; i < req->num_tasks; ++i) {
        req->tasks[i].expected_crc32 = spec->items[i].expected_crc32;
        memcpy(req->tasks[i].expected_sha256, spec->items[i].expected_sha256, 32);
    }
    atomic_init(&req->done, 0); atomic_init(&req->detached, 0);
    atomic_init(&req->refs, 2); atomic_fetch_add(&ctx->refs, 1);
    rc = enqueue_request(ctx, 1, req);
    if (rc) { read_request_put(req); read_request_put(req); return rc; }
    *out = req; return 0;
}

int npu_nvme_submit_transfer(NPUNVMEContext *ctx,
        const NPUNVMETransferSpec *spec, NPUNVMERequest **out) {
    if (!out) return -EINVAL;
    *out = NULL;
    if (!spec || spec->struct_size != sizeof(*spec) ||
        spec->version != NPU_NVME_TRANSFER_VERSION ||
        spec->memory_kind > NPU_NVME_MEMORY_HOST ||
        (spec->checksum_flags & ~(NPU_NVME_CHECK_CRC32 | NPU_NVME_CHECK_SHA256)))
        return -EINVAL;
    if (spec->operation > NPU_NVME_TRANSFER_FLUSH) return -ENOTSUP;
    if (spec->operation >= NPU_NVME_TRANSFER_META_READ) {
        if (spec->memory_kind != NPU_NVME_MEMORY_HOST || spec->checksum_flags) return -EINVAL;
        if (spec->operation == NPU_NVME_TRANSFER_FLUSH) {
            if (spec->item_count || spec->items) return -EINVAL;
            return submit_meta_owned(ctx,0,0,0,1,NULL,out);
        }
        if (spec->item_count != 1 || !spec->items || spec->items[0].reserved ||
            spec->items[0].length > UINT32_MAX) return -EINVAL;
        return submit_meta_owned(ctx,spec->items[0].offset,spec->items[0].length,
            spec->operation==NPU_NVME_TRANSFER_META_READ,0,spec->items[0].address,out);
    }
    if (!spec->items || !spec->item_count || spec->item_count > MAX_BATCH_ITEMS) return -EINVAL;
    size_t count = spec->item_count;
    void **ptrs = calloc(count, sizeof(*ptrs));
    uint64_t *offsets = calloc(count, sizeof(*offsets));
    size_t *sizes = calloc(count, sizeof(*sizes));
    int rc = -ENOMEM;
    if (!ptrs || !offsets || !sizes) goto finish;
    for (size_t i = 0; i < count; ++i) {
        if (spec->items[i].reserved || spec->items[i].length > SIZE_MAX) {
            rc = -EINVAL; goto finish;
        }
        ptrs[i] = spec->items[i].address;
        offsets[i] = spec->items[i].offset;
        sizes[i] = spec->items[i].length;
    }
    if (spec->operation == NPU_NVME_TRANSFER_READ) {
        rc = submit_read_owned(ctx, ptrs, offsets, sizes, spec, out);
        goto finish;
    }
    rc = submit_write_owned(ctx, ptrs, offsets, sizes, count,
        spec->memory_kind == NPU_NVME_MEMORY_HOST,
        spec->checksum_flags & NPU_NVME_CHECK_CRC32, true,
        spec->checksum_flags & NPU_NVME_CHECK_SHA256, out);
finish:
    free(ptrs); free(offsets); free(sizes);
    return rc;
}

int npu_nvme_get_transfer_receipt(NPUNVMERequest *req,
        NPUNVMETransferReceipt *out, uint32_t size) {
    if (!req || !out || size != sizeof(*out)) return -EINVAL;
    if (!atomic_load_explicit(&req->done, memory_order_acquire)) return -EAGAIN;
    memset(out, 0, sizeof(*out));
    out->done = 1; out->result = req->result;
    out->operation = req->operation; out->item_count = req->num_tasks;
    for (int i = 0; i < req->num_tasks; ++i) out->logical_bytes += req->tasks[i].size;
    if (req->num_tasks) out->request_id = req->tasks[0].request_id;
    else { out->request_id=req->request_id; out->logical_bytes=req->total_bytes;
        out->item_count=req->is_flush ? 0 : 1; }
    out->data_durable = req->is_flush && req->result==0;
    /* FAILED is terminal only after all asynchronous borrowers are stopped. */
    out->source_safe = out->transport_safe = 1;
    return 0;
}

int npu_nvme_get_transfer_digests(NPUNVMERequest *req,
        NPUNVMETransferDigest *out, uint32_t capacity) {
    if (!req || !out || capacity < (uint32_t)req->num_tasks) return -EINVAL;
    if (!atomic_load_explicit(&req->done, memory_order_acquire)) return -EAGAIN;
    if (req->result) return req->result;
    if (!req->compute_crc && !req->compute_sha256) return -ENODATA;
    for (int i = 0; i < req->num_tasks; ++i) {
        memset(&out[i], 0, sizeof(out[i]));
        out[i].crc32 = req->tasks[i].crc32;
        memcpy(out[i].sha256, req->tasks[i].sha256, sizeof(out[i].sha256));
    }
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

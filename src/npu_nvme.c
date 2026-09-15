/* npu_nvme: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

uint32_t npu_nvme_get_abi_version(void) { return NPU_NVME_ABI_VERSION; }

int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                          uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    NPUNVMERequest *req = NULL;
    int rc = submit_write_common(ctx, npu_ptrs, nvme_offsets, sizes, num_items,
                                 false, false, &req);
    if (rc != 0) return rc;
    rc = npu_nvme_wait_request(req, ctx->io_timeout_ms);
    /* Legacy raw-pointer ABI cannot release the caller buffer on timeout.
     * Keep draining here until callers migrate to submit/poll ownership. */
    if (rc == -ETIMEDOUT) {
        while (!atomic_load_explicit(&req->done, memory_order_acquire)) usleep(1000);
    }
    npu_nvme_release_request(req);
    int result = rc;
    return result;
}

int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **ptrs,
                               uint64_t *nvme_offsets, size_t *sizes,
                               int num_items) {
    if (validate_io_batch(ctx, ptrs, nvme_offsets, sizes, num_items) != 0)
        return -1;

    NPUNVMERequest *req = NULL;
    int rc = submit_write_common(ctx, ptrs, nvme_offsets, sizes, num_items,
                                 true, false, &req);
    if (rc != 0) return rc;
    rc = npu_nvme_wait_request(req, ctx->io_timeout_ms);
    /* Legacy raw-pointer ABI cannot release the caller buffer on timeout.
     * Keep draining here until callers migrate to submit/poll ownership. */
    if (rc == -ETIMEDOUT) {
        while (!atomic_load_explicit(&req->done, memory_order_acquire)) usleep(1000);
    }
    npu_nvme_release_request(req);
    return rc;
}

int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                         uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (validate_io_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items) != 0)
        return -1;
    if (aclrtSetCurrentContext(ctx->acl.acl_ctx) != ACL_SUCCESS) return -1;

    read_request_t *req = calloc(1, sizeof(read_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, npu_ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    identify_tasks(ctx, req->tasks, num_items);
    req->num_tasks = num_items;
    req->is_host = false;
    req->done = 0;
    req->result = 0;

    req->ctx = ctx;
    atomic_init(&req->refs, 2);
    atomic_fetch_add(&ctx->refs, 1);
    int enqueue_rc = enqueue_request(ctx, 1, req);
    if (enqueue_rc != 0) { read_request_put(req); read_request_put(req); return enqueue_rc; }

    if (wait_request_done(ctx, &req->done) != 0) {
        /* Preserve the legacy destination-buffer lifetime until migration. */
        while (!atomic_load_explicit(&req->done, memory_order_acquire)) usleep(1000);
        read_request_put(req);
        return -ETIMEDOUT;
    }

    int result = req->result;
    read_request_put(req);
    return result;
}

int npu_nvme_read_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes,
                             int num_items) {
    if (validate_io_batch(ctx, host_ptrs, nvme_offsets, sizes, num_items) != 0)
        return -1;

    read_request_t *req = calloc(1, sizeof(read_request_t));
    if (!req) return -1;

    io_task_t *tasks = create_io_tasks(num_items, host_ptrs, nvme_offsets, sizes);
    if (!tasks) { free(req); return -1; }

    req->tasks = tasks;
    identify_tasks(ctx, req->tasks, num_items);
    req->num_tasks = num_items;
    req->is_host = true;
    req->done = 0;
    req->result = 0;

    req->ctx = ctx;
    atomic_init(&req->refs, 2);
    atomic_fetch_add(&ctx->refs, 1);
    int enqueue_rc = enqueue_request(ctx, 1, req);
    if (enqueue_rc != 0) { read_request_put(req); read_request_put(req); return enqueue_rc; }

    if (wait_request_done(ctx, &req->done) != 0) {
        /* Preserve the legacy destination-buffer lifetime until migration. */
        while (!atomic_load_explicit(&req->done, memory_order_acquire)) usleep(1000);
        read_request_put(req);
        return -ETIMEDOUT;
    }
    /* Host reads skip profiling. */
    int result = req->result;
    read_request_put(req);
    return result;
}

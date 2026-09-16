/* Compile the production implementation, replacing only external device calls.
 * No SPDK/ACL initialization, NVMe access, or device memory is used. */
#include <assert.h>
#include <limits.h>
#include <sys/mman.h>
#include "../../src/runtime.c"
#include "../../src/reactor.c"
#include "../../src/request.c"
#include "../../src/validation.c"
#include "../../src/dma.c"
#include "../../src/write_pipeline.c"
#include "../../src/read_pipeline.c"
#include "../../src/metadata_io.c"
#include "../../src/step_listener.c"
#include "../../src/metrics.c"
#include "../../src/npu_nvme.c"

static int copy_result, record_result, sync_result;
static unsigned enqueues;
static int query_result, event_sync_result;
static spdk_nvme_cmd_cb pending_callback;
static void *pending_callback_arg;
struct spdk_ring { void *object; };
size_t spdk_ring_count(struct spdk_ring *r) { return r && r->object ? 1 : 0; }
size_t spdk_ring_enqueue(struct spdk_ring *r, void **objects, size_t n, size_t *free_space) {
    (void)free_space; ++enqueues;
    if (!r || r->object || n != 1) return 0;
    r->object = objects[0]; return 1;
}
size_t spdk_ring_dequeue(struct spdk_ring *r, void **objects, size_t n) {
    if (!r || !r->object || n != 1) return 0;
    objects[0] = r->object; r->object = NULL; return 1;
}
aclError aclrtMemcpyAsync(void *d, size_t cap, const void *s, size_t n,
                        aclrtMemcpyKind kind, aclrtStream stream) {
    (void)d; (void)cap; (void)s; (void)n; (void)kind; (void)stream;
    return copy_result;
}
aclError aclrtMemcpy(void *d, size_t cap, const void *s, size_t n, aclrtMemcpyKind kind) {
    (void)cap; (void)kind; memcpy(d, s, n); return ACL_SUCCESS;
}
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    (void)event; (void)stream; return record_result;
}
aclError aclrtSynchronizeStream(aclrtStream stream) { (void)stream; return sync_result; }
aclError aclrtSetDevice(int32_t device) { (void)device; return ACL_SUCCESS; }
aclError aclrtSetCurrentContext(aclrtContext context) { (void)context; return ACL_SUCCESS; }
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    (void)event; *status = ACL_EVENT_RECORDED_STATUS_COMPLETE; return query_result;
}
aclError aclrtSynchronizeEvent(aclrtEvent event) { (void)event; return event_sync_result; }
aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
    (void)event; (void)stream; return ACL_SUCCESS;
}
int spdk_nvme_ns_cmd_write(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair,
        void *payload, uint64_t lba, uint32_t count, spdk_nvme_cmd_cb callback,
        void *arg, uint32_t flags) {
    (void)ns; (void)qpair; (void)payload; (void)lba; (void)count; (void)flags;
    assert(!pending_callback); pending_callback = callback; pending_callback_arg = arg;
    return 0;
}
int32_t spdk_nvme_qpair_process_completions(struct spdk_nvme_qpair *qpair, uint32_t max) {
    (void)qpair; (void)max;
    if (pending_callback) {
        spdk_nvme_cmd_cb callback = pending_callback;
        struct spdk_nvme_cpl completion = {0};
        pending_callback = NULL; callback(pending_callback_arg, &completion); return 1;
    }
    return 0;
}

static NPUNVMEContext context(void) {
    NPUNVMEContext ctx = {0};
    ctx.block_size = 4096; ctx.total_blocks = 1024; ctx.dma.chunk_size = 4096;
    ctx.io_timeout_ms = 20;
    atomic_init(&ctx.refs, 1);
    return ctx;
}
static atomic_int release_test_reactor;
static void *finish_test_reactor(void *arg) {
    NPUNVMEContext *ctx = arg;
    while (!atomic_load(&release_test_reactor)) usleep(1000);
    atomic_store_explicit(&ctx->reactor_exited, 1, memory_order_release);
    return NULL;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    NPUNVMEContext ctx = context();
    char bytes[4096] = {0};
    void *ptrs[] = {bytes}; uint64_t offsets[] = {0}; size_t sizes[] = {4};
    if (!strcmp(argv[1], "delta_overflow")) {
        assert(npu_nvme_delta_init(&ctx, 0, UINT64_C(1) << 63, 2) != 0);
        assert(ctx.delta.slot_count == 0);
    } else if (!strcmp(argv[1], "batch_bound")) {
        /* None of these arrays may be dereferenced before count validation. */
        void *guard = mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(guard != MAP_FAILED);
        assert(validate_io_batch(&ctx, guard, guard, guard, INT_MAX) != 0);
        munmap(guard, 4096);
    } else if (!strcmp(argv[1], "batch_bytes")) {
        enum { COUNT = 65 };
        void *many_ptrs[COUNT]; uint64_t many_offsets[COUNT]; size_t many_sizes[COUNT];
        ctx.total_blocks = UINT64_C(1) << 30;
        ctx.dma.chunk_size = UINT32_C(1) << 30;
        for (int i = 0; i < COUNT; ++i) {
            many_ptrs[i] = bytes; many_offsets[i] = 0; many_sizes[i] = ctx.dma.chunk_size;
        }
        assert(validate_io_batch(&ctx, many_ptrs, many_offsets, many_sizes, 64) == 0);
        assert(validate_io_batch(&ctx, many_ptrs, many_offsets, many_sizes, COUNT) != 0);
    } else if (!strcmp(argv[1], "geometry_overflow")) {
        ctx.total_blocks = UINT64_MAX / 4096 + 2;
        assert(validate_io_batch(&ctx, ptrs, offsets, sizes, 1) != 0);
    } else if (!strcmp(argv[1], "metadata_capacity")) {
        ctx.meta_dma_buf = bytes;
        assert(npu_nvme_sync_meta_io(&ctx, 1024 * 4096, 4096, 1, bytes) != 0);
        assert(enqueues == 0);
    } else if (!strcmp(argv[1], "legal_bounds")) {
        offsets[0] = 1023 * 4096;
        assert(validate_io_batch(&ctx, ptrs, offsets, sizes, 1) == 0);
        assert(npu_nvme_delta_init(&ctx, 0, 4096, 1024) == 0);
    } else if (!strcmp(argv[1], "queued_close") || !strcmp(argv[1], "early_release") ||
               !strcmp(argv[1], "late_completion")) {
        struct spdk_ring ring = {0}; ctx.write_ring = &ring;
        pthread_mutex_init(&ctx.state_lock, NULL); ctx.state_lock_initialized = true;
        NPUNVMERequest *req = NULL;
        assert(npu_nvme_submit_write_batch_host(&ctx, ptrs, offsets, sizes, 1, &req) == 0);
        assert(atomic_load(&ctx.refs) == 2 && atomic_load(&ctx.pending_requests) == 1);
        if (!strcmp(argv[1], "early_release")) {
            npu_nvme_release_request(req);
            assert(atomic_load(&ctx.refs) == 2); /* queue still owns it */
            cancel_queued_requests(&ctx);
        } else if (!strcmp(argv[1], "late_completion")) {
            assert(npu_nvme_wait_request(req, 1) == -ETIMEDOUT);
            int done = 9;
            assert(npu_nvme_poll_request(req, &done) == 0 && !done);
            cancel_queued_requests(&ctx);
            assert(npu_nvme_poll_request(req, &done) == -ECANCELED && done);
            npu_nvme_release_request(req);
        } else {
            assert(npu_nvme_close(&ctx, 1) == 0);
            cancel_queued_requests(&ctx);
            int done = 0;
            assert(npu_nvme_poll_request(req, &done) == -ECANCELED && done);
            NPUNVMERequest *other = NULL;
            assert(npu_nvme_submit_write_batch_host(&ctx, ptrs, offsets, sizes, 1, &other) != 0);
            npu_nvme_release_request(req);
        }
        assert(atomic_load(&ctx.refs) == 1 && atomic_load(&ctx.pending_requests) == 0);
        pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1], "close_deadline")) {
        ctx.reactor_pthread_started = true; /* no completed reactor to join */
        uint64_t start = get_time_us();
        assert(npu_nvme_close(&ctx, 2) == -ETIMEDOUT);
        assert(get_time_us() - start < 1000000);
        assert(ctx.reactor_pthread_started && atomic_load(&ctx.app_should_stop));
    } else if (!strcmp(argv[1], "close_retry")) {
        atomic_init(&release_test_reactor, 0);
        assert(pthread_create(&ctx.reactor_pthread, NULL, finish_test_reactor, &ctx) == 0);
        ctx.reactor_pthread_started = true;
        assert(npu_nvme_close(&ctx, 2) == -ETIMEDOUT);
        assert(ctx.reactor_pthread_started && atomic_load(&ctx.app_should_stop));
        atomic_store(&release_test_reactor, 1);
        assert(npu_nvme_close(&ctx, 1000) == 0);
        assert(!ctx.reactor_pthread_started && atomic_load(&ctx.reactor_exited));
        assert(npu_nvme_close(&ctx, 1) == 0); /* idempotent until cleanup */
    } else if (!strcmp(argv[1], "metadata_timeout")) {
        struct spdk_ring ring = {0}; ctx.meta_ring = &ring; ctx.meta_dma_buf = bytes;
        pthread_mutex_init(&ctx.state_lock, NULL); ctx.state_lock_initialized = true;
        assert(npu_nvme_sync_meta_io(&ctx, 0, 4096, 1, bytes) == -ETIMEDOUT);
        assert(atomic_load(&ctx.refs) == 2 && atomic_load(&ctx.pending_requests) == 1);
        cancel_queued_requests(&ctx);
        assert(atomic_load(&ctx.refs) == 1 && atomic_load(&ctx.pending_requests) == 0);
        pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1], "write_completion") || !strcmp(argv[1], "query_quarantine") ||
               !strcmp(argv[1], "lost_completion")) {
        struct spdk_ring ring = {0}; ctx.write_ring = &ring;
        pthread_mutex_init(&ctx.state_lock, NULL); ctx.state_lock_initialized = true;
        dma_buf_t buffer = {.buf=bytes}; aclrtEvent event = (void *)1;
        ctx.dma.pool = &buffer; ctx.dma.max_pipe_depth = 1; ctx.acl.events = &event;
        assert(ring_init(&ctx.dma.free_ring, 2) == 0);
        assert(ring_push(&ctx.dma.free_ring, 0) == 0);
        NPUNVMERequest *req = NULL;
        assert(npu_nvme_submit_write_batch(&ctx, ptrs, offsets, sizes, 1, &req) == 0);
        assert(write_fsm_poller_fn(&ctx) == 0); /* accepted DMA */
        assert(atomic_load(&ctx.dma_inflight) == 1);
        if (!strcmp(argv[1], "query_quarantine")) {
            query_result = event_sync_result = sync_result = ACL_ERROR_FAILURE;
            assert(write_fsm_poller_fn(&ctx) == 0);
            assert(atomic_load(&ctx.quarantined));
            assert(ring_is_empty(&ctx.dma.free_ring));
            assert(!atomic_load(&req->done));
            assert(npu_nvme_wait_request(req, 1) == -ETIMEDOUT);
            assert(npu_nvme_close(&ctx, 1) == -EIO);
            /* Teardown only in the host fake, which never schedules real DMA. */
            ctx.write_fsm.req = NULL;
            write_request_put(req);
        } else {
            if (!strcmp(argv[1], "lost_completion")) {
                assert(npu_nvme_wait_request(req, 1) == -ETIMEDOUT);
                assert(!atomic_load(&req->done) && atomic_load(&ctx.admission_closed));
                NPUNVMERetainedSlot retained[1]; uint32_t count = 0;
                assert(npu_nvme_get_retained_slots(&ctx, retained, 1, &count) == 0);
                assert(count == 1 && retained[0].reason == 3 && retained[0].request_id != 0);
                NPUNVMERequest *other = NULL;
                assert(npu_nvme_submit_write_batch(&ctx, ptrs, offsets, sizes, 1, &other) != 0);
            }
            /* Caller releases before NVMe callback; reactor owns final access. */
            npu_nvme_release_request(req); req = NULL;
            assert(write_fsm_poller_fn(&ctx) == 0); /* enqueue NVMe callback */
            assert(write_fsm_poller_fn(&ctx) == 0); /* callback and final release */
            assert(ctx.write_fsm.req == NULL);
            assert(atomic_load(&ctx.refs) == 1 && atomic_load(&ctx.pending_requests) == 0);
            assert(atomic_load(&ctx.dma_inflight) == 0);
        }
        if (req) npu_nvme_release_request(req);
        ring_free(&ctx.dma.free_ring); pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1], "dma_quarantine") || !strcmp(argv[1], "dma_stopped") ||
               !strcmp(argv[1], "copy_rejected")) {
        dma_buf_t buffer = {.buf=bytes}; aclrtEvent event = (void *)1;
        ctx.dma.pool = &buffer; ctx.dma.max_pipe_depth = 1; ctx.acl.events = &event;
        assert(ring_init(&ctx.dma.free_ring, 2) == 0);
        assert(ring_push(&ctx.dma.free_ring, 0) == 0);
        io_task_t task = {.npu_ptr=bytes,.size=4,.request_id=42};
        record_result = ACL_ERROR_FAILURE;
        if (!strcmp(argv[1], "copy_rejected")) copy_result = ACL_ERROR_FAILURE;
        sync_result = !strcmp(argv[1], "dma_quarantine") ? ACL_ERROR_FAILURE : ACL_SUCCESS;
        assert(try_submit_async(&ctx, &task, false, true) < 0);
        if (sync_result != ACL_SUCCESS) {
            assert(ring_is_empty(&ctx.dma.free_ring));
            assert(atomic_load(&ctx.dma_inflight) == 1);
            assert(atomic_load(&ctx.quarantined));
            NPUNVMERetainedSlot retained[1]; uint32_t count = 0;
            assert(npu_nvme_get_retained_slots(&ctx, retained, 1, &count) == 0);
            assert(count == 1 && retained[0].request_id == 42 && retained[0].bytes == 4);
            assert(retained[0].slot == 0 && retained[0].reason == 1);
            assert(npu_nvme_close(&ctx, 1) == -EIO);
            assert(npu_nvme_close(&ctx, 1) == -EIO);
        } else {
            assert(!ring_is_empty(&ctx.dma.free_ring));
            assert(atomic_load(&ctx.dma_inflight) == 0);
        }
        ring_free(&ctx.dma.free_ring);
    } else return 2;
    puts("PASS");
    return 0;
}

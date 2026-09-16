/* Compile the production implementation, replacing only external device calls.
 * No SPDK/ACL initialization, NVMe access, or device memory is used. */
#include <assert.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/wait.h>
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
static int emulate_copy, event_not_ready, completion_failed;
static const void *simulated_disk;
static unsigned enqueues;
static int query_result, event_sync_result;
static spdk_nvme_cmd_cb pending_callback;
static void *pending_callback_arg;
struct spdk_thread { int state, polls, destroyed; };
static struct spdk_thread *test_current_thread;
struct spdk_thread *spdk_get_thread(void) { return test_current_thread; }
void spdk_set_thread(struct spdk_thread *thread) { test_current_thread = thread; }
int spdk_thread_exit(struct spdk_thread *thread) {
    assert(test_current_thread == thread && !thread->destroyed);
    thread->state = 1; return 0;
}
bool spdk_thread_is_exited(struct spdk_thread *thread) { return thread->state == 2; }
int spdk_thread_poll(struct spdk_thread *thread, uint32_t max, uint64_t now) {
    (void)max; (void)now;
    assert(test_current_thread == thread && thread->state == 1 && !thread->destroyed);
    if (++thread->polls == 3) thread->state = 2;
    return 0;
}
void spdk_thread_destroy(struct spdk_thread *thread) {
    assert(thread->state == 2 && !thread->destroyed);
    thread->destroyed++;
}
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
    if (emulate_copy && copy_result == ACL_SUCCESS) memcpy(d, s, n);
    return copy_result;
}
aclError aclrtMemcpy(void *d, size_t cap, const void *s, size_t n, aclrtMemcpyKind kind) {
    (void)cap; (void)kind; memcpy(d, s, n); return ACL_SUCCESS;
}
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    (void)event; (void)stream; return record_result;
}
aclError aclrtSynchronizeStream(aclrtStream stream) { (void)stream; return sync_result; }
aclError aclrtMallocHost(void **p, size_t n) { (void)p; (void)n; assert(!"unexpected ACL Host allocation"); return 1; }
aclError aclrtSetDevice(int32_t device) { (void)device; return ACL_SUCCESS; }
aclError aclrtSetCurrentContext(aclrtContext context) { (void)context; return ACL_SUCCESS; }
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    (void)event;
    *status = event_not_ready ? ACL_EVENT_RECORDED_STATUS_NOT_READY : ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return query_result;
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
static unsigned flush_commands;
int spdk_nvme_ns_cmd_flush(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair,
        spdk_nvme_cmd_cb callback, void *arg) {
    (void)ns; (void)qpair;
    assert(!pending_callback); ++flush_commands;
    pending_callback=callback; pending_callback_arg=arg; return 0;
}
int spdk_nvme_ns_cmd_read(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair,
        void *payload, uint64_t lba, uint32_t count, spdk_nvme_cmd_cb callback,
        void *arg, uint32_t flags) {
    (void)ns; (void)qpair; (void)lba; (void)flags;
    assert(simulated_disk && !pending_callback);
    memcpy(payload, simulated_disk, count * 4096);
    pending_callback = callback; pending_callback_arg = arg;
    return 0;
}
int32_t spdk_nvme_qpair_process_completions(struct spdk_nvme_qpair *qpair, uint32_t max) {
    (void)qpair; (void)max;
    if (pending_callback) {
        spdk_nvme_cmd_cb callback = pending_callback;
        struct spdk_nvme_cpl completion = {0};
        completion.status.sc=completion_failed ? 1 : 0;
        pending_callback = NULL; callback(pending_callback_arg, &completion); return 1;
    }
    return 0;
}

static NPUNVMEContext context(void) {
    NPUNVMEContext ctx = {0};
    ctx.block_size = 4096; ctx.total_blocks = 1024; ctx.dma.chunk_size = 4096;
    ctx.io_timeout_ms = 20;
    ctx.qpair=(void *)1; /* mock attached controller; COPY_RANK clears it */
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
static void *close_test_context(void *arg) {
    assert(npu_nvme_close(arg, 1000) == 0);
    return NULL;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    NPUNVMEContext ctx = context();
    char bytes[4096] = {0};
    void *ptrs[] = {bytes}; uint64_t offsets[] = {0}; size_t sizes[] = {4};
    if (!strcmp(argv[1],"role_permissions")) {
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,.operation=4,.memory_kind=1};
        NPUNVMECopySpec copy={.struct_size=sizeof(copy),.version=1,.operation=5,
                             .source=bytes,.destination=bytes,.length=4};
        NPUNVMERequest *req=NULL;
        NPUNVMECapabilities caps;
        ctx.role=NPU_NVME_ROLE_HOST_OWNER;
        assert(npu_nvme_submit_copy(&ctx,&copy,&req)==-ENOTSUP && !req);
        assert(npu_nvme_set_step_ptr(&ctx,bytes,1)==-ENOTSUP);
        assert(npu_nvme_submit_write_batch(&ctx,ptrs,offsets,sizes,1,&req)==-ENOTSUP);
        assert(npu_nvme_get_capabilities(&ctx,&caps,sizeof(caps))==0 && caps.operation_mask==31);
        ctx.role=NPU_NVME_ROLE_COPY_RANK;ctx.qpair=NULL;ctx.ns=NULL;
        assert(npu_nvme_get_capabilities(&ctx,&caps,sizeof(caps))==0);
        assert(caps.operation_mask==96 && !caps.namespace_bytes);
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==-ENOTSUP && !req);
        assert(submit_meta_owned(&ctx,0,0,0,1,NULL,&req)==-ENOTSUP && !req);
    } else if (!strcmp(argv[1],"metadata_failed_flush")) {
        struct spdk_ring ring={0};ctx.meta_ring=&ring;ctx.meta_qpair=(void *)1;ctx.meta_dma_buf=bytes;
        pthread_mutex_init(&ctx.state_lock,NULL);ctx.state_lock_initialized=true;
        NPUNVMETransferItem item={.address=bytes,.length=4096};
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,.operation=3,.memory_kind=1,.item_count=1,.items=&item};
        NPUNVMERequest *req=NULL;completion_failed=1;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==0);meta_poller_fn(&ctx);
        assert(atomic_load(&req->done) && req->result && ctx.prior_write_error);
        npu_nvme_release_request(req);completion_failed=0;
        spec.operation=4;spec.items=NULL;spec.item_count=0;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==0);meta_poller_fn(&ctx);
        NPUNVMETransferReceipt receipt;
        assert(npu_nvme_get_transfer_receipt(req,&receipt,sizeof(receipt))==0);
        assert(receipt.result && !receipt.data_durable && !flush_commands);
        npu_nvme_release_request(req);assert(atomic_load(&ctx.refs)==1);
        pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1],"pointer_overflow")) {
        void *bad[]={ (void *)(UINTPTR_MAX-1) };
        assert(validate_io_batch(&ctx,bad,offsets,sizes,1)!=0 && !enqueues);
    } else if (!strcmp(argv[1],"copy_d2h") || !strcmp(argv[1],"copy_h2d") || !strcmp(argv[1],"copy_corrupt")) {
        bool d2h=!strcmp(argv[1],"copy_d2h");
        char source[4096]="123456789",target[4096];memset(target,'Z',sizeof(target));
        struct spdk_ring ring={0};if (d2h) ctx.write_ring=&ring;else ctx.read_ring=&ring;
        pthread_mutex_init(&ctx.state_lock,NULL);ctx.state_lock_initialized=true;
        dma_buf_t buffer={.buf=bytes};aclrtEvent event=(void *)1;
        ctx.dma.pool=&buffer;ctx.dma.max_pipe_depth=1;ctx.acl.events=&event;emulate_copy=1;
        assert(ring_init(&ctx.dma.free_ring,2)==0 && ring_push(&ctx.dma.free_ring,0)==0);
        NPUNVMECopySpec spec={.struct_size=sizeof(spec),.version=1,.operation=d2h ? 5 : 6,
            .checksum_flags=3,.source=source,.destination=target,.length=9,.expected_crc32=0xcbf43926U};
        SHA256((void *)source,9,spec.expected_sha256);
        if (!strcmp(argv[1],"copy_corrupt")) spec.expected_sha256[0]^=1;
        NPUNVMERequest *req=NULL;
        assert(npu_nvme_submit_copy(&ctx,&spec,&req)==0 && !ctx.accepted_writes);
        spec.source=NULL;spec.destination=NULL; /* descriptors have been copied */
        int (*pump)(void *)=d2h ? write_fsm_poller_fn : read_fsm_poller_fn;
        event_not_ready=1;pump(&ctx);pump(&ctx);pump(&ctx);
        if (!strcmp(argv[1],"copy_corrupt")) {
            assert(atomic_load(&req->done) && req->result==-EBADMSG);
            for (int i=0;i<4096;++i) assert(target[i]=='Z');
        } else {
            assert(!atomic_load(&req->done) && ring_is_empty(&ctx.dma.free_ring));
            if (d2h) for (int i=0;i<4096;++i) assert(target[i]=='Z');
            event_not_ready=0;
            for (int i=0;i<10 && !atomic_load(&req->done);++i) pump(&ctx);
            NPUNVMETransferReceipt receipt;
            assert(npu_nvme_get_transfer_receipt(req,&receipt,sizeof(receipt))==0);
            assert(receipt.operation==(d2h ? 5 : 6) && !receipt.result && !receipt.data_durable);
            assert(!memcmp(target,source,9) && target[9]=='Z');
        }
        assert(!atomic_load(&ctx.nvme_submit_count) && !pending_callback);
        npu_nvme_release_request(req);
        assert(atomic_load(&ctx.refs)==1 && !ctx.prior_write_error);
        ring_free(&ctx.dma.free_ring);pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1],"streaming_windows")) {
        ctx.dma.chunk_size=1024*1024;ctx.total_blocks=UINT64_C(1)<<28;
        size_t *lengths=malloc(65536*sizeof(*lengths));
        uint64_t *positions=malloc(65536*sizeof(*positions));
        void **addresses=malloc(65536*sizeof(*addresses));
        assert(lengths && positions && addresses);
        uint64_t total=0;unsigned items=0;
        /* Geometry-only synthetic stream: no device pointers are dereferenced. */
        for (unsigned base=0;base<95520;) {
            unsigned count=95520-base; if (count>65536) count=65536;
            for (unsigned i=0;i<count;++i) {
                lengths[i]=1024*1024;positions[i]=(uint64_t)(base+i)*1024*1024;addresses[i]=bytes;
            }
            assert(validate_io_batch(&ctx,addresses,positions,lengths,count)==0);
            base+=count;items+=count;total+=(uint64_t)count*1024*1024;
        }
        assert(items>65536 && total>MAX_BATCH_BYTES);
        free(lengths);free(positions);free(addresses);
    } else if (!strcmp(argv[1],"flush_prior_failure")) {
        struct spdk_ring wring={0},mring={0};ctx.write_ring=&wring;ctx.meta_ring=&mring;ctx.meta_qpair=(void *)1;
        pthread_mutex_init(&ctx.state_lock,NULL);ctx.state_lock_initialized=true;
        NPUNVMERequest *write=NULL,*flush=NULL;
        assert(npu_nvme_submit_write_batch_host(&ctx,ptrs,offsets,sizes,1,&write)==0);
        cancel_queued_requests(&ctx); npu_nvme_release_request(write);
        assert(ctx.prior_write_error==-ECANCELED);
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,.operation=4,.memory_kind=1};
        assert(npu_nvme_submit_transfer(&ctx,&spec,&flush)==0 && flush->dependency_count==0);
        meta_poller_fn(&ctx);
        NPUNVMETransferReceipt receipt;
        assert(npu_nvme_get_transfer_receipt(flush,&receipt,sizeof(receipt))==0);
        assert(receipt.result==-ECANCELED && !receipt.data_durable && !flush_commands);
        npu_nvme_release_request(flush);assert(atomic_load(&ctx.refs)==1);
        pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1],"capabilities")) {
        ctx.dma.max_pipe_depth=64;
        NPUNVMECapabilities caps;
        assert(npu_nvme_get_capabilities(&ctx,&caps,sizeof(caps)-1)==-EINVAL);
        assert(npu_nvme_get_capabilities(&ctx,&caps,sizeof(caps))==0);
        assert(caps.pipe_depth==64 && caps.quantum_items==64);
        assert(caps.dma_pool_bytes==64*4096 && caps.max_request_items==65536);
        assert(caps.max_request_bytes==(UINT64_C(64)<<30));
        assert(caps.checksum_bytes_per_tick==65536 && caps.operation_mask==127);
        uint32_t value;
        setenv("NPU_NVME_TEST_BUDGET","64",1);
        assert(scheduler_value("NPU_NVME_TEST_BUDGET",4,64,&value)==0 && value==64);
        const char *bad[]={"0","65","-1","+1"," 1","1x","18446744073709551616",""};
        for (unsigned i=0;i<sizeof(bad)/sizeof(*bad);++i) {
            setenv("NPU_NVME_TEST_BUDGET",bad[i],1);
            assert(scheduler_value("NPU_NVME_TEST_BUDGET",4,64,&value)==-EINVAL);
        }
        unsetenv("NPU_NVME_TEST_BUDGET");
        assert(scheduler_value("NPU_NVME_TEST_BUDGET",4,64,&value)==0 && value==4);
    } else if (!strcmp(argv[1],"pending_limit")) {
        struct spdk_ring wring={0},rring={0}; ctx.write_ring=&wring;ctx.read_ring=&rring;
        ctx.max_pending_requests=1;
        pthread_mutex_init(&ctx.state_lock,NULL);ctx.state_lock_initialized=true;
        NPUNVMERequest *first=NULL,*second=NULL;
        assert(npu_nvme_submit_write_batch_host(&ctx,ptrs,offsets,sizes,1,&first)==0);
        NPUNVMETransferItem item={.address=bytes,.length=4};
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,
            .operation=1,.memory_kind=1,.item_count=1,.items=&item};
        assert(npu_nvme_submit_transfer(&ctx,&spec,&second)==-EBUSY && !second);
        assert(atomic_load(&ctx.pending_requests)==1 && atomic_load(&ctx.refs)==2);
        cancel_queued_requests(&ctx); npu_nvme_release_request(first);
        assert(atomic_load(&ctx.refs)==1 && !atomic_load(&ctx.pending_requests));
        pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1],"write_rotation") || !strcmp(argv[1],"read_rotation")) {
        bool reading=!strcmp(argv[1],"read_rotation");
        struct spdk_ring ring={0};
        if (reading) ctx.read_ring=&ring; else ctx.write_ring=&ring;
        pthread_mutex_init(&ctx.state_lock,NULL); ctx.state_lock_initialized=true;
        dma_buf_t buffer={.buf=bytes}; ctx.dma.pool=&buffer; ctx.dma.max_pipe_depth=1;
        assert(ring_init(&ctx.dma.free_ring,2)==0 && ring_push(&ctx.dma.free_ring,0)==0);
        char source[4096]="data"; simulated_disk=source;
        NPUNVMETransferItem items[9]={0};
        for (int i=0;i<9;++i) {items[i].address=source;items[i].offset=i*4096;items[i].length=4;}
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,
            .memory_kind=NPU_NVME_MEMORY_HOST,.operation=reading ? 1 : 0,.item_count=9,.items=items};
        NPUNVMERequest *big=NULL,*small=NULL;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&big)==0);
        int (*pump)(void *)=reading ? read_fsm_poller_fn : write_fsm_poller_fn;
        pump(&ctx); spec.item_count=1;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&small)==0);
        for (int i=0;i<100 && !atomic_load(&small->done);++i) pump(&ctx);
        assert(atomic_load(&small->done) && !small->result);
        assert(!atomic_load(&big->done)); /* Small request beats the ninth big chunk. */
        for (int i=0;i<100 && !atomic_load(&big->done);++i) pump(&ctx);
        assert(atomic_load(&big->done) && !big->result);
        npu_nvme_release_request(big); npu_nvme_release_request(small);
        assert(atomic_load(&ctx.refs)==1 && !atomic_load(&ctx.pending_requests));
        assert(!ctx.ready_head[reading] && !ctx.ready_tail[reading]);
        ring_free(&ctx.dma.free_ring); pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1],"checksum_slices")) {
        size_t size=3*65536+517;
        unsigned char *payload=malloc(size); assert(payload);
        for (size_t i=0;i<size;++i) payload[i]=(unsigned char)(i*17);
        dma_buf_t buffer={.buf=payload}; ctx.dma.pool=&buffer;
        io_task_t task={.size=size,.buf_idx=0};
        write_request_t req={.compute_crc=true,.compute_sha256=true};
        size_t budget=0; assert(!checksum_task(&ctx,&req,&task,&budget));
        for (int i=0;i<3;++i) {
            budget=65536; assert(!checksum_task(&ctx,&req,&task,&budget));
            assert(!budget && task.checksum_offset==(size_t)(i+1)*65536);
        }
        budget=65536; assert(checksum_task(&ctx,&req,&task,&budget));
        assert(budget==65536-517 && task.crc32==crc32_buffer(payload,size));
        unsigned char sha[32]; SHA256(payload,size,sha); assert(!memcmp(sha,task.sha256,32));
        size_t remaining=budget; assert(checksum_task(&ctx,&req,&task,&budget) && budget==remaining);
        free(payload);
    } else if (!strncmp(argv[1], "read_", 5)) {
        char disk[4096] = "123456789", target[4096]; memset(target, 0x5a, sizeof(target));
        simulated_disk = disk; emulate_copy = 1;
        struct spdk_ring ring = {0}; ctx.read_ring = &ring;
        pthread_mutex_init(&ctx.state_lock, NULL); ctx.state_lock_initialized = true;
        dma_buf_t buffer = {.buf=bytes}; aclrtEvent event = (void *)1;
        ctx.dma.pool = &buffer; ctx.dma.max_pipe_depth = 1; ctx.acl.events = &event;
        assert(ring_init(&ctx.dma.free_ring, 2) == 0);
        assert(ring_push(&ctx.dma.free_ring, 0) == 0);
        NPUNVMETransferItem item = {.address=target, .length=9, .expected_crc32=0xcbf43926U};
        SHA256((void *)disk, 9, item.expected_sha256);
        NPUNVMETransferSpec spec = {.struct_size=sizeof(spec), .version=1,
            .operation=NPU_NVME_TRANSFER_READ, .checksum_flags=3, .item_count=1, .items=&item};
        if (!strcmp(argv[1], "read_corrupt")) item.expected_sha256[0] ^= 1;
        NPUNVMERequest *req = NULL;
        assert(npu_nvme_submit_transfer(&ctx, &spec, &req) == 0);
        assert(read_fsm_poller_fn(&ctx) == 0); /* NVMe submitted */
        assert(!atomic_load(&req->done) && atomic_load(&ctx.nvme_outstanding)==1);
        if (!strcmp(argv[1], "read_record_quarantine")) record_result = sync_result = ACL_ERROR_FAILURE;
        assert(read_fsm_poller_fn(&ctx) == 0); /* NVMe completion, integrity, H2D */
        if (!strcmp(argv[1], "read_corrupt")) {
            assert(atomic_load(&req->done) && req->result==-EBADMSG);
            for (int i=0;i<4096;++i) assert((unsigned char)target[i]==0x5a);
            assert(atomic_load(&ctx.async_dma_submit_count)==0);
        } else if (!strcmp(argv[1], "read_record_quarantine") || !strcmp(argv[1], "read_query_quarantine")) {
            if (!strcmp(argv[1], "read_query_quarantine")) {
                query_result=event_sync_result=sync_result=ACL_ERROR_FAILURE;
                assert(read_fsm_poller_fn(&ctx)==0);
            }
            assert(atomic_load(&ctx.quarantined) && atomic_load(&ctx.dma_inflight)==1);
            assert(!atomic_load(&req->done) && ring_is_empty(&ctx.dma.free_ring));
            assert(npu_nvme_close(&ctx,1)==-EIO);
            /* Fake-only teardown: no real DMA was ever scheduled. */
            ctx.read_fsm.req=NULL;read_request_put(req);
        } else {
            event_not_ready=1;
            assert(read_fsm_poller_fn(&ctx)==0);
            assert(!atomic_load(&req->done) && ring_is_empty(&ctx.dma.free_ring));
            if (!strcmp(argv[1], "read_late")) assert(npu_nvme_wait_request(req,1)==-ETIMEDOUT);
            event_not_ready=0;assert(read_fsm_poller_fn(&ctx)==0);
            NPUNVMETransferReceipt receipt;
            assert(npu_nvme_get_transfer_receipt(req,&receipt,sizeof(receipt))==0);
            assert(receipt.operation==NPU_NVME_TRANSFER_READ && receipt.result==0 && receipt.transport_safe);
            assert(memcmp(target,disk,9)==0 && (unsigned char)target[9]==0x5a);
            assert(atomic_load(&ctx.dma_inflight)==0);
        }
        npu_nvme_release_request(req);
        assert(atomic_load(&ctx.refs)==1);
        ring_free(&ctx.dma.free_ring);pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1], "transfer_digest")) {
        char source[] = "123456789";
        struct spdk_ring ring = {0}; ctx.write_ring = &ring;
        pthread_mutex_init(&ctx.state_lock, NULL); ctx.state_lock_initialized = true;
        dma_buf_t buffer = {.buf=bytes}; aclrtEvent event = (void *)1;
        ctx.dma.pool = &buffer; ctx.dma.max_pipe_depth = 1; ctx.acl.events = &event;
        assert(ring_init(&ctx.dma.free_ring, 2) == 0);
        assert(ring_push(&ctx.dma.free_ring, 0) == 0);
        NPUNVMETransferItem item = {.address=source, .length=9};
        NPUNVMETransferSpec spec = {.struct_size=sizeof(spec), .version=1,
            .memory_kind=NPU_NVME_MEMORY_HOST, .checksum_flags=3, .item_count=1, .items=&item};
        NPUNVMERequest *req = NULL;
        assert(npu_nvme_submit_transfer(&ctx, &spec, &req) == 0);
        /* Descriptors are copied before submit returns. */
        item.address = NULL; item.length = UINT64_MAX;
        NPUNVMETransferReceipt receipt; NPUNVMETransferDigest digest;
        assert(npu_nvme_get_transfer_receipt(req, &receipt, sizeof(receipt)) == -EAGAIN);
        for (int i=0; i<8 && !atomic_load(&req->done); ++i) write_fsm_poller_fn(&ctx);
        assert(npu_nvme_get_transfer_receipt(req, &receipt, sizeof(receipt)) == 0);
        assert(receipt.done && receipt.source_safe && receipt.transport_safe && !receipt.data_durable);
        assert(receipt.logical_bytes == 9 && receipt.result == 0);
        assert(npu_nvme_get_transfer_digests(req, &digest, 0) == -EINVAL);
        assert(npu_nvme_get_transfer_digests(req, &digest, 1) == 0);
        assert(digest.crc32 == 0xcbf43926U);
        unsigned char expected[32]; SHA256((void *)source, 9, expected);
        assert(memcmp(expected, digest.sha256, 32) == 0);
        assert(memcmp(bytes, source, 9) == 0 && bytes[9] == 0);
        npu_nvme_release_request(req);
        assert(atomic_load(&ctx.refs) == 1 && atomic_load(&ctx.pending_requests) == 0);
        ring_free(&ctx.dma.free_ring); pthread_mutex_destroy(&ctx.state_lock);
    } else if (!strcmp(argv[1], "transfer_invalid")) {
        NPUNVMETransferItem item = {.address=bytes,.length=4};
        NPUNVMETransferSpec spec = {.struct_size=sizeof(spec),.version=1,.item_count=1,.items=&item};
        NPUNVMERequest *req = (void *)1;
        spec.version=2; assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==-EINVAL && !req);
        spec.version=1; spec.item_count=65537;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==-EINVAL && !req);
        spec.item_count=1; spec.checksum_flags=4;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==-EINVAL && !req);
        spec.checksum_flags=0; item.reserved=1;
        assert(npu_nvme_submit_transfer(&ctx,&spec,&req)==-EINVAL && !req);
        assert(enqueues==0);
    } else if (!strcmp(argv[1], "closed_admission")) {
        NPUNVMERequest *request = (void *)1;
        atomic_store(&ctx.app_should_stop, 1);
        assert(npu_nvme_submit_write_batch_host(&ctx, ptrs, offsets, sizes, 1, &request) == -ESHUTDOWN);
        assert(request == NULL);
        atomic_store(&ctx.app_should_stop, 0);
        atomic_store(&ctx.quarantined, 1);
        assert(npu_nvme_submit_write_batch_host(&ctx, ptrs, offsets, sizes, 1, &request) == -EIO);
        assert(request == NULL);
    } else if (!strcmp(argv[1], "native_owner")) {
        NPUNVMEContext second = context();
        assert(acquire_native_owner(&ctx, "0000:ff:1f.7") == 0);
        assert(acquire_native_owner(&second, "0000:ff:1f.7") == -EBUSY);
        pid_t child = fork();
        assert(child >= 0);
        if (child == 0) {
            release_native_owner(&ctx); /* cannot unlock the parent's file */
            native_owner = NULL; /* exercise the cross-process file lock */
            assert(acquire_native_owner(&second, "0000:ff:1f.7") == -EBUSY);
            _exit(0);
        }
        int status;
        assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        release_native_owner(&ctx);
        assert(acquire_native_owner(&second, "0000:ff:1f.7") == 0);
        release_native_owner(&second);
    } else if (!strcmp(argv[1], "concurrent_close")) {
        struct spdk_thread thread = {.state = 2};
        ctx.reactor_thread = &thread;
        atomic_init(&release_test_reactor, 0);
        assert(pthread_create(&ctx.reactor_pthread, NULL, finish_test_reactor, &ctx) == 0);
        ctx.reactor_pthread_started = true;
        pthread_t a, b;
        assert(pthread_create(&a, NULL, close_test_context, &ctx) == 0);
        assert(pthread_create(&b, NULL, close_test_context, &ctx) == 0);
        atomic_store(&release_test_reactor, 1);
        pthread_join(a, NULL); pthread_join(b, NULL);
        assert(thread.destroyed == 1 && !ctx.reactor_pthread_started);
    } else if (!strcmp(argv[1], "spdk_exit")) {
        struct spdk_thread thread = {0}, previous = {0};
        ctx.reactor_thread = &thread;
        test_current_thread = &previous;
        assert(npu_nvme_close(&ctx, 1) == -EBUSY);
        assert(!thread.destroyed && ctx.reactor_thread == &thread);
        reactor_finish_thread(&thread);
        assert(thread.polls == 3 && test_current_thread == &previous);
        assert(npu_nvme_close(&ctx, 1) == 0);
        assert(thread.destroyed == 1 && ctx.reactor_thread == NULL);
        assert(npu_nvme_close(&ctx, 1) == 0 && thread.destroyed == 1);
    } else if (!strcmp(argv[1], "delta_overflow")) {
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
    } else if (!strcmp(argv[1], "flush_dependency") || !strcmp(argv[1], "flush_failed_dependency")) {
        struct spdk_ring wring={0}, mring={0};
        ctx.write_ring=&wring; ctx.meta_ring=&mring; ctx.meta_qpair=(void *)1;
        pthread_mutex_init(&ctx.state_lock,NULL); ctx.state_lock_initialized=true;
        NPUNVMERequest *write=NULL, *flush=NULL;
        assert(npu_nvme_submit_write_batch(&ctx,ptrs,offsets,sizes,1,&write)==0);
        NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,
            .operation=NPU_NVME_TRANSFER_FLUSH,.memory_kind=NPU_NVME_MEMORY_HOST};
        assert(npu_nvme_submit_transfer(&ctx,&spec,&flush)==0);
        assert(flush->dependency_count==1 && flush->dependencies[0]==write);
        npu_nvme_release_request(write); /* Dependency retains caller-released write. */
        assert(meta_poller_fn(&ctx)==0 && flush_commands==0 && !atomic_load(&flush->done));
        void *obj=NULL; assert(spdk_ring_dequeue(&wring,&obj,1)==1);
        atomic_fetch_sub(&ctx.queued_writes,1);
        write=obj; write->result=!strcmp(argv[1],"flush_failed_dependency") ? -EIO : 0;
        forget_accepted_write(&ctx,write);
        atomic_store_explicit(&write->done,1,memory_order_release);
        atomic_fetch_sub(&ctx.pending_requests,1); write_request_put(write);
        assert(meta_poller_fn(&ctx)==0 && atomic_load(&flush->done));
        NPUNVMETransferReceipt receipt;
        assert(npu_nvme_get_transfer_receipt(flush,&receipt,sizeof(receipt))==0);
        bool failed=!strcmp(argv[1],"flush_failed_dependency");
        assert(receipt.result==(failed ? -EIO : 0));
        assert(receipt.data_durable==!failed && flush_commands==!failed);
        npu_nvme_release_request(flush);
        assert(atomic_load(&ctx.refs)==1 && atomic_load(&ctx.pending_requests)==0);
        pthread_mutex_destroy(&ctx.state_lock);
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

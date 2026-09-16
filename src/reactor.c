/* reactor: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

NPUNVMEContext *g_reactor_ctx;

/* SPDK exit is asynchronous even after unregistering every poller.  Keep the
 * owning pthread alive until SPDK has consumed its remaining messages and
 * removed the unregistered pollers.  The caller's close deadline only bounds
 * observation; it must not interrupt this drain or free the thread early. */
void reactor_finish_thread(struct spdk_thread *thread) {
    if (!thread) return;
    struct spdk_thread *previous = spdk_get_thread();
    spdk_set_thread(thread);
    spdk_thread_exit(thread);
    while (!spdk_thread_is_exited(thread)) {
        spdk_thread_poll(thread, 0, 0);
        usleep(100);
    }
    spdk_set_thread(previous);
}

int enqueue_request(NPUNVMEContext *ctx, int direction, void *obj) {
    int rc = -ESHUTDOWN;
    pthread_mutex_lock(&ctx->state_lock);
    if (!atomic_load(&ctx->app_should_stop) && !atomic_load(&ctx->admission_closed) &&
        !atomic_load(&ctx->quarantined)) {
        struct spdk_ring *ring = direction == 0 ? ctx->write_ring :
                                 direction == 1 ? ctx->read_ring : ctx->meta_ring;
        if (ring) {
            atomic_fetch_add(&ctx->pending_requests, 1);
            if (direction == 0) atomic_fetch_add(&ctx->queued_writes, 1);
            if (spdk_ring_enqueue(ring, &obj, 1, NULL) == 1) rc = 0;
            else {
                atomic_fetch_sub(&ctx->pending_requests, 1);
                if (direction == 0) atomic_fetch_sub(&ctx->queued_writes, 1);
                rc = -EBUSY;
            }
        }
    }
    pthread_mutex_unlock(&ctx->state_lock);
    return rc;
}

void *reactor_loop(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    spdk_set_thread(ctx->reactor_thread);

    /* Reset registered tasks to IDLE. */
    pthread_mutex_lock(&ctx->state_lock);
    for (int i = 0; i < ctx->listener.num_registered_tasks; i++) {
        ctx->listener.registered_tasks[i].state = CHUNK_IDLE;
        ctx->listener.registered_tasks[i].buf_idx = -1;
    }
    pthread_mutex_unlock(&ctx->state_lock);

    ctx->last_step_seen = -1;

    /* Create MPSC rings for Python → reactor requests (16 slots each). */
    ctx->write_ring = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 16,
                                        SPDK_ENV_SOCKET_ID_ANY);
    ctx->read_ring  = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 16,
                                        SPDK_ENV_SOCKET_ID_ANY);
    ctx->meta_ring  = spdk_ring_create(SPDK_RING_TYPE_MP_SC, 4,
                                        SPDK_ENV_SOCKET_ID_ANY);

    if (!ctx->write_ring || !ctx->read_ring || !ctx->meta_ring) {
        fprintf(stderr, "[Fatal] Reactor request-ring allocation failed.\n");
        ctx->reactor_init_result = -1;
        pthread_barrier_wait(&ctx->init_barrier);
        goto reactor_cleanup;
    }

    /* Init FSMs as idle. */
    ctx->write_fsm.state = WRITE_FSM_IDLE;
    ctx->write_fsm.req = NULL;
    ctx->read_fsm.state = READ_FSM_IDLE;
    ctx->read_fsm.req = NULL;
    ctx->meta_req = NULL;

    /* Register pollers. */
    ctx->step_poller      = spdk_poller_register(step_poller_fn, ctx,
                                                  STEP_POLLER_PERIOD_US);
    ctx->write_fsm_poller = spdk_poller_register(write_fsm_poller_fn, ctx, 0);
    ctx->read_fsm_poller  = spdk_poller_register(read_fsm_poller_fn, ctx, 0);
    ctx->meta_poller      = spdk_poller_register(meta_poller_fn, ctx, 0);
    fprintf(stderr, "[Diag] reactor: rings w=%p r=%p m=%p pollers s=%p w=%p r=%p m=%p\n",
            (void *)ctx->write_ring, (void *)ctx->read_ring, (void *)ctx->meta_ring,
            (void *)ctx->step_poller, (void *)ctx->write_fsm_poller,
            (void *)ctx->read_fsm_poller, (void *)ctx->meta_poller);

    if (!ctx->step_poller || !ctx->write_fsm_poller ||
        !ctx->read_fsm_poller || !ctx->meta_poller) {
        fprintf(stderr, "[Fatal] Reactor poller registration failed.\n");
        ctx->reactor_init_result = -1;
        pthread_barrier_wait(&ctx->init_barrier);
        goto reactor_cleanup;
    }

    pthread_barrier_wait(&ctx->init_barrier);

    /* Once shutdown is requested, stop accepting new step triggers but keep
     * polling until any in-flight data request has returned its DMA slots. */
    while (!ctx->app_should_stop ||
           ctx->write_fsm.state != WRITE_FSM_IDLE ||
           ctx->read_fsm.state != READ_FSM_IDLE ||
           ctx->meta_req != NULL) {
        struct timespec cpu_before, cpu_after;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_before);
        spdk_thread_poll(ctx->reactor_thread, 0, 0);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_after);
        uint64_t before_us = (uint64_t)cpu_before.tv_sec * 1000000ULL +
                             (uint64_t)cpu_before.tv_nsec / 1000ULL;
        uint64_t after_us = (uint64_t)cpu_after.tv_sec * 1000000ULL +
                            (uint64_t)cpu_after.tv_nsec / 1000ULL;
        if (after_us >= before_us)
            atomic_fetch_add_explicit(&ctx->reactor_cpu_us,
                                      after_us - before_us,
                                      memory_order_relaxed);
        usleep(100);
    }

reactor_cleanup:
    cancel_queued_requests(ctx);
    if (ctx->step_poller) spdk_poller_unregister(&ctx->step_poller);
    if (ctx->write_fsm_poller) spdk_poller_unregister(&ctx->write_fsm_poller);
    if (ctx->read_fsm_poller) spdk_poller_unregister(&ctx->read_fsm_poller);
    if (ctx->meta_poller) spdk_poller_unregister(&ctx->meta_poller);
    if (ctx->meta_qpair) {
        spdk_nvme_ctrlr_free_io_qpair(ctx->meta_qpair);
        ctx->meta_qpair = NULL;
    }
    if (ctx->write_ring) spdk_ring_free(ctx->write_ring);
    if (ctx->read_ring) spdk_ring_free(ctx->read_ring);
    if (ctx->meta_ring) spdk_ring_free(ctx->meta_ring);
    ctx->write_ring = NULL;
    ctx->read_ring = NULL;
    ctx->meta_ring = NULL;
    reactor_finish_thread(ctx->reactor_thread);
    atomic_store_explicit(&ctx->reactor_exited, 1, memory_order_release);
    return NULL;
}

void reactor_new_thread_fn(struct spdk_thread *thread, void *arg) {
    (void)arg;  /* unused — ctx comes from g_reactor_ctx */
    NPUNVMEContext *ctx = g_reactor_ctx;
    ctx->reactor_thread = thread;

    pthread_attr_t attr;
    int rc = pthread_attr_init(&attr);
    if (rc != 0) {
        ctx->reactor_init_result = -1;
        fprintf(stderr, "[Fatal] Failed to initialize pthread attributes (rc=%d).\n",
                rc);
        return;
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    rc = pthread_create(&ctx->reactor_pthread, &attr, reactor_loop, ctx);
    if (rc == 0) {
        ctx->reactor_pthread_started = true;
    } else {
        ctx->reactor_init_result = -1;
        fprintf(stderr, "[Fatal] Failed to create Reactor pthread (rc=%d).\n", rc);
    }
    pthread_attr_destroy(&attr);
}

/* write_pipeline: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void nvme_write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "[NPU-NVMe] NVMe write error for task %d: "
                "status=%d type=%d\n",
                cb_arg->task->task_idx,
                completion->status.sc, completion->status.sct);
        *cb_arg->result = -1;
        atomic_fetch_add_explicit(&cb_arg->ctx->completion_error_count, 1,
                                  memory_order_relaxed);
    }
    if (test_fault("NPU_NVME_TEST_FAIL_NVME_COMPLETION")) {
        *cb_arg->result = -1;
        atomic_fetch_add_explicit(&cb_arg->ctx->completion_error_count, 1,
                                  memory_order_relaxed);
    }
    if (cb_arg->ctx->enable_profiling) cb_arg->task->ts_spdk_done = get_time_us();
    cb_arg->task->state = CHUNK_DONE;
    (*cb_arg->completed_counter)++;
    release_dma_slot(cb_arg->ctx, cb_arg->task->buf_idx);
    dma_inflight_dec(cb_arg->ctx);
    nvme_outstanding_dec(cb_arg->ctx);
    if (cb_arg->ctx->enable_profiling)
        cb_arg->task->ts_slot_release = get_time_us();
    free(cb_arg);
}

int submit_to_spdk_write(NPUNVMEContext *ctx, io_task_t *task,
                          int *completed_counter, int *result) {
    static atomic_ulong test_write_submissions = 0;
    if (test_fault("NPU_NVME_TEST_FAIL_NVME_SUBMIT")) return -EIO;
    const char *delay_env = getenv("NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS");
    if (delay_env && delay_env[0]) {
        char *end = NULL;
        unsigned long delay_ms = strtoul(delay_env, &end, 10);
        if (end != delay_env && *end == '\0' && delay_ms > 0)
            usleep(delay_ms * 1000UL);
    }
    const char *fail_at_env = getenv("NPU_NVME_TEST_FAIL_WRITE_AT");
    if (fail_at_env && fail_at_env[0]) {
        char *end = NULL;
        unsigned long fail_at = strtoul(fail_at_env, &end, 10);
        unsigned long sequence = atomic_fetch_add(&test_write_submissions, 1) + 1;
        if (end != fail_at_env && *end == '\0' && fail_at == sequence)
            return -EIO;
    }
    size_t aligned_sz = ALIGN_4K(task->size);
    uint64_t lba = task->nvme_offset / ctx->block_size;
    uint32_t lba_count = aligned_sz / ctx->block_size;

    spdk_cb_arg_t *cb_arg = malloc(sizeof(spdk_cb_arg_t));
    if (!cb_arg) return -1;
    cb_arg->ctx = ctx; cb_arg->task = task;
    cb_arg->completed_counter = completed_counter;
    cb_arg->result = result;

    int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair,
                                     ctx->dma.pool[task->buf_idx].buf,
                                     lba, lba_count,
                                     nvme_write_complete_cb, cb_arg, 0);
    if (rc != 0) { free(cb_arg); return rc; }
    nvme_outstanding_inc(ctx);
    if (ctx->enable_profiling) task->ts_spdk_submit = get_time_us();
    task->state = CHUNK_SPDK_WRITING;
    return 0;
}

void initiate_write_fsm(NPUNVMEContext *ctx, write_request_t *req) {
    write_fsm_ctx_t *fsm = &ctx->write_fsm;
    fsm->state = WRITE_FSM_RUNNING;
    fsm->req = req;
    fsm->next_submit_idx = 0;
    fsm->next_spdk_submit_idx = 0;
    fsm->completed_count = 0;
    req->done = 0;
    req->result = 0;
    req->ts_batch_start = get_time_us();
}

void write_fsm_tick(NPUNVMEContext *ctx) {
    write_fsm_ctx_t *fsm = &ctx->write_fsm;
    write_request_t *req = fsm->req;
    if (!req || atomic_load(&ctx->quarantined)) return;

    /* Ensure ACL context is bound to the reactor thread before any
     * aclrtMemcpy calls (needed for HBM to host DMA).  Idempotent
     * if already bound. */
    ensure_acl_context(ctx);

    /* 1. Process SPDK completions (triggers callbacks that update task
     *    state and increment fsm->completed_count). */
    spdk_nvme_qpair_process_completions(ctx->qpair, 0);

    /* 2. Reap completed ACL events without synchronizing the reactor.  ACL
     * copy_stream preserves submission order, so an incomplete event also
     * bounds the completed prefix. */
    for (int i = fsm->next_spdk_submit_idx; i < fsm->next_submit_idx; ++i) {
        io_task_t *task = &req->tasks[i];
        if (task->state != CHUNK_NPU_COPYING) continue;
        atomic_fetch_add_explicit(&ctx->async_event_query_count, 1,
                                  memory_order_relaxed);
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        aclError ret = aclrtQueryEventStatus(ctx->acl.events[task->buf_idx],
                                             &status);
        if (test_fault("NPU_NVME_TEST_FAIL_EVENT_QUERY"))
            ret = ACL_ERROR_FAILURE;
        if (ret != ACL_SUCCESS) {
            atomic_fetch_add_explicit(&ctx->async_event_query_error_count, 1,
                                      memory_order_relaxed);
            req->result = -1;
            aclError sync_ret = aclrtSynchronizeEvent(
                ctx->acl.events[task->buf_idx]);
            if (sync_ret != ACL_SUCCESS) {
                if (aclrtSynchronizeStream(ctx->acl.copy_stream) != ACL_SUCCESS) {
                    task->state = CHUNK_QUARANTINED;
                    atomic_store(&ctx->safety_reason, 2);
                    atomic_store_explicit(&ctx->quarantined, 1, memory_order_release);
                    return;
                }
                task->state = CHUNK_DONE;
                fsm->completed_count++;
                release_dma_slot(ctx, task->buf_idx);
                dma_inflight_dec(ctx);
                if (ctx->enable_profiling)
                    task->ts_slot_release = get_time_us();
                continue;
            }
            status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
        }
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) break;
        if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
        if (req->compute_crc)
            task->crc32 = crc32_buffer(
                ctx->dma.pool[task->buf_idx].buf, task->size);
        (void)aclrtResetEvent(ctx->acl.events[task->buf_idx],
                              ctx->acl.copy_stream);
        task->state = CHUNK_NPU_DONE;
    }

    /* 3. Submit NPU_DONE chunks to SPDK.  The cursor makes this O(N) over
     * the whole request instead of rescanning every prior chunk on each
     * reactor tick. */
    while (fsm->next_spdk_submit_idx < fsm->next_submit_idx) {
        io_task_t *task = &req->tasks[fsm->next_spdk_submit_idx];
        if (task->state == CHUNK_NPU_DONE) {
            int rc = submit_to_spdk_write(ctx, task, &fsm->completed_count,
                                          &req->result);
            /* On queue-full (-ENOMEM/-EAGAIN), retry this task next tick. */
            if (rc == -ENOMEM || rc == -EAGAIN) break;
            if (rc != 0) {
                req->result = -1;
                task->state = CHUNK_DONE;
                fsm->completed_count++;
                release_dma_slot(ctx, task->buf_idx);
                dma_inflight_dec(ctx);
                if (ctx->enable_profiling)
                    task->ts_slot_release = get_time_us();
            }
        } else if (task->state == CHUNK_NPU_COPYING) {
            break;
        }
        fsm->next_spdk_submit_idx++;
    }

    /* 4. Fill all currently free DMA slots.  Submission itself is
     * non-blocking for NPU buffers; SPDK completion and ACL event polling on
     * later ticks create the actual DMA/NVMe overlap. */
    while (fsm->next_submit_idx < req->num_tasks) {
        io_task_t *task = &req->tasks[fsm->next_submit_idx];
        size_t aligned_sz = ALIGN_4K(task->size);

        if (task->size == 0 || aligned_sz > ctx->dma.chunk_size) {
            /* Defensive fallback: public entry points reject these chunks. */
            req->result = -1;
            task->state = CHUNK_DONE;
            fsm->completed_count++;
            fsm->next_submit_idx++;
        } else if (!ring_is_empty(&ctx->dma.free_ring)) {
            int rc = try_submit_async(ctx, task, req->is_host,
                                      req->async_dma);
            if (rc == 0) {
                if (req->compute_crc && task->state == CHUNK_NPU_DONE)
                    task->crc32 = crc32_buffer(
                        ctx->dma.pool[task->buf_idx].buf, task->size);
                fsm->next_submit_idx++;
            } else if (rc == -3) {
                req->result = -EIO;
                return;
            } else if (rc == -2) {
                req->result = -1;
                task->state = CHUNK_DONE;
                fsm->completed_count++;
                fsm->next_submit_idx++;
            }
            /* On ring-full: retry next tick. */
        } else break;
    }

    /* 5. Check if all chunks are complete. */
    if (fsm->completed_count >= req->num_tasks) {
        req->ts_batch_end = get_time_us();
        ctx->last_write_io_us = req->ts_batch_end - req->ts_batch_start;
        write_profiling_csv(ctx, req->tasks, req->num_tasks, PIPELINE_WRITE);
        fsm->state = WRITE_FSM_IDLE;
        fsm->req = NULL;
        atomic_store_explicit(&req->done, 1, memory_order_release);
    }
}

int write_fsm_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    write_fsm_ctx_t *fsm = &ctx->write_fsm;

    /* Refuse to stop mid-write; only stop when idle. */
    if (ctx->app_should_stop && fsm->state == WRITE_FSM_IDLE) return -1;

    /* Phase 1: check for new requests from Python (ring). */
    if (fsm->state == WRITE_FSM_IDLE) {
        void *obj = NULL;
        if (spdk_ring_dequeue(ctx->write_ring, &obj, 1) == 1) {
            atomic_fetch_sub(&ctx->queued_writes, 1);
            initiate_write_fsm(ctx, (write_request_t *)obj);
        }
    }

    /* Phase 2: advance the current FSM. */
    if (fsm->state == WRITE_FSM_RUNNING) {
        bool was_faf = (fsm->req == &fsm->faf_req);
        write_request_t *finished_req = fsm->req;
        uint32_t faf_step = fsm->faf_step;

        write_fsm_tick(ctx);

        /* If just completed: signal probe flag for FaF, free deferred tasks. */
        if (fsm->state == WRITE_FSM_IDLE) {
            if (was_faf && fsm->faf_req.result == 0) {
                ensure_acl_context(ctx);
                signal_probe_flag(ctx, faf_step);
            } else if (was_faf) {
                fprintf(stderr,
                        "[NPU-NVMe] FaF checkpoint step %u failed; "
                        "probe flag was not advanced.\n", faf_step);
            }
            pthread_mutex_lock(&ctx->state_lock);
            if (ctx->listener.old_tasks) {
                free(ctx->listener.old_tasks);
                ctx->listener.old_tasks = NULL;
            }
            pthread_mutex_unlock(&ctx->state_lock);
            atomic_fetch_sub(&ctx->pending_requests, 1);
            if (!was_faf) write_request_put(finished_req);
        }
    }

    return 0;
}

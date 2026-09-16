/* read_pipeline: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void initiate_read_fsm(NPUNVMEContext *ctx, read_request_t *req) {
    read_fsm_ctx_t *fsm = &ctx->read_fsm;
    fsm->state = READ_FSM_RUNNING;
    fsm->req = req;
    fsm->next_submit_idx = req->resume_index;
    fsm->next_copy_idx = req->resume_index;
    fsm->completed_count = req->resume_index;
    if (!req->resume_index) {
        req->done = 0; req->result = 0;
        req->ts_batch_start = get_time_us();
    }
}

static void finish_read_chunk(NPUNVMEContext *ctx, read_fsm_ctx_t *fsm, io_task_t *task) {
    task->state = CHUNK_DONE;
    fsm->completed_count++;
    release_dma_slot(ctx, task->buf_idx);
    dma_inflight_dec(ctx);
    if (ctx->enable_profiling) task->ts_slot_release = get_time_us();
}

static bool retain_read_chunk(NPUNVMEContext *ctx, read_request_t *req,
                              io_task_t *task, unsigned reason) {
    req->result = -EIO; task->state = CHUNK_QUARANTINED;
    atomic_store(&ctx->safety_reason, reason);
    atomic_store_explicit(&ctx->quarantined, 1, memory_order_release);
    return false;
}

/* Returns false while the slot is still borrowed by ACL. Completion/error
 * publication must follow a positive event or independent stream-stop proof. */
static bool reap_read_copy(NPUNVMEContext *ctx, read_fsm_ctx_t *fsm, io_task_t *task) {
    read_request_t *req = fsm->req;
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    atomic_fetch_add(&ctx->async_event_query_count, 1);
    aclError rc = aclrtQueryEventStatus(ctx->acl.events[task->buf_idx], &status);
    if (test_fault("NPU_NVME_TEST_FAIL_EVENT_QUERY")) rc = ACL_ERROR_FAILURE;
    if (rc != ACL_SUCCESS) {
        req->result = -EIO; atomic_fetch_add(&ctx->async_event_query_error_count, 1);
        aclError stop = aclrtSynchronizeEvent(ctx->acl.events[task->buf_idx]);
        if (test_fault("NPU_NVME_TEST_FAIL_EVENT_SYNC")) stop = ACL_ERROR_FAILURE;
        if (stop != ACL_SUCCESS) {
            atomic_fetch_add(&ctx->stream_sync_fallback_count, 1);
            stop = aclrtSynchronizeStream(ctx->acl.copy_stream);
            if (test_fault("NPU_NVME_TEST_FAIL_STREAM_SYNC")) stop = ACL_ERROR_FAILURE;
        }
        if (stop != ACL_SUCCESS) return retain_read_chunk(ctx, req, task, 2);
        status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    }
    if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) return false;
    if (ctx->enable_profiling) task->ts_npu_done = get_time_us();
    if (aclrtResetEvent(ctx->acl.events[task->buf_idx], ctx->acl.copy_stream) != ACL_SUCCESS) {
        req->result = -EIO; atomic_store(&ctx->admission_closed, 1);
    }
    finish_read_chunk(ctx, fsm, task);
    return true;
}

void read_fsm_tick(NPUNVMEContext *ctx) {
    read_fsm_ctx_t *fsm = &ctx->read_fsm;
    read_request_t *req = fsm->req;
    if (!req || atomic_load(&ctx->quarantined)) return;

    size_t checksum_budget = EFFECTIVE(ctx->checksum_bytes_per_tick,65536);
    size_t copy_budget = EFFECTIVE(ctx->copy_bytes_per_tick,65536);
    int quantum=EFFECTIVE(ctx->quantum_items,DEFAULT_QUANTUM(ctx));
    int quantum_end=req->num_tasks;
    if (quantum_end-req->resume_index>quantum) quantum_end=req->resume_index+quantum;
    if (!req->is_host) ensure_acl_context(ctx);

    /* 1. Process SPDK completions (triggers callbacks). */
    if (!test_fault("NPU_NVME_TEST_HOLD_NVME_COMPLETIONS"))
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);

    /* 2. Copy completed DMA buffers to NPU/Host.  Consume in task order with
     * a monotonic cursor, avoiding an O(N^2) scan for small model chunks. */
    while (fsm->next_copy_idx < fsm->next_submit_idx) {
        io_task_t *task = &req->tasks[fsm->next_copy_idx];
        if (task->state == CHUNK_DONE) {
            fsm->next_copy_idx++;
            continue;
        }
        if (task->state == CHUNK_H2D_COPYING) {
            if (!reap_read_copy(ctx, fsm, task)) break;
            fsm->next_copy_idx++; continue;
        }
        if (task->state != CHUNK_SPDK_DONE) break;
        if (!checksum_task(ctx,req,task,&checksum_budget)) break;
        if ((req->compute_crc && task->crc32 != task->expected_crc32) ||
            (req->compute_sha256 && memcmp(task->sha256, task->expected_sha256, 32))) {
            req->result = -EBADMSG;
            finish_read_chunk(ctx, fsm, task); fsm->next_copy_idx++; continue;
        }
        aclError rc = ACL_SUCCESS;
        if (req->is_host) {
            size_t bytes=task->size-task->host_copy_offset;
            if (bytes>copy_budget) bytes=copy_budget;
            memcpy((char *)task->npu_ptr+task->host_copy_offset,
                   (char *)ctx->dma.pool[task->buf_idx].buf+task->host_copy_offset,bytes);
            copy_budget-=bytes; task->host_copy_offset+=bytes;
            if (task->host_copy_offset<task->size) break;
        } else if (req->async_dma) {
            atomic_fetch_add(&ctx->async_dma_submit_count, 1);
            rc = test_fault("NPU_NVME_TEST_FAIL_ACL_COPY") ? ACL_ERROR_FAILURE :
                aclrtMemcpyAsync(task->npu_ptr, task->size,
                    ctx->dma.pool[task->buf_idx].buf, task->size,
                    ACL_MEMCPY_HOST_TO_DEVICE, ctx->acl.copy_stream);
            if (rc == ACL_SUCCESS) {
                if (ctx->enable_profiling) task->ts_h2d_submit = get_time_us();
                rc = test_fault("NPU_NVME_TEST_FAIL_EVENT_RECORD") ? ACL_ERROR_FAILURE :
                    aclrtRecordEvent(ctx->acl.events[task->buf_idx], ctx->acl.copy_stream);
                if (rc == ACL_SUCCESS) { task->state = CHUNK_H2D_COPYING; break; }
                atomic_fetch_add(&ctx->stream_sync_fallback_count, 1);
                aclError stop = aclrtSynchronizeStream(ctx->acl.copy_stream);
                if (test_fault("NPU_NVME_TEST_FAIL_STREAM_SYNC")) stop = ACL_ERROR_FAILURE;
                if (stop != ACL_SUCCESS) { retain_read_chunk(ctx, req, task, 1); return; }
            }
        } else {
            rc = aclrtMemcpy(task->npu_ptr, task->size,
                ctx->dma.pool[task->buf_idx].buf, task->size, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (rc != ACL_SUCCESS) req->result = -EIO;
        finish_read_chunk(ctx, fsm, task);
        fsm->next_copy_idx++;
    }
    if (atomic_load(&ctx->quarantined)) return;

    /* 3. Submit one more read to SPDK. */
    if (fsm->next_submit_idx < quantum_end) {
        io_task_t *task = &req->tasks[fsm->next_submit_idx];
        size_t aligned_sz = ALIGN_4K(task->size);

        if (task->size == 0 || aligned_sz > ctx->dma.chunk_size) {
            req->result = -1;
            task->state = CHUNK_DONE;
            fsm->completed_count++;
            fsm->next_submit_idx++;
        } else if (!ring_is_empty(&ctx->dma.free_ring)) {
            int buf_idx = -1;
            if (ring_pop(&ctx->dma.free_ring, &buf_idx) != 0 ||
                buf_idx < 0 || buf_idx >= ctx->dma.max_pipe_depth) {
                req->result = -1;
                task->state = CHUNK_DONE;
                fsm->completed_count++;
                fsm->next_submit_idx++;
            } else {
                task->buf_idx = buf_idx;
                own_dma_slot(ctx, task, buf_idx);
                dma_inflight_inc(ctx);

            spdk_cb_arg_t *cb_arg = malloc(sizeof(spdk_cb_arg_t));
            if (cb_arg) {
                cb_arg->ctx = ctx; cb_arg->task = task;
                cb_arg->completed_counter = &fsm->completed_count;
                cb_arg->result = &req->result;

                uint64_t lba = task->nvme_offset / ctx->block_size;
                uint32_t lba_count = aligned_sz / ctx->block_size;
                int rc = test_fault("NPU_NVME_TEST_FAIL_NVME_READ_SUBMIT") ? -EIO :
                    spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair,
                                                ctx->dma.pool[buf_idx].buf,
                                                lba, lba_count,
                                                nvme_read_complete_cb, cb_arg, 0);
                if (rc == 0) {
                    nvme_outstanding_inc(ctx);
                    if (ctx->enable_profiling) task->ts_submit = task->ts_spdk_submit = get_time_us();
                    task->state = CHUNK_SPDK_READING;
                    fsm->next_submit_idx++;
                } else if (rc == -ENOMEM || rc == -EAGAIN) {
                    /* A qpair can transiently reject a submission when its
                     * command ring is full.  Keep the task at CHUNK_IDLE and
                     * retry on the next reactor tick, matching the write FSM
                     * behavior; treating this as a permanent I/O error makes
                     * high pipeline-depth reads fail spuriously. */
                    atomic_fetch_add_explicit(&ctx->spdk_retry_count, 1,
                                              memory_order_relaxed);
                    release_dma_slot(ctx, buf_idx);
                    dma_inflight_dec(ctx);
                    free(cb_arg);
                } else {
                    req->result = -1;
                    release_dma_slot(ctx, buf_idx);
                    dma_inflight_dec(ctx);
                    free(cb_arg);
                    task->state = CHUNK_DONE;
                    fsm->completed_count++;
                    fsm->next_submit_idx++;
                }
                } else {
                    req->result = -1;
                    release_dma_slot(ctx, buf_idx);
                    dma_inflight_dec(ctx);
                    task->state = CHUNK_DONE;
                    fsm->completed_count++;
                    fsm->next_submit_idx++;
                }
            }
        }
    }

    if (fsm->completed_count==quantum_end && quantum_end<req->num_tasks) {
        rotate_request(ctx,1,req,quantum_end);
        fsm->req=NULL; fsm->state=READ_FSM_IDLE; return;
    }

    /* 4. Check completion. */
    if (fsm->completed_count >= req->num_tasks) {
        req->ts_batch_end = get_time_us();
        ctx->last_read_io_us = req->ts_batch_end - req->ts_batch_start;
        write_profiling_csv(ctx, req->tasks, req->num_tasks, PIPELINE_READ);
        fsm->state = READ_FSM_IDLE;
        fsm->req = NULL;
        atomic_store_explicit(&req->done, 1, memory_order_release);
    }
}

int read_fsm_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    read_fsm_ctx_t *fsm = &ctx->read_fsm;

    if (ctx->app_should_stop && fsm->state == READ_FSM_IDLE &&
        !ctx->ready_head[1] && !spdk_ring_count(ctx->read_ring)) return -1;

    if (fsm->state == READ_FSM_IDLE) {
        read_request_t *req=take_ready_request(ctx,1);
        if (req) initiate_read_fsm(ctx,req);
    }

    if (fsm->state == READ_FSM_RUNNING) {
        read_request_t *finished_req = fsm->req;
        read_fsm_tick(ctx);
        if (fsm->state == READ_FSM_IDLE && atomic_load(&finished_req->done)) {
            atomic_fetch_sub(&ctx->pending_requests, 1);
            read_request_put(finished_req);
        }
    }

    return 0;
}

void nvme_read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    spdk_cb_arg_t *cb_arg = (spdk_cb_arg_t *)arg;
    io_task_t *task = cb_arg->task;
    NPUNVMEContext *ctx = cb_arg->ctx;

    if (spdk_nvme_cpl_is_error(completion) || test_fault("NPU_NVME_TEST_FAIL_NVME_READ_COMPLETION")) {
        fprintf(stderr, "[NPU-NVMe] NVMe read error for task %d: "
                "status=%d type=%d\n",
                task->task_idx, completion->status.sc, completion->status.sct);
        *cb_arg->result = -1;
        task->state = CHUNK_DONE;
        (*cb_arg->completed_counter)++;
        release_dma_slot(ctx, task->buf_idx);
        dma_inflight_dec(ctx);
        nvme_outstanding_dec(ctx);
    } else {
        task->state = CHUNK_SPDK_DONE;
        nvme_outstanding_dec(ctx);
    }
    if (ctx->enable_profiling) task->ts_spdk_done = get_time_us();
    free(cb_arg);
}

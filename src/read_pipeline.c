/* read_pipeline: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

void initiate_read_fsm(NPUNVMEContext *ctx, read_request_t *req) {
    read_fsm_ctx_t *fsm = &ctx->read_fsm;
    fsm->state = READ_FSM_RUNNING;
    fsm->req = req;
    fsm->next_submit_idx = 0;
    fsm->next_copy_idx = 0;
    fsm->completed_count = 0;
    req->done = 0;
    req->result = 0;
    req->ts_batch_start = get_time_us();
}

void read_fsm_tick(NPUNVMEContext *ctx) {
    read_fsm_ctx_t *fsm = &ctx->read_fsm;
    read_request_t *req = fsm->req;
    if (!req) return;

    if (!req->is_host) ensure_acl_context(ctx);

    /* 1. Process SPDK completions (triggers callbacks). */
    spdk_nvme_qpair_process_completions(ctx->qpair, 0);

    /* 2. Copy completed DMA buffers to NPU/Host.  Consume in task order with
     * a monotonic cursor, avoiding an O(N^2) scan for small model chunks. */
    while (fsm->next_copy_idx < fsm->next_submit_idx) {
        io_task_t *task = &req->tasks[fsm->next_copy_idx];
        if (task->state == CHUNK_DONE) {
            fsm->next_copy_idx++;
            continue;
        }
        if (task->state != CHUNK_SPDK_DONE) break;

        /* For host reads: memcpy.  For NPU reads: use aclrtMemcpy
         * (synchronous - one chunk per tick keeps latency bounded). */
        aclError ret = ACL_SUCCESS;
        if (req->is_host) {
            memcpy(task->npu_ptr, ctx->dma.pool[task->buf_idx].buf,
                   task->size);
        } else {
            ret = aclrtMemcpy(task->npu_ptr, task->size,
                              ctx->dma.pool[task->buf_idx].buf,
                              task->size, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret != ACL_SUCCESS) req->result = -1;
        task->state = CHUNK_DONE;
        fsm->completed_count++;
        release_dma_slot(ctx, task->buf_idx);
        dma_inflight_dec(ctx);
        fsm->next_copy_idx++;
    }

    /* 3. Submit one more read to SPDK. */
    if (fsm->next_submit_idx < req->num_tasks) {
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
                int rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair,
                                                ctx->dma.pool[buf_idx].buf,
                                                lba, lba_count,
                                                nvme_read_complete_cb, cb_arg, 0);
                if (rc == 0) {
                    nvme_outstanding_inc(ctx);
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

    /* 4. Check completion. */
    if (fsm->completed_count >= req->num_tasks) {
        req->ts_batch_end = get_time_us();
        ctx->last_read_io_us = req->ts_batch_end - req->ts_batch_start;
        if (!req->is_host) write_profiling_csv(ctx, req->tasks, req->num_tasks, PIPELINE_READ);
        fsm->state = READ_FSM_IDLE;
        fsm->req = NULL;
        atomic_store_explicit(&req->done, 1, memory_order_release);
    }
}

int read_fsm_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;
    read_fsm_ctx_t *fsm = &ctx->read_fsm;

    if (ctx->app_should_stop && fsm->state == READ_FSM_IDLE) return -1;

    if (fsm->state == READ_FSM_IDLE) {
        void *obj = NULL;
        if (spdk_ring_dequeue(ctx->read_ring, &obj, 1) == 1) {
            initiate_read_fsm(ctx, (read_request_t *)obj);
        }
    }

    if (fsm->state == READ_FSM_RUNNING) {
        read_request_t *finished_req = fsm->req;
        read_fsm_tick(ctx);
        if (fsm->state == READ_FSM_IDLE) {
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

    if (spdk_nvme_cpl_is_error(completion)) {
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

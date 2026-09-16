/* step_listener: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

int npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr) {
    if (!ctx) return -1;

    pthread_mutex_lock(&ctx->state_lock);

    if (!ctx->listener.probe_flag_host &&
        aclrtMallocHost(&ctx->listener.probe_flag_host, 4) != ACL_SUCCESS) {
        pthread_mutex_unlock(&ctx->state_lock);
        return -1;
    }

    if (ctx->listener.owns_probe_flag &&
        ctx->listener.probe_flag_dev_ptr) {
        aclrtFree(ctx->listener.probe_flag_dev_ptr);
        ctx->listener.probe_flag_dev_ptr = NULL;
        ctx->listener.owns_probe_flag = false;
    }

    if (dev_ptr) {
        ctx->listener.probe_flag_dev_ptr = dev_ptr;
        ctx->listener.owns_probe_flag = false;
    } else {
        /* Self-allocate a 4-byte HBM buffer for the listener to poll */
        aclError ret = aclrtMalloc(&ctx->listener.probe_flag_dev_ptr, 4,
                                    ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU-NVMe] Failed to allocate probe flag device memory.\n");
            pthread_mutex_unlock(&ctx->state_lock);
            return -1;
        }
        ctx->listener.owns_probe_flag = true;
        uint32_t zero = 0;
        if (aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &zero, 4,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            aclrtFree(ctx->listener.probe_flag_dev_ptr);
            ctx->listener.probe_flag_dev_ptr = NULL;
            ctx->listener.owns_probe_flag = false;
            pthread_mutex_unlock(&ctx->state_lock);
            return -1;
        }
    }

    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

int npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value) {
    if (!ctx || !ctx->listener.probe_flag_dev_ptr) return -1;
    pthread_mutex_lock(&ctx->state_lock);
    ensure_acl_context(ctx);
    aclError ret = aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &value, 4,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    pthread_mutex_unlock(&ctx->state_lock);
    return (ret == ACL_SUCCESS) ? 0 : -1;
}

void *npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx) {
    return ctx ? ctx->listener.probe_flag_dev_ptr : NULL;
}

void signal_probe_flag(NPUNVMEContext *ctx, uint32_t value) {
    if (!ctx->listener.probe_flag_dev_ptr) return;
    pthread_mutex_lock(&ctx->state_lock);
    ensure_acl_context(ctx);
    aclrtMemcpy(ctx->listener.probe_flag_dev_ptr, 4, &value, 4,
                 ACL_MEMCPY_HOST_TO_DEVICE);
    /* Mirror to host buffer for polling */
    if (ctx->listener.probe_flag_host) {
        aclrtMemcpy(ctx->listener.probe_flag_host, 4,
                     ctx->listener.probe_flag_dev_ptr, 4,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    }
    pthread_mutex_unlock(&ctx->state_lock);
}

int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval) {
    if (!ctx || !dev_ptr || ckpt_interval <= 0) return -1;
    pthread_mutex_lock(&ctx->state_lock);

    if (!ctx->listener.step_poll_buf) {
        if (aclrtMallocHost(&ctx->listener.step_poll_buf, 4) != ACL_SUCCESS) {
            pthread_mutex_unlock(&ctx->state_lock);
            return -1;
        }
    }
    ctx->listener.dev_step_ptr = dev_ptr;
    ctx->listener.ckpt_interval = ckpt_interval;
    ctx->last_step_seen = -1;
    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

int step_poller_fn(void *arg) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)arg;

    if (ctx->app_should_stop) return -1;
    if (!ctx->listener.dev_step_ptr) return 0;

    /* At-most-1 in-flight: skip if previous write still running. */
    if (ctx->write_fsm.state != WRITE_FSM_IDLE) return 0;

    ensure_acl_context(ctx);

    int cur_step = 0;
    aclError ret = aclrtMemcpy(ctx->listener.step_poll_buf, 4,
                                ctx->listener.dev_step_ptr, 4,
                                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
        cur_step = *(int *)ctx->listener.step_poll_buf;
    }

    if (cur_step > ctx->last_step_seen &&
        cur_step % ctx->listener.ckpt_interval == 0 &&
        cur_step != 0) {
        ctx->last_step_seen = cur_step;

        /* Snapshot registered tasks under state_lock, then initiate FSM.
         * state_lock is released before any DMA/SPDK I/O — the FSM runs
         * unlocked on the reactor thread. */
        pthread_mutex_lock(&ctx->state_lock);
        for (int i = 0; i < ctx->listener.num_registered_tasks; i++) {
            ctx->listener.registered_tasks[i].state = CHUNK_IDLE;
            ctx->listener.registered_tasks[i].buf_idx = -1;
        }
        /* Use pre-allocated FaF request (same reactor thread, no ring). */
        ctx->write_fsm.faf_req.tasks = ctx->listener.registered_tasks;
        ctx->write_fsm.faf_req.num_tasks = ctx->listener.num_registered_tasks;
        ctx->write_fsm.faf_req.is_host = false;
        ctx->write_fsm.faf_step = (uint32_t)cur_step;
        identify_tasks(ctx, ctx->listener.registered_tasks, ctx->listener.num_registered_tasks);
        atomic_fetch_add(&ctx->pending_requests, 1);
        initiate_write_fsm(ctx, &ctx->write_fsm.faf_req);
        pthread_mutex_unlock(&ctx->state_lock);
        /* I/O runs asynchronously in write_fsm_poller_fn.
         * probe flag is signalled on completion. */
    }

    return 0;
}

int npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes, int num_items) {
    if (validate_io_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items) != 0)
        return -1;

    pthread_mutex_lock(&ctx->state_lock);

    /* Allocate new array before freeing the old one, so that a failed
     * allocation leaves the listener with a valid (stale) task list
     * rather than a NULL pointer + stale count. */
    io_task_t *new_tasks = calloc(num_items, sizeof(io_task_t));
    if (!new_tasks) {
        pthread_mutex_unlock(&ctx->state_lock);
        return -1;
    }

    for (int i = 0; i < num_items; i++) {
        new_tasks[i].task_idx = i;
        new_tasks[i].buf_idx = -1;
        new_tasks[i].state = CHUNK_IDLE;
        new_tasks[i].npu_ptr = npu_ptrs[i];
        new_tasks[i].nvme_offset = nvme_offsets[i];
        new_tasks[i].size = sizes[i];
    }

    /* Swap: defer-free the old array so the async FSM can safely finish
     * any in-flight write using the old pointer.  The old array is freed
     * on the next register_tasks call or when the FSM goes idle. */
    if (ctx->listener.old_tasks) {
        free(ctx->listener.old_tasks);
    }
    ctx->listener.old_tasks = ctx->listener.registered_tasks;
    ctx->listener.registered_tasks = new_tasks;
    ctx->listener.num_registered_tasks = num_items;
    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

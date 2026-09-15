/* metrics: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"

uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL +
           (uint64_t)ts.tv_nsec / 1000ULL;
}

void update_peak_uint(atomic_uint *peak, unsigned value) {
    unsigned current = atomic_load_explicit(peak, memory_order_relaxed);
    while (value > current && !atomic_compare_exchange_weak_explicit(
            peak, &current, value, memory_order_relaxed, memory_order_relaxed)) {
        /* current is refreshed by compare_exchange */
    }
}

unsigned nvme_outstanding_inc(NPUNVMEContext *ctx) {
    unsigned value = atomic_fetch_add_explicit(&ctx->nvme_outstanding, 1,
                                                memory_order_relaxed) + 1;
    update_peak_uint(&ctx->nvme_outstanding_peak, value);
    atomic_fetch_add_explicit(&ctx->nvme_submit_count, 1, memory_order_relaxed);
    return value;
}

void nvme_outstanding_dec(NPUNVMEContext *ctx) {
    atomic_fetch_sub_explicit(&ctx->nvme_outstanding, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->nvme_complete_count, 1, memory_order_relaxed);
}

void dma_inflight_inc(NPUNVMEContext *ctx) {
    unsigned value = atomic_fetch_add_explicit(&ctx->dma_inflight, 1,
                                                memory_order_relaxed) + 1;
    update_peak_uint(&ctx->dma_inflight_peak, value);
}

void dma_inflight_dec(NPUNVMEContext *ctx) {
    atomic_fetch_sub_explicit(&ctx->dma_inflight, 1, memory_order_relaxed);
}

void write_profiling_csv(NPUNVMEContext *ctx, io_task_t *tasks,
                          int num_items, pipeline_dir_t dir) {
    if (!ctx->enable_profiling) return;

    char path[512];
    const char *fname = (dir == PIPELINE_WRITE) ? "time_write.csv" : "time_read.csv";
    snprintf(path, sizeof(path), "%s/%s", ctx->profiling_dir, fname);
    const bool append = dir == PIPELINE_WRITE;
    FILE *f = fopen(path, append ? "a+" : "w");
    if (!f) {
        fprintf(stderr, "[Warning] Could not open profiling file: %s\n", path);
        return;
    }

    bool write_header = true;
    if (append) {
        if (fseek(f, 0, SEEK_END) == 0)
            write_header = ftell(f) == 0;
    }

    if (dir == PIPELINE_WRITE) {
        if (write_header) {
            fprintf(f, "item,buf_idx,queue_depth,ts_slot_wait_us,"
                    "ts_slot_acquire_us,ts_dma_submit_us,ts_dma_done_us,"
                    "ts_nvme_submit_us,ts_nvme_done_us,ts_slot_release_us,"
                    "slot_wait_us,dma_us,nvme_us,total_e2e_us\n");
        }
        for (int i = 0; i < num_items; ++i) {
            uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_submit)
                ? (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
            uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_spdk_submit)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_spdk_submit) : 0;
            uint64_t total_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
            uint64_t wait_us = (tasks[i].ts_slot_acquire > tasks[i].ts_slot_wait)
                ? tasks[i].ts_slot_acquire - tasks[i].ts_slot_wait : 0;
            fprintf(f, "%d,%d,%u,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
                    i, tasks[i].buf_idx, tasks[i].queue_depth,
                    tasks[i].ts_slot_wait, tasks[i].ts_slot_acquire,
                    tasks[i].ts_submit, tasks[i].ts_npu_done,
                    tasks[i].ts_spdk_submit, tasks[i].ts_spdk_done,
                    tasks[i].ts_slot_release, wait_us, npu_us, spdk_us,
                    total_us);
        }
    } else {
        fprintf(f, "item,buf_idx,ts_submit_us,ts_spdk_done_us,ts_npu_done_us,"
                "spdk_nvme_us,npu_async_us,total_e2e_us\n");
        for (int i = 0; i < num_items; ++i) {
            uint64_t spdk_us = (tasks[i].ts_spdk_done > tasks[i].ts_submit)
                ? (tasks[i].ts_spdk_done - tasks[i].ts_submit) : 0;
            uint64_t npu_us = (tasks[i].ts_npu_done > tasks[i].ts_spdk_done)
                ? (tasks[i].ts_npu_done - tasks[i].ts_spdk_done) : 0;
            uint64_t total_us = (tasks[i].ts_npu_done > tasks[i].ts_submit)
                ? (tasks[i].ts_npu_done - tasks[i].ts_submit) : 0;
            fprintf(f, "%d,%d,%lu,%lu,%lu,%lu,%lu,%lu\n", i, tasks[i].buf_idx,
                    tasks[i].ts_submit, tasks[i].ts_spdk_done,
                    tasks[i].ts_npu_done,
                    spdk_us, npu_us, total_us);
        }
    }
    fclose(f);
}

uint64_t npu_nvme_get_last_io_us(NPUNVMEContext *ctx, int is_read) {
    if (!ctx) return 0;
    return is_read ? ctx->last_read_io_us : ctx->last_write_io_us;
}

int npu_nvme_get_stats(NPUNVMEContext *ctx, NPUNVMEStats *out_stats) {
    if (!ctx || !out_stats) return -EINVAL;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->nvme_submit_count = atomic_load_explicit(&ctx->nvme_submit_count, memory_order_relaxed);
    out_stats->nvme_complete_count = atomic_load_explicit(&ctx->nvme_complete_count, memory_order_relaxed);
    out_stats->nvme_outstanding = atomic_load_explicit(&ctx->nvme_outstanding, memory_order_relaxed);
    out_stats->nvme_outstanding_peak = atomic_load_explicit(&ctx->nvme_outstanding_peak, memory_order_relaxed);
    out_stats->dma_inflight = atomic_load_explicit(&ctx->dma_inflight, memory_order_relaxed);
    out_stats->dma_inflight_peak = atomic_load_explicit(&ctx->dma_inflight_peak, memory_order_relaxed);
    out_stats->request_ring_depth = atomic_load(&ctx->queued_writes);
    out_stats->request_ring_peak = atomic_load_explicit(&ctx->request_ring_peak, memory_order_relaxed);
    out_stats->async_dma_submit_count = atomic_load_explicit(&ctx->async_dma_submit_count, memory_order_relaxed);
    out_stats->async_event_query_count = atomic_load_explicit(&ctx->async_event_query_count, memory_order_relaxed);
    out_stats->async_event_query_error_count = atomic_load_explicit(&ctx->async_event_query_error_count, memory_order_relaxed);
    out_stats->stream_sync_fallback_count = atomic_load_explicit(&ctx->stream_sync_fallback_count, memory_order_relaxed);
    out_stats->spdk_retry_count = atomic_load_explicit(&ctx->spdk_retry_count, memory_order_relaxed);
    out_stats->completion_error_count = atomic_load_explicit(&ctx->completion_error_count, memory_order_relaxed);
    out_stats->reactor_cpu_us = atomic_load_explicit(&ctx->reactor_cpu_us, memory_order_relaxed);
    return 0;
}

int npu_nvme_get_retained_slots(NPUNVMEContext *ctx, NPUNVMERetainedSlot *slots,
                                 uint32_t capacity, uint32_t *count) {
    if (!ctx || !count || capacity > MAX_PIPE_DEPTH || (capacity && !slots)) return -EINVAL;
    *count = 0;
    uint32_t reason = atomic_load(&ctx->safety_reason);
    for (int i = 0; ctx->dma.pool && i < ctx->dma.max_pipe_depth; ++i) {
        uint64_t owner = atomic_load_explicit(&ctx->dma.pool[i].owner_request_id, memory_order_acquire);
        if (!owner) continue;
        if (*count >= capacity) return -ENOSPC;
        slots[*count] = (NPUNVMERetainedSlot){.slot=(uint32_t)i, .reason=reason,
            .request_id=owner, .bytes=atomic_load(&ctx->dma.pool[i].owned_bytes),
            .nvme_offset=atomic_load(&ctx->dma.pool[i].owned_offset)};
        ++*count;
    }
    return 0;
}

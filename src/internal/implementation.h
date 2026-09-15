/* Private cross-module contracts. No new public exports from moved statics. */
#ifndef NPU_NVME_IMPLEMENTATION_H
#define NPU_NVME_IMPLEMENTATION_H
#include "npu_nvme.h"

/* Internal headers */
#include "internal/ring_buffer.h"
#include "internal/io_task.h"
#include "internal/pipeline.h"
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include "internal/context.h"

/* SPDK */
#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"

/* System */
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

/* ---- Hugepage pool auto-expansion for DPDK/SPDK ---- */
#define HUGEPAGE_2MB_PATH "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"


#define MAX_BATCH_ITEMS 65536
#define MAX_BATCH_BYTES (UINT64_C(64) * 1024 * 1024 * 1024)

extern __attribute__((visibility("hidden"))) NPUNVMEContext *g_reactor_ctx;

__attribute__((visibility("hidden"))) int read_int_from_file(const char *path);
__attribute__((visibility("hidden"))) void ensure_hugepages(void);
__attribute__((visibility("hidden"))) bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                    struct spdk_nvme_ctrlr_opts *opts);
__attribute__((visibility("hidden"))) void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts);
uint64_t get_time_us(void);
__attribute__((visibility("hidden"))) void update_peak_uint(atomic_uint *peak, unsigned value);
__attribute__((visibility("hidden"))) unsigned nvme_outstanding_inc(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void nvme_outstanding_dec(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void dma_inflight_inc(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void dma_inflight_dec(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void own_dma_slot(NPUNVMEContext *ctx, io_task_t *task, int slot);
__attribute__((visibility("hidden"))) void release_dma_slot(NPUNVMEContext *ctx, int slot);
__attribute__((visibility("hidden"))) void identify_tasks(NPUNVMEContext *ctx, io_task_t *tasks, int count);
__attribute__((visibility("hidden"))) bool test_fault(const char *name);
__attribute__((visibility("hidden"))) void context_put(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void write_request_put(write_request_t *req);
__attribute__((visibility("hidden"))) void read_request_put(read_request_t *req);
__attribute__((visibility("hidden"))) void meta_request_put(meta_request_t *req);
__attribute__((visibility("hidden"))) int enqueue_request(NPUNVMEContext *ctx, int direction, void *obj);
__attribute__((visibility("hidden"))) uint32_t finite_timeout(uint32_t timeout);
__attribute__((visibility("hidden"))) int wait_request_done(NPUNVMEContext *ctx, atomic_int *done);
__attribute__((visibility("hidden"))) int wait_reactor_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms);
__attribute__((visibility("hidden"))) void cancel_queued_requests(NPUNVMEContext *ctx);
io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes);
int try_submit_async(NPUNVMEContext *ctx, io_task_t *task, bool is_host,
                     bool async_dma);
void nvme_write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);
int submit_to_spdk_write(NPUNVMEContext *ctx, io_task_t *task,
                          int *completed_counter, int *result);
void write_profiling_csv(NPUNVMEContext *ctx, io_task_t *tasks,
                          int num_items, pipeline_dir_t dir);
__attribute__((visibility("hidden"))) int namespace_capacity(NPUNVMEContext *ctx, uint64_t *capacity);
__attribute__((visibility("hidden"))) int validate_io_batch(NPUNVMEContext *ctx, void **ptrs,
                             uint64_t *nvme_offsets, size_t *sizes,
                             int num_items);
__attribute__((visibility("hidden"))) int submit_write_common(NPUNVMEContext *ctx, void **ptrs,
                               uint64_t *nvme_offsets, size_t *sizes,
                               int num_items, bool is_host,
                               bool async_dma,
                               NPUNVMERequest **out_request);
__attribute__((visibility("hidden"))) int ensure_acl_context(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void *reactor_loop(void *arg);
__attribute__((visibility("hidden"))) void reactor_new_thread_fn(struct spdk_thread *thread, void *arg);
__attribute__((visibility("hidden"))) void initiate_write_fsm(NPUNVMEContext *ctx, write_request_t *req);
__attribute__((visibility("hidden"))) void write_fsm_tick(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) int write_fsm_poller_fn(void *arg);
__attribute__((visibility("hidden"))) void initiate_read_fsm(NPUNVMEContext *ctx, read_request_t *req);
__attribute__((visibility("hidden"))) void read_fsm_tick(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) int read_fsm_poller_fn(void *arg);
__attribute__((visibility("hidden"))) void meta_io_complete_cb(void *arg, const struct spdk_nvme_cpl *cpl);
__attribute__((visibility("hidden"))) int wait_meta_request_done(NPUNVMEContext *ctx, meta_request_t *req);
__attribute__((visibility("hidden"))) int meta_poller_fn(void *arg);
void nvme_read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);
__attribute__((visibility("hidden"))) void reactor_finish_thread(struct spdk_thread *thread);
#endif

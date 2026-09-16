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
#include <openssl/sha.h>

/* ---- Hugepage pool auto-expansion for DPDK/SPDK ---- */
#define STEP_POLLER_PERIOD_US  10000   /* step counter poll interval (10 ms) */
#define HUGEPAGE_2MB_PATH "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"


#define EFFECTIVE(v, fallback) ((v) ? (v) : (fallback))
#define DEFAULT_QUANTUM(ctx) ((ctx)->dma.max_pipe_depth > 4 ? (uint32_t)(ctx)->dma.max_pipe_depth : 4U)
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
__attribute__((visibility("hidden"))) void forget_accepted_write(NPUNVMEContext *ctx, write_request_t *req);
__attribute__((visibility("hidden"))) int submit_meta_owned(NPUNVMEContext *ctx, uint64_t offset,
    uint32_t bytes, int read, int flush, void *buffer, NPUNVMERequest **out);
__attribute__((visibility("hidden"))) int enqueue_request(NPUNVMEContext *ctx, int direction, void *obj);
__attribute__((visibility("hidden"))) uint32_t finite_timeout(uint32_t timeout);
__attribute__((visibility("hidden"))) int wait_request_done(NPUNVMEContext *ctx, atomic_int *done);
__attribute__((visibility("hidden"))) int wait_reactor_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms);
__attribute__((visibility("hidden"))) void cancel_queued_requests(NPUNVMEContext *ctx);
io_task_t *create_io_tasks(int num_tasks, void **npu_ptrs,
                            uint64_t *nvme_offsets, size_t *sizes);
__attribute__((visibility("hidden"))) bool checksum_task(NPUNVMEContext *ctx, write_request_t *req, io_task_t *task, size_t *budget);
__attribute__((visibility("hidden"))) uint32_t crc32_buffer(const void *data, size_t size);
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
                               int num_items, bool is_host, bool compute_crc,
                               bool async_dma,
                               NPUNVMERequest **out_request);
int npu_nvme_submit_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                                uint64_t *nvme_offsets, size_t *sizes,
                                int num_items, NPUNVMERequest **out_request);
int npu_nvme_submit_write_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                                     uint64_t *nvme_offsets, size_t *sizes,
                                     int num_items,
                                     NPUNVMERequest **out_request);
int npu_nvme_poll_request(NPUNVMERequest *request, int *done);
int npu_nvme_wait_request(NPUNVMERequest *request, uint32_t timeout_ms);
void npu_nvme_release_request(NPUNVMERequest *request);
int npu_nvme_get_max_transfer(NPUNVMEContext *ctx);
uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx);
uint64_t npu_nvme_get_last_io_us(NPUNVMEContext *ctx, int is_read);
int npu_nvme_get_stats(NPUNVMEContext *ctx, NPUNVMEStats *out_stats);
int npu_nvme_set_io_timeout_ms(NPUNVMEContext *ctx, uint32_t timeout_ms);
uint32_t npu_nvme_get_io_timeout_ms(NPUNVMEContext *ctx);
int npu_nvme_wait_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms);
__attribute__((visibility("hidden"))) int ensure_acl_context(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void *reactor_loop(void *arg);
__attribute__((visibility("hidden"))) void reactor_new_thread_fn(struct spdk_thread *thread, void *arg);
int npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr);
int npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value);
void *npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void signal_probe_flag(NPUNVMEContext *ctx, uint32_t value);
int npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval);
__attribute__((visibility("hidden"))) int step_poller_fn(void *arg);
int npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes, int num_items);
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
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset,
                           uint32_t total_bytes, int is_read, void *meta_buffer);
int npu_nvme_flush(NPUNVMEContext *ctx);
__attribute__((visibility("hidden"))) void reactor_finish_thread(struct spdk_thread *thread);
int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                  int pipe_depth, uint32_t chunk_size, bool enable_profiling,
                  const char *prof_dir);
int npu_nvme_get_retained_slots(NPUNVMEContext *ctx, NPUNVMERetainedSlot *slots,
                                 uint32_t capacity, uint32_t *count);
int npu_nvme_close(NPUNVMEContext *ctx, uint32_t timeout_ms);
void npu_nvme_cleanup(NPUNVMEContext *ctx);
int npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t area_offset,
                         uint64_t delta_slot_size, uint32_t delta_slot_count);
uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx);
uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx);
uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx);

__attribute__((visibility("hidden"))) write_request_t *take_ready_request(NPUNVMEContext *ctx, int direction);
__attribute__((visibility("hidden"))) void rotate_request(NPUNVMEContext *ctx, int direction, write_request_t *req, int prefix);

#endif

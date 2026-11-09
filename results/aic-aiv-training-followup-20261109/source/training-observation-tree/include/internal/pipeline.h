/* Internal: unified dual-polling I/O pipeline.
 *
 * Provides shared helpers for the Reactor-driven read/write FSMs.
 */
#ifndef NPU_NVME_PIPELINE_H
#define NPU_NVME_PIPELINE_H

#include "io_task.h"

/* SPDK completion callback — the callback struct is defined in spdk/nvme.h,
 * which pulls in the full SPDK headers.  Only declare the signature here;
 * the implementation in npu_nvme.c includes spdk/nvme.h first. */
struct spdk_nvme_cpl;

struct NPUNVMEContext;

/* --- Profiling CSV export --- */

/* Write per-chunk micro-benchmark data to a CSV file.
 * Column order depends on direction (write: npu,spdk,e2e; read: spdk,npu,e2e). */
void write_profiling_csv(struct NPUNVMEContext *ctx, io_task_t *tasks,
                          int num_items, pipeline_dir_t dir);

/* --- SPDK submission helpers --- */

int submit_to_spdk_write(struct NPUNVMEContext *ctx, io_task_t *task,
                          int *completed_counter, int *result);
void nvme_write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);
void nvme_read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);

/* --- NPU DMA submission --- */

/* Submit a single NPU-to-DMA copy.  Returns 0 on success, -1 if the ring
 * buffer is full, -2 on ACL error. */
int try_submit_async(struct NPUNVMEContext *ctx, io_task_t *task, bool is_host,
                     bool async_dma);

/* --- Timestamp --- */
uint64_t get_time_us(void);

#endif

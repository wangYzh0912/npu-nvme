#ifndef NPU_NVME_H
#define NPU_NVME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npu_nvme_context npu_nvme_context_t;

int npu_nvme_init(npu_nvme_context_t **ctx,
                  const char *nvme_pci_addr,
                  int npu_device_id,
                  int pipeline_depth,
                  int chunk_size,
                  bool enable_profiling,
                  const char *profiling_dir);

void npu_nvme_cleanup(npu_nvme_context_t *ctx);

size_t npu_nvme_get_max_transfer(npu_nvme_context_t *ctx);

/* 新增：获取盘的总块数 (用于 Python 层计算盘尾的栈区) */
uint64_t npu_nvme_get_total_blocks(npu_nvme_context_t *ctx);

/* 新增：元数据专属同步 I/O 接口 
 * start_lba: 写入的物理块号
 * num_blocks: 写入的块数 (确保 num_blocks * block_size <= 128KB)
 * is_read: true 为读，false 为写
 * meta_buffer: Python 传进来的连续内存 (ctypes buffer)
 */
int npu_nvme_sync_meta_io(npu_nvme_context_t *ctx, 
                          uint64_t byte_offset, 
                          uint32_t total_bytes, 
                          int is_read, 
                          void *meta_buffer);

int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                         void **npu_ptrs,
                         uint64_t *nvme_offsets,
                         size_t *sizes,
                         int num_items);

int npu_nvme_read_batch(npu_nvme_context_t *ctx,
                        void **npu_ptrs,
                        uint64_t *nvme_offsets,
                        size_t *sizes,
                        int num_items);

#ifdef __cplusplus
}
#endif
#endif
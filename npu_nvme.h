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
                  size_t chunk_size,
                  bool enable_profiling);

void npu_nvme_cleanup(npu_nvme_context_t *ctx);

size_t npu_nvme_get_max_transfer(npu_nvme_context_t *ctx);

/* 批量写：将 chunks 写到 NVMe
 * npu_ptrs[i]: NPU 端地址（起始指针，已含 chunk 内偏移）
 * nvme_offsets[i]: NVMe 字节偏移
 * sizes[i]: 本次要写的大小（不超过 max_transfer）
 * 返回 0 成功，<0 失败
 */
int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                         void **npu_ptrs,
                         uint64_t *nvme_offsets,
                         size_t *sizes,
                         int num_items);

/* 批量读：从 NVMe 读回 NPU */
int npu_nvme_read_batch(npu_nvme_context_t *ctx,
                        void **npu_ptrs,
                        uint64_t *nvme_offsets,
                        size_t *sizes,
                        int num_items);

#ifdef __cplusplus
}
#endif
#endif

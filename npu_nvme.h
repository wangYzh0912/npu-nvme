#ifndef NPU_NVME_ASYNC_H
#define NPU_NVME_ASYNC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque 句柄，Python 层不需要知道其内部结构
 */
typedef struct NPUNVMEContext NPUNVMEContext;

/**
 * @brief 初始化 NPU 和 NVMe SPDK 异步环境
 * * @param out_ctx 输出的全局上下文指针
 * @param pci_addr NVMe 硬盘的 PCIe 地址 (如 "0000:83:00.0")
 * @param npu_id 绑定的 Ascend NPU 设备 ID
 * @param pipe_depth 异步 Ring Buffer 的深度 (推荐 4-16)
 * @param chunk_size 单个数据块切片大小 (推荐 4MB = 4194304)
 * @param enable_profiling 是否开启纳秒级微观性能打点
 * @param prof_dir Profiling CSV 文件的输出目录
 * @return int 0 成功, -1 失败
 */
int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id, 
                  int pipe_depth, int chunk_size, bool enable_profiling, const char *prof_dir);

/**
 * @brief 释放所有软硬件资源，销毁 Stream 与 Event
 */
void npu_nvme_cleanup(NPUNVMEContext *ctx);

/**
 * @brief 获取 NVMe 硬盘总字节容量 (用于 Python 层容量防爆校验)
 */
uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx);

/**
 * @brief 获取配置的最大单次传输大小 (Chunk Size)
 */
int npu_nvme_get_max_transfer(NPUNVMEContext *ctx);

/**
 * @brief 同步读写元数据 (Superblock & JSON Ledger)
 * * @param ctx 全局上下文
 * @param byte_offset 物理硬盘上的绝对字节偏移量
 * @param total_bytes 读写总字节数
 * @param is_read 1 为读，0 为写
 * @param meta_buffer 主机内存中的 Buffer 指针
 * @return int 0 成功, -1 失败
 */
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset, uint32_t total_bytes, int is_read, void *meta_buffer);

/**
 * @brief 【核心】全异步 Zero-Bubble 批量张量直写 (NPU -> NVMe)
 * * @param ctx 全局上下文
 * @param npu_ptrs NPU 显存源地址数组
 * @param nvme_offsets NVMe 目标物理偏移量数组
 * @param sizes 每个张量切片的大小数组
 * @param num_items 任务总数
 * @return int 0 成功, -1 失败
 */
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                         uint64_t *nvme_offsets, size_t *sizes, int num_items);

/**
 * @brief 【核心】全异步 Zero-Bubble 批量张量直读 (NVMe -> NPU)
 * * @param ctx 全局上下文
 * @param npu_ptrs NPU 显存目标地址数组
 * @param nvme_offsets NVMe 源物理偏移量数组
 * @param sizes 每个张量切片的大小数组
 * @param num_items 任务总数
 * @return int 0 成功, -1 失败
 */
int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs, 
                        uint64_t *nvme_offsets, size_t *sizes, int num_items);

#ifdef __cplusplus
}
#endif

#endif // NPU_NVME_ASYNC_H
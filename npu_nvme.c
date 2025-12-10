/**
 * @file npu_nvme.c
 * @brief Implementation of NPU-NVMe zero-copy transfer
 */

#include "npu_nvme.h"
#include "spdk/stdinc.h"
#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/vmd.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SINGLE_TRANSFER (4 * 1024 * 1024)

// ==================================================
// Internal Data Structures
// ==================================================

// Pipeline slot状态
typedef enum {
    SLOT_FREE = 0,        // 空闲
    SLOT_COPYING,         // 正在从NPU拷贝
    SLOT_READY,          // 拷贝完成，准备提交
    SLOT_SUBMITTED,      // 已提交到NVMe
    SLOT_COMPLETED       // NVMe写完成
} slot_state_t;

// Pipeline中的一个slot
struct pipeline_slot {
    void *host_buffer;           // SPDK DMA buffer
    size_t buffer_size;          // Buffer大小
    
    // 当前chunk信息
    size_t chunk_size;           // 实际数据大小
    uint64_t nvme_lba;          // NVMe起始LBA
    uint32_t num_blocks;         // 块数量
    
    // 状态
    slot_state_t state;
    int error;
    
    // 用于回调识别
    int slot_id;
};

struct npu_nvme_context {
    // SPDK/NVMe
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns    *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;
    
    // ACL/NPU
    int npu_device_id;
    
    // I/O completion
    volatile int io_completed;
    volatile int io_error;
    
    // Staging buffer
    void *host_buffer;
    size_t host_buffer_size;

    struct pipeline_slot *pipeline_slots;
    int pipeline_depth;
    volatile int *slot_completed;  // 每个slot的完成标志
};

// ==================================================
// Helper Functions
// ==================================================

static void pipeline_write_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    struct pipeline_slot *slot = (struct pipeline_slot *)arg;
    
    if (spdk_nvme_cpl_is_error(cpl)) {
        fprintf(stderr, "[Pipeline] Slot %d write error\n", slot->slot_id);
        slot->error = 1;
    }
    
    slot->state = SLOT_COMPLETED;
}

static void io_complete_callback(void *arg, const struct spdk_nvme_cpl *cpl) {
    struct npu_nvme_context *ctx = (struct npu_nvme_context *)arg;
    
    if (spdk_nvme_cpl_is_error(cpl)) {
        fprintf(stderr, "[NVMe Error] %s\n",
                spdk_nvme_cpl_get_status_string(&cpl->status));
        ctx->io_error = 1;
    }
    
    ctx->io_completed = 1;
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    return true;
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    struct npu_nvme_context *ctx = (struct npu_nvme_context *)cb_ctx;
    int nsid;
    
    // Use first active namespace
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); 
         nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        
        struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == NULL || ! spdk_nvme_ns_is_active(ns)) {
            continue;
        }
        
        ctx->ctrlr = ctrlr;
        ctx->ns = ns;
        ctx->block_size = spdk_nvme_ns_get_sector_size(ns);
        ctx->total_blocks = spdk_nvme_ns_get_num_sectors(ns);
        
        printf("[NPU-NVMe] Attached to NVMe namespace\n");
        printf("  Block size: %u bytes\n", ctx->block_size);
        printf("  Capacity: %lu GB\n", 
               ctx->total_blocks * ctx->block_size / 1024 / 1024 / 1024);
        break;
    }
}

// ==================================================
// Public API Implementation
// ==================================================

int npu_nvme_init(npu_nvme_context_t **ctx, const char *nvme_pci_addr) {
    int rc;
    struct spdk_env_opts opts;
    struct spdk_nvme_transport_id trid;
    
    if (!ctx || !nvme_pci_addr) {
        return -1;
    }
    
    printf("\n========================================\n");
    printf("Initializing NPU-NVMe Environment\n");
    printf("========================================\n");
    
    // Allocate context
    *ctx = (struct npu_nvme_context *)calloc(1, sizeof(struct npu_nvme_context));
    if (!*ctx) {
        fprintf(stderr, "[Error] Failed to allocate context\n");
        return -1;
    }
    
    (*ctx)->npu_device_id = 7;
    
    // Initialize SPDK environment (only first time)
    static int spdk_initialized = 0;
    if (! spdk_initialized) {
        spdk_env_opts_init(&opts);
        opts.name = "npu_nvme";
        opts.opts_size = sizeof(opts);
        
        if (spdk_env_init(&opts) < 0) {
            fprintf(stderr, "[Error] Failed to initialize SPDK\n");
            free(*ctx);
            return -1;
        }
        spdk_initialized = 1;
        printf("[SPDK] Environment initialized\n");
    }
    /*
    // Initialize ACL
    printf("[ACL] Initializing.. .\n");
    aclError acl_ret = aclInit(NULL);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] ACL initialization failed: %d\n", acl_ret);
        free(*ctx);
        return -1;
    }
    */
    aclError acl_ret = aclrtSetDevice((*ctx)->npu_device_id);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] Failed to set NPU device: %d\n", acl_ret);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    printf("[ACL] Using NPU device %d\n", (*ctx)->npu_device_id);
    
    // Parse NVMe address
    memset(&trid, 0, sizeof(trid));
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", nvme_pci_addr);
    
    // Probe NVMe device
    printf("[NVMe] Probing device at %s.. .\n", nvme_pci_addr);
    rc = spdk_nvme_probe(&trid, *ctx, probe_cb, attach_cb, NULL);
    if (rc != 0 || (*ctx)->ctrlr == NULL) {
        fprintf(stderr, "[Error] Failed to probe NVMe device\n");
        aclrtResetDevice((*ctx)->npu_device_id);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    // Allocate I/O queue pair
    printf("[NVMe] Allocating I/O queue pair...\n");
    (*ctx)->qpair = spdk_nvme_ctrlr_alloc_io_qpair((*ctx)->ctrlr, NULL, 0);
    if (! (*ctx)->qpair) {
        fprintf(stderr, "[Error] Failed to allocate I/O queue pair\n");
        aclrtResetDevice((*ctx)->npu_device_id);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    // Allocate staging buffer (will be resized as needed)
    (*ctx)->host_buffer = NULL;
    (*ctx)->host_buffer_size = 0;
    
    printf("\n========================================\n");
    printf("Initialization Complete\n");
    printf("  Mode: Zero-Copy (NPU <-> NVMe)\n");
    printf("========================================\n\n");
    
    return 0;
}

int npu_nvme_write(npu_nvme_context_t *ctx,
                   void *npu_buffer,
                   uint64_t nvme_offset,
                   size_t size) {
    if (! ctx || !npu_buffer || size == 0) {
        fprintf(stderr, "[Error] Invalid parameters: ctx=%p, npu_buffer=%p, size=%zu\n",
                ctx, npu_buffer, size);
        return -1;
    }
    
    printf("\n[NPU-NVMe Write] Starting transfer\n");
    printf("  Total size: %zu bytes (%.2f MB)\n", size, size / 1024.0 / 1024.0);
    printf("  NPU buffer: %p\n", npu_buffer);
    printf("  NVMe offset: %lu bytes\n", nvme_offset);
    printf("  Block size: %u bytes\n", ctx->block_size);
    
    size_t remaining = size;
    size_t transferred = 0;
    int chunk_num = 0;
    
    while (remaining > 0) {
        chunk_num++;
        size_t chunk_size = (remaining > MAX_SINGLE_TRANSFER) 
                            ? MAX_SINGLE_TRANSFER 
                            : remaining;
        
        printf("\n[Chunk %d] Processing %.2f MB\n", chunk_num, chunk_size / 1024.0 / 1024.0);
        
        // 计算对齐大小
        size_t aligned_size = (chunk_size + ctx->block_size - 1) & ~(ctx->block_size - 1);
        uint64_t current_offset = nvme_offset + transferred;
        uint64_t lba = current_offset / ctx->block_size;
        uint32_t num_blocks = aligned_size / ctx->block_size;
        
        printf("  Chunk size: %zu bytes (%.2f MB)\n", chunk_size, chunk_size / 1024.0 / 1024.0);
        printf("  Aligned size: %zu bytes (%.2f MB)\n", aligned_size, aligned_size / 1024.0 / 1024.0);
        printf("  NVMe offset: %lu bytes (LBA %lu)\n", current_offset, lba);
        printf("  Num blocks: %u\n", num_blocks);
        printf("  End LBA: %lu (device has %lu blocks)\n", lba + num_blocks, ctx->total_blocks);
        
        // 边界检查
        if (lba + num_blocks > ctx->total_blocks) {
            fprintf(stderr, "[Error] Write would exceed device capacity\n");
            fprintf(stderr, "  Requested: LBA %lu + %u blocks = %lu\n", 
                    lba, num_blocks, lba + num_blocks);
            fprintf(stderr, "  Available: %lu blocks\n", ctx->total_blocks);
            return -1;
        }
        
        // 分配或调整 host buffer
        if (ctx->host_buffer_size < aligned_size) {
            printf("  Allocating host buffer: %.2f MB\n", aligned_size / 1024.0 / 1024.0);
            
            if (ctx->host_buffer) {
                spdk_dma_free(ctx->host_buffer);
            }
            
            ctx->host_buffer = spdk_dma_zmalloc(aligned_size, 4096, NULL);
            if (!ctx->host_buffer) {
                fprintf(stderr, "[Error] Failed to allocate host buffer\n");
                return -1;
            }
            ctx->host_buffer_size = aligned_size;
            
            // 验证buffer
            uint64_t phys = spdk_vtophys(ctx->host_buffer, NULL);
            if (phys == SPDK_VTOPHYS_ERROR) {
                fprintf(stderr, "[Error] Buffer vtophys failed\n");
                spdk_dma_free(ctx->host_buffer);
                ctx->host_buffer = NULL;
                return -1;
            }
            printf("  Host buffer: virt=%p, phys=0x%lx\n", ctx->host_buffer, phys);
        }
        
        // NPU -> Host
        printf("  Step 1/2: NPU -> Host (aclrtMemcpy)...\n");
        aclError acl_ret = aclrtMemcpy(ctx->host_buffer, aligned_size,
                                       (char *)npu_buffer + transferred, chunk_size,
                                       ACL_MEMCPY_DEVICE_TO_HOST);
        if (acl_ret != ACL_SUCCESS) {
            fprintf(stderr, "[Error] aclrtMemcpy failed: %d\n", acl_ret);
            return -1;
        }
        printf("  Step 1/2: Complete\n");
        
        // Host -> NVMe
        printf("  Step 2/2: Host -> NVMe (spdk_nvme_ns_cmd_write)...\n");
        ctx->io_completed = 0;
        ctx->io_error = 0;
        
        int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair,
                                        ctx->host_buffer,
                                        lba, num_blocks,
                                        io_complete_callback, ctx, 0);
        
        printf("  spdk_nvme_ns_cmd_write returned: %d\n", rc);
        
        if (rc != 0) {
            fprintf(stderr, "[Error] Failed to submit write command (rc=%d)\n", rc);
            fprintf(stderr, "  This usually means:\n");
            fprintf(stderr, "    1. Queue is full (try increasing io_queue_size)\n");
            fprintf(stderr, "    2. Invalid parameters (LBA out of range)\n");
            fprintf(stderr, "    3. Namespace is in error state\n");
            return -1;
        }
        
        printf("  Waiting for completion...\n");
        int poll_count = 0;
        while (! ctx->io_completed) {
            spdk_nvme_qpair_process_completions(ctx->qpair, 0);
            poll_count++;
            
            // 超时检测
            if (poll_count > 1000000) {
                fprintf(stderr, "[Error] I/O timeout after %d polls\n", poll_count);
                return -1;
            }
        }
        printf("  Polled %d times\n", poll_count);
        
        if (ctx->io_error) {
            fprintf(stderr, "[Error] Write operation failed (I/O error)\n");
            return -1;
        }
        
        printf("  Step 2/2: Complete\n");
        
        transferred += chunk_size;
        remaining -= chunk_size;
        
        printf("[Chunk %d] Success - Progress: %.2f / %.2f MB (%.1f%%)\n",
               chunk_num,
               transferred / 1024.0 / 1024.0,
               size / 1024.0 / 1024.0,
               100.0 * transferred / size);
    }
    
    printf("\n[NPU-NVMe Write] All chunks completed successfully\n");
    printf("  Total transferred: %.2f MB in %d chunks\n", 
           size / 1024.0 / 1024.0, chunk_num);
    
    return 0;
}

int npu_nvme_read(npu_nvme_context_t *ctx,
                  void *npu_buffer,
                  uint64_t nvme_offset,
                  size_t size) {
    if (!ctx || !npu_buffer || size == 0) {
        fprintf(stderr, "[Error] Invalid parameters\n");
        return -1;
    }
    
    printf("[NPU-NVMe Read] Total size: %zu bytes (%.2f MB)\n", 
           size, size / 1024.0 / 1024.0);
    
    size_t remaining = size;
    size_t transferred = 0;
    
    while (remaining > 0) {
        size_t chunk_size = (remaining > MAX_SINGLE_TRANSFER) 
                            ? MAX_SINGLE_TRANSFER 
                            : remaining;
        
        size_t aligned_size = (chunk_size + ctx->block_size - 1) 
                              & ~(ctx->block_size - 1);
        uint64_t lba = (nvme_offset + transferred) / ctx->block_size;
        uint32_t num_blocks = aligned_size / ctx->block_size;
        
        if (lba + num_blocks > ctx->total_blocks) {
            fprintf(stderr, "[Error] Read exceeds device capacity\n");
            return -1;
        }
        
        if (ctx->host_buffer_size < aligned_size) {
            if (ctx->host_buffer) {
                spdk_dma_free(ctx->host_buffer);
            }
            ctx->host_buffer = spdk_dma_zmalloc(aligned_size, 4096, NULL);
            if (!ctx->host_buffer) {
                fprintf(stderr, "[Error] Failed to allocate host buffer\n");
                return -1;
            }
            ctx->host_buffer_size = aligned_size;
        }
        
        memset(ctx->host_buffer, 0, aligned_size);
        
        // NVMe -> Host
        ctx->io_completed = 0;
        ctx->io_error = 0;
        
        int rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair,
                                       ctx->host_buffer,
                                       lba, num_blocks,
                                       io_complete_callback, ctx, 0);
        if (rc != 0) {
            fprintf(stderr, "[Error] Failed to submit read command at offset %zu\n", 
                    transferred);
            return -1;
        }
        
        while (! ctx->io_completed) {
            spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        }
        
        if (ctx->io_error) {
            fprintf(stderr, "[Error] Read operation failed at offset %zu\n", 
                    transferred);
            return -1;
        }
        
        // Host -> NPU
        aclError ret = aclrtMemcpy((char *)npu_buffer + transferred, chunk_size,
                                   ctx->host_buffer, chunk_size,
                                   ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Error] Host to NPU transfer failed at offset %zu: %d\n", 
                    transferred, ret);
            return -1;
        }
        
        transferred += chunk_size;
        remaining -= chunk_size;
        
        if (size > 100 * 1024 * 1024) {
            printf("[Progress] %.2f / %.2f MB (%.1f%%)\n",
                   transferred / 1024.0 / 1024.0,
                   size / 1024.0 / 1024.0,
                   100.0 * transferred / size);
        }
    }
    
    printf("[NPU-NVMe Read] Completed: %.2f MB\n", size / 1024.0 / 1024.0);
    return 0;
}

int npu_nvme_write_pipeline(npu_nvme_context_t *ctx,
                            void *npu_buffer,
                            uint64_t nvme_offset,
                            size_t size,
                            int pipeline_depth) {
    if (!  ctx || ! npu_buffer || size == 0) {
        return -1;
    }
    
    if (pipeline_depth < 1 || pipeline_depth > 16) {
        fprintf(stderr, "[Pipeline] Invalid depth %d, using 4\n", pipeline_depth);
        pipeline_depth = 4;
    }
    
    printf("\n[Pipeline Write] Starting\n");
    printf("  Total size: %.2f MB\n", size / 1024.0 / 1024.0);
    printf("  Pipeline depth: %d\n", pipeline_depth);
    printf("  Chunk size: %.2f MB\n", MAX_SINGLE_TRANSFER / 1024.0 / 1024.0);
    
    // 分配pipeline slots
    struct pipeline_slot *slots = calloc(pipeline_depth, sizeof(struct pipeline_slot));
    if (! slots) {
        fprintf(stderr, "[Pipeline] Failed to allocate slots\n");
        return -1;
    }
    
    // 初始化每个slot
    for (int i = 0; i < pipeline_depth; i++) {
        slots[i].slot_id = i;
        slots[i].state = SLOT_FREE;
        slots[i].error = 0;
        slots[i].buffer_size = 0;
        slots[i]. host_buffer = NULL;
    }
    
    // Pipeline状态
    size_t total_submitted = 0;
    int num_chunks = (size + MAX_SINGLE_TRANSFER - 1) / MAX_SINGLE_TRANSFER;
    int next_chunk_to_prepare = 0;    // 下一个要准备的chunk编号
    int next_chunk_to_submit = 0;     // 下一个要提交的chunk编号
    int chunks_completed = 0;          // 已完成的chunk数量
    
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    printf("[Pipeline] Total chunks: %d\n", num_chunks);
    
    // Pipeline主循环
    while (chunks_completed < num_chunks) {
        
        // === Stage 1: 准备新的chunk（NPU -> Host拷贝）===
        if (next_chunk_to_prepare < num_chunks) {
            // 找一个空闲的slot
            for (int i = 0; i < pipeline_depth; i++) {
                if (slots[i].  state == SLOT_FREE) {
                    
                    // 计算这个chunk的大小和偏移
                    size_t chunk_offset = (size_t)next_chunk_to_prepare * MAX_SINGLE_TRANSFER;
                    size_t remaining = size - chunk_offset;
                    size_t chunk_size = (remaining > MAX_SINGLE_TRANSFER) 
                                        ? MAX_SINGLE_TRANSFER 
                                        : remaining;
                    size_t aligned_size = (chunk_size + ctx->block_size - 1) 
                                          & ~(ctx->block_size - 1);
                    
                    // 分配或调整buffer
                    if (slots[i].buffer_size < aligned_size) {
                        if (slots[i].  host_buffer) {
                            spdk_dma_free(slots[i]. host_buffer);
                        }
                        slots[i].host_buffer = spdk_dma_zmalloc(aligned_size, 4096, NULL);
                        if (! slots[i].host_buffer) {
                            fprintf(stderr, "[Pipeline] Failed to allocate buffer\n");
                            goto cleanup_error;
                        }
                        slots[i].buffer_size = aligned_size;
                    }
                    
                    // NPU -> Host 拷贝
                    aclError ret = aclrtMemcpy(
                        slots[i].host_buffer,
                        aligned_size,
                        (char *)npu_buffer + chunk_offset,
                        chunk_size,
                        ACL_MEMCPY_DEVICE_TO_HOST
                    );
                    
                    if (ret != ACL_SUCCESS) {
                        fprintf(stderr, "[Pipeline] NPU copy failed (chunk %d)\n", 
                                next_chunk_to_prepare);
                        goto cleanup_error;
                    }
                    
                    // 设置chunk信息
                    slots[i]. chunk_size = chunk_size;
                    slots[i].nvme_lba = (nvme_offset + chunk_offset) / ctx->block_size;
                    slots[i].num_blocks = aligned_size / ctx->block_size;
                    slots[i]. state = SLOT_READY;
                    slots[i].error = 0;
                    
                    next_chunk_to_prepare++;
                    
                    // 只准备一个，下次循环再继续
                    break;
                }
            }
        }
        
        // === Stage 2: 提交准备好的chunk到NVMe ===
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i]. state == SLOT_READY) {
                
                // 提交写命令
                int rc = spdk_nvme_ns_cmd_write(
                    ctx->ns,
                    ctx->qpair,
                    slots[i].host_buffer,
                    slots[i].nvme_lba,
                    slots[i].num_blocks,
                    pipeline_write_complete,
                    &slots[i],
                    0
                );
                
                if (rc != 0) {
                    fprintf(stderr, "[Pipeline] Failed to submit slot %d (rc=%d)\n", i, rc);
                    goto cleanup_error;
                }
                
                slots[i].state = SLOT_SUBMITTED;
                next_chunk_to_submit++;
                
                // 显示进度
                if (next_chunk_to_submit % 10 == 0 || next_chunk_to_submit == num_chunks) {
                    gettimeofday(&current_time, NULL);
                    double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                                    (current_time.tv_usec - start_time.tv_usec) / 1e6;
                    double progress_size = (double)next_chunk_to_submit * MAX_SINGLE_TRANSFER;
                    if (progress_size > size) progress_size = size;
                    double speed = progress_size / elapsed / 1024.0 / 1024.0;
                    
                    printf("[Pipeline] Progress:  %d/%d chunks, %.1f MB/s\n",
                           next_chunk_to_submit, num_chunks, speed);
                }
            }
        }
        
        // === Stage 3: Poll完成事件并回收slot ===
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        
        // 检查哪些slot完成了
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i].state == SLOT_COMPLETED) {
                if (slots[i].error) {
                    fprintf(stderr, "[Pipeline] Slot %d completed with error\n", i);
                    goto cleanup_error;
                }
                
                // 标记为FREE，可以复用
                slots[i].state = SLOT_FREE;
                chunks_completed++;
            }
        }
        
        // 超时检测
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1e6;
        if (elapsed > 60.0) {
            fprintf(stderr, "[Pipeline] Timeout after %.1f seconds\n", elapsed);
            fprintf(stderr, "[Pipeline] Status: prepared=%d, submitted=%d, completed=%d/%d\n",
                    next_chunk_to_prepare, next_chunk_to_submit, chunks_completed, num_chunks);
            
            // 打印每个slot的状态
            for (int i = 0; i < pipeline_depth; i++) {
                const char *state_str = "UNKNOWN";
                switch (slots[i].state) {
                    case SLOT_FREE: state_str = "FREE"; break;
                    case SLOT_COPYING: state_str = "COPYING"; break;
                    case SLOT_READY: state_str = "READY"; break;
                    case SLOT_SUBMITTED: state_str = "SUBMITTED"; break;
                    case SLOT_COMPLETED: state_str = "COMPLETED"; break;
                }
                fprintf(stderr, "[Pipeline] Slot %d: %s\n", i, state_str);
            }
            
            goto cleanup_error;
        }
    }
    
    // 计算最终统计
    gettimeofday(&current_time, NULL);
    double total_time = (current_time.tv_sec - start_time.tv_sec) + 
                       (current_time.tv_usec - start_time.tv_usec) / 1e6;
    double avg_speed = size / total_time / 1024.0 / 1024.0;
    
    printf("\n[Pipeline Write] Completed!\n");
    printf("  Total:  %.2f MB in %.3f seconds\n", 
           size / 1024.0 / 1024.0, total_time);
    printf("  Average speed: %.2f MB/s\n", avg_speed);
    printf("  Chunks:  prepared=%d, submitted=%d, completed=%d\n",
           next_chunk_to_prepare, next_chunk_to_submit, chunks_completed);
    
    // 清理
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i]. host_buffer) {
            spdk_dma_free(slots[i].host_buffer);
        }
    }
    free(slots);
    
    return 0;

cleanup_error:
    // 错误清理
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i].host_buffer) {
            spdk_dma_free(slots[i].host_buffer);
        }
    }
    free(slots);
    return -1;
}

void npu_nvme_cleanup(npu_nvme_context_t *ctx) {
    if (!ctx) {
        return;
    }
    
    printf("\n========================================\n");
    printf("Cleaning up NPU-NVMe\n");
    printf("========================================\n");
    
    // Free host buffer
    if (ctx->host_buffer) {
        spdk_dma_free(ctx->host_buffer);
        ctx->host_buffer = NULL;
    }
    
    // Free I/O queue pair
    if (ctx->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
        ctx->qpair = NULL;
    }
    
    // Detach controller
    if (ctx->ctrlr) {
        spdk_nvme_detach(ctx->ctrlr);
        ctx->ctrlr = NULL;
    }
    
    // Reset NPU device
    aclrtResetDevice(ctx->npu_device_id);
    aclFinalize();
    
    free(ctx);
    
    printf("Cleanup complete\n");
    printf("========================================\n\n");
}
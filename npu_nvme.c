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
    SLOT_COMPLETED,       // NVMe写完成
} slot_state_t;

typedef enum {
    SLOT_STATE_FREE = 0,
    SLOT_STATE_COPYING_NPU,
    SLOT_STATE_COPY_DONE,
    SLOT_STATE_NVME_SUBMITTED,
    SLOT_STATE_NVME_COMPLETED
} pipeline_slot_state_t;


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

typedef struct {
    void *npu_buffer;      // NPU地址
    uint64_t nvme_offset;  // NVMe偏移
    size_t size;           // 大小
    int status;            // 0=pending, 1=done, -1=error
} batch_transfer_item_t;

typedef struct {
    batch_transfer_item_t *items;
    int num_items;
    int current_item;              // 当前处理的item
    size_t current_item_offset;    // 当前item内的偏移
    int total_chunks;
    int chunks_prepared;
    int chunks_submitted;
    int chunks_completed;
} batch_context_t;

typedef struct pipeline_slot {
    // Buffer
    void *host_buffer;
    size_t buffer_size;
    
    // Chunk info
    size_t chunk_size;
    uint64_t nvme_lba;
    uint32_t num_blocks;
    
    // ACL async
    aclrtStream acl_stream;
    aclrtEvent acl_event;
    
    // State
    pipeline_slot_state_t state;
    int error;
    int slot_id;
} pipeline_slot_t;

// ==================================================
// Helper Functions
// ==================================================

static void async_pipeline_write_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    pipeline_slot_t *slot = (pipeline_slot_t *)arg;
    
    printf("[Callback] Slot %d:  write completed\n", slot->slot_id);
    
    if (spdk_nvme_cpl_is_error(cpl)) {
        fprintf(stderr, "[Callback] Slot %d:  NVMe error\n", slot->slot_id);
        slot->error = 1;
    }
    
    // 更新状态
    slot->state = SLOT_STATE_NVME_COMPLETED;
}

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
                   size_t size, 
                   size_t chunk_size_max){
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
        size_t chunk_size = (remaining > chunk_size_max) 
                            ? chunk_size_max 
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
                  size_t size,
                  size_t chunk_size_max) {
    if (!ctx || !npu_buffer || size == 0) {
        fprintf(stderr, "[Error] Invalid parameters\n");
        return -1;
    }
    
    printf("[NPU-NVMe Read] Total size: %zu bytes (%.2f MB)\n", 
           size, size / 1024.0 / 1024.0);
    
    size_t remaining = size;
    size_t transferred = 0;
    
    while (remaining > 0) {
        size_t chunk_size = (remaining > chunk_size_max) 
                            ? chunk_size_max 
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
                            int pipeline_depth,
                            size_t chunk_size_max) {
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
    printf("  Chunk size: %.2f MB\n", chunk_size_max / 1024.0 / 1024.0);
    
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
    int num_chunks = (size + chunk_size_max - 1) / chunk_size_max;
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
                    size_t chunk_offset = (size_t)next_chunk_to_prepare * chunk_size_max;
                    size_t remaining = size - chunk_offset;
                    size_t chunk_size = (remaining > chunk_size_max) 
                                        ? chunk_size_max 
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
                    slots[i].chunk_size = chunk_size;
                    slots[i].nvme_lba = (nvme_offset + chunk_offset) / ctx->block_size;
                    slots[i].num_blocks = aligned_size / ctx->block_size;
                    slots[i].state = SLOT_READY;
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
                    double progress_size = (double)next_chunk_to_submit * chunk_size_max;
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

int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                         void **npu_buffers,
                         uint64_t *nvme_offsets,
                         size_t *sizes,
                         int num_items,
                         int pipeline_depth,
                         size_t chunk_size_max) {
    
    if (! ctx || !npu_buffers || !nvme_offsets || !sizes || num_items <= 0) {
        return -1;
    }
    
    if (pipeline_depth < 1 || pipeline_depth > 16) {
        pipeline_depth = 4;
    }
    
    printf("\n[Batch Pipeline Write] Starting\n");
    printf("  Number of items: %d\n", num_items);
    printf("  Pipeline depth: %d\n", pipeline_depth);
    
    // 计算总大小和总chunk数
    size_t total_size = 0;
    int total_chunks = 0;
    for (int i = 0; i < num_items; i++) {
        total_size += sizes[i];
        total_chunks += (sizes[i] + chunk_size_max - 1) / chunk_size_max;
    }
    
    printf("  Total size:  %.2f MB\n", total_size / 1024.0 / 1024.0);
    printf("  Total chunks: %d\n", total_chunks);
    
    // 初始化batch context
    batch_context_t batch_ctx = {
        .num_items = num_items,
        .current_item = 0,
        .current_item_offset = 0,
        . total_chunks = total_chunks,
        .chunks_prepared = 0,
        .chunks_submitted = 0,
        .chunks_completed = 0
    };
    
    // 分配pipeline slots
    struct pipeline_slot *slots = calloc(pipeline_depth, sizeof(struct pipeline_slot));
    if (!slots) {
        fprintf(stderr, "[Batch] Failed to allocate slots\n");
        return -1;
    }
    
    for (int i = 0; i < pipeline_depth; i++) {
        slots[i].slot_id = i;
        slots[i].state = SLOT_FREE;
        slots[i].error = 0;
        slots[i].buffer_size = 0;
        slots[i]. host_buffer = NULL;
    }
    
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    int last_progress = 0;
    
    // ========================================
    // Pipeline主循环（跨参数）
    // ========================================
    while (batch_ctx.chunks_completed < total_chunks) {
        
        // === Stage 1: 准备新chunk（可能来自不同参数）===
        if (batch_ctx.chunks_prepared < total_chunks) {
            for (int slot_idx = 0; slot_idx < pipeline_depth; slot_idx++) {
                if (slots[slot_idx].state == SLOT_FREE) {
                    
                    // 跳过已经处理完的item
                    while (batch_ctx.current_item < num_items && 
                           batch_ctx.current_item_offset >= sizes[batch_ctx.current_item]) {
                        batch_ctx.current_item++;
                        batch_ctx.current_item_offset = 0;
                    }
                    
                    if (batch_ctx.current_item >= num_items) {
                        break;  // 所有item都已准备
                    }
                    
                    // 计算当前chunk
                    int item_idx = batch_ctx.current_item;
                    size_t item_offset = batch_ctx.current_item_offset;
                    size_t remaining = sizes[item_idx] - item_offset;
                    
                    size_t chunk_size = (remaining > chunk_size_max) 
                                        ? chunk_size_max 
                                        : remaining;
                    size_t aligned_size = (chunk_size + ctx->block_size - 1) 
                                          & ~(ctx->block_size - 1);
                    
                    // 分配buffer
                    if (slots[slot_idx].buffer_size < aligned_size) {
                        if (slots[slot_idx].host_buffer) {
                            spdk_dma_free(slots[slot_idx].host_buffer);
                        }
                        slots[slot_idx]. host_buffer = spdk_dma_zmalloc(aligned_size, 4096, NULL);
                        if (!slots[slot_idx].host_buffer) {
                            fprintf(stderr, "[Batch] Failed to allocate buffer\n");
                            goto cleanup_error;
                        }
                        slots[slot_idx]. buffer_size = aligned_size;
                    }
                    
                    // NPU -> Host拷贝
                    aclError ret = aclrtMemcpy(
                        slots[slot_idx].host_buffer,
                        aligned_size,
                        (char *)npu_buffers[item_idx] + item_offset,
                        chunk_size,
                        ACL_MEMCPY_DEVICE_TO_HOST
                    );
                    
                    if (ret != ACL_SUCCESS) {
                        fprintf(stderr, "[Batch] NPU copy failed (item %d, offset %zu)\n", 
                                item_idx, item_offset);
                        goto cleanup_error;
                    }
                    
                    // 设置chunk信息
                    uint64_t nvme_lba = (nvme_offsets[item_idx] + item_offset) / ctx->block_size;
                    uint32_t num_blocks = aligned_size / ctx->block_size;
                    
                    slots[slot_idx]. chunk_size = chunk_size;
                    slots[slot_idx]. nvme_lba = nvme_lba;
                    slots[slot_idx].num_blocks = num_blocks;
                    slots[slot_idx].state = SLOT_READY;
                    slots[slot_idx].error = 0;
                    
                    batch_ctx.current_item_offset += chunk_size;
                    batch_ctx.chunks_prepared++;
                    
                    break;  // 每次准备一个
                }
            }
        }
        
        // === Stage 2: 提交准备好的chunk ===
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i].state == SLOT_READY) {
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
                    fprintf(stderr, "[Batch] Failed to submit slot %d (rc=%d)\n", i, rc);
                    goto cleanup_error;
                }
                
                slots[i].state = SLOT_SUBMITTED;
                batch_ctx.chunks_submitted++;
            }
        }
        
        // === Stage 3: Poll完成事件 ===
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i]. state == SLOT_COMPLETED) {
                if (slots[i].error) {
                    fprintf(stderr, "[Batch] Slot %d completed with error\n", i);
                    goto cleanup_error;
                }
                slots[i].state = SLOT_FREE;
                batch_ctx.chunks_completed++;
            }
        }
        
        // 显示进度（每10%）
        int progress_pct = (batch_ctx.chunks_completed * 100) / total_chunks;
        if (progress_pct >= last_progress + 10) {
            gettimeofday(&current_time, NULL);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                           (current_time.tv_usec - start_time.tv_usec) / 1e6;
            double completed_size = 0;
            for (int i = 0; i <= batch_ctx.current_item && i < num_items; i++) {
                if (i < batch_ctx.current_item) {
                    completed_size += sizes[i];
                } else {
                    completed_size += batch_ctx.current_item_offset;
                }
            }
            double speed = completed_size / elapsed / 1024.0 / 1024.0;
            
            printf("[Batch] Progress:  %d%% (%d/%d chunks), %.1f MB/s\n",
                   progress_pct, batch_ctx.chunks_completed, total_chunks, speed);
            last_progress = progress_pct;
        }
        
        // 超时检测
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1e6;
        if (elapsed > 120.0) {
            fprintf(stderr, "[Batch] Timeout after %.1fs\n", elapsed);
            goto cleanup_error;
        }
    }
    
    // 计算统计
    gettimeofday(&current_time, NULL);
    double total_time = (current_time.tv_sec - start_time.tv_sec) + 
                       (current_time.tv_usec - start_time.tv_usec) / 1e6;
    double avg_speed = total_size / total_time / 1024.0 / 1024.0;
    
    printf("\n[Batch Pipeline Write] Completed!\n");
    printf("  Total:  %.2f MB in %.3f seconds\n", 
           total_size / 1024.0 / 1024.0, total_time);
    printf("  Average speed: %.2f MB/s\n", avg_speed);
    printf("  Items:  %d, Chunks: prepared=%d, submitted=%d, completed=%d\n",
           num_items, batch_ctx.chunks_prepared, batch_ctx.chunks_submitted, 
           batch_ctx.chunks_completed);
    
    // 清理
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i]. host_buffer) {
            spdk_dma_free(slots[i].host_buffer);
        }
    }
    free(slots);
    
    return 0;

cleanup_error:
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i].host_buffer) {
            spdk_dma_free(slots[i].host_buffer);
        }
    }
    free(slots);
    return -1;
}

int npu_nvme_write_batch_async(npu_nvme_context_t *ctx,
                                void **npu_buffers,
                                uint64_t *nvme_offsets,
                                size_t *sizes,
                                int num_items,
                                int pipeline_depth,
                                size_t chunk_size_max) {
    
    printf("\n[Async Batch Pipeline] Starting\n");
    printf("  Items: %d, Pipeline depth: %d\n", num_items, pipeline_depth);
    
    // 计算总信息
    size_t total_size = 0;
    int total_chunks = 0;
    for (int i = 0; i < num_items; i++) {
        total_size += sizes[i];
        total_chunks += (sizes[i] + chunk_size_max - 1) / chunk_size_max;
    }
    printf("  Total:  %.2f MB, %d chunks\n", total_size/1024.0/1024.0, total_chunks);
    
    // 分配slots
    pipeline_slot_t *slots = calloc(pipeline_depth, sizeof(pipeline_slot_t));
    if (!slots) {
        fprintf(stderr, "[Pipeline] Failed to allocate slots\n");
        return -1;
    }
    
    // 初始化slots
    for (int i = 0; i < pipeline_depth; i++) {
        slots[i].slot_id = i;
        slots[i].state = SLOT_STATE_FREE;
        slots[i]. error = 0;
        slots[i].buffer_size = 0;
        slots[i]. host_buffer = NULL;
        slots[i].acl_stream = NULL;
        slots[i].acl_event = NULL;
        
        // 创建stream
        aclError ret = aclrtCreateStream(&slots[i].acl_stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Pipeline] Failed to create stream %d: %d\n", i, ret);
            goto init_cleanup;
        }
        
        // 创建event
        ret = aclrtCreateEvent(&slots[i].acl_event);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Pipeline] Failed to create event %d: %d\n", i, ret);
            goto init_cleanup;
        }
        
        printf("[Pipeline] Slot %d initialized\n", i);
    }
    
    // Pipeline状态
    int current_item = 0;
    size_t current_item_offset = 0;
    int chunks_prepared = 0;
    int chunks_submitted = 0;
    int chunks_completed = 0;
    
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    printf("[Pipeline] Starting main loop.. .\n");
    
    int loop_count = 0;
    
    // 主循环
    while (chunks_completed < total_chunks) {
        loop_count++;
        
        // 每1000次循环输出一次心跳
        if (loop_count % 10000 == 0) {
            printf("[Pipeline] Loop %d:  prep=%d, sub=%d, comp=%d\n",
                   loop_count, chunks_prepared, chunks_submitted, chunks_completed);
        }
        
        // ========================================
        // Stage 1: 发起NPU拷贝
        // ========================================
        if (chunks_prepared < total_chunks) {
            for (int i = 0; i < pipeline_depth; i++) {
                if (slots[i].state == SLOT_STATE_FREE) {
                    
                    // 跳过已完成的item
                    while (current_item < num_items && 
                           current_item_offset >= sizes[current_item]) {
                        current_item++;
                        current_item_offset = 0;
                    }
                    
                    if (current_item >= num_items) break;
                    
                    // 计算chunk
                    size_t remaining = sizes[current_item] - current_item_offset;
                    size_t chunk_size = (remaining > chunk_size_max) 
                                        ? chunk_size_max : remaining;
                    size_t aligned_size = (chunk_size + ctx->block_size - 1) 
                                          & ~(ctx->block_size - 1);
                    
                    // 分配buffer
                    if (slots[i].buffer_size < aligned_size) {
                        if (slots[i].host_buffer) {
                            spdk_dma_free(slots[i].host_buffer);
                        }
                        slots[i]. host_buffer = spdk_dma_zmalloc(aligned_size, 4096, NULL);
                        if (!slots[i].host_buffer) {
                            fprintf(stderr, "[Pipeline] Failed to allocate buffer\n");
                            goto cleanup;
                        }
                        slots[i].buffer_size = aligned_size;
                    }
                    
                    // 异步拷贝
                    void *npu_src = (char *)npu_buffers[current_item] + current_item_offset;
                    
                    aclError acl_ret = aclrtMemcpyAsync(
                        slots[i].host_buffer,
                        aligned_size,
                        npu_src,
                        chunk_size,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                        slots[i]. acl_stream
                    );
                    
                    if (acl_ret != ACL_SUCCESS) {
                        fprintf(stderr, "[Pipeline] aclrtMemcpyAsync failed: %d\n", acl_ret);
                        goto cleanup;
                    }
                    
                    // 记录event
                    acl_ret = aclrtRecordEvent(slots[i].acl_event, slots[i].acl_stream);
                    if (acl_ret != ACL_SUCCESS) {
                        fprintf(stderr, "[Pipeline] aclrtRecordEvent failed: %d\n", acl_ret);
                        goto cleanup;
                    }
                    
                    // 设置chunk信息
                    slots[i].chunk_size = chunk_size;
                    slots[i]. nvme_lba = (nvme_offsets[current_item] + current_item_offset) 
                                        / ctx->block_size;
                    slots[i].num_blocks = aligned_size / ctx->block_size;
                    slots[i].state = SLOT_STATE_COPYING_NPU;
                    
                    current_item_offset += chunk_size;
                    chunks_prepared++;
                    
                    printf("[Pipeline] Slot %d:  started NPU copy (chunk %d/%d)\n",
                           i, chunks_prepared, total_chunks);
                    
                    break;
                }
            }
        }
        
        // ========================================
        // Stage 2: 检查NPU拷贝完成
        // ========================================
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i]. state == SLOT_STATE_COPYING_NPU) {
                aclrtEventStatus event_status;
                aclError ret = aclrtQueryEvent(slots[i].acl_event, &event_status);
                
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[Pipeline] aclrtQueryEvent failed: %d\n", ret);
                    slots[i].error = 1;
                    goto cleanup;
                }
                
                if (event_status == ACL_EVENT_STATUS_COMPLETE) {
                    slots[i].state = SLOT_STATE_COPY_DONE;
                    printf("[Pipeline] Slot %d: NPU copy completed\n", i);
                }
            }
        }
        
        // ========================================
        // Stage 3: 提交NVMe写
        // ========================================
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i]. state == SLOT_STATE_COPY_DONE) {
                
                printf("[Pipeline] Slot %d:  submitting NVMe write (LBA=%lu, blocks=%u)\n",
                       i, slots[i].nvme_lba, slots[i].num_blocks);
                
                int rc = spdk_nvme_ns_cmd_write(
                    ctx->ns,
                    ctx->qpair,
                    slots[i].host_buffer,
                    slots[i].nvme_lba,
                    slots[i].num_blocks,
                    async_pipeline_write_complete,  // 使用明确命名的回调
                    &slots[i],
                    0
                );
                
                if (rc != 0) {
                    fprintf(stderr, "[Pipeline] Failed to submit NVMe write (rc=%d)\n", rc);
                    goto cleanup;
                }
                
                slots[i].state = SLOT_STATE_NVME_SUBMITTED;
                chunks_submitted++;
                
                printf("[Pipeline] Slot %d:  NVMe write submitted (total %d/%d)\n",
                       i, chunks_submitted, total_chunks);
            }
        }
        
        // ========================================
        // Stage 4: Poll NVMe完成
        // ========================================
        int completions = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        if (completions > 0) {
            printf("[Pipeline] Polled %d completions\n", completions);
        }
        
        for (int i = 0; i < pipeline_depth; i++) {
            if (slots[i]. state == SLOT_STATE_NVME_COMPLETED) {
                if (slots[i].error) {
                    fprintf(stderr, "[Pipeline] Slot %d completed with error\n", i);
                    goto cleanup;
                }
                
                printf("[Pipeline] Slot %d:  fully completed, marking FREE\n", i);
                slots[i].state = SLOT_STATE_FREE;
                chunks_completed++;
            }
        }
        
        // 超时检测
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start_time.tv_sec) + 
                        (now.tv_usec - start_time.tv_usec) / 1e6;
        if (elapsed > 30.0) {  // 降低到30秒，更容易调试
            fprintf(stderr, "[Pipeline] Timeout after %.1f seconds\n", elapsed);
            fprintf(stderr, "[Pipeline] Status: prep=%d, sub=%d, comp=%d/%d\n",
                    chunks_prepared, chunks_submitted, chunks_completed, total_chunks);
            
            for (int i = 0; i < pipeline_depth; i++) {
                const char *state_str = "UNKNOWN";
                switch (slots[i].state) {
                    case SLOT_STATE_FREE:  state_str = "FREE"; break;
                    case SLOT_STATE_COPYING_NPU:  state_str = "COPYING_NPU"; break;
                    case SLOT_STATE_COPY_DONE: state_str = "COPY_DONE"; break;
                    case SLOT_STATE_NVME_SUBMITTED: state_str = "NVME_SUBMITTED"; break;
                    case SLOT_STATE_NVME_COMPLETED: state_str = "NVME_COMPLETED"; break;
                    default: state_str = "INVALID"; break;
                }
                fprintf(stderr, "  Slot %d: state=%d (%s), error=%d\n", 
                        i, slots[i]. state, state_str, slots[i].error);
            }
            
            goto cleanup;
        }
    }
    
    // 成功完成
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    double total_time = (end_time. tv_sec - start_time. tv_sec) + 
                       (end_time.tv_usec - start_time.tv_usec) / 1e6;
    double speed = total_size / total_time / 1024.0 / 1024.0;
    
    printf("\n[Async Batch Pipeline] Completed!\n");
    printf("  Total:  %.2f MB in %.3f seconds\n", 
           total_size/1024.0/1024.0, total_time);
    printf("  Average speed: %.2f MB/s\n", speed);
    
    // 清理
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i]. acl_event) aclrtDestroyEvent(slots[i].acl_event);
        if (slots[i]. acl_stream) aclrtDestroyStream(slots[i].acl_stream);
        if (slots[i].host_buffer) spdk_dma_free(slots[i].host_buffer);
    }
    free(slots);
    return 0;

init_cleanup:
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i].acl_event) aclrtDestroyEvent(slots[i]. acl_event);
        if (slots[i].acl_stream) aclrtDestroyStream(slots[i].acl_stream);
    }
    free(slots);
    return -1;

cleanup:
    for (int i = 0; i < pipeline_depth; i++) {
        if (slots[i].acl_stream) aclrtSynchronizeStream(slots[i]. acl_stream);
        if (slots[i].acl_event) aclrtDestroyEvent(slots[i].acl_event);
        if (slots[i].acl_stream) aclrtDestroyStream(slots[i]. acl_stream);
        if (slots[i].host_buffer) spdk_dma_free(slots[i].host_buffer);
    }
    free(slots);
    return -1;
}
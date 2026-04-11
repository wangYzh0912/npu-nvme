#include "npu_nvme.h"
#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h> // 引入 usleep

#define MIN_PIPE_DEPTH   1
#define MAX_PIPE_DEPTH   16
#define ALIGN_4K(x) (((x) + 4095ULL) & ~4095ULL)
#define META_DMA_BUF_SIZE (1024 * 1024)

/* =========================
 * SPSC ring (单生产单消费)
 * ========================= */
typedef struct {
    int *slots;
    int capacity;
    int head;
    int tail;
} ring_t;

static int ring_init(ring_t *r, int cap) {
    r->slots = calloc(cap + 1, sizeof(int));
    if (!r->slots) return -1;
    r->capacity = cap + 1;
    r->head = r->tail = 0;
    return 0;
}
static void ring_free(ring_t *r) {
    free(r->slots);
    r->slots = NULL;
}
static bool ring_is_full(ring_t *r) {
    return ((r->tail + 1) % r->capacity) == r->head;
}
static bool ring_is_empty(ring_t *r) {
    return r->head == r->tail;
}
static bool ring_push(ring_t *r, int v) {
    if (ring_is_full(r)) return false;
    r->slots[r->tail] = v;
    r->tail = (r->tail + 1) % r->capacity;
    return true;
}
static bool ring_pop(ring_t *r, int *out) {
    if (ring_is_empty(r)) return false;
    *out = r->slots[r->head];
    r->head = (r->head + 1) % r->capacity;
    return true;
}

typedef struct dma_buf {
    void *buf;
    size_t size;
} dma_buf_t;

typedef struct {
    int      buf_idx;
    int      state;
    uint64_t copy_us;
    uint64_t submit_ts;
    uint64_t done_ts;
} item_stat_t;

typedef struct {
    int           item;
    int          *flag_ptr;
    item_stat_t  *stat_ptr;
} cb_ctx_t;

static inline uint64_t tv_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/* 修复了原本强转 arg 为 int* 导致的内存踩踏 Bug */
static void io_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    cb_ctx_t *c = (cb_ctx_t *)arg;
    int err = spdk_nvme_cpl_is_error(cpl) ? -1 : 1;
    *(c->flag_ptr) = err;
    c->stat_ptr[c->item].state   = 2;
    c->stat_ptr[c->item].done_ts = tv_us();
}

/* 元数据专属的简化回调 */
static void io_complete_meta(void *arg, const struct spdk_nvme_cpl *cpl) {
    int *flag = (int *)arg;
    *flag = spdk_nvme_cpl_is_error(cpl) ? -1 : 1;
}

struct npu_nvme_context {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns    *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;

    int npu_device_id;

    /* 数据面 DMA buffer pool */
    dma_buf_t *pool;
    int pool_size;       
    ring_t free_ring;

    /* 控制面（元数据）专属 DMA buffer */
    void *meta_dma_buf;

    int max_transfer;
    int mdts_limit;

    int pipeline_depth;
    bool enable_profiling;
    char profiling_dir[256];
};

static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    npu_nvme_context_t *ctx = cb_ctx;
    int nsid;
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr);
         nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (!ns || !spdk_nvme_ns_is_active(ns)) continue;
        ctx->ctrlr = ctrlr;
        ctx->ns = ns;
        ctx->block_size = spdk_nvme_ns_get_sector_size(ns);
        ctx->total_blocks = spdk_nvme_ns_get_num_sectors(ns);
        break;
    }
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    return true;
}

int npu_nvme_init(npu_nvme_context_t **pctx,
                  const char *nvme_pci_addr,
                  int npu_device_id,
                  int pipeline_depth,
                  int chunk_size,
                  bool enable_profiling,
                  const char *profiling_dir) {
    if (!pctx || !nvme_pci_addr) return -1;
    
    if (pipeline_depth < MIN_PIPE_DEPTH) pipeline_depth = MIN_PIPE_DEPTH;
    if (pipeline_depth > MAX_PIPE_DEPTH) pipeline_depth = MAX_PIPE_DEPTH;

    npu_nvme_context_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return -1;
    ctx->pipeline_depth = pipeline_depth;
    ctx->enable_profiling = enable_profiling;
    if (profiling_dir) {
        strncpy(ctx->profiling_dir, profiling_dir, sizeof(ctx->profiling_dir) - 1);
    } else {
        strcpy(ctx->profiling_dir, ".");
    }
    ctx->mdts_limit = 0; 

    static int spdk_inited = 0;
    if (!spdk_inited) {
        struct spdk_env_opts opts;
        spdk_env_opts_init(&opts);
        opts.name = "npu_nvme";
        const char *shm = getenv("SPDK_SHM_ID");
        if (shm) opts.shm_id = atoi(shm);
        if (spdk_env_init(&opts) < 0) {
            free(ctx);
            return -1;
        }
        spdk_inited = 1;
    }

    aclrtSetDevice(npu_device_id);
    ctx->npu_device_id = npu_device_id;

    struct spdk_nvme_transport_id trid;
    memset(&trid, 0, sizeof(trid));
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", nvme_pci_addr);

    if (spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, NULL) != 0 || !ctx->ctrlr) {
        aclrtResetDevice(ctx->npu_device_id);
        aclFinalize();
        free(ctx);
        return -1;
    }

    ctx->max_transfer = (chunk_size == 0) ? (4 * 1024 * 1024ULL) : chunk_size;

    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, NULL, 0);
    if (!ctx->qpair) goto fail;

    /* 初始化数据面 DMA Pool */
    ctx->pool_size = pipeline_depth; 
    ctx->pool = calloc(ctx->pool_size, sizeof(dma_buf_t));
    if (!ctx->pool) goto fail;
    if (ring_init(&ctx->free_ring, ctx->pool_size) != 0) goto fail;

    for (int i = 0; i < ctx->pool_size; ++i) {
        size_t sz = ALIGN_4K(ctx->max_transfer);
        ctx->pool[i].buf = spdk_dma_zmalloc(sz, 4096, NULL);
        ctx->pool[i].size = sz;
        if (!ctx->pool[i].buf) goto fail;
        ring_push(&ctx->free_ring, i);
    }

    /* 分配元数据专属 DMA Buffer (128KB) */
    ctx->meta_dma_buf = spdk_dma_zmalloc(META_DMA_BUF_SIZE, 4096, NULL);
    if (!ctx->meta_dma_buf) goto fail;

    *pctx = ctx;
    return 0;

fail:
    npu_nvme_cleanup(ctx);
    return -1;
}

void npu_nvme_cleanup(npu_nvme_context_t *ctx) {
    if (!ctx) return;
    if (ctx->pool) {
        for (int i = 0; i < ctx->pool_size; ++i) {
            if (ctx->pool[i].buf) spdk_dma_free(ctx->pool[i].buf);
        }
        free(ctx->pool);
    }
    if (ctx->meta_dma_buf) {
        spdk_dma_free(ctx->meta_dma_buf);
    }
    ring_free(&ctx->free_ring);
    if (ctx->qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    if (ctx->ctrlr) spdk_nvme_detach(ctx->ctrlr);
    aclrtResetDevice(ctx->npu_device_id);
    aclFinalize();
    free(ctx);
}

size_t npu_nvme_get_max_transfer(npu_nvme_context_t *ctx) {
    return ctx ? ctx->max_transfer : 0;
}

uint64_t npu_nvme_get_total_blocks(npu_nvme_context_t *ctx) {
    return ctx ? (ctx->total_blocks * ctx->block_size) : 0;
}

/* =========================================================
 * 核心新增：同步元数据 I/O 引擎 (分离控制面)
 * ========================================================= */
int npu_nvme_sync_meta_io(npu_nvme_context_t *ctx, uint64_t byte_offset, uint32_t total_bytes, int is_read, void *meta_buffer) {
    if (!ctx || !meta_buffer) return -1;
    
    // C 层自动根据底层真实的 block_size 换算 LBA 和物理块数！Python 彻底解脱！
    uint64_t start_lba = byte_offset / ctx->block_size;
    uint32_t num_blocks = (total_bytes + ctx->block_size - 1) / ctx->block_size;
    size_t size = num_blocks * ctx->block_size;
    
    if (size > META_DMA_BUF_SIZE) return -1; 

    int flag = 0;
    if (is_read == 0) {
        memcpy(ctx->meta_dma_buf, meta_buffer, size);
        spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair, ctx->meta_dma_buf, start_lba, num_blocks, io_complete_meta, &flag, 0);
    } else {
        spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair, ctx->meta_dma_buf, start_lba, num_blocks, io_complete_meta, &flag, 0);
    }

    while (flag == 0) {
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
    }

    if (is_read != 0 && flag == 1) {
        memcpy(meta_buffer, ctx->meta_dma_buf, size);
    }

    return flag == 1 ? 0 : -1;
}


/* =========================================================
 * 数据面狂飙引擎
 * ========================================================= */
int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                         void **npu_ptrs,
                         uint64_t *nvme_offsets,
                         size_t *sizes,
                         int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;

    int submitted = 0, completed = 0;
    int idx;
    int ret = 0;

    int *flags = calloc(num_items, sizeof(int));
    int *buf_idx = calloc(num_items, sizeof(int));
    bool *reclaimed = calloc(num_items, sizeof(bool));
    item_stat_t *stat = calloc(num_items, sizeof(item_stat_t));
    cb_ctx_t *cb_ctx = calloc(num_items, sizeof(cb_ctx_t));
    if (!stat || !cb_ctx || !flags || !buf_idx || !reclaimed) { ret = -1; goto cleanup; }

    while (completed < num_items) {
        while (submitted < num_items) {
            if (!ring_pop(&ctx->free_ring, &idx)) break;
            size_t sz = sizes[submitted];
            if (sz == 0 || sz > ctx->max_transfer) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++; completed++; 
                continue;
            }
            size_t aligned = ALIGN_4K(sz);
            if (aligned > ctx->pool[idx].size) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++; completed++;
                continue;
            }

            uint64_t t1 = tv_us();
            aclError acret = aclrtMemcpy(ctx->pool[idx].buf, aligned,
                                         npu_ptrs[submitted], sz,
                                         ACL_MEMCPY_DEVICE_TO_HOST);
            uint64_t t2 = tv_us();
            if (acret != ACL_SUCCESS) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++; completed++;
                ring_push(&ctx->free_ring, idx);
                continue;
            }

            uint64_t lba = nvme_offsets[submitted] / ctx->block_size;
            uint32_t nblk = (uint32_t)(aligned / ctx->block_size);

            stat[submitted].copy_us   = t2 - t1;
            stat[submitted].buf_idx   = idx;
            stat[submitted].state     = 1;
            stat[submitted].submit_ts = tv_us();

            cb_ctx[submitted].item     = submitted;
            cb_ctx[submitted].flag_ptr = &flags[submitted];
            cb_ctx[submitted].stat_ptr = stat;

            flags[submitted] = 0;
            buf_idx[submitted] = idx;
            int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair,
                                            ctx->pool[idx].buf,
                                            lba, nblk,
                                            io_complete, &cb_ctx[submitted], 0);
            if (rc != 0) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++; completed++;
                ring_push(&ctx->free_ring, idx);
                continue;
            }
            submitted++;
        }

        int cpl = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        
        // 恢复原来的 usleep 退坡逻辑
        if (cpl == 0 && submitted >= num_items && completed < num_items) {
            usleep(50);
        }

        for (int i = 0; i < num_items; ++i) {
            if (!reclaimed[i] && stat[i].state == 2) { 
                ring_push(&ctx->free_ring, stat[i].buf_idx);
                reclaimed[i] = true;
                if (flags[i] != 1) ret = -1;
                completed++;
            }
        }
    }

    if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_write.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "item,buf_idx,copy_us,nvme_us\n");
            for (int i = 0; i < num_items; ++i) {
                if (stat[i].state == 2) {
                    uint64_t nvme_us = (stat[i].done_ts >= stat[i].submit_ts)
                                    ? (stat[i].done_ts - stat[i].submit_ts)
                                    : 0;
                    fprintf(f, "%d,%d,%lu,%lu\n",
                            i, stat[i].buf_idx, stat[i].copy_us, nvme_us);
                }
            }
            fclose(f);
        }
    }

cleanup:
    free(stat);
    free(cb_ctx);
    free(flags);
    free(buf_idx);
    free(reclaimed);
    return ret;
}

int npu_nvme_read_batch(npu_nvme_context_t *ctx,
                        void **npu_ptrs,
                        uint64_t *nvme_offsets,
                        size_t *sizes,
                        int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;

    int submitted = 0, completed = 0;
    int idx;
    int ret = 0;
    int *flags = calloc(num_items, sizeof(int));
    int *buf_idx = calloc(num_items, sizeof(int));
    bool *reclaimed = calloc(num_items, sizeof(bool));
    cb_ctx_t *cb_ctx = calloc(num_items, sizeof(cb_ctx_t));
    item_stat_t *stat = calloc(num_items, sizeof(item_stat_t)); 
    if (!flags || !buf_idx || !reclaimed || !cb_ctx || !stat) { ret = -1; goto cleanup; }

    while (completed < num_items) {
        while (submitted < num_items) {
            if (!ring_pop(&ctx->free_ring, &idx)) break;

            size_t sz = sizes[submitted];
            if (sz == 0 || sz > ctx->max_transfer) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++;
                continue;
            }
            size_t aligned = ALIGN_4K(sz);
            if (aligned > ctx->pool[idx].size) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++;
                continue;
            }

            uint64_t lba = nvme_offsets[submitted] / ctx->block_size;
            uint32_t nblk = (uint32_t)(aligned / ctx->block_size);

            flags[submitted] = 0;
            buf_idx[submitted] = idx;

            stat[submitted].buf_idx   = idx;
            stat[submitted].state     = 1;
            stat[submitted].submit_ts = tv_us();

            cb_ctx[submitted].item = submitted;
            cb_ctx[submitted].flag_ptr = &flags[submitted];
            cb_ctx[submitted].stat_ptr = stat; 

            int rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair,
                                           ctx->pool[idx].buf,
                                           lba, nblk,
                                           io_complete, &cb_ctx[submitted], 0);
            if (rc != 0) {
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                ring_push(&ctx->free_ring, idx);
            }
            submitted++;
        }

        int cpl = spdk_nvme_qpair_process_completions(ctx->qpair, 0);
        
        // 恢复原来的 usleep 退坡逻辑
        if (cpl == 0 && submitted >= num_items && completed < num_items) {
            usleep(50);
        }

        for (int i = 0; i < num_items; ++i) {
            if (!reclaimed[i] && flags[i] != 0) {
                if (flags[i] == 1) {
                    size_t sz = sizes[i];
                    uint64_t t1 = tv_us();
                    aclError acret = aclrtMemcpy(npu_ptrs[i], sz,
                                                 ctx->pool[buf_idx[i]].buf, sz,
                                                 ACL_MEMCPY_HOST_TO_DEVICE);
                    uint64_t t2 = tv_us();
                    if (acret != ACL_SUCCESS) {
                        ret = -1;
                    }
                    stat[i].copy_us = t2 - t1;
                } else {
                    ret = -1;
                }
                ring_push(&ctx->free_ring, buf_idx[i]);
                reclaimed[i] = true;
                completed++;
            }
        }
    }

    if (ctx->enable_profiling) {
        char path[512];
        snprintf(path, sizeof(path), "%s/time_read.csv", ctx->profiling_dir);
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "item,buf_idx,copy_us,nvme_us\n");
            for (int i = 0; i < num_items; ++i) {
                if (stat[i].state == 2) {
                    uint64_t nvme_us = (stat[i].done_ts >= stat[i].submit_ts)
                                    ? (stat[i].done_ts - stat[i].submit_ts)
                                    : 0;
                    fprintf(f, "%d,%d,%lu,%lu\n",
                            i, stat[i].buf_idx, stat[i].copy_us, nvme_us);
                }
            }
            fclose(f);
        }
    }

cleanup:
    free(flags);
    free(buf_idx);
    free(reclaimed);
    free(cb_ctx);
    free(stat);
    return ret;
}
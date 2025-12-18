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

#define MIN_PIPE_DEPTH   1
#define MAX_PIPE_DEPTH   16
#define ALIGN_4K(x) (((x) + 4095ULL) & ~4095ULL)

/* =========================
 * SPSC ring (单生产单消费)
 * ========================= */
typedef struct {
    int *slots;      /* 存放 buffer index */
    int capacity;
    int head;        /* 消费者读 */
    int tail;        /* 生产者写 */
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
    void *buf;       /* host DMA buffer */
    size_t size;     /* 已分配大小 */
} dma_buf_t;

static void io_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    int *flag = (int *)arg;
    if (spdk_nvme_cpl_is_error(cpl)) {
        *flag = -1;
    } else {
        *flag = 1;
    }
}

struct npu_nvme_context {
    /* SPDK/NVMe */
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns    *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t block_size;
    uint64_t total_blocks;

    /* ACL/NPU */
    int npu_device_id;

    /* DMA buffer pool */
    dma_buf_t *pool;
    int pool_size;       
    ring_t free_ring;    /* 可用 buffer 索引 */

    /* 设备限制 */
    size_t max_transfer; /* 由 MDTS 推导 */
    size_t mdts_limit;

    /* 管理参数 */
    int pipeline_depth;
};

/* 计算 MDTS 得到 max_transfer */
static size_t get_mdts_bytes(const struct spdk_nvme_ctrlr_data *cdata) {
    /* 2^(12 + mdts) 字节；mdts=0 表示无限制，取 4MB 保险值 */
    if (cdata->mdts == 0) return 4 * 1024 * 1024ULL;
    uint64_t sz = 1ULL << (12 + cdata->mdts);
    /* 根据你的设备测试，4MB 安全，若需要可改大/小 */
    if (sz > 4 * 1024 * 1024ULL) sz = 4 * 1024 * 1024ULL;
    return (size_t)sz;
}

/* attach 回调 */
static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    npu_nvme_context_t *ctx = cb_ctx;
    const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(ctrlr);
    size_t mdts_limit = get_mdts_bytes(cdata);

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
        ctx->mdts_limit = mdts_limit;
        printf("[NVMe] block=%u, total_blocks=%lu, max_xfer=%.2f MB\n",
               ctx->block_size, ctx->total_blocks, ctx->max_transfer/1024.0/1024.0);
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
                  size_t chunk_size) {
    if (!pctx || !nvme_pci_addr) return -1;

    if (pipeline_depth < MIN_PIPE_DEPTH) pipeline_depth = MIN_PIPE_DEPTH;
    if (pipeline_depth > MAX_PIPE_DEPTH) pipeline_depth = MAX_PIPE_DEPTH;

    npu_nvme_context_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return -1;
    ctx->pipeline_depth = pipeline_depth;
    ctx->mdts_limit = 0; 

    /* SPDK env init (once) */
    static int spdk_inited = 0;
    if (!spdk_inited) {
        struct spdk_env_opts opts;
        spdk_env_opts_init(&opts);
        opts.name = "npu_nvme";
        if (spdk_env_init(&opts) < 0) {
            fprintf(stderr, "spdk_env_init failed\n");
            free(ctx);
            return -1;
        }
        spdk_inited = 1;
    }

    /* ACL init */
    /* 
    ctx->npu_device_id = npu_device_id;
    if (aclInit(NULL) != ACL_SUCCESS ||
        aclrtSetDevice(ctx->npu_device_id) != ACL_SUCCESS) {
        fprintf(stderr, "ACL init failed\n");
        free(ctx);
        return -1;
    }
    */
    aclrtSetDevice(ctx->npu_device_id);


    /* NVMe probe */
    struct spdk_nvme_transport_id trid;
    memset(&trid, 0, sizeof(trid));
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", nvme_pci_addr);

    if (spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, NULL) != 0 || !ctx->ctrlr) {
        fprintf(stderr, "nvme probe failed\n");
        aclrtResetDevice(ctx->npu_device_id);
        aclFinalize();
        free(ctx);
        return -1;
    }

    if (chunk_size == 0) {
        ctx->max_transfer = ctx->mdts_limit;
    } else {
        ctx->max_transfer = chunk_size;
        if (ctx->max_transfer > ctx->mdts_limit) {
            ctx->max_transfer = ctx->mdts_limit;
        }
    }

    /* qpair */
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, NULL, 0);
    if (!ctx->qpair) {
        fprintf(stderr, "alloc io qpair failed\n");
        aclrtResetDevice(ctx->npu_device_id);
        aclFinalize();
        free(ctx);
        return -1;
    }

    /* buffer pool = depth */
    ctx->pool_size = pipeline_depth; 
    ctx->pool = calloc(ctx->pool_size, sizeof(dma_buf_t));
    if (!ctx->pool) {
        fprintf(stderr, "buffer pool alloc failed\n");
        goto fail;
    }
    if (ring_init(&ctx->free_ring, ctx->pool_size) != 0) {
        fprintf(stderr, "ring init failed\n");
        goto fail;
    }
    for (int i = 0; i < ctx->pool_size; ++i) {
        //size_t sz = ALIGN_4K(ctx->max_transfer);
        size_t sz = ALIGN_4K(chunk_size);
        ctx->pool[i].buf = spdk_dma_zmalloc(sz, 4096, NULL);
        printf("[Init] Allocated DMA buf %d at %p, size=%zu\n", i, ctx->pool[i].buf, sz);
        ctx->pool[i].size = sz;
        if (!ctx->pool[i].buf) {
            fprintf(stderr, "dma buf alloc failed at %d\n", i);
            goto fail;
        }
        ring_push(&ctx->free_ring, i);
    }

    *pctx = ctx;
    ctx->max_transfer = chunk_size;
    return 0;

fail:
    if (ctx->pool) {
        for (int i = 0; i < ctx->pool_size; ++i) {
            if (ctx->pool[i].buf) spdk_dma_free(ctx->pool[i].buf);
        }
        free(ctx->pool);
    }
    ring_free(&ctx->free_ring);
    if (ctx->qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    if (ctx->ctrlr) spdk_nvme_detach(ctx->ctrlr);
    aclrtResetDevice(ctx->npu_device_id);
    aclFinalize();
    free(ctx);
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
int npu_nvme_write_batch(npu_nvme_context_t *ctx,
                         void **npu_ptrs,
                         uint64_t *nvme_offsets,
                         size_t *sizes,
                         int num_items) {
    if (!ctx || !npu_ptrs || !nvme_offsets || !sizes || num_items <= 0) return -1;

    int submitted = 0, completed = 0;
    int idx;
    int ret = 0;

    int *flags = calloc(num_items, sizeof(int));          // -1 error, 1 ok, 0 pending
    int *buf_idx = calloc(num_items, sizeof(int));        // 记录每个提交用的buffer
    bool *reclaimed = calloc(num_items, sizeof(bool));    // 是否已回收
    if (!flags || !buf_idx || !reclaimed) { ret = -1; goto cleanup; }

    while (completed < num_items) {
        /* Stage1: 尽量提交新的 I/O */
        while (submitted < num_items) {
            if (!ring_pop(&ctx->free_ring, &idx)) break; /* 无空闲 buffer，去 poll */

            size_t sz = sizes[submitted];
            if (sz == 0 || sz > ctx->max_transfer) {
                fprintf(stderr, "size invalid: %zu, max_transfer: %zu\n", sz, ctx->max_transfer);
                flags[submitted] = -1;
                reclaimed[submitted] = true; // 不占用 buffer
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

            /* NPU -> host */
            aclError acret = aclrtMemcpy(ctx->pool[idx].buf, aligned,
                                         npu_ptrs[submitted], sz,
                                         ACL_MEMCPY_DEVICE_TO_HOST);
            if (acret != ACL_SUCCESS) {
                fprintf(stderr, "aclrtMemcpy failed item %d\n", submitted);
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                submitted++;
                ring_push(&ctx->free_ring, idx);
                continue;
            }

            uint64_t lba = nvme_offsets[submitted] / ctx->block_size;
            uint32_t nblk = (uint32_t)(aligned / ctx->block_size);

            flags[submitted] = 0;
            buf_idx[submitted] = idx;
            int rc = spdk_nvme_ns_cmd_write(ctx->ns, ctx->qpair,
                                            ctx->pool[idx].buf,
                                            lba, nblk,
                                            io_complete, &flags[submitted], 0);
            if (rc != 0) {
                fprintf(stderr, "spdk_nvme_ns_cmd_write failed %d\n", rc);
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                ring_push(&ctx->free_ring, idx);
            }
            submitted++;
        }

        /* Stage2: poll completions */
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);

        /* Stage3: 回收已完成的 buffer */
        for (int i = 0; i < num_items; ++i) {
            if (!reclaimed[i] && flags[i] != 0) { // 已完成或出错且未回收
                if (flags[i] != 1) ret = -1; // 记录错误

                ring_push(&ctx->free_ring, buf_idx[i]);
                reclaimed[i] = true;
                completed++;
            }
        }

        /* 避免忙等 */
        if (submitted >= num_items && completed < num_items) {
            usleep(50);
        }
    }

cleanup:
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
    if (!flags || !buf_idx || !reclaimed) { ret = -1; goto out; }

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
            int rc = spdk_nvme_ns_cmd_read(ctx->ns, ctx->qpair,
                                           ctx->pool[idx].buf,
                                           lba, nblk,
                                           io_complete, &flags[submitted], 0);
            if (rc != 0) {
                fprintf(stderr, "spdk_nvme_ns_cmd_read failed %d\n", rc);
                flags[submitted] = -1;
                reclaimed[submitted] = true;
                ring_push(&ctx->free_ring, idx);
            }
            submitted++;
        }

        spdk_nvme_qpair_process_completions(ctx->qpair, 0);

        for (int i = 0; i < num_items; ++i) {
            if (!reclaimed[i] && flags[i] != 0) {
                /* host -> NPU */
                if (flags[i] == 1) {
                    size_t sz = sizes[i];
                    size_t aligned = ALIGN_4K(sz);
                    aclError acret = aclrtMemcpy(npu_ptrs[i], sz,
                                                 ctx->pool[buf_idx[i]].buf, sz,
                                                 ACL_MEMCPY_HOST_TO_DEVICE);
                    if (acret != ACL_SUCCESS) {
                        fprintf(stderr, "aclrtMemcpy H2D failed item %d\n", i);
                        ret = -1;
                    }
                } else {
                    ret = -1;
                }
                ring_push(&ctx->free_ring, buf_idx[i]);
                reclaimed[i] = true;
                completed++;
            }
        }

        if (submitted >= num_items && completed < num_items) {
            usleep(50);
        }
    }

out:
    free(flags);
    free(buf_idx);
    free(reclaimed);
    return ret;
}

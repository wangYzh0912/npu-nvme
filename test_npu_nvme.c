#include "npu_nvme.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>

#define DEFAULT_PCI_ADDR     "0000:83:00.0"
#define DEFAULT_PIPE_DEPTH   4
#define DEFAULT_CHUNK_SIZE   (4 * 1024 * 1024ULL) /* 4MB */

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + a - 1) & ~(a - 1);
}

int main(int argc, char **argv) {
    const char *nvme_addr = (argc > 1) ? argv[1] : DEFAULT_PCI_ADDR;
    int npu_device_id   = (argc > 2) ? atoi(argv[2]) : 0;
    int pipeline_depth    = (argc > 3) ? atoi(argv[3]) : DEFAULT_PIPE_DEPTH;
    size_t req_chunk_size = (argc > 4) ? strtoull(argv[4], NULL, 10) : DEFAULT_CHUNK_SIZE;
    bool enable_profile = false;

    printf("======================================\n");
    printf("NPU-NVMe Batch Test\n");
    printf("PCIe addr      : %s\n", nvme_addr);
    printf("pipeline depth : %d\n", pipeline_depth);
    printf("req chunk size : %zu bytes (%.2f MB)\n",
           req_chunk_size, req_chunk_size / 1024.0 / 1024.0);
    printf("======================================\n\n");

    /* Init */
    npu_nvme_context_t *ctx = NULL;
    if (npu_nvme_init(&ctx, nvme_addr, npu_device_id, pipeline_depth, req_chunk_size)) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }

    size_t max_xfer = npu_nvme_get_max_transfer(ctx);
    printf("[Init] max_transfer (effective) = %zu bytes (%.2f MB)\n\n",
           max_xfer, max_xfer / 1024.0 / 1024.0);

    /* 构造 3 个 chunk，覆盖大/中/小 */
    size_t sz0 = max_xfer;                  /* 约 4MB */
    size_t sz1 = max_xfer / 2;              /* 约 2MB */
    size_t sz2 = max_xfer / 3;              /* 约 1.3MB */
    size_t sizes[3] = { sz0, sz1, sz2 };

    /* NVMe 偏移对齐到 4K/block */
    uint64_t off0 = 0;
    uint64_t off1 = align_up(off0 + sz0, 4096);
    uint64_t off2 = align_up(off1 + sz1, 4096);
    uint64_t offsets[3] = { off0, off1, off2 };

    /* 计算总大小与对齐后的末尾 */
    uint64_t total_span = align_up(off2 + sz2, 4096);
    printf("[Plan] chunk sizes: %.2f / %.2f / %.2f MB\n",
           sz0 / 1024.0 / 1024.0,
           sz1 / 1024.0 / 1024.0,
           sz2 / 1024.0 / 1024.0);
    printf("[Plan] offsets   : %lu / %lu / %lu (bytes)\n",
           offsets[0], offsets[1], offsets[2]);
    printf("[Plan] total span: %.2f MB\n\n", total_span / 1024.0 / 1024.0);

    /* 分配 NPU 缓冲，大小为 total_span */
    void *npu_buf = NULL;
    size_t npu_alloc = (size_t)total_span;
    if (aclrtMalloc(&npu_buf, npu_alloc, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "Failed to allocate NPU buffer (%zu bytes)\n", npu_alloc);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    printf("[Alloc] NPU buffer: %p, size: %.2f MB\n",
           npu_buf, npu_alloc / 1024.0 / 1024.0);

    /* 准备 host 测试数据并拷到 NPU */
    uint8_t *host_init = (uint8_t *)malloc(npu_alloc);
    if (!host_init) {
        fprintf(stderr, "Failed to alloc host_init\n");
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    /* 为不同 chunk 填不同模式 */
    memset(host_init + off0, 0x11, sz0);
    memset(host_init + off1, 0x22, sz1);
    memset(host_init + off2, 0x33, sz2);

    if (aclrtMemcpy(npu_buf, npu_alloc, host_init, npu_alloc,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        fprintf(stderr, "Memcpy H2D failed\n");
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    printf("[Init] Patterns copied to NPU\n");

    /* 写测试：batch */
    void *write_ptrs[3] = {
        (uint8_t *)npu_buf + off0,
        (uint8_t *)npu_buf + off1,
        (uint8_t *)npu_buf + off2
    };

    printf("\n[Write] Starting batch write...\n");
    double t0 = now_ms();
    int rc = npu_nvme_write_batch(ctx, write_ptrs, offsets, sizes, 3);
    double t1 = now_ms();
    if (rc != 0) {
        fprintf(stderr, "[Write] Failed\n");
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    double write_ms = t1 - t0;
    double write_bw = (total_span / 1024.0 / 1024.0) / (write_ms / 1000.0);
    printf("[Write] Done: %.2f ms, %.2f MB/s\n", write_ms, write_bw);

    /* 清零 NPU buffer，准备读回验证 */
    memset(host_init, 0, npu_alloc);
    if (aclrtMemcpy(npu_buf, npu_alloc, host_init, npu_alloc,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        fprintf(stderr, "Clear NPU buffer failed\n");
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }

    /* 读测试：batch */
    void *read_ptrs[3] = {
        (uint8_t *)npu_buf + off0,
        (uint8_t *)npu_buf + off1,
        (uint8_t *)npu_buf + off2
    };

    printf("\n[Read] Starting batch read...\n");
    t0 = now_ms();
    rc = npu_nvme_read_batch(ctx, read_ptrs, offsets, sizes, 3);
    t1 = now_ms();
    if (rc != 0) {
        fprintf(stderr, "[Read] Failed\n");
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    double read_ms = t1 - t0;
    double read_bw = (total_span / 1024.0 / 1024.0) / (read_ms / 1000.0);
    printf("[Read] Done: %.2f ms, %.2f MB/s\n", read_ms, read_bw);

    /* 验证 */
    uint8_t *host_verify = (uint8_t *)malloc(npu_alloc);
    if (!host_verify) {
        fprintf(stderr, "Failed to alloc host_verify\n");
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    if (aclrtMemcpy(host_verify, npu_alloc, npu_buf, npu_alloc,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        fprintf(stderr, "Memcpy D2H failed\n");
        free(host_verify);
        free(host_init);
        aclrtFree(npu_buf);
        npu_nvme_cleanup(ctx);
        return 1;
    }

    size_t errs = 0;
    for (size_t i = 0; i < sz0; ++i) {
        if (host_verify[off0 + i] != 0x11) { errs++; break; }
    }
    for (size_t i = 0; i < sz1; ++i) {
        if (host_verify[off1 + i] != 0x22) { errs++; break; }
    }
    for (size_t i = 0; i < sz2; ++i) {
        if (host_verify[off2 + i] != 0x33) { errs++; break; }
    }

    if (errs == 0) {
        printf("\n[Verify] ✓ Data verification passed!\n");
    } else {
        printf("\n[Verify] ✗ Data verification failed (mismatches found)\n");
    }

    /* 清理 */
    free(host_verify);
    free(host_init);
    aclrtFree(npu_buf);
    npu_nvme_cleanup(ctx);

    printf("\n======================================\n");
    printf("Test completed\n");
    printf("======================================\n\n");

    return errs == 0 ? 0 : 1;
}

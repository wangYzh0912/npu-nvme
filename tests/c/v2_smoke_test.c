/* V2 smoke test: verify init + host write/read + cleanup work with reactor.
 * Build: make -C build v2_smoke_test && LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH sudo build/v2_smoke_test
 */
#include "npu_nvme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(int argc, char **argv) {
    NPUNVMEContext *ctx = NULL;
    const char *pci_addr = (argc > 1) ? argv[1] : "0000:83:00.0";
    int npu_id = (argc > 2) ? atoi(argv[2]) : 1;

    printf("[V2-smoke] calling npu_nvme_init...\n"); fflush(stdout);
    int rc = npu_nvme_init(&ctx, pci_addr, npu_id, 4, 4194304, false, ".");
    if (rc != 0 || !ctx) {
        fprintf(stderr, "[V2-smoke] FAIL: init returned %d\n", rc);
        return 1;
    }
    printf("[V2-smoke] init OK (ctx=%p)\n", (void*)ctx); fflush(stdout);

    /* Simple host write + read test */
    size_t buf_size = 1048576;  /* 1 MB */
    void *host_buf = malloc(buf_size);
    void *read_buf = malloc(buf_size);
    if (!host_buf || !read_buf) {
        fprintf(stderr, "[V2-smoke] FAIL: host allocation failed\n");
        npu_nvme_cleanup(ctx);
        free(host_buf);
        free(read_buf);
        return 1;
    }
    memset(host_buf, 0xAB, buf_size);

    void *ptrs[1] = { host_buf };
    /* Keep the smoke payload in the unallocated V2 gap, away from metadata,
     * FULL slots, and the tail Delta ring. */
    uint64_t offsets[1] = { 64ULL * 1024 * 1024 * 1024 };
    size_t sizes[1] = { buf_size };

    printf("[V2-smoke] writing %zu bytes...\n", buf_size); fflush(stdout);
    rc = npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1);
    printf("[V2-smoke] write -> %d\n", rc); fflush(stdout);
    if (rc != 0) {
        fprintf(stderr, "[V2-smoke] FAIL: host write returned %d\n", rc);
        npu_nvme_cleanup(ctx);
        free(host_buf);
        free(read_buf);
        return 1;
    }

    printf("[V2-smoke] reading back...\n"); fflush(stdout);
    memset(read_buf, 0, buf_size);
    ptrs[0] = read_buf;
    rc = npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, 1);
    printf("[V2-smoke] read -> %d\n", rc); fflush(stdout);
    if (rc != 0) {
        fprintf(stderr, "[V2-smoke] FAIL: host read returned %d\n", rc);
        npu_nvme_cleanup(ctx);
        free(host_buf);
        free(read_buf);
        return 1;
    }

    /* Verify */
    if (memcmp(host_buf, read_buf, buf_size) == 0) {
        printf("[V2-smoke] DATA MATCH ✓\n");
    } else {
        printf("[V2-smoke] DATA MISMATCH ✗\n");
        for (size_t i = 0; i < 64; i++) {
            printf("%02x ", ((unsigned char*)read_buf)[i]);
        }
        printf("\n");
        npu_nvme_cleanup(ctx);
        free(host_buf);
        free(read_buf);
        return 1;
    }

    printf("[V2-smoke] calling cleanup...\n"); fflush(stdout);
    npu_nvme_cleanup(ctx);

    free(host_buf);
    free(read_buf);
    printf("[V2-smoke] === PASS ===\n");
    return 0;
}

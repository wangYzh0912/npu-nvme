/* Reactor host I/O roundtrip test: verify init, host write/read, and cleanup.
 * Build: cmake --build build --target reactor_host_io_roundtrip_test
 * Run:   sudo env LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH build/reactor_host_io_roundtrip_test
 */
#include "npu_nvme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    NPUNVMEContext *ctx = NULL;
    const uint64_t scratch_offset = 1024ULL * 1024ULL * 1024ULL;  /* 1 GiB */

    printf("[reactor-host-io] calling npu_nvme_init...\n"); fflush(stdout);
    int rc = npu_nvme_init(&ctx, "0000:83:00.0", 1, 4, 4194304, false, ".");
    if (rc != 0 || !ctx) {
        fprintf(stderr, "[reactor-host-io] FAIL: init returned %d\n", rc);
        return 1;
    }
    printf("[reactor-host-io] init OK (ctx=%p)\n", (void*)ctx); fflush(stdout);

    /* Simple host write + read test */
    size_t buf_size = 1048576;  /* 1 MB */
    void *host_buf = malloc(buf_size);
    void *read_buf = malloc(buf_size);
    memset(host_buf, 0xAB, buf_size);

    void *ptrs[1] = { host_buf };
    uint64_t offsets[1] = { scratch_offset };
    size_t sizes[1] = { buf_size };

    printf("[reactor-host-io] writing %zu bytes...\n", buf_size); fflush(stdout);
    rc = npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1);
    printf("[reactor-host-io] write -> %d\n", rc); fflush(stdout);

    printf("[reactor-host-io] reading back...\n"); fflush(stdout);
    memset(read_buf, 0, buf_size);
    ptrs[0] = read_buf;
    rc = npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, 1);
    printf("[reactor-host-io] read -> %d\n", rc); fflush(stdout);

    /* Verify */
    if (memcmp(host_buf, read_buf, buf_size) == 0) {
        printf("[reactor-host-io] DATA MATCH\n");
    } else {
        printf("[reactor-host-io] DATA MISMATCH\n");
        for (size_t i = 0; i < 64; i++) {
            printf("%02x ", ((unsigned char*)read_buf)[i]);
        }
        printf("\n");
    }

    printf("[reactor-host-io] calling cleanup...\n"); fflush(stdout);
    npu_nvme_cleanup(ctx);

    free(host_buf);
    free(read_buf);
    printf("[reactor-host-io] === PASS ===\n");
    return 0;
}

/* V2 smoke test: verify init + host write/read + cleanup work with reactor.
 * Build: make -C build v2_smoke_test && LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH sudo build/v2_smoke_test
 */
#include "npu_nvme.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>

static int transfer(NPUNVMEContext *ctx, void *pointer, uint64_t offset, size_t size, int read) {
    NPUNVMETransferItem item={.address=pointer,.offset=offset,.length=size};
    NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,.operation=read,
        .memory_kind=NPU_NVME_MEMORY_HOST,.item_count=1,.items=&item};
    NPUNVMERequest *request=NULL;
    int rc=npu_nvme_submit_transfer(ctx,&spec,&request);
    if (rc) return rc;
    rc=npu_nvme_wait_request(request,0);
    if (rc==-ETIMEDOUT) {
        int done=0;
        while (!done) {npu_nvme_poll_request(request,&done);if (!done) usleep(1000);}
    }
    npu_nvme_release_request(request);return rc;
}

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
    rc = transfer(ctx,ptrs[0],offsets[0],sizes[0],0);
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
    rc = transfer(ctx,ptrs[0],offsets[0],sizes[0],1);
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

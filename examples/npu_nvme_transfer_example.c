/*
 * Example: use the npu_nvme_transfer C API to write host data to NVMe.
 *
 * Usage:
 *   ./npu_nvme_transfer_example <PCI_ADDR> [NPU_ID]
 *
 * Notes:
 * - This example focuses on the host write path (write_batch_host).
 * - NVMe access may require root privileges and proper hugepage/SPDK setup.
 * - For NPU read/write, allocate device memory via ACL and use the NPU batch APIs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "npu_nvme_transfer.h"

static int parse_int(const char *s, int default_value) {
    if (!s || !*s) {
        return default_value;
    }
    return (int)strtol(s, NULL, 10);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <PCI_ADDR> [NPU_ID]\n", argv[0]);
        return 1;
    }

    const char *pci_addr = argv[1];
    int npu_id = parse_int(argc > 2 ? argv[2] : NULL, 0);

    /* Tune these parameters based on workload and device capabilities. */
    const int pipe_depth = 2;
    const int chunk_size = 1 << 20; /* 1 MiB */

    NPUNVMEContext *ctx = NULL;
    int rc = npu_nvme_transfer_init(
        &ctx,
        pci_addr,
        npu_id,
        pipe_depth,
        chunk_size,
        /* enable_profiling */ 0,
        /* prof_dir */ NULL
    );
    if (rc != 0) {
        fprintf(stderr, "init failed: rc=%d\n", rc);
        return 2;
    }

    uint64_t total_blocks = npu_nvme_transfer_get_total_blocks(ctx);
    int max_transfer = npu_nvme_transfer_get_max_transfer(ctx);
    printf("total_blocks=%" PRIu64 ", max_transfer=%d bytes\n", total_blocks, max_transfer);

    /* Example: write a single host buffer to NVMe. */
    const size_t payload_size = 4096;
    if (max_transfer == 0) {
        fprintf(stderr, "[Warn] max_transfer is 0, continue without enforcing limit\n");
    }

    void *host_buf = NULL;
    if (posix_memalign(&host_buf, 4096, payload_size) != 0 || !host_buf) {
        fprintf(stderr, "failed to allocate host buffer\n");
        npu_nvme_transfer_cleanup(ctx);
        return 4;
    }
    memset(host_buf, 0xAB, payload_size);

    void *ptrs[1] = { host_buf };
    uint64_t offsets[1] = { 0 }; /* Byte offset into NVMe; align to device sector size. */
    size_t sizes[1] = { payload_size };

    rc = npu_nvme_transfer_write_batch_host(ctx, ptrs, offsets, sizes, 1);
    if (rc != 0) {
        fprintf(stderr, "write_batch_host failed: rc=%d\n", rc);
        free(host_buf);
        npu_nvme_transfer_cleanup(ctx);
        return 5;
    }
    printf("write_batch_host ok\n");

    /* Example: sync a small metadata block (read path). */
    unsigned char meta_buf[512] = {0};
    rc = npu_nvme_transfer_sync_meta_io(ctx, 0, sizeof(meta_buf), /* is_read */ 1, meta_buf);
    if (rc != 0) {
        fprintf(stderr, "sync_meta_io(read) failed: rc=%d\n", rc);
    } else {
        printf("sync_meta_io(read) ok\n");
    }

    /*
     * Optional: NPU read/write path (requires device memory).
     *
     * void *npu_ptrs[1] = { npu_device_ptr };
     * uint64_t npu_offsets[1] = { 0 };
     * size_t npu_sizes[1] = { payload_size };
     * npu_nvme_transfer_read_batch(ctx, npu_ptrs, npu_offsets, npu_sizes, 1);
     */

    free(host_buf);
    npu_nvme_transfer_cleanup(ctx);
    return 0;
}

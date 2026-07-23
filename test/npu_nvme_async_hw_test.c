/*
 * C hardware integration test for asynchronous request handles and
 * device-side async copy.
 *
 * Build: cmake --build build --target npu_nvme_async_hw_test
 * Run:   sudo env LD_LIBRARY_PATH=install/lib:$LD_LIBRARY_PATH \
 *        install/bin/npu_nvme_async_hw_test 0000:83:00.0 1
 */
#include "npu_nvme.h"
#include "internal/disk_layout.h"

#include <acl/acl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define DEFAULT_TEST_BYTES (64ULL * 1024ULL * 1024ULL)
#define DEFAULT_CHUNK_BYTES (4U * 1024U * 1024U)
#define DEFAULT_PIPE_DEPTH 4
#define SUBMIT_LATENCY_LIMIT_US 10000ULL

static uint64_t now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static void fill_pattern(uint8_t *buf, size_t size, uint32_t seed)
{
    for (size_t i = 0; i < size; i++) {
        buf[i] = (uint8_t)((i * 131U + seed) & 0xffU);
    }
}

static void busy_work_us(uint64_t duration_us)
{
    volatile uint64_t acc = 0;
    uint64_t start = now_us();
    while (now_us() - start < duration_us) {
        acc += now_us() & 0xffU;
    }
    (void)acc;
}

static int wait_request(const char *name, npu_nvme_request_t *req,
                        uint64_t *elapsed_us)
{
    uint64_t start = now_us();
    int rc = npu_nvme_request_wait(req, 0);
    *elapsed_us = now_us() - start;
    if (rc != 0) {
        fprintf(stderr, "[async-hw] %s wait failed: rc=%d result=%d\n",
                name, rc, npu_nvme_request_result(req));
        return -1;
    }
    if (npu_nvme_request_poll(req) != 1) {
        fprintf(stderr, "[async-hw] %s poll did not report success after wait\n",
                name);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *pci_addr = argc > 1 ? argv[1] : "0000:83:00.0";
    int npu_id = argc > 2 ? atoi(argv[2]) : 1;
    size_t test_bytes = argc > 3 ? strtoull(argv[3], NULL, 0) : DEFAULT_TEST_BYTES;

    NPUNVMEContext *ctx = NULL;
    void *dev_src = NULL;
    void *dev_dst = NULL;
    void *host_src = NULL;
    void *host_dst = NULL;
    int exit_code = 1;

    if (test_bytes == 0 || test_bytes % 4096U != 0) {
        fprintf(stderr, "[async-hw] test_bytes must be non-zero and 4 KiB aligned\n");
        return 1;
    }

    int rc = npu_nvme_init(&ctx, pci_addr, npu_id, DEFAULT_PIPE_DEPTH,
                           DEFAULT_CHUNK_BYTES, false, ".");
    if (rc != 0 || !ctx) {
        fprintf(stderr, "[async-hw] npu_nvme_init failed: rc=%d\n", rc);
        return 1;
    }

    uint64_t capacity = npu_nvme_get_total_blocks(ctx);
    uint64_t scratch_offset = NPU_NVME_RAW_IO_START_OFFSET + 1024ULL * 1024ULL * 1024ULL;
    if (scratch_offset + 2ULL * test_bytes > capacity) {
        fprintf(stderr, "[async-hw] insufficient capacity for scratch test range\n");
        goto out;
    }

    if (aclrtSetDevice(npu_id) != ACL_SUCCESS) {
        fprintf(stderr, "[async-hw] aclrtSetDevice failed\n");
        goto out;
    }
    if (aclrtMalloc(&dev_src, test_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dev_dst, test_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "[async-hw] aclrtMalloc failed\n");
        goto out;
    }
    if (aclrtMallocHost(&host_src, test_bytes) != ACL_SUCCESS ||
        aclrtMallocHost(&host_dst, test_bytes) != ACL_SUCCESS) {
        fprintf(stderr, "[async-hw] aclrtMallocHost failed\n");
        goto out;
    }

    fill_pattern((uint8_t *)host_src, test_bytes, 0x37U);
    memset(host_dst, 0, test_bytes);
    if (aclrtMemcpy(dev_src, test_bytes, host_src, test_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        fprintf(stderr, "[async-hw] initial H2D failed\n");
        goto out;
    }

    int num_items = (int)((test_bytes + DEFAULT_CHUNK_BYTES - 1U) / DEFAULT_CHUNK_BYTES);
    void **write_ptrs = calloc((size_t)num_items, sizeof(void *));
    void **read_ptrs = calloc((size_t)num_items, sizeof(void *));
    uint64_t *offsets = calloc((size_t)num_items, sizeof(uint64_t));
    uint64_t *offsets_overlap = calloc((size_t)num_items, sizeof(uint64_t));
    size_t *sizes = calloc((size_t)num_items, sizeof(size_t));
    if (!write_ptrs || !read_ptrs || !offsets || !offsets_overlap || !sizes) {
        fprintf(stderr, "[async-hw] chunk array allocation failed\n");
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    for (int i = 0; i < num_items; i++) {
        size_t offset = (size_t)i * DEFAULT_CHUNK_BYTES;
        size_t remaining = test_bytes - offset;
        size_t chunk_size = remaining < DEFAULT_CHUNK_BYTES ? remaining : DEFAULT_CHUNK_BYTES;
        write_ptrs[i] = (uint8_t *)dev_src + offset;
        read_ptrs[i] = (uint8_t *)dev_dst + offset;
        offsets[i] = scratch_offset + offset;
        offsets_overlap[i] = scratch_offset + test_bytes + offset;
        sizes[i] = chunk_size;
    }

    npu_nvme_request_t *write_req = NULL;
    uint64_t submit_start = now_us();
    rc = npu_nvme_write_batch_async(ctx, write_ptrs, offsets, sizes,
                                    num_items, &write_req);
    uint64_t write_submit_us = now_us() - submit_start;
    if (rc != 0 || !write_req) {
        fprintf(stderr, "[async-hw] write async submit failed: rc=%d\n", rc);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    if (write_submit_us > SUBMIT_LATENCY_LIMIT_US) {
        fprintf(stderr, "[async-hw] write submit too slow: %lu us\n", write_submit_us);
        npu_nvme_request_wait(write_req, 0);
        npu_nvme_request_free(write_req);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }

    uint64_t write_wait_us = 0;
    if (wait_request("write", write_req, &write_wait_us) != 0) {
        npu_nvme_request_free(write_req);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    npu_nvme_request_free(write_req);

    npu_nvme_request_t *read_req = NULL;
    submit_start = now_us();
    rc = npu_nvme_read_batch_async(ctx, read_ptrs, offsets, sizes,
                                   num_items, &read_req);
    uint64_t read_submit_us = now_us() - submit_start;
    if (rc != 0 || !read_req) {
        fprintf(stderr, "[async-hw] read async submit failed: rc=%d\n", rc);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    if (read_submit_us > SUBMIT_LATENCY_LIMIT_US) {
        fprintf(stderr, "[async-hw] read submit too slow: %lu us\n", read_submit_us);
        npu_nvme_request_wait(read_req, 0);
        npu_nvme_request_free(read_req);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }

    uint64_t read_wait_us = 0;
    if (wait_request("read", read_req, &read_wait_us) != 0) {
        npu_nvme_request_free(read_req);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    npu_nvme_request_free(read_req);

    if (aclrtMemcpy(host_dst, test_bytes, dev_dst, test_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        fprintf(stderr, "[async-hw] final D2H failed\n");
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    if (memcmp(host_src, host_dst, test_bytes) != 0) {
        fprintf(stderr, "[async-hw] roundtrip data mismatch\n");
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }

    npu_nvme_request_t *overlap_req = NULL;
    uint64_t overlap_start = now_us();
    rc = npu_nvme_write_batch_async(ctx, write_ptrs, offsets_overlap, sizes,
                                    num_items, &overlap_req);
    if (rc != 0 || !overlap_req) {
        fprintf(stderr, "[async-hw] overlap write submit failed: rc=%d\n", rc);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    uint64_t busy_us = write_wait_us / 2U;
    if (busy_us < 1000U) busy_us = 1000U;
    busy_work_us(busy_us);
    uint64_t overlap_wait_us = 0;
    if (wait_request("overlap-write", overlap_req, &overlap_wait_us) != 0) {
        npu_nvme_request_free(overlap_req);
        free(write_ptrs); free(read_ptrs); free(offsets); free(offsets_overlap); free(sizes);
        goto out;
    }
    uint64_t overlap_total_us = now_us() - overlap_start;
    npu_nvme_request_free(overlap_req);

    double mib = (double)test_bytes / 1024.0 / 1024.0;
    printf("[async-hw] bytes=%zu chunks=%d\n", test_bytes, num_items);
    printf("[async-hw] write submit=%lu us wait=%lu us bw=%.2f MiB/s\n",
           write_submit_us, write_wait_us, mib * 1000000.0 / (double)write_wait_us);
    printf("[async-hw] read submit=%lu us wait=%lu us bw=%.2f MiB/s\n",
           read_submit_us, read_wait_us, mib * 1000000.0 / (double)read_wait_us);
    printf("[async-hw] overlap diagnostic: busy=%lu us total=%lu us post-busy-wait=%lu us\n",
           busy_us, overlap_total_us, overlap_wait_us);
    printf("[async-hw] === PASS ===\n");

    free(write_ptrs);
    free(read_ptrs);
    free(offsets);
    free(offsets_overlap);
    free(sizes);
    exit_code = 0;

out:
    if (host_src) aclrtFreeHost(host_src);
    if (host_dst) aclrtFreeHost(host_dst);
    if (dev_src) aclrtFree(dev_src);
    if (dev_dst) aclrtFree(dev_dst);
    if (ctx) npu_nvme_cleanup(ctx);
    return exit_code;
}

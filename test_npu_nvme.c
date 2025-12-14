/**
 * @file test_npu_nvme.c
 * @brief Simple test for NPU-NVMe API
 */

#include "npu_nvme.h"
#include <acl/acl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define TEST_SIZE (4 * 1024 * 1024)  // 4MB
#define CHUNK_SIZE (4 * 1024 * 1024) // 4MB
#define NVME_OFFSET 0

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    const char *nvme_addr = "0000:83:00.0";
    if (argc > 1) {
        nvme_addr = argv[1];
    }
    
    printf("======================================\n");
    printf("NPU-NVMe Simple Test\n");
    printf("======================================\n\n");
    
    // Initialize
    npu_nvme_context_t *ctx = NULL;
    int rc = npu_nvme_init(&ctx, nvme_addr);
    if (rc != 0) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }
    
    // Allocate NPU buffer
    void *npu_buffer = NULL;
    aclError ret = aclrtMalloc(&npu_buffer, TEST_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "Failed to allocate NPU buffer\n");
        npu_nvme_cleanup(ctx);
        return 1;
    }
    printf("[Test] Allocated NPU buffer: %p, size: %d MB\n", 
           npu_buffer, TEST_SIZE / 1024 / 1024);
    
    // Prepare test data on host
    uint32_t *host_data = (uint32_t *)malloc(TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE / sizeof(uint32_t); i++) {
        host_data[i] = 0x12345678 + i;
    }
    
    // Copy test data to NPU
    ret = aclrtMemcpy(npu_buffer, TEST_SIZE, host_data, TEST_SIZE, 
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "Failed to copy data to NPU\n");
        free(host_data);
        aclrtFree(npu_buffer);
        npu_nvme_cleanup(ctx);
        return 1;
    }
    printf("[Test] Initialized test data on NPU\n");
    
    // Test Write
    printf("\n[Test] Writing %d MB from NPU to NVMe...\n", TEST_SIZE / 1024 / 1024);
    double start = get_time_ms();
    rc = npu_nvme_write(ctx, npu_buffer, NVME_OFFSET, TEST_SIZE, CHUNK_SIZE);
    double write_time = get_time_ms() - start;
    
    if (rc != 0) {
        fprintf(stderr, "Write failed\n");
    } else {
        double bw = (TEST_SIZE / 1024.0 / 1024.0) / (write_time / 1000.0);
        printf("[Test] Write completed: %.2f ms, %.2f MB/s\n", write_time, bw);
    }
    
    // Clear NPU buffer
    memset(host_data, 0, TEST_SIZE);
    aclrtMemcpy(npu_buffer, TEST_SIZE, host_data, TEST_SIZE, 
                ACL_MEMCPY_HOST_TO_DEVICE);
    
    // Test Read
    printf("\n[Test] Reading %d MB from NVMe to NPU...\n", TEST_SIZE / 1024 / 1024);
    start = get_time_ms();
    rc = npu_nvme_read(ctx, npu_buffer, NVME_OFFSET, TEST_SIZE, CHUNK_SIZE);
    double read_time = get_time_ms() - start;
    
    if (rc != 0) {
        fprintf(stderr, "Read failed\n");
    } else {
        double bw = (TEST_SIZE / 1024.0 / 1024.0) / (read_time / 1000.0);
        printf("[Test] Read completed: %.2f ms, %.2f MB/s\n", read_time, bw);
    }
    
    // Verify data
    printf("\n[Test] Verifying data...\n");
    uint32_t *verify_data = (uint32_t *)malloc(TEST_SIZE);
    aclrtMemcpy(verify_data, TEST_SIZE, npu_buffer, TEST_SIZE,
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    int errors = 0;
    for (size_t i = 0; i < TEST_SIZE / sizeof(uint32_t) && errors < 10; i++) {
        uint32_t expected = 0x12345678 + i;
        if (verify_data[i] != expected) {
            fprintf(stderr, "  Mismatch at [%zu]: expected 0x%08x, got 0x%08x\n",
                    i, expected, verify_data[i]);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("[Test] ✓ Data verification passed!\n");
    } else {
        printf("[Test] ✗ Data verification failed (%d errors)\n", errors);
    }
    
    // Cleanup
    free(verify_data);
    free(host_data);
    aclrtFree(npu_buffer);
    npu_nvme_cleanup(ctx);
    
    printf("\n======================================\n");
    printf("Test completed\n");
    printf("======================================\n\n");
    
    return errors == 0 ? 0 : 1;
}
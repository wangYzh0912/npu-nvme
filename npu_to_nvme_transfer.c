/*
 * NPU to NVMe Transfer Test - Single Buffer Strategy
 * 
 * Test if ACL DMA can work with SPDK-allocated hugepage memory
 * If successful, this enables true zero-copy NPU <-> NVMe transfer
 */

#include "spdk/stdinc.h"
#include "spdk/nvme.h"
#include "spdk_internal/nvme_util.h"
#include "spdk/vmd.h"
#include "spdk/nvme_zns.h"
#include "spdk/env.h"
#include "spdk/string.h"
#include "spdk/log.h"
#include <acl/acl.h>

// ==================================================
// Configuration
// ==================================================
#define TEST_DATA_SIZE (4096)
#define NUM_BLOCKS 1
#define STARTING_LBA 0

// ==================================================
// Data Structures
// ==================================================

struct ctrlr_entry {
    struct spdk_nvme_ctrlr      *ctrlr;
    TAILQ_ENTRY(ctrlr_entry)    link;
    char                        name[1024];
};

struct ns_entry {
    struct spdk_nvme_ctrlr  *ctrlr;
    struct spdk_nvme_ns     *ns;
    TAILQ_ENTRY(ns_entry)   link;
    struct spdk_nvme_qpair  *qpair;
};

typedef struct {
    struct ns_entry *ns_entry;
    void            *npu_buffer;        // NPU device memory
    void            *host_buffer;       // SPDK-allocated host memory (shared for both ACL and NVMe)
    size_t          buffer_size;
    uint64_t        lba_start;
    uint32_t        lba_count;
    int             is_completed;
    int             error_occurred;
    bool            initialized;
    bool            acl_compatible;     // ACL DMA works with SPDK memory? 
} npu_nvme_context_t;

// Global variables
static TAILQ_HEAD(, ctrlr_entry) g_controllers = TAILQ_HEAD_INITIALIZER(g_controllers);
static TAILQ_HEAD(, ns_entry) g_namespaces = TAILQ_HEAD_INITIALIZER(g_namespaces);
static struct spdk_nvme_transport_id g_trid = {};
static bool g_vmd = false;

// Forward declarations
int npu_nvme_init(npu_nvme_context_t **ctx, size_t data_size, 
                  uint64_t lba_start, uint32_t lba_count);
int npu_nvme_write(npu_nvme_context_t *ctx, bool verify);
int npu_nvme_read(npu_nvme_context_t *ctx, bool verify);
void npu_nvme_cleanup(npu_nvme_context_t *ctx);
int npu_nvme_init_data(npu_nvme_context_t *ctx, int pattern, uint32_t seed);

// ==================================================
// Helper Functions
// ==================================================

static aclError checkAclError(aclError ret, const char* funcName) {
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ACL Error] %s failed with error code: %d\n", funcName, ret);
    }
    return ret;
}

static void* allocateNPUMemory(size_t size) {
    void* d_ptr = NULL;
    aclError ret = aclrtMalloc(&d_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    checkAclError(ret, "aclrtMalloc");
    
    if (ret == ACL_SUCCESS && d_ptr != NULL) {
        printf("[NPU] Allocated %zu bytes at %p\n", size, d_ptr);
    } else {
        fprintf(stderr, "[NPU] Memory allocation failed\n");
    }
    return d_ptr;
}

static void* allocateSPDKHostMemory(size_t size) {
    // Use SPDK's DMA-capable memory allocator
    // This memory is backed by hugepages and registered with DPDK
    void* ptr = spdk_dma_zmalloc(size, 4096, NULL);
    
    if (ptr != NULL) {
        printf("[SPDK Host] Allocated %zu bytes at %p\n", size, ptr);
        
        // Verify physical address translation works
        uint64_t phys_addr = spdk_vtophys(ptr, NULL);
        if (phys_addr != SPDK_VTOPHYS_ERROR) {
            printf("[SPDK Host] Physical address: 0x%lx\n", phys_addr);
            printf("[SPDK Host] Memory is DMA-capable (backed by hugepages)\n");
        } else {
            fprintf(stderr, "[SPDK Host] Warning: vtophys failed\n");
        }
    } else {
        fprintf(stderr, "[SPDK Host] Memory allocation failed\n");
    }
    
    return ptr;
}

static void freeNPUMemory(void* d_ptr) {
    if (d_ptr != NULL) {
        aclrtFree(d_ptr);
        printf("[NPU] Freed memory at %p\n", d_ptr);
    }
}

static void freeSPDKHostMemory(void* ptr) {
    if (ptr != NULL) {
        spdk_dma_free(ptr);
        printf("[SPDK Host] Freed memory at %p\n", ptr);
    }
}

/**
 * @brief Test if ACL DMA can access SPDK-allocated memory
 * 
 * This is the key test: can aclrtMemcpy work with SPDK hugepage memory? 
 */
static bool test_acl_spdk_compatibility(void *npu_buf, void *spdk_buf, size_t size) {
    printf("\n========================================\n");
    printf("Testing ACL <-> SPDK Memory Compatibility\n");
    printf("========================================\n");
    
    // Prepare test pattern
    uint32_t *test_data = (uint32_t *)spdk_buf;
    size_t num_words = size / sizeof(uint32_t);
    
    printf("[Test] Initializing SPDK buffer with test pattern...\n");
    for (size_t i = 0; i < num_words; i++) {
        test_data[i] = 0xDEADBEEF + i;
    }
    printf("[Test] First value in SPDK buffer: 0x%08x\n", test_data[0]);
    
    // Test 1: Host -> NPU (using SPDK buffer)
    printf("\n[Test 1/2] SPDK Host -> NPU transfer...\n");
    aclError ret = aclrtMemcpy(npu_buf, size, spdk_buf, size, ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError(ret, "aclrtMemcpy (SPDK Host -> NPU)");
    
    if (ret != ACL_SUCCESS) {
        printf("[Test 1/2] ✗ FAILED: ACL cannot copy from SPDK memory to NPU\n");
        return false;
    }
    printf("[Test 1/2] ✓ SUCCESS: Host -> NPU transfer completed\n");
    
    // Clear SPDK buffer
    memset(spdk_buf, 0, size);
    printf("[Test] Cleared SPDK buffer (first value now: 0x%08x)\n", test_data[0]);
    
    // Test 2: NPU -> Host (to SPDK buffer)
    printf("\n[Test 2/2] NPU -> SPDK Host transfer...\n");
    ret = aclrtMemcpy(spdk_buf, size, npu_buf, size, ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError(ret, "aclrtMemcpy (NPU -> SPDK Host)");
    
    if (ret != ACL_SUCCESS) {
        printf("[Test 2/2] ✗ FAILED: ACL cannot copy from NPU to SPDK memory\n");
        return false;
    }
    printf("[Test 2/2] ✓ SUCCESS: NPU -> Host transfer completed\n");
    
    // Verify data integrity
    printf("\n[Verification] Checking data integrity...\n");
    bool data_correct = true;
    for (size_t i = 0; i < num_words && i < 10; i++) {
        uint32_t expected = 0xDEADBEEF + i;
        if (test_data[i] != expected) {
            fprintf(stderr, "[Verification] ✗ Mismatch at [%zu]: expected 0x%08x, got 0x%08x\n",
                    i, expected, test_data[i]);
            data_correct = false;
        } else {
            printf("[Verification] [%zu] = 0x%08x ✓\n", i, test_data[i]);
        }
    }
    
    if (data_correct) {
        printf("\n========================================\n");
        printf("✓ COMPATIBILITY TEST PASSED!\n");
        printf("ACL DMA works with SPDK hugepage memory\n");
        printf("Zero-copy NPU <-> NVMe transfer is possible!\n");
        printf("========================================\n\n");
        return true;
    } else {
        printf("\n========================================\n");
        printf("✗ COMPATIBILITY TEST FAILED\n");
        printf("Data corruption detected\n");
        printf("========================================\n\n");
        return false;
    }
}

// ==================================================
// SPDK Callbacks
// ==================================================

static void read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    npu_nvme_context_t *ctx = (npu_nvme_context_t *)arg;
    
    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(ctx->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Read error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        ctx->error_occurred = 1;
    } else {
        printf("[NVMe] Read completed successfully\n");
    }
    ctx->is_completed = 1;
}

static void write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    npu_nvme_context_t *ctx = (npu_nvme_context_t *)arg;
    
    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(ctx->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Write error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        ctx->error_occurred = 1;
    } else {
        printf("[NVMe] Write completed successfully\n");
    }
    ctx->is_completed = 1;
}

static void reset_zone_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    npu_nvme_context_t *ctx = (npu_nvme_context_t *)arg;
    
    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(ctx->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Reset zone error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        ctx->error_occurred = 1;
    }
    ctx->is_completed = 1;
}

// ==================================================
// SPDK Helper Functions
// ==================================================

static void register_ns(struct spdk_nvme_ctrlr *ctrlr, struct spdk_nvme_ns *ns) {
    struct ns_entry *entry;
    
    if (! spdk_nvme_ns_is_active(ns)) {
        return;
    }
    
    entry = (struct ns_entry *)malloc(sizeof(struct ns_entry));
    if (entry == NULL) {
        perror("ns_entry malloc");
        exit(1);
    }
    
    entry->ctrlr = ctrlr;
    entry->ns = ns;
    entry->qpair = NULL;
    TAILQ_INSERT_TAIL(&g_namespaces, entry, link);
    
    printf("[NVMe] Registered namespace ID: %d, Size: %ju GB\n", 
           spdk_nvme_ns_get_id(ns),
           spdk_nvme_ns_get_size(ns) / 1000000000);
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    printf("[NVMe] Probing controller at %s\n", trid->traddr);
    return true;
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr, 
                      const struct spdk_nvme_ctrlr_opts *opts) {
    int nsid;
    struct ctrlr_entry *entry;
    struct spdk_nvme_ns *ns;
    const struct spdk_nvme_ctrlr_data *cdata;
    
    entry = (struct ctrlr_entry *)malloc(sizeof(struct ctrlr_entry));
    if (entry == NULL) {
        perror("ctrlr_entry malloc");
        exit(1);
    }
    
    printf("[NVMe] Attached to controller at %s\n", trid->traddr);
    
    cdata = spdk_nvme_ctrlr_get_data(ctrlr);
    snprintf(entry->name, sizeof(entry->name), "%-20. 20s (%-20.20s)", 
             cdata->mn, cdata->sn);
    
    entry->ctrlr = ctrlr;
    TAILQ_INSERT_TAIL(&g_controllers, entry, link);
    
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == NULL) {
            continue;
        }
        register_ns(ctrlr, ns);
    }
}

static void cleanup_nvme(void) {
    struct ns_entry *ns_entry, *tmp_ns_entry;
    struct ctrlr_entry *ctrlr_entry, *tmp_ctrlr_entry;
    struct spdk_nvme_detach_ctx *detach_ctx = NULL;
    
    TAILQ_FOREACH_SAFE(ns_entry, &g_namespaces, link, tmp_ns_entry) {
        TAILQ_REMOVE(&g_namespaces, ns_entry, link);
        free(ns_entry);
    }
    
    TAILQ_FOREACH_SAFE(ctrlr_entry, &g_controllers, link, tmp_ctrlr_entry) {
        TAILQ_REMOVE(&g_controllers, ctrlr_entry, link);
        spdk_nvme_detach_async(ctrlr_entry->ctrlr, &detach_ctx);
        free(ctrlr_entry);
    }
    
    if (detach_ctx) {
        spdk_nvme_detach_poll(detach_ctx);
    }
}

// ==================================================
// Core API Functions
// ==================================================

/**
 * @brief Initialize with single-buffer strategy
 * Tests ACL compatibility with SPDK memory
 */
int npu_nvme_init(npu_nvme_context_t **ctx, size_t data_size, 
                  uint64_t lba_start, uint32_t lba_count) {
    int rc;
    aclError acl_ret;
    struct ns_entry *ns_entry;
    
    printf("\n========================================\n");
    printf("Initializing NPU-NVMe Environment\n");
    printf("Testing Single-Buffer Strategy\n");
    printf("========================================\n");
    
    *ctx = (npu_nvme_context_t *)calloc(1, sizeof(npu_nvme_context_t));
    if (*ctx == NULL) {
        fprintf(stderr, "[Error] Failed to allocate context\n");
        return -1;
    }
    
    (*ctx)->buffer_size = data_size;
    (*ctx)->lba_start = lba_start;
    (*ctx)->lba_count = lba_count;
    (*ctx)->initialized = false;
    (*ctx)->acl_compatible = false;
    
    // 1. Initialize ACL
    printf("\n[Step 1/5] Initializing ACL...\n");
    acl_ret = aclInit(NULL);
    checkAclError(acl_ret, "aclInit");
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] ACL initialization failed\n");
        free(*ctx);
        return -1;
    }
    
    acl_ret = aclrtSetDevice(0);
    checkAclError(acl_ret, "aclrtSetDevice");
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] Failed to set NPU device\n");
        aclFinalize();
        free(*ctx);
        return -1;
    }
    printf("[ACL] Using NPU device 0\n");
    
    // 2. Probe NVMe
    printf("\n[Step 2/5] Probing NVMe devices...\n");
    rc = spdk_nvme_probe(&g_trid, NULL, probe_cb, attach_cb, NULL);
    if (rc != 0 || TAILQ_EMPTY(&g_namespaces)) {
        fprintf(stderr, "[Error] No NVMe namespace found\n");
        aclrtResetDevice(0);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    ns_entry = TAILQ_FIRST(&g_namespaces);
    (*ctx)->ns_entry = ns_entry;
    
    // 3. Allocate IO queue pair
    printf("\n[Step 3/5] Allocating NVMe IO queue pair...\n");
    ns_entry->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ns_entry->ctrlr, NULL, 0);
    if (ns_entry->qpair == NULL) {
        fprintf(stderr, "[Error] Failed to allocate IO qpair\n");
        aclrtResetDevice(0);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    printf("[NVMe] IO queue pair allocated\n");
    
    // 4. Allocate NPU memory
    printf("\n[Step 4/5] Allocating NPU memory...\n");
    (*ctx)->npu_buffer = allocateNPUMemory(data_size);
    if ((*ctx)->npu_buffer == NULL) {
        fprintf(stderr, "[Error] NPU memory allocation failed\n");
        spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
        aclrtResetDevice(0);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    // 5. Allocate SPDK host memory (the key test!)
    printf("\n[Step 5/5] Allocating SPDK host memory (hugepages)...\n");
    (*ctx)->host_buffer = allocateSPDKHostMemory(data_size);
    if ((*ctx)->host_buffer == NULL) {
        fprintf(stderr, "[Error] SPDK host memory allocation failed\n");
        freeNPUMemory((*ctx)->npu_buffer);
        spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
        aclrtResetDevice(0);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    // 6. TEST: Can ACL DMA work with SPDK memory?
    (*ctx)->acl_compatible = test_acl_spdk_compatibility(
        (*ctx)->npu_buffer, 
        (*ctx)->host_buffer, 
        data_size
    );
    
    if (!(*ctx)->acl_compatible) {
        fprintf(stderr, "\n[Error] ACL is NOT compatible with SPDK memory!\n");
        fprintf(stderr, "[Error] This architecture requires dual-buffer strategy\n");
        freeSPDKHostMemory((*ctx)->host_buffer);
        freeNPUMemory((*ctx)->npu_buffer);
        spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
        aclrtResetDevice(0);
        aclFinalize();
        free(*ctx);
        return -1;
    }
    
    // 7. Reset zone if ZNS
    if (spdk_nvme_ns_get_csi(ns_entry->ns) == SPDK_NVME_CSI_ZNS) {
        printf("\n[ZNS] Resetting zone at LBA %lu.. .\n", lba_start);
        (*ctx)->is_completed = 0;
        (*ctx)->error_occurred = 0;
        
        rc = spdk_nvme_zns_reset_zone(ns_entry->ns, ns_entry->qpair,
                                      lba_start, false,
                                      reset_zone_complete_cb, *ctx);
        if (rc != 0) {
            fprintf(stderr, "[Error] Zone reset submission failed\n");
            freeSPDKHostMemory((*ctx)->host_buffer);
            freeNPUMemory((*ctx)->npu_buffer);
            spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
            aclrtResetDevice(0);
            aclFinalize();
            free(*ctx);
            return -1;
        }
        
        while (!(*ctx)->is_completed) {
            spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
        }
        
        if ((*ctx)->error_occurred) {
            freeSPDKHostMemory((*ctx)->host_buffer);
            freeNPUMemory((*ctx)->npu_buffer);
            spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
            aclrtResetDevice(0);
            aclFinalize();
            free(*ctx);
            return -1;
        }
    }
    
    (*ctx)->initialized = true;
    
    printf("\n========================================\n");
    printf("Initialization Complete!\n");
    printf("========================================\n");
    printf("  Architecture: ZERO-COPY (Single Buffer)\n");
    printf("  NPU Buffer:   %p (%zu bytes)\n", (*ctx)->npu_buffer, data_size);
    printf("  Host Buffer:  %p (%zu bytes)\n", (*ctx)->host_buffer, data_size);
    printf("  Buffer Type:  SPDK hugepage (ACL-compatible)\n");
    printf("  LBA Range:    %lu - %lu\n", lba_start, lba_start + lba_count - 1);
    printf("========================================\n\n");
    
    return 0;
}

/**
 * @brief Zero-copy write: NPU -> SPDK buffer -> NVMe
 */
int npu_nvme_write(npu_nvme_context_t *ctx, bool verify) {
    int rc;
    
    if (!ctx || !ctx->initialized || !ctx->acl_compatible) {
        fprintf(stderr, "[Error] Context not properly initialized\n");
        return -1;
    }
    
    printf("\n========================================\n");
    printf("NPU -> NVMe Write (ZERO-COPY)\n");
    printf("========================================\n");
    
    // Single transfer: NPU -> SPDK host buffer
    printf("\n[Step 1/2] Transferring data from NPU to SPDK host buffer...\n");
    aclError ret = aclrtMemcpy(ctx->host_buffer, ctx->buffer_size,
                               ctx->npu_buffer, ctx->buffer_size,
                               ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError(ret, "aclrtMemcpy (NPU -> SPDK Host)");
    
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] NPU to Host transfer failed\n");
        return -1;
    }
    printf("[Transfer] NPU -> SPDK Host: %zu bytes\n", ctx->buffer_size);
    
    // Write to NVMe (buffer is already DMA-capable)
    printf("\n[Step 2/2] Writing SPDK buffer to NVMe (LBA %lu)...\n", ctx->lba_start);
    
    ctx->is_completed = 0;
    ctx->error_occurred = 0;
    
    rc = spdk_nvme_ns_cmd_write(ctx->ns_entry->ns, ctx->ns_entry->qpair,
                                ctx->host_buffer,
                                ctx->lba_start, ctx->lba_count,
                                write_complete_cb, ctx, 0);
    if (rc != 0) {
        fprintf(stderr, "[Error] Write submission failed\n");
        return -1;
    }
    
    while (!ctx->is_completed) {
        spdk_nvme_qpair_process_completions(ctx->ns_entry->qpair, 0);
    }
    
    if (ctx->error_occurred) {
        fprintf(stderr, "[Error] Write operation failed\n");
        return -1;
    }
    
    printf("\n========================================\n");
    printf("Write Completed Successfully!  (ZERO-COPY)\n");
    printf("  Data path: NPU -> SPDK hugepage -> NVMe\n");
    printf("  No intermediate memcpy required!\n");
    printf("  Wrote %zu bytes to LBA %lu\n", ctx->buffer_size, ctx->lba_start);
    printf("========================================\n\n");
    
    if (verify) {
        return npu_nvme_read(ctx, true);
    }
    
    return 0;
}

/**
 * @brief Zero-copy read: NVMe -> SPDK buffer -> NPU
 */
int npu_nvme_read(npu_nvme_context_t *ctx, bool verify) {
    int rc;
    void *verify_buffer = NULL;
    bool data_match = true;
    
    if (!ctx || !ctx->initialized || !ctx->acl_compatible) {
        fprintf(stderr, "[Error] Context not properly initialized\n");
        return -1;
    }
    
    printf("\n========================================\n");
    printf("NVMe -> NPU Read (ZERO-COPY)\n");
    printf("========================================\n");
    
    // Save data for verification
    if (verify) {
        verify_buffer = malloc(ctx->buffer_size);
        if (verify_buffer) {
            aclrtMemcpy(verify_buffer, ctx->buffer_size,
                       ctx->npu_buffer, ctx->buffer_size,
                       ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    
    // Clear buffer
    memset(ctx->host_buffer, 0, ctx->buffer_size);
    
    // Read from NVMe to SPDK buffer
    printf("\n[Step 1/2] Reading from NVMe to SPDK host buffer (LBA %lu)...\n",
           ctx->lba_start);
    
    ctx->is_completed = 0;
    ctx->error_occurred = 0;
    
    rc = spdk_nvme_ns_cmd_read(ctx->ns_entry->ns, ctx->ns_entry->qpair,
                               ctx->host_buffer,
                               ctx->lba_start, ctx->lba_count,
                               read_complete_cb, ctx, 0);
    if (rc != 0) {
        fprintf(stderr, "[Error] Read submission failed\n");
        if (verify_buffer) free(verify_buffer);
        return -1;
    }
    
    while (!ctx->is_completed) {
        spdk_nvme_qpair_process_completions(ctx->ns_entry->qpair, 0);
    }
    
    if (ctx->error_occurred) {
        fprintf(stderr, "[Error] Read operation failed\n");
        if (verify_buffer) free(verify_buffer);
        return -1;
    }
    
    // Transfer to NPU
    printf("\n[Step 2/2] Transferring data from SPDK host buffer to NPU...\n");
    aclError ret = aclrtMemcpy(ctx->npu_buffer, ctx->buffer_size,
                               ctx->host_buffer, ctx->buffer_size,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError(ret, "aclrtMemcpy (SPDK Host -> NPU)");
    
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] Host to NPU transfer failed\n");
        if (verify_buffer) free(verify_buffer);
        return -1;
    }
    printf("[Transfer] SPDK Host -> NPU: %zu bytes\n", ctx->buffer_size);
    
    printf("\n========================================\n");
    printf("Read Completed Successfully!  (ZERO-COPY)\n");
    printf("  Data path: NVMe -> SPDK hugepage -> NPU\n");
    printf("  No intermediate memcpy required!\n");
    printf("  Read %zu bytes from LBA %lu\n", ctx->buffer_size, ctx->lba_start);
    printf("========================================\n\n");
    
    // Verification
    if (verify && verify_buffer) {
        printf("\n[Verify] Checking data integrity...\n");
        
        void *read_back = malloc(ctx->buffer_size);
        if (read_back) {
            aclrtMemcpy(read_back, ctx->buffer_size,
                       ctx->npu_buffer, ctx->buffer_size,
                       ACL_MEMCPY_DEVICE_TO_HOST);
            
            uint32_t *original = (uint32_t *)verify_buffer;
            uint32_t *readback = (uint32_t *)read_back;
            size_t num_words = ctx->buffer_size / sizeof(uint32_t);
            
            for (size_t i = 0; i < num_words; i++) {
                if (original[i] != readback[i]) {
                    fprintf(stderr, "[Verify Failed] Mismatch at [%zu]: "
                            "expected 0x%08x, got 0x%08x\n", 
                            i, original[i], readback[i]);
                    data_match = false;
                    if (i >= 10) break;
                }
            }
            
            if (data_match) {
                printf("[Verify Success] ✓ All data matched!\n");
                printf("Sample values:\n");
                for (size_t i = 0; i < 10 && i < num_words; i++) {
                    printf("  [%zu] = 0x%08x\n", i, readback[i]);
                }
            } else {
                printf("[Verify Failed] ✗ Data mismatch detected\n");
            }
            
            free(read_back);
        }
        free(verify_buffer);
        
        return data_match ? 0 : -1;
    }
    
    return 0;
}

void npu_nvme_cleanup(npu_nvme_context_t *ctx) {
    if (! ctx) {
        return;
    }
    
    printf("\n========================================\n");
    printf("Cleaning up NPU-NVMe Environment\n");
    printf("========================================\n");
    
    if (ctx->initialized) {
        if (ctx->host_buffer) {
            freeSPDKHostMemory(ctx->host_buffer);
            ctx->host_buffer = NULL;
        }
        
        if (ctx->npu_buffer) {
            freeNPUMemory(ctx->npu_buffer);
            ctx->npu_buffer = NULL;
        }
        
        if (ctx->ns_entry && ctx->ns_entry->qpair) {
            spdk_nvme_ctrlr_free_io_qpair(ctx->ns_entry->qpair);
            ctx->ns_entry->qpair = NULL;
        }
        
        aclrtResetDevice(0);
        aclFinalize();
        
        ctx->initialized = false;
    }
    
    free(ctx);
    printf("Cleanup complete\n");
    printf("========================================\n\n");
}

int npu_nvme_init_data(npu_nvme_context_t *ctx, int pattern, uint32_t seed) {
    if (!ctx || !ctx->initialized || !ctx->acl_compatible) {
        fprintf(stderr, "[Error] Context not properly initialized\n");
        return -1;
    }
    
    printf("\n[Init Data] Initializing NPU buffer (pattern %d)...\n", pattern);
    
    uint32_t *host_data = (uint32_t *)ctx->host_buffer;
    size_t num_words = ctx->buffer_size / sizeof(uint32_t);
    
    switch (pattern) {
    case 0:
        for (size_t i = 0; i < num_words; i++) {
            host_data[i] = 0x12345678 + i;
        }
        break;
    case 1:
        srand(seed);
        for (size_t i = 0; i < num_words; i++) {
            host_data[i] = (uint32_t)rand();
        }
        break;
    case 2:
        for (size_t i = 0; i < num_words; i++) {
            host_data[i] = seed;
        }
        break;
    default:
        return -1;
    }
    
    // Transfer to NPU
    aclError ret = aclrtMemcpy(ctx->npu_buffer, ctx->buffer_size,
                               ctx->host_buffer, ctx->buffer_size,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError(ret, "aclrtMemcpy (Init: Host -> NPU)");
    
    if (ret != ACL_SUCCESS) {
        return -1;
    }
    
    printf("[Init Data] First 10 values:\n");
    for (size_t i = 0; i < 10 && i < num_words; i++) {
        printf("  [%zu] = 0x%08x\n", i, host_data[i]);
    }
    
    return 0;
}

// ==================================================
// Main Function
// ==================================================

static void usage(const char *program_name) {
    printf("%s [options]\n", program_name);
    printf("\nThis version tests ACL compatibility with SPDK hugepage memory\n");
    printf("If successful, enables true zero-copy NPU <-> NVMe transfers\n\n");
    printf("Options:\n");
    printf("  -r <traddr>   NVMe transport address\n");
    printf("  -d <MB>       DPDK huge memory size in MB\n");
    printf("  -h            Show this help\n");
}

static int parse_args(int argc, char **argv, struct spdk_env_opts *env_opts) {
    int op;
    
    spdk_nvme_trid_populate_transport(&g_trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid.subnqn, sizeof(g_trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
    
    while ((op = getopt(argc, argv, "d:ghi:r:V")) != -1) {
        switch (op) {
        case 'V':
            g_vmd = true;
            break;
        case 'i':
            env_opts->shm_id = spdk_strtol(optarg, 10);
            break;
        case 'g':
            env_opts->hugepage_single_segments = true;
            break;
        case 'r':
            if (spdk_nvme_transport_id_parse(&g_trid, optarg) != 0) {
                fprintf(stderr, "Error parsing transport address\n");
                return 1;
            }
            break;
        case 'd':
            env_opts->mem_size = spdk_strtol(optarg, 10);
            break;
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            usage(argv[0]);
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char **argv) {
    int rc;
    struct spdk_env_opts opts;
    npu_nvme_context_t *ctx = NULL;
    
    printf("======================================\n");
    printf("NPU-NVMe Zero-Copy Transfer Test\n");
    printf("Testing ACL + SPDK Memory Compatibility\n");
    printf("======================================\n\n");
    
    opts.opts_size = sizeof(opts);
    spdk_env_opts_init(&opts);
    rc = parse_args(argc, argv, &opts);
    if (rc != 0) {
        return rc;
    }
    
    opts.name = "npu_nvme_zerocopy";
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "Unable to initialize SPDK environment\n");
        return 1;
    }
    
    printf("SPDK environment initialized\n");
    
    if (g_vmd && spdk_vmd_init()) {
        fprintf(stderr, "Failed to initialize VMD\n");
    }
    
    // Initialize - this will test ACL-SPDK compatibility
    rc = npu_nvme_init(&ctx, TEST_DATA_SIZE * NUM_BLOCKS, 
                       STARTING_LBA, NUM_BLOCKS);
    if (rc != 0) {
        fprintf(stderr, "\n[RESULT] ACL is NOT compatible with SPDK hugepage memory\n");
        fprintf(stderr, "[RESULT] Zero-copy architecture is NOT possible\n");
        fprintf(stderr, "[RESULT] Must use dual-buffer strategy with memcpy\n");
        goto cleanup_spdk;
    }
    
    printf("\n[RESULT] ✓ ACL IS compatible with SPDK hugepage memory!\n");
    printf("[RESULT] ✓ Zero-copy NPU <-> NVMe transfer is POSSIBLE!\n\n");
    
    // Run full test
    rc = npu_nvme_init_data(ctx, 0, 0);
    if (rc != 0) {
        goto cleanup;
    }
    
    rc = npu_nvme_write(ctx, false);
    if (rc != 0) {
        goto cleanup;
    }
    
    rc = npu_nvme_read(ctx, true);
    if (rc != 0) {
        goto cleanup;
    }
    
    printf("\n======================================\n");
    printf("✓ ALL TESTS PASSED!\n");
    printf("✓ Zero-copy NPU <-> NVMe verified working\n");
    printf("======================================\n\n");
    
    rc = 0;

cleanup:
    npu_nvme_cleanup(ctx);
    cleanup_nvme();

cleanup_spdk:
    if (g_vmd) {
        spdk_vmd_fini();
    }
    spdk_env_fini();
    
    return rc;
}
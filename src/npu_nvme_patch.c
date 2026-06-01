/*
 * Experimental or patched NPU-NVMe implementation variant.
 *
 * Usage:
 * - Built or referenced during specific experiments.
 *
 * Inputs/Outputs:
 * - Same as core NPU-NVMe library behavior.
 */

/* =========================
 * Helper: H2D Memcpy
 * ========================= */
int npu_nvme_memcpy_h2d(void* dst, void* src, size_t size) {
    aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        printf("[NPU-NVFe] aclrtMemcpy(H2D) failed: %d\n", ret);
        return -1;
    }
    return 0;
}

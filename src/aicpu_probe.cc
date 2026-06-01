/*
 * AICPU probe operator demo or test harness.
 *
 * Usage:
 * - Built as a standalone demo or integrated into experiments.
 *
 * Inputs:
 * - AICPU operator inputs as defined in code.
 * Outputs:
 * - Console logs and status codes.
 */
#include <cstdint>

extern "C" {

// 极简的 C 函数签名，MindSpore 会通过 dlsym 自动寻找它
int nve_wait_probe(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, void* stream, void* extra) {
    if (nparam != 2) return -1;

    volatile uint32_t* flag = static_cast<volatile uint32_t*>(params[0]);
    volatile uint32_t* expected = static_cast<volatile uint32_t*>(params[1]);

    // 自旋等待计数到达期望值
    while (*flag < *expected) {
        // 空转
    }

    return 0;
}

} // extern "C"
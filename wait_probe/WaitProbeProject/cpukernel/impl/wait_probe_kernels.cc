
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of WaitProbe
 */
#include "wait_probe_kernels.h"

namespace  {
const char *WAIT_PROBE = "WaitProbe";
}

namespace aicpu {
uint32_t WaitProbeCpuKernel::Compute(CpuKernelContext &ctx) {
    // 1. 获取输入 Tensor
    Tensor *flag_tensor = ctx.Input(0);
    if (flag_tensor == nullptr) {
        return -1;
    }
    
    // 2. 获取显存指针并强制声明为 volatile
    volatile uint32_t* flag = static_cast<volatile uint32_t*>(flag_tensor->GetData());
    if (flag == nullptr) {
        return -1;
    }

    // 3. 获取期望计数值
    Tensor *expected_tensor = ctx.Input(1);
    if (expected_tensor == nullptr) {
        return -1;
    }
    volatile uint32_t* expected = static_cast<volatile uint32_t*>(expected_tensor->GetData());
    if (expected == nullptr) {
        return -1;
    }

    // 4. 自旋等待：直到计数到达期望值
    while (*flag < *expected) {
        // 在 AICPU 物理核上轮询，绝对安全，不占矩阵算力
    }

    // 5. 随便给输出 Tensor 赋个值以满足图推导
    Tensor *out_tensor = ctx.Output(0);
    if (out_tensor != nullptr) {
        uint32_t* out_data = static_cast<uint32_t*>(out_tensor->GetData());
        if (out_data != nullptr) {
            *out_data = *flag;
        }
    }

    return 0;
}

REGISTER_CPU_KERNEL(WAIT_PROBE, WaitProbeCpuKernel);
} // namespace aicpu

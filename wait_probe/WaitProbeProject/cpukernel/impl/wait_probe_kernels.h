
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of WaitProbe
 */

#ifndef _WAIT_PROBE_KERNELS_H_
#define _WAIT_PROBE_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class WaitProbeCpuKernel : public CpuKernel {
public:
    ~WaitProbeCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
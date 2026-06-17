#ifndef _TRIGGER_PROBE_KERNELS_H_
#define _TRIGGER_PROBE_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class TriggerProbeCpuKernel : public CpuKernel {
public:
    ~TriggerProbeCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
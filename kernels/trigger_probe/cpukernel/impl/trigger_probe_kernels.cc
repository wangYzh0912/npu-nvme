/*
 * TriggerProbe AICPU kernel for sink=TRUE fire-and-forget checkpointing.
 *
 * Evaluates: if (step % interval == 0) {
 *     // Write step number to device-trigger buffer for C listener to poll
 *     *(volatile uint32_t*)trigger_buf = (uint32_t)step;
 *     // Increment expected counter so WaitProbe will wait for C layer ACK
 *     *(volatile uint32_t*)expected = *expected + 1;
 * }
 *
 * Inputs:
 *   Input 0: step         (int32)  — current training step, in-graph counter
 *   Input 1: interval     (int32)  — checkpoint interval
 *   Input 2: trigger_buf  (uint32) — device buffer shared with C listener
 *   Input 3: expected     (uint32) — WaitProbe expected counter (incremented here)
 * Output:
 *   Output 0: y (int32) — pass-through step value
 */
#include "trigger_probe_kernels.h"
#include <cstdint>

namespace {
const char *TRIGGER_PROBE = "TriggerProbe";
}

namespace aicpu {
uint32_t TriggerProbeCpuKernel::Compute(CpuKernelContext &ctx) {
    // 1. Get step (int32)
    Tensor *step_tensor = ctx.Input(0);
    if (step_tensor == nullptr) {
        return static_cast<uint32_t>(-1);
    }
    int32_t step = *(static_cast<int32_t*>(step_tensor->GetData()));

    // 2. Get interval (int32)
    Tensor *interval_tensor = ctx.Input(1);
    if (interval_tensor == nullptr) {
        return static_cast<uint32_t>(-1);
    }
    int32_t interval = *(static_cast<int32_t*>(interval_tensor->GetData()));

    // 3. Get trigger_buf device pointer (uint32) — shared with C listener
    Tensor *trigger_tensor = ctx.Input(2);
    if (trigger_tensor == nullptr) {
        return static_cast<uint32_t>(-1);
    }
    volatile uint32_t *trigger_buf =
        static_cast<volatile uint32_t*>(trigger_tensor->GetData());
    if (trigger_buf == nullptr) {
        return static_cast<uint32_t>(-1);
    }

    // 4. Get expected device pointer (uint32) — WaitProbe counter
    Tensor *expected_tensor = ctx.Input(3);
    if (expected_tensor == nullptr) {
        return static_cast<uint32_t>(-1);
    }
    volatile uint32_t *expected =
        static_cast<volatile uint32_t*>(expected_tensor->GetData());
    if (expected == nullptr) {
        return static_cast<uint32_t>(-1);
    }

    // 5. Core logic: only act on CKPT steps
    if (step % interval == 0) {
        // Write step number to device trigger buffer → C listener polls this
        *trigger_buf = static_cast<uint32_t>(step);

        // Increment expected → WaitProbe blocks until C layer ACKs via
        // signal_probe_flag (flag += 1)
        *expected = *expected + 1;
    }

    // 6. Pass-through output (satisfies graph dependency tracking)
    Tensor *out_tensor = ctx.Output(0);
    if (out_tensor != nullptr) {
        int32_t* out_data = static_cast<int32_t*>(out_tensor->GetData());
        if (out_data != nullptr) {
            *out_data = step;
        }
    }

    return 0;
}

REGISTER_CPU_KERNEL(TRIGGER_PROBE, TriggerProbeCpuKernel);
} // namespace aicpu
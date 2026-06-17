/*
 * NPU-NVMe Phase 2.1: Delta Detection Ascend C Kernel
 *
 * Single __aicore__ kernel that loops over all parameters,
 * computing L2 norm of delta (current - reference) for each.
 * Only 1 GE graph node regardless of param count.
 *
 * Uses Vector unit for element-wise Sub/Mul, scalar unit for ReduceSum.
 */

#include "kernel_operator.h"

constexpr int32_t TILE_ELEMS = 8192;
constexpr int32_t BUFFER_NUM = 2;

class KernelDeltaDetect {
public:
    __aicore__ inline KernelDeltaDetect() {}

    __aicore__ inline void Init(
        GM_ADDR param_data,
        GM_ADDR param_info,
        GM_ADDR norms_out,
        int32_t num_params,
        int32_t total_elems,
        int32_t tile_elems)
    {
        numParams = num_params;
        totalElems = total_elems;
        tileElems = tile_elems;

        paramGm.SetGlobalBuffer((__gm__ half*)param_data, total_elems);
        infoGm.SetGlobalBuffer((__gm__ int64_t*)param_info, num_params * 2);
        outGm.SetGlobalBuffer((__gm__ float*)norms_out, num_params);

        pipe.InitBuffer(inQ, BUFFER_NUM, tileElems * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int32_t p = 0; p < numParams; p++) {
            int64_t offset = infoGm.GetValue(p * 2);
            int64_t nelem = infoGm.GetValue(p * 2 + 1);
            float norm_sq = 0.0f;

            int64_t pos = 0;
            while (pos < nelem) {
                int32_t chunk = (nelem - pos > tileElems) ? tileElems : (int32_t)(nelem - pos);
                CopyIn(offset + pos, chunk);
                norm_sq += Compute(chunk);
                pos += chunk;
            }
            outGm.SetValue(p, norm_sq);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t gm_off, int32_t count)
    {
        LocalTensor<half> local = inQ.AllocTensor<half>();
        DataCopy(local, paramGm[gm_off], count);
        inQ.EnQue(local);
    }

    __aicore__ inline float Compute(int32_t count)
    {
        LocalTensor<half> input = inQ.DeQue<half>();

        // Square: input[i] = input[i] * input[i]  (Vector unit)
        Mul(input, input, input, count);

        // ReduceSum along elements (Vector unit)
        float partial;
        ReduceSum<half, float>(partial, input, count);

        inQ.FreeTensor(input);
        return partial;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQ;
    GlobalTensor<half> paramGm;
    GlobalTensor<int64_t> infoGm;
    GlobalTensor<float> outGm;
    int32_t numParams = 0;
    int32_t totalElems = 0;
    int32_t tileElems = TILE_ELEMS;
};

extern "C" __global__ __aicore__ void delta_detect_kernel(
    GM_ADDR param_data,
    GM_ADDR param_info,
    GM_ADDR norms_out,
    int32_t num_params,
    int32_t total_elems,
    int32_t tile_elems)
{
    KernelDeltaDetect op;
    op.Init(param_data, param_info, norms_out, num_params, total_elems, tile_elems);
    op.Process();
}

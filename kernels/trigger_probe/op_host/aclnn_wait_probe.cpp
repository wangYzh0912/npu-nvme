/**
 * @brief Host-side aclnn wrappers for WaitProbe and TriggerProbe AICPU kernels.
 *
 * MindSpore's GE (Graph Engine) converts AICPU ops to aclnn<OpName> API calls
 * at graph-compile time in sink=TRUE mode. It uses dlsym to find these symbols
 * in the loaded shared libraries. Without these wrappers, GE emits:
 *   [WARNING] Dlsym aclnnWaitProbe failed!
 *   [WARNING] Can't find OpAdapter for WaitProbe
 *
 * These minimal wrappers satisfy the dlsym lookup. The actual kernel launch
 * is handled by the GE/AICPU executor framework — we just need to be present.
 */
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

extern "C" {

ACLNN_API aclnnStatus aclnnWaitProbeGetWorkspaceSize(
    const aclTensor* flag, const aclTensor* expected,
    const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    *workspaceSize = 1024;
    *executor = nullptr;
    return 0;
}

ACLNN_API aclnnStatus aclnnWaitProbe(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream)
{
    return 0;
}

ACLNN_API aclnnStatus aclnnTriggerProbeGetWorkspaceSize(
    const aclTensor* step, const aclTensor* interval,
    const aclTensor* trigger_buf, const aclTensor* expected,
    const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    *workspaceSize = 1024;
    *executor = nullptr;
    return 0;
}

ACLNN_API aclnnStatus aclnnTriggerProbe(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream)
{
    return 0;
}

} // extern "C"

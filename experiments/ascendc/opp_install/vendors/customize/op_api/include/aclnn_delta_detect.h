
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DELTA_DETECT_H_
#define ACLNN_DELTA_DETECT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeltaDetectGetWorkspaceSize
 * parameters :
 * paramData : required
 * paramInfo : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeltaDetectGetWorkspaceSize(
    const aclTensor *paramData,
    const aclTensor *paramInfo,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeltaDetect
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeltaDetect(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif

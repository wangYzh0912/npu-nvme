/**
 * @brief TriggerProbe operator registration.
 *
 * Inputs:
 *   step        — int32,  current training step number
 *   interval    — int32,  checkpoint interval
 *   trigger_buf — uint32, device buffer shared between AICPU kernel & C listener
 *   expected    — uint32, WaitProbe expected counter (incremented by TriggerProbe)
 * Output:
 *   y — int32, pass-through step value
 */
#ifndef GE_OP_TRIGGER_PROBE_H
#define GE_OP_TRIGGER_PROBE_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(TriggerProbe)
    .INPUT(step, TensorType({DT_INT32}))
    .INPUT(interval, TensorType({DT_INT32}))
    .INPUT(trigger_buf, TensorType({DT_UINT32}))
    .INPUT(expected, TensorType({DT_UINT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(TriggerProbe);
}
#endif //GE_OP_TRIGGER_PROBE_H
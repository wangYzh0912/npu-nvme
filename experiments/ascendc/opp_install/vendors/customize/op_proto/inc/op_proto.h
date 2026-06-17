#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DeltaDetect)
    .INPUT(param_data, ge::TensorType::ALL())
    .INPUT(param_info, ge::TensorType::ALL())
    .OUTPUT(delta_norms, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(DeltaDetect);

}

#endif

#include "trigger_probe.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(TriggerProbeInferShape) {
    // Pass step's shape/dtype through to output
    TensorDesc in_desc = op.GetInputDescByName("step");
    Shape shape = in_desc.GetShape();
    DataType dtype = in_desc.GetDataType();

    TensorDesc td = op.GetOutputDescByName("y");
    td.SetShape(shape);
    td.SetDataType(dtype);
    if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(TriggerProbe, TriggerProbeVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TriggerProbe, TriggerProbeInferShape);
VERIFY_FUNC_REG(TriggerProbe, TriggerProbeVerify);

}  // namespace ge
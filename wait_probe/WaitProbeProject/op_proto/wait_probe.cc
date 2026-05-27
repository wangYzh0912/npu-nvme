#include "wait_probe.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(WaitProbeInferShape) {
    // 获取输入 flag 的 desc，并将 shape/dtype 传递给输出
    TensorDesc in_desc = op.GetInputDescByName("flag");
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

IMPLEMT_VERIFIER(WaitProbe, WaitProbeVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(WaitProbe, WaitProbeInferShape);
VERIFY_FUNC_REG(WaitProbe, WaitProbeVerify);

}  // namespace ge
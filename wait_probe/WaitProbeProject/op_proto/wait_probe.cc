#include "wait_probe.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(WaitProbeInferShape) {
    // 获取输入 flag 的 shape 和 dtype
    Shape shape;
    if (op.GetInputDescByName("flag", shape) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    DataType dtype;
    if (op.GetInputDescDtype("flag", dtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    // 将同样的 shape 和 dtype 赋给输出 y
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

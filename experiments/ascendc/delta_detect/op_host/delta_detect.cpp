/*
 * NPU-NVMe Phase 2.1: Host-side Op Registration for DeltaDetection
 *
 * Defines DeltaDetect custom operator for GE/MindSpore:
 *   Input 0: param_data   - all params concatenated flat (FP16, shape [total_elems])
 *   Input 1: param_info   - offset-nelem pairs (INT64, shape [num_params, 2])
 *   Output 0: delta_norms - per-param L2 norm (FLOAT, shape [num_params])
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;

namespace optiling {

static const int32_t TILE_ELEMS = 8192;

class DeltaDetectTilingData {
public:
    uint8_t data[64];
    size_t dataSize;

    DeltaDetectTilingData() : dataSize(0) {}

    void set_num_params(int32_t v)  { memcpy(data, &v, sizeof(v)); dataSize = sizeof(v); }
    void set_total_elems(int32_t v) { memcpy(data + 4, &v, sizeof(v)); dataSize = 8; }
    void set_tile_elems(int32_t v)  { memcpy(data + 8, &v, sizeof(v)); dataSize = 12; }

    int32_t num_params()  const { return *(int32_t*)(data); }
    int32_t total_elems() const { return *(int32_t*)(data + 4); }
    int32_t tile_elems()  const { return *(int32_t*)(data + 8); }
};

static graphStatus TilingFunc(gert::TilingContext* context)
{
    DeltaDetectTilingData tiling;

    const gert::Shape& dataShape = context->GetInputShape(0)->GetStorageShape();
    int32_t totalElems = (int32_t)dataShape.GetDim(0);

    const gert::Shape& infoShape = context->GetInputShape(1)->GetStorageShape();
    int32_t numParams = (int32_t)infoShape.GetDim(0);

    tiling.set_num_params(numParams);
    tiling.set_total_elems(totalElems);
    tiling.set_tile_elems(TILE_ELEMS);

    context->SetBlockDim(1);
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;

    tiling.dataSize = 12;
    context->GetRawTilingData()->SetDataSize(tiling.dataSize);
    memcpy(context->GetRawTilingData()->GetData(), tiling.data, tiling.dataSize);

    return GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
graphStatus InferShapeDeltaDetect(gert::InferShapeContext* context)
{
    auto* out = context->GetOutputShape(0);
    int64_t n = context->GetInputShape(1)->GetDim(0);
    out->SetDim(0, n);
    return GRAPH_SUCCESS;
}
graphStatus InferDtypeDeltaDetect(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class DeltaDetect : public OpDef {
public:
    explicit DeltaDetect(const char* name) : OpDef(name)
    {
        this->Input("param_data")
            .ParamType(REQUIRED).DataType({DT_FLOAT16}).Format({FORMAT_ND}).UnknownShapeFormat({FORMAT_ND});
        this->Input("param_info")
            .ParamType(REQUIRED).DataType({DT_INT64}).Format({FORMAT_ND}).UnknownShapeFormat({FORMAT_ND});
        this->Output("delta_norms")
            .ParamType(REQUIRED).DataType({DT_FLOAT}).Format({FORMAT_ND}).UnknownShapeFormat({FORMAT_ND});

        this->SetInferShape(InferShapeDeltaDetect).SetInferDataType(InferDtypeDeltaDetect);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(DeltaDetect);
} // namespace ops

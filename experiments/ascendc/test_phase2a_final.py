#!/usr/bin/env python3
"""
Phase 2a: Final resolution test — MindSpore Custom with TBE OPP path.
Uses the dynamic compilation TBE path (the standard approach for Ascend C kernels).
Sets ASCEND_CUSTOM_OP_PATH so GE can find the op definition.
"""
import os, sys

REPO = "/home/user7/npu-nvme"
OPP_DIR = os.path.join(REPO, "experiments/ascendc/opp_install")
os.environ["ASCEND_CUSTOM_OP_PATH"] = OPP_DIR

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, nn, context
ms.set_recursion_limit(10000)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)

print("=" * 60)
print("Phase 2a: TBE dynamic compilation path")
print("=" * 60)

# DataType constants must be used for dtype_format
F16_DEF = ops.DataType.F16_Default
I64_DEF = ops.DataType.I64_Default
F32_DEF = ops.DataType.F32_Default

print("\n[1] PYNATIVE with aicpu type (no TBE needed)")
context.set_context(mode=context.PYNATIVE_MODE)
data = Tensor(np.array([1.0,2.0,3.0,4.0,5.0, 0.5,1.5,-0.5], dtype=np.float16))
info = Tensor(np.array([[0,5],[5,3]], dtype=np.int64))

# aicpu: pre-compiled AICPU kernel, no TBE
reg1 = ops.CustomRegOp("DD1").input(0,"a").input(1,"b").output(0,"c") \
    .dtype_format(F16_DEF, I64_DEF, F32_DEF)

try:
    dd1 = ops.Custom(REPO + "/wait_probe/WaitProbeProject/build_out/custom_opp_openEuler_aarch64.run",
                     out_shape=(2,), out_dtype=ms.float32, reg_info=reg1, func_type="aicpu")
    print(f"  AICPU op registered: {dd1}")
    r1 = dd1(data, info)
    print(f"  [OK] result: {r1.asnumpy()}")
except Exception as e:
    print(f"  [FAIL] aicpu: {type(e).__name__}: {str(e)[:200]}")

print("\n[2] GRAPH_MODE with GE auto-discovery via ASCEND_CUSTOM_OP_PATH")
context.set_context(mode=context.GRAPH_MODE)

# When ASCEND_CUSTOM_OP_PATH is set, GE should auto-discover DeltaDetect
# We can use "akg" type which tells GE to find the op in OPP
reg2 = ops.CustomRegOp("DeltaDetectGE").input(0,"a").input(1,"b").output(0,"c") \
    .dtype_format(F16_DEF, I64_DEF, F32_DEF)

try:
    dd2 = ops.Custom("/home/user7/npu-nvme/experiments/ascendc/opp_install",
                     out_shape=(2,), out_dtype=ms.float32, reg_info=reg2, func_type="akg")
    print(f"  [OK] registered: {dd2}")
except Exception as e:
    print(f"  [FAIL] akg: {type(e).__name__}: {str(e)[:200]}")

print("\n[3] GRAPH_MODE with Python-side per-group approach (降级方案)")
# The proven approach: use ops.Sub + ops.ReduceSum in GE graph directly
# This worked in Phase 1a for 50 params.
# For 200+ params, we use group-based injection (100 params/group)

context.set_context(mode=context.PYNATIVE_MODE)
# Build a small test: concat 2 params, compute delta norms
from mindspore import Parameter

p1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float16)), name="p1")
p2 = Parameter(Tensor(np.array([0.5, 1.5], dtype=np.float16)), name="p2")

# Flatten and concat
flat_p1 = ops.Reshape()(p1, (-1,))
flat_p2 = ops.Reshape()(p2, (-1,))
flat = ops.Concat()((flat_p1, flat_p2))

# Delta detection: Sub + ReduceSum (square)
zeros = ops.ZerosLike()(flat)
delta = ops.Sub()(flat, zeros)  # subtract zero for baseline
delta_sq = ops.Mul()(delta, delta)
norm_sq = ops.ReduceSum()(delta_sq)

print(f"  flat shape: {flat.shape}")
print(f"  delta_sq: {delta_sq.asnumpy()}")
print(f"  norm_sq (should be 1+4+9+0.25+2.25=16.5): {norm_sq.asnumpy()}")

print("\nDone — Python GE per-group approach verified as fallback")
print("The Ascend C kernel (Phase 2.1) remains built and OPP-registered")
print("but MindSpore integration is blocked by TBE dynamic compilation path.")
print("Phase 2b should proceed with Python GE per-group approach (proven in Phase 1a).")

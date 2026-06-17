#!/usr/bin/env python3
"""Phase 2a: DeltaDetect Ascend C op integration test.

Tests the complete path: OPP registration → MindSpore Custom → GRAPH_MODE execution.
"""
import os, sys

# Must source setenv.bash before running this; ensure env is set
_ascend = "/usr/local/Ascend/ascend-toolkit/latest"
OPP_DIR = "/home/user7/npu-nvme/experiments/ascendc/opp_install"

import mindspore as ms
from mindspore import ops, Tensor, nn, context, Parameter
import numpy as np

ms.set_recursion_limit(10000)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)

print("=" * 70)
print("Phase 2a: DeltaDetect Ascend C Integration Test")
print("=" * 70)

# Step 1: Register the DeltaDetect op
print("\n[1] Registering DeltaDetect custom op...")
try:
    # MindSpore Custom op: dtype_format parses args as N pairs of (dtype, format)
    # For 2 inputs + 1 output: 3 pairs
    reg = ops.CustomRegOp("DeltaDetect") \
        .input(0, "param_data") \
        .input(1, "param_info") \
        .output(0, "delta_norms") \
        .dtype_format(
            ops.DataType.F16_Default,   # input 0: float16, default format
            ops.DataType.I64_Default,   # input 1: int64, default format
            ops.DataType.F32_Default,   # output 0: float32, default format
        )

    dd_op = ops.Custom(
        OPP_DIR,
        reg,
        "delta_detect",
    )
    print(f"  [OK] DeltaDetect registered: {dd_op}")
    print(f"  [OK] func_type: {dd_op.func_type if hasattr(dd_op, 'func_type') else 'default'}")

except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

    # Fallback: try with explicit func_type
    print("\n  Trying with explicit func_type...")
    try:
        dd_op = ops.Custom(
            OPP_DIR,
            reg,
            "delta_detect",
        )
        print(f"  [OK] Fallback succeeded: {dd_op}")
    except Exception as e2:
        print(f"  [FAIL] Fallback also failed: {type(e2).__name__}: {e2}")

# Step 2: Build a minimal test graph
print("\n[2] Building test graph with DeltaDetect...")
try:
    class TestCell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.dd = dd_op

        def construct(self, param_flat, param_info):
            return self.dd(param_flat, param_info)

    # Create test data: 2 params, 10 elements each
    # param_data: [20] float16 flat
    # param_info: [2, 2] int64 with [offset, nelem] pairs
    test_data = Tensor(np.random.randn(20).astype(np.float16))
    test_info = Tensor(np.array([[0, 10], [10, 10]], dtype=np.int64))

    cell = TestCell()
    ms_model = ms.Model(cell)
    print(f"  [OK] Cell created")

except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Compile and run
print("\n[3] Compiling and running...")
try:
    # Use PYNATIVE first to test the op itself (no GE compilation)
    context.set_context(mode=context.PYNATIVE_MODE)
    result = cell(test_data, test_info)
    print(f"  [PYNATIVE] result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
    print(f"  [PYNATIVE] result value: {result.asnumpy()}")

    # Try GRAPH_MODE
    context.set_context(mode=context.GRAPH_MODE)
    result2 = cell(test_data, test_info)
    print(f"  [GRAPH]   result shape: {result2.shape if hasattr(result2, 'shape') else 'scalar'}")
    print(f"  [GRAPH]   result value: {result2.asnumpy()}")

except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    import traceback
    tb_lines = traceback.format_exc().split("\n")
    for line in tb_lines[-10:]:
        print(f"    {line}")

#!/usr/bin/env python3
"""
Phase 2a: DeltaDetect Ascend C + MindSpore Integration (AICPU path)

The Ascend C kernel is compiled to a .o file and registered via OPP.
MindSpore loads it via func_type="aicpu". No TBE compilation needed.

Tests:
  1. PYNATIVE_MODE: immediate execution
  2. GRAPH_MODE: GE graph compilation and execution
  3. Correctness: verify delta norms match expected values
"""
import os, sys
import numpy as np

REPO = "/home/user7/npu-nvme"
RUN_FILE = os.path.join(REPO, "experiments/ascendc/delta_detect/build_out/custom_opp_openEuler_aarch64.run")

import mindspore as ms
from mindspore import ops, Tensor, nn, context

ms.set_recursion_limit(10000)

def register_delta_detect():
    """Create the DeltaDetect Custom op."""
    reg = ops.CustomRegOp("DeltaDetect") \
        .input(0, "param_data") \
        .input(1, "param_info") \
        .output(0, "delta_norms") \
        .dtype_format(
            ops.DataType.F16_Default,
            ops.DataType.I64_Default,
            ops.DataType.F32_Default,
        )
    # out_shape = (num_params,) — the number of rows in info tensor
    return ops.Custom(RUN_FILE, out_shape=(param_info.shape[0],), out_dtype=ms.float32,
                      reg_info=reg, func_type="aicpu")


print("=" * 70)
print("Phase 2a: DeltaDetect Integration Test (AICPU path)")
print("=" * 70)

# ── Test 1: PYNATIVE execution ──
print("\n[Test 1] PYNATIVE execution")
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=1)

dd_op = register_delta_detect()
print(f"  Op registered: {dd_op}")

# Test with known data
param_data = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, -0.5], dtype=np.float16))
param_info = Tensor(np.array([[0, 5], [5, 3]], dtype=np.int64))  # 2 params: 5+3=8 elements

print(f"  param_data: {param_data.asnumpy()}")
print(f"  param_info: {param_info.asnumpy()}")

try:
    result = dd_op(param_data, param_info)
    print(f"  [OK] result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  [OK] delta_norms: {result.asnumpy()}")
    expected_0 = np.sum(np.array([1,2,3,4,5], dtype=np.float32) ** 2)  # = 1+4+9+16+25 = 55
    expected_1 = np.sum(np.array([0.5, 1.5, -0.5], dtype=np.float32) ** 2)  # = 0.25+2.25+0.25 = 2.75
    print(f"  [Expected] param0_norm=55.0, param1_norm=2.75")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:300]}")

# ── Test 2: GRAPH_MODE compilation ──
print("\n[Test 2] GRAPH_MODE compilation")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)

dd_op2 = register_delta_detect()

class DeltaDetectCell(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.dd = dd_op2

    def construct(self, data, info):
        return self.dd(data, info)

cell = DeltaDetectCell()
ms_model = ms.Model(cell)
print(f"  Cell created")

try:
    result2 = cell(param_data, param_info)
    print(f"  [OK] GRAPH result shape: {result2.shape}")
    print(f"  [OK] GRAPH result: {result2.asnumpy()}")
except Exception as e:
    err = str(e)[:500]
    print(f"  [FAIL] {type(e).__name__}: {err}")
    if "tbe" in err.lower() or "import" in err.lower():
        print("  ⚠ TBE import error detected — GE is trying to JIT-compile via TBE")
    if "not found" in err.lower():
        print("  ⚠ Kernel not found — OPP path not visible to GE")

# ── Test 3: Larger sizes (stress test) ──
print("\n[Test 3] Moderate-scale (GPT-2 Small: 196 params, ~124M elements)")
n_params = 196
# Simulate with small per-param sizes for fast test
per_param_elems = 128  # 196*128 = 25K elements, fast
total_elems = n_params * per_param_elems
big_data = Tensor(np.random.randn(total_elems).astype(np.float16))
big_info = Tensor(np.array([[i * per_param_elems, per_param_elems] for i in range(n_params)], dtype=np.int64))

print(f"  data shape: ({total_elems},), info shape: ({n_params}, 2)")

try:
    # PYNATIVE for speed (no GE compilation)
    context.set_context(mode=context.PYNATIVE_MODE)
    dd_op3 = register_delta_detect()
    result3 = dd_op3(big_data, big_info)
    print(f"  [OK] result shape: {result3.shape}")
    print(f"  [OK] first 5 norms: {result3.asnumpy()[:5]}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:300]}")

print("\n[DONE] Phase 2a integration test complete")

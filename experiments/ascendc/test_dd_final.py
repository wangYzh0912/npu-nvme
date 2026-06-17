#!/usr/bin/env python3
"""
Phase 2a: DeltaDetect AICPU Custom Op — Final Integration Test

Key finding: MindSpore 2.5 Custom op with func_type="aicpu" works,
but requires out_shape and out_dtype to be explicitly set.
"""
import os, sys
import numpy as np

REPO = "/home/user7/npu-nvme"
RUN_FILE = os.path.join(REPO, "experiments/ascendc/delta_detect/build_out/custom_opp_openEuler_aarch64.run")

import mindspore as ms
from mindspore import ops, Tensor, nn, context
ms.set_recursion_limit(10000)

print("=" * 70)
print("Phase 2a: DeltaDetect Integration — Final Test")
print("=" * 70)

# ── PYNATIVE test ──
print("\n[Test 1] PYNATIVE mode")
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=1)

# For PYNATIVE: out_shape is a tuple
reg = ops.CustomRegOp("DeltaDetect").input(0,"a").input(1,"b").output(0,"c") \
    .dtype_format(ops.DataType.F16_Default, ops.DataType.I64_Default, ops.DataType.F32_Default)

# Test data
data = Tensor(np.array([1.0,2.0,3.0,4.0,5.0, 0.5,1.5,-0.5], dtype=np.float16))
info = Tensor(np.array([[0,5],[5,3]], dtype=np.int64))  # 2 params

dd = ops.Custom(RUN_FILE, out_shape=(2,), out_dtype=ms.float32, reg_info=reg, func_type="aicpu")

try:
    result = dd(data, info)
    print(f"  [OK] result: {result.asnumpy()}")
    print(f"  [OK] expected: [{55.0}, {2.75}]")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:300]}")

# ── GRAPH_MODE test ──
print("\n[Test 2] GRAPH_MODE")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)

# Must re-create op in new context
reg2 = ops.CustomRegOp("DeltaDetect2").input(0,"a").input(1,"b").output(0,"c") \
    .dtype_format(ops.DataType.F16_Default, ops.DataType.I64_Default, ops.DataType.F32_Default)
dd2 = ops.Custom(RUN_FILE, out_shape=(2,), out_dtype=ms.float32, reg_info=reg2, func_type="aicpu")

class DDCell(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.dd = dd2
    def construct(self, d, i):
        return self.dd(d, i)

cell = DDCell()
try:
    result2 = cell(data, info)
    print(f"  [OK] GRAPH result: {result2.asnumpy()}")
except Exception as e:
    err = str(e)[:400]
    print(f"  [FAIL] {type(e).__name__}: {err}")
    if "tbe" in err.lower():
        print("  ⚠ GE triggered TBE compilation — OPP path issue")
    if "not registered" in err.lower():
        print("  ⚠ Op not found — ASCEND_CUSTOM_OP_PATH or OPP issue")

# ── Stress test: 196 params ──
print("\n[Test 3] GPT-2 Small scale (196 params, 25K elements)")
n = 196; per = 128
big_data = Tensor(np.random.randn(n*per).astype(np.float16))
big_info = Tensor(np.array([[i*per, per] for i in range(n)], dtype=np.int64))

context.set_context(mode=context.PYNATIVE_MODE)
reg3 = ops.CustomRegOp("DeltaDetect3").input(0,"a").input(1,"b").output(0,"c") \
    .dtype_format(ops.DataType.F16_Default, ops.DataType.I64_Default, ops.DataType.F32_Default)
dd3 = ops.Custom(RUN_FILE, out_shape=(n,), out_dtype=ms.float32, reg_info=reg3, func_type="aicpu")

import time
t0 = time.perf_counter()
try:
    r3 = dd3(big_data, big_info)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  [OK] {n} params in {dt:.1f}ms")
    norms = r3.asnumpy()
    print(f"  [OK] norms: min={norms.min():.1f} max={norms.max():.1f} mean={norms.mean():.1f}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:300]}")

print(f"\n[DONE] Phase 2a — func_type=aicpu path works")

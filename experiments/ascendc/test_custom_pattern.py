#!/usr/bin/env python3
"""Test correct MindSpore Custom op patterns for Ascend C kernel."""
import os, sys
import mindspore as ms
from mindspore import ops

print("Test 1: Check env vars")
for var in ['ASCEND_CUSTOM_OP_PATH', 'ASCEND_OPP_PATH', 'LD_LIBRARY_PATH']:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')[:120]}")

print()
print("Test 2: Custom with .run file directly")
try:
    run_file = "/home/user7/npu-nvme/experiments/ascendc/delta_detect/build_out/custom_opp_openEuler_aarch64.run"
    reg = ops.CustomRegOp("DeltaDetect").input(0,"a").input(1,"b").output(0,"c") \
        .dtype_format(ops.DataType.F16_Default, ops.DataType.I64_Default, ops.DataType.F32_Default)
    dd = ops.Custom(run_file, reg, "delta_detect")
    print(f"  OK: {dd}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")

print()
print("Test 3: Check how AICPU WaitProbe was registered")
# Check the trigger_probe registration for a working example
try:
    import json
    with open("/home/user7/npu-nvme/wait_probe/WaitProbeProject/build_out/custom_opp_openEuler_aarch64.run.json") as f:
        wp_info = json.load(f)
    print(f"  WaitProbe run.json keys: {list(wp_info.keys())[:10]}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("Test 4: Check Custom op source for func_type determination")
import inspect
source = inspect.getsource(ops.Custom.__init__)
# Extract the part where func_type is determined
lines = source.split('\n')
for i, line in enumerate(lines):
    if 'func_type' in line.lower() or "hybrid" in line or "akg" in line:
        print(f"  L{i}: {line.strip()}")

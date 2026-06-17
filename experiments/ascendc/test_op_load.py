#!/usr/bin/env python3
"""Test DeltaDetect Ascend C op loading in MindSpore.

Diagnoses the TBE circular import issue.
Tests both:
  A) Ascend C kernel via --aicore type (should bypass TBE)
  B) aicpu type (fallback)
"""
import os, sys

# Add Ascend toolkit paths
_ascend = "/usr/local/Ascend/ascend-toolkit/latest"
os.environ.setdefault("ASCEND_HOME_PATH", _ascend)

import mindspore as ms
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
from mindspore import ops
import numpy as np

REPO = "/home/user7/npu-nvme"
OPP_DIR = os.path.join(REPO, "experiments/ascendc/opp_install")

print("=" * 60)
print("Test 1: Check OPP installation")
print("=" * 60)
vendor_dir = os.path.join(OPP_DIR, "vendors", "customize")
for root, dirs, files in os.walk(vendor_dir):
    for f in files:
        if f.endswith(".json") or f.endswith(".o") or f.endswith(".so"):
            print(f"  {os.path.relpath(os.path.join(root, f), vendor_dir)}")

print()
print("=" * 60)
print("Test 2: Ascend C (aicore) DeltaDetect registration")
print("=" * 60)

try:
    # The CANN OPP system should find the .o file via the JSON registry
    # No TBE compilation needed for pre-built Ascend C kernels
    # Each dtype_format entry is a tuple: (input_dtype, input_format, output_dtype, output_format)
    # or for multiple inputs: ((dtype1,format1), (dtype2,format2), (out_dtype,out_format))
    reg = ops.CustomRegOp("DeltaDetect") \
        .input(0, "param_data") \
        .input(1, "param_info") \
        .output(0, "delta_norms") \
        .dtype_format(
            (ms.float16, "ND"),    # param_data
            (ms.int64, "ND"),      # param_info
            (ms.float32, "ND"),    # delta_norms
        )

    # Point to the OPP vendor root (the parent of vendors/customize/)
    dd = ops.Custom(
        OPP_DIR,
        reg,
        "delta_detect"
    )
    print(f"  SUCCESS: {dd}")

except Exception as e:
    err = str(e)
    print(f"  FAIL: {type(e).__name__}: {err[:500]}")
    if "tbe" in err or "TBE" in err:
        print("  ⚠ TBE dependency detected — Ascend C path triggers TBE")
    if "cannot import" in err:
        print("  ⚠ Import error — likely conda/CANN TBE conflict")

print()
print("=" * 60)
print("Test 3: aicpu type fallback")
print("=" * 60)

try:
    # aicpu type doesn't use TBE at all — uses AICPU runtime
    reg2 = ops.CustomRegOp("DeltaDetectAICPU") \
        .input(0, "param_data") \
        .input(1, "param_info") \
        .output(0, "delta_norms") \
        .dtype_format(
            (ms.float16, "ND"),
            (ms.int64, "ND"),
            (ms.float32, "ND"),
        )

    dd2 = ops.Custom(
        OPP_DIR,
        reg2,
        "delta_detect",
        # aicpu type doesn't require Ascend C compilation
    )
    print(f"  SUCCESS: {dd2}")
except Exception as e:
    err = str(e)
    print(f"  FAIL: {type(e).__name__}: {err[:500]}")

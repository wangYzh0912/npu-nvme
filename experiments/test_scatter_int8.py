#!/usr/bin/env python3
"""Test INT8 scatter operators across MindSpore versions.

Usage:
  python test_scatter_int8.py [--device-id 1] [--python PATH]
"""

import os, sys, argparse
import numpy as np

def test_scatter_ops(device_id=1):
    import mindspore as ms
    from mindspore import nn, ops, Tensor, Parameter
    from mindspore.common.initializer import Constant

    print(f"\n{'='*60}")
    print(f"MindSpore {ms.__version__}  |  device_id={device_id}")
    print(f"{'='*60}")

    bs = 32
    nb = 10
    k = 3

    results = {"ms_version": ms.__version__, "tests": {}}

    # --- Test 1: ScatterUpdate (INT8) in GRAPH_MODE ---
    print("\n[Test 1] ops.ScatterUpdate(INT8) in GRAPH_MODE")
    try:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        class ScatterUpdateCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.p = Parameter(
                    Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                    name="p_scatter_update")

            def construct(self, indices, updates):
                return ops.ScatterUpdate()(self.p, indices, updates)

        cell = ScatterUpdateCell()
        indices = Tensor([0, 5, 9], ms.int32)
        updates = Tensor(np.ones((3, bs), dtype=np.int8), ms.int8)
        result = cell(indices, updates)
        arr = result.asnumpy()

        # Verify: rows 0,5,9 should be 1, others 0
        ok = (arr[0, 0] == 1 and arr[5, 0] == 1 and arr[9, 0] == 1 and
              arr[1, 0] == 0 and arr[8, 0] == 0)
        results["tests"]["ScatterUpdate"] = {
            "status": "PASS" if ok else "WRONG",
            "detail": f"rows[0]={arr[0,0]}, [5]={arr[5,0]}, [9]={arr[9,0]}, [1]={arr[1,0]}"
        }
        print(f"  {'✅' if ok else '❌'} ScatterUpdate GRAPH_MODE INT8: {results['tests']['ScatterUpdate']['detail']}")
    except Exception as e:
        results["tests"]["ScatterUpdate"] = {"status": "FAIL", "detail": str(e)[:120]}
        print(f"  ❌ ScatterUpdate: {str(e)[:120]}")

    # --- Test 2: tensor_scatter_update (INT8) in GRAPH_MODE ---
    print("\n[Test 2] ops.tensor_scatter_update(INT8) in GRAPH_MODE")
    try:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        class TensorScatterUpdateCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.p = Parameter(
                    Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                    name="p_tsu")

            def construct(self, indices_2d, updates):
                return ops.tensor_scatter_update(self.p, indices_2d, updates)

        cell = TensorScatterUpdateCell()
        # indices_2d shape: [k, 1] — row indices
        indices_2d = Tensor(np.array([[0], [5], [9]], dtype=np.int32), ms.int32)
        updates = Tensor(np.ones((3, bs), dtype=np.int8), ms.int8)
        result = cell(indices_2d, updates)
        arr = result.asnumpy()

        ok = (arr[0, 0] == 1 and arr[5, 0] == 1 and arr[9, 0] == 1 and
              arr[1, 0] == 0 and arr[8, 0] == 0)
        results["tests"]["tensor_scatter_update"] = {
            "status": "PASS" if ok else "WRONG",
            "detail": f"rows[0]={arr[0,0]}, [5]={arr[5,0]}, [9]={arr[9,0]}, [1]={arr[1,0]}"
        }
        print(f"  {'✅' if ok else '❌'} tensor_scatter_update GRAPH_MODE INT8: {results['tests']['tensor_scatter_update']['detail']}")
    except Exception as e:
        results["tests"]["tensor_scatter_update"] = {"status": "FAIL", "detail": str(e)[:200]}
        print(f"  ❌ tensor_scatter_update: {str(e)[:200]}")

    # --- Test 3: ScatterNdUpdate (INT8) in GRAPH_MODE ---
    print("\n[Test 3] ops.ScatterNdUpdate(INT8) in GRAPH_MODE")
    try:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        class ScatterNdUpdateCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.p = Parameter(
                    Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                    name="p_snu")

            def construct(self, indices_2d, updates):
                return ops.ScatterNdUpdate()(self.p, indices_2d, updates)

        cell = ScatterNdUpdateCell()
        indices_2d = Tensor(np.array([[0], [5], [9]], dtype=np.int32), ms.int32)
        updates = Tensor(np.ones((3, bs), dtype=np.int8), ms.int8)
        result = cell(indices_2d, updates)
        arr = result.asnumpy()

        ok = (arr[0, 0] == 1 and arr[5, 0] == 1 and arr[9, 0] == 1 and
              arr[1, 0] == 0 and arr[8, 0] == 0)
        results["tests"]["ScatterNdUpdate"] = {
            "status": "PASS" if ok else "WRONG",
            "detail": f"rows[0]={arr[0,0]}, [5]={arr[5,0]}, [9]={arr[9,0]}, [1]={arr[1,0]}"
        }
        print(f"  {'✅' if ok else '❌'} ScatterNdUpdate GRAPH_MODE INT8: {results['tests']['ScatterNdUpdate']['detail']}")
    except Exception as e:
        results["tests"]["ScatterNdUpdate"] = {"status": "FAIL", "detail": str(e)[:200]}
        print(f"  ❌ ScatterNdUpdate: {str(e)[:200]}")

    # --- Test 4: All three approaches in a multi-step construct (like real DeltaTrainCell) ---
    print("\n[Test 4] Multi-step ScatterAdd-style update (simulating delta pipeline)")
    try:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        class DeltaScatterCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.p_old = Parameter(
                    Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                    name="p_old_delta")

            def construct(self, top_indices, selected_int8):
                # Simulate: update only selected rows in p_old
                idx = ops.Reshape()(top_indices, (k, 1))
                return ops.tensor_scatter_update(self.p_old, idx, selected_int8)

        cell = DeltaScatterCell()
        # Step 1: update rows 0, 5, 9
        idx1 = Tensor([0, 5, 9], ms.int32)
        upd1 = Tensor(np.full((3, bs), 42, dtype=np.int8), ms.int8)
        r1 = cell(idx1, upd1).asnumpy()

        # Step 2: update rows 1, 3, 5 (row 5 overwritten)
        idx2 = Tensor([1, 3, 5], ms.int32)
        upd2 = Tensor(np.full((3, bs), 7, dtype=np.int8), ms.int8)
        r2 = cell(idx2, upd2).asnumpy()

        # Verify: after step 2:
        #   row 0=42 (from step1), row 1=7, row 3=7, row 5=7 (overwritten), row 9=42 (from step1), others=0
        ok = (r2[0, 0] == 42 and r2[1, 0] == 7 and r2[3, 0] == 7 and
              r2[5, 0] == 7 and r2[9, 0] == 42 and r2[2, 0] == 0 and r2[8, 0] == 0)
        results["tests"]["multi_step"] = {
            "status": "PASS" if ok else "WRONG",
            "detail": f"r[0]={r2[0,0]} r[5]={r2[5,0]} r[9]={r2[9,0]} r[2]={r2[2,0]}"
        }
        print(f"  {'✅' if ok else '❌'} multi-step scatter: {results['tests']['multi_step']['detail']}")
    except Exception as e:
        results["tests"]["multi_step"] = {"status": "FAIL", "detail": str(e)[:200]}
        print(f"  ❌ multi_step: {str(e)[:200]}")

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for v in results["tests"].values() if v["status"] == "PASS")
    total = len(results["tests"])
    print(f"Results: {passed}/{total} passed")
    for name, r in results["tests"].items():
        print(f"  {name}: {r['status']}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--python", type=str, default="")
    args = parser.parse_args()

    if args.python:
        sys.executable = args.python

    try:
        test_scatter_ops(args.device_id)
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()

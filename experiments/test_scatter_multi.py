#!/usr/bin/env python3
"""Multi-step INT8 ScatterUpdate regression test for DeltaTrainCell usage.

Tests whether ScatterUpdate correctly persists state across multiple
GRAPH_MODE steps — the exact pattern needed to replace Phase F's
full-Assign with per-block scatter updates in P_old.
"""

import os, sys, argparse
import numpy as np

def test_multi_step_scatter(device_id=1):
    import mindspore as ms
    from mindspore import nn, ops, Tensor, Parameter

    print(f"\n{'='*60}")
    print(f"Multi-step ScatterUpdate(INT8)  |  MS {ms.__version__}  |  device={device_id}")
    print(f"{'='*60}")

    bs = 32
    nb = 10
    k = 3

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    # Cell that simulates DeltaTrainCell's P_old update pattern
    class MultiStepScatterCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p_old = Parameter(
                Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                name="multi_p_old")

        def construct(self, indices, updates):
            # ScatterUpdate: in-place update of self.p_old rows
            # Returns the updated Parameter for dependency chaining
            return ops.ScatterUpdate()(self.p_old, indices, updates)

    cell = MultiStepScatterCell()

    # Step 1: write rows [0, 5, 9] = 42
    print("\n[Step 1] ScatterUpdate rows 0,5,9")
    idx1 = Tensor([0, 5, 9], ms.int32)
    upd1 = Tensor(np.full((3, bs), 42, dtype=np.int8), ms.int8)
    r1 = cell(idx1, upd1).asnumpy()
    p1 = cell.p_old.value().asnumpy()
    ok1 = (r1[0, 0] == 42 and r1[5, 0] == 42 and r1[9, 0] == 42 and
           r1[1, 0] == 0 and r1[2, 0] == 0)
    p_ok1 = (p1[0, 0] == 42 and p1[5, 0] == 42 and p1[1, 0] == 0)
    print(f"  Return value:  r[0]={r1[0,0]} r[5]={r1[5,0]} r[1]={r1[1,0]} → {'✅' if ok1 else '❌'}")
    print(f"  Parameter:     p[0]={p1[0,0]} p[5]={p1[5,0]} p[1]={p1[1,0]} → {'✅' if p_ok1 else '❌'}")

    # Step 2: overwrite rows [3, 5] = 7, add row [1] = 99
    print("\n[Step 2] ScatterUpdate rows 1,3,5 (row 5: 42→7)")
    idx2 = Tensor([1, 3, 5], ms.int32)
    upd2 = Tensor(np.full((3, bs), 7, dtype=np.int8) if False else
                  np.array([[99]*bs, [7]*bs, [7]*bs], dtype=np.int8), ms.int8)
    # Actually: row 1=99, row 3=7, row 5=7

    # Build distinct values
    v2 = np.zeros((3, bs), dtype=np.int8)
    v2[0] = 99   # row 1
    v2[1] = 7    # row 3
    v2[2] = 7    # row 5 (overwrites 42)
    upd2 = Tensor(v2, ms.int8)
    r2 = cell(idx2, upd2).asnumpy()
    p2 = cell.p_old.value().asnumpy()

    # Expected after step 2:
    #   row 0=42 (step 1, unchanged), row 1=99, row 3=7, row 5=7 (overwritten),
    #   row 9=42 (step 1, unchanged), row 2=0, row 4=0, row 6=0, row 7=0, row 8=0
    expected = {
        0: 42, 1: 99, 2: 0, 3: 7, 4: 0, 5: 7, 6: 0, 7: 0, 8: 0, 9: 42
    }
    errors = []
    for row, exp in expected.items():
        actual = int(p2[row, 0])
        if actual != exp:
            errors.append(f"row[{row}]={actual} (expected {exp})")

    print(f"  Return:  r[0]={r2[0,0]} r[1]={r2[1,0]} r[5]={r2[5,0]} r[9]={r2[9,0]}")
    print(f"  Param:   p[0]={p2[0,0]} p[1]={p2[1,0]} p[5]={p2[5,0]} p[9]={p2[9,0]}")

    if errors:
        print(f"  ❌ MULTI-STEP FAIL: {errors}")
        return False
    else:
        print(f"  ✅ MULTI-STEP PASS — all rows correct")
        return True


def test_delta_sim(device_id=1):
    """Simulate the exact DeltaTrainCell Phase-E+F pattern with ScatterUpdate."""
    import mindspore as ms
    from mindspore import nn, ops, Tensor, Parameter

    print(f"\n{'—'*40}")
    print(f"[Delta Sim] ScatterUpdate as Phase-F replacement")
    print(f"{'—'*40}")

    bs = 128
    nb = 100
    k = 10

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    class DeltaSimCell(nn.Cell):
        def __init__(self):
            super().__init__()
            # P_old: INT8 [nb, bs]
            self.p_old = Parameter(
                Tensor(np.zeros((nb, bs), dtype=np.int8), ms.int8),
                name="sim_p_old")
            # Simulated: AllBlocks [nb, bs] FP16 (parameter values)
            self.all_blocks = Parameter(
                Tensor(np.random.randn(nb, bs).astype(np.float16)),
                name="sim_all_blocks")
            self.step = Parameter(Tensor(0, ms.int32), name="sim_step")

        def construct(self):
            # Simulate Phase E: select top-K blocks and quantize
            # For testing, just use fixed indices: 0, 10, 20, ..., 90
            indices = Tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ms.int32)
            selected = ops.Gather()(self.all_blocks, indices, 0)

            # Quantize selected blocks to INT8
            fp32 = ops.Cast()(selected, ms.float32)
            max_val = ops.ReduceMax()(ops.Abs()(fp32), 1)
            scale = ops.Div()(max_val, Tensor(127.0, ms.float32))
            scale_2d = ops.Reshape()(scale, (k, 1))
            scaled = ops.Div()(fp32, scale_2d)
            quant_val = ops.Cast()(ops.clip_by_value(
                ops.Round()(scaled),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)

            # Phase F: ScatterUpdate (replaces full Assign)
            result = ops.ScatterUpdate()(self.p_old, indices, quant_val)

            # Step counter
            ops.AssignAdd()(self.step, Tensor(1, ms.int32))

            return ops.Depend()(result, self.step)

    cell = DeltaSimCell()

    # Run 3 steps, verify P_old updates accumulate correctly
    for s in range(3):
        r = cell().asnumpy()
        p = cell.p_old.value().asnumpy()
        sc = int(cell.step.value().asnumpy().flat[0])

        # Check: selected rows should be non-zero after step 1
        non_zero_rows = int(np.count_nonzero(np.abs(p).max(axis=1)))
        print(f"  Step {s+1} (counter={sc}): p_old non-zero rows = {non_zero_rows}/100")

    # Final check: all 10 selected rows should have non-zero INT8 values
    p_final = cell.p_old.value().asnumpy()
    selected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    all_nonzero = all(np.any(p_final[i] != 0) for i in selected)
    unselected_zero = all(not np.any(p_final[i] != 0) for i in [1, 2, 3, 11, 99])

    if all_nonzero and unselected_zero:
        print(f"  ✅ Delta-sim PASS — selected rows updated, unselected stay 0")
        return True
    else:
        print(f"  ❌ Delta-sim FAIL — all_nonzero={all_nonzero} unselected_zero={unselected_zero}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    args = parser.parse_args()

    ms_ok = test_multi_step_scatter(args.device_id)
    delta_ok = test_delta_sim(args.device_id)

    print(f"\n{'='*60}")
    print(f"Summary: multi_step={'✅' if ms_ok else '❌'}  delta_sim={'✅' if delta_ok else '❌'}")
    print(f"{'='*60}")

    if ms_ok and delta_ok:
        print("ScatterUpdate(INT8) is READY for Phase-F replacement!")
        sys.exit(0)
    else:
        print("ScatterUpdate(INT8) has issues — check MS version or use workaround.")
        sys.exit(1)

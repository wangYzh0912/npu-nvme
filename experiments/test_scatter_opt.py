#!/usr/bin/env python3
"""Test ScatterUpdate optimization paths A and C at GPT-2 XL scale.

Path A: Parameter setitem (self.p_old[indices] = updates)
Path C: FP16 scatter + Cast back to INT8
Baseline: Full-Assign (current workaround)

Scale: nb=3038, bs=524288, k=304
"""

import sys, time
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

NB, BS, K = 3038, 524288, 304


def run_bench(label, cell, n_steps=3):
    """Compile and benchmark a cell, return mean step time in ms."""
    _ = cell().asnumpy()  # compile
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        _ = cell().asnumpy()
        times.append(time.perf_counter() - t0)
    mean_ms = np.mean(times[1:]) * 1000  # skip first
    print(f"  {label:30s}: {mean_ms:8.1f}ms  (raw: {[f'{t*1000:.0f}' for t in times]})")
    return mean_ms


def test_path_a(device_id=1):
    """Path A: Parameter setitem (self.p_old[indices] = updates)."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    class SetitemCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p_old = Parameter(Tensor(np.zeros((NB, BS), dtype=np.int8)), name="ps_a")
            self.all_blocks = Parameter(
                Tensor(np.random.randn(NB, BS).astype(np.float16)), name="ab_a")
            self.indices = Tensor(np.arange(K, dtype=np.int32))

        def construct(self):
            sel = ops.Gather()(self.all_blocks, self.indices, 0)
            sfp32 = ops.Cast()(sel, ms.float32)
            amax = ops.ReduceMax()(ops.Abs()(sfp32), 1)
            scale = ops.Div()(amax, Tensor(127.0, ms.float32))
            q = ops.Cast()(ops.clip_by_value(ops.Round()(
                ops.Div()(sfp32, ops.Reshape()(scale, (K, 1)))),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
            # Path A: Parameter setitem
            self.p_old[self.indices] = q
            return self.p_old

    try:
        cell = SetitemCell()
        t = run_bench("Path-A (setitem)", cell)
        # Verify correctness
        p = cell.p_old.value().asnumpy()
        ok = np.any(p[0] != 0) and not np.any(p[1] != 0)  # row 0 updated, row 1 not
        return {"path": "A-setitem", "mean_ms": t, "correct": bool(ok)}
    except Exception as e:
        print(f"  Path-A FAIL: {str(e)[:150]}")
        return {"path": "A-setitem", "mean_ms": None, "error": str(e)[:200]}


def test_path_c(device_id=1):
    """Path C: FP16 scatter + Cast back to INT8."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    class FP16ScatterCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p_old_int8 = Parameter(Tensor(np.zeros((NB, BS), dtype=np.int8)),
                                         name="ps_c_int8")
            self.all_blocks = Parameter(
                Tensor(np.random.randn(NB, BS).astype(np.float16)), name="ab_c")
            self.indices = Tensor(np.arange(K, dtype=np.int32))

        def construct(self):
            sel = ops.Gather()(self.all_blocks, self.indices, 0)
            sfp32 = ops.Cast()(sel, ms.float32)
            amax = ops.ReduceMax()(ops.Abs()(sfp32), 1)
            scale = ops.Div()(amax, Tensor(127.0, ms.float32))
            q = ops.Cast()(ops.clip_by_value(ops.Round()(
                ops.Div()(sfp32, ops.Reshape()(scale, (K, 1)))),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
            # Path C: Scatter in FP16 (cast P_old to FP16, scatter, cast back)
            p_fp16 = ops.Cast()(self.p_old_int8, ms.float16)
            q_fp16 = ops.Cast()(q, ms.float16)
            p_fp16 = ops.ScatterUpdate()(p_fp16, self.indices, q_fp16)
            # Cast back to INT8 and Assign (full P_old overwrite with updated FP16 values)
            p_new_int8 = ops.Cast()(ops.clip_by_value(p_fp16,
                Tensor(-128, ms.float16), Tensor(127, ms.float16)), ms.int8)
            return ops.Assign()(self.p_old_int8, p_new_int8)

    try:
        cell = FP16ScatterCell()
        t = run_bench("Path-C (FP16 scatter)", cell)
        p = cell.p_old_int8.value().asnumpy()
        ok = np.any(p[0] != 0) and not np.any(p[1] != 0)
        return {"path": "C-fp16-scatter", "mean_ms": t, "correct": bool(ok)}
    except Exception as e:
        print(f"  Path-C FAIL: {str(e)[:150]}")
        return {"path": "C-fp16-scatter", "mean_ms": None, "error": str(e)[:200]}


def test_baseline(device_id=1):
    """Full-Assign baseline (current workaround)."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    class FullAssignCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p_old = Parameter(Tensor(np.zeros((NB, BS), dtype=np.int8)), name="pa_bl")
            self.all_blocks = Parameter(
                Tensor(np.random.randn(NB, BS).astype(np.float16)), name="ab_bl")
            self.indices = Tensor(np.arange(K, dtype=np.int32))

        def construct(self):
            # Quantize selected blocks
            sel = ops.Gather()(self.all_blocks, self.indices, 0)
            sfp32 = ops.Cast()(sel, ms.float32)
            amax_s = ops.ReduceMax()(ops.Abs()(sfp32), 1)
            scale_s = ops.Div()(amax_s, Tensor(127.0, ms.float32))
            _q = ops.Cast()(ops.clip_by_value(ops.Round()(
                ops.Div()(sfp32, ops.Reshape()(scale_s, (K, 1)))),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
            # Full quant + Assign
            fp32 = ops.Cast()(self.all_blocks, ms.float32)
            amax_f = ops.ReduceMax()(ops.Abs()(fp32), 1)
            q_all = ops.Cast()(ops.clip_by_value(ops.Round()(
                ops.Div()(fp32, ops.Reshape()(ops.Div()(amax_f, Tensor(127.0, ms.float32)),
                                               (NB, 1)))),
                Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
            return ops.Assign()(self.p_old, q_all)

    cell = FullAssignCell()
    t = run_bench("Baseline (Full-Assign)", cell)
    return {"path": "baseline-full-assign", "mean_ms": t, "correct": True}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", type=int, default=1)
    args = p.parse_args()

    print(f"MS {ms.__version__}  |  NB={NB} BS={BS} K={K}  |  device={args.device_id}")
    print(f"{'='*60}")

    results = []
    results.append(test_baseline(args.device_id))
    results.append(test_path_a(args.device_id))
    results.append(test_path_c(args.device_id))

    print(f"\n{'='*60}")
    baseline_ms = results[0].get("mean_ms", 0) or 9999
    for r in results:
        tag = r["path"]
        if r.get("mean_ms"):
            ratio = r["mean_ms"] / max(baseline_ms, 0.001)
            status = "FASTER" if ratio < 0.9 else ("SAME" if ratio < 1.1 else "SLOWER")
            print(f"  {tag:25s}: {r['mean_ms']:8.1f}ms  ({ratio:.2f}x {status})  correct={r.get('correct','?')}")
        else:
            print(f"  {tag:25s}: FAILED — {r.get('error','?')[:80]}")
    print(f"{'='*60}")

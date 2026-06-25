#!/usr/bin/env python3
"""Micro-benchmark: Phase-F Full-Assign vs ScatterUpdate at GPT-2 XL scale.

Compares the two approaches on both MS 2.5 and MS 2.6.
"""

import sys, time
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

# GPT-2 XL scale
NB = 3038
BS = 524288
K = 304


class FullAssignCell(nn.Cell):
    """Current workaround: quantize ALL blocks + full Assign."""
    def __init__(self):
        super().__init__()
        self.p_old = Parameter(Tensor(np.zeros((NB, BS), dtype=np.int8)), name="pa")
        self.all_blocks = Parameter(
            Tensor(np.random.randn(NB, BS).astype(np.float16)), name="ab")
        self.indices = Tensor(np.arange(K, dtype=np.int32))

    def construct(self):
        # Phase E: quantize selected blocks
        sel = ops.Gather()(self.all_blocks, self.indices, 0)
        sfp32 = ops.Cast()(sel, ms.float32)
        amax = ops.ReduceMax()(ops.Abs()(sfp32), 1)
        scale = ops.Div()(amax, Tensor(127.0, ms.float32))
        q = ops.Cast()(ops.clip_by_value(ops.Round()(
            ops.Div()(sfp32, ops.Reshape()(scale, (K, 1)))),
            Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
        # Phase F: full quant + Assign
        fp32 = ops.Cast()(self.all_blocks, ms.float32)
        amax_f = ops.ReduceMax()(ops.Abs()(fp32), 1)
        q_all = ops.Cast()(ops.clip_by_value(ops.Round()(
            ops.Div()(fp32, ops.Reshape()(ops.Div()(amax_f, Tensor(127.0, ms.float32)), (NB, 1)))),
            Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
        return ops.Assign()(self.p_old, q_all)


class ScatterCell(nn.Cell):
    """Proposed: ScatterUpdate only top-K rows."""
    def __init__(self):
        super().__init__()
        self.p_old = Parameter(Tensor(np.zeros((NB, BS), dtype=np.int8)), name="ps")
        self.all_blocks = Parameter(
            Tensor(np.random.randn(NB, BS).astype(np.float16)), name="abs")
        self.indices = Tensor(np.arange(K, dtype=np.int32))

    def construct(self):
        sel = ops.Gather()(self.all_blocks, self.indices, 0)
        sfp32 = ops.Cast()(sel, ms.float32)
        amax = ops.ReduceMax()(ops.Abs()(sfp32), 1)
        scale = ops.Div()(amax, Tensor(127.0, ms.float32))
        q = ops.Cast()(ops.clip_by_value(ops.Round()(
            ops.Div()(sfp32, ops.Reshape()(scale, (K, 1)))),
            Tensor(-128, ms.float32), Tensor(127, ms.float32)), ms.int8)
        # Phase F: ScatterUpdate (no full quant needed)
        return ops.ScatterUpdate()(self.p_old, self.indices, q)


def run_bench(device_id=1):
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)

    print(f"MS {ms.__version__}  |  NB={NB} BS={BS} K={K}")
    results = {}

    for name, CellCls in [("Full-Assign", FullAssignCell), ("ScatterUpdate", ScatterCell)]:
        cell = CellCls()
        # Compile
        _ = cell().asnumpy()
        # Time 5 steps
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = cell().asnumpy()
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times[1:]) * 1000  # skip first (cold)
        std_ms = np.std(times[1:]) * 1000
        results[name] = {"mean_ms": mean_ms, "std_ms": std_ms, "all_ms": [t*1000 for t in times]}
        print(f"  {name:20s}: {mean_ms:8.1f}ms ± {std_ms:.1f}ms  (raw: {[f'{t*1000:.0f}' for t in times]})")

    if results["ScatterUpdate"]["mean_ms"] < results["Full-Assign"]["mean_ms"]:
        ratio = results["Full-Assign"]["mean_ms"] / max(results["ScatterUpdate"]["mean_ms"], 0.001)
        print(f"  ==> ScatterUpdate is {ratio:.1f}x FASTER than Full-Assign")
    else:
        ratio = results["ScatterUpdate"]["mean_ms"] / max(results["Full-Assign"]["mean_ms"], 0.001)
        print(f"  ==> ScatterUpdate is {ratio:.1f}x SLOWER than Full-Assign")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    args = parser.parse_args()
    run_bench(args.device_id)

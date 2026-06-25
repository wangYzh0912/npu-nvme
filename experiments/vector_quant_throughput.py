#!/usr/bin/env python3
"""
Vector Engine Quantization Throughput Benchmark — Q1 (pure Cast), Q2 (full pipeline), Q3 (GPT scale).

Output: experiments/output/vector_quant_bench.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /home/user7/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/vector_quant_throughput.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, gc, argparse
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vector_quant_bench.json")


def bench_op(op_name, sizes_mb, fn_factory, device_id=1):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)
    results = {}
    for mb in sizes_mb:
        elems = int(mb * 1024 * 1024 / 2)  # FP16 = 2 bytes/elem
        x = Tensor(np.random.randn(elems).astype(np.float16))
        cell = fn_factory()
        # warmup
        for _ in range(3): _ = cell(x)
        times = []
        for _ in range(15):
            t0 = time.perf_counter(); _ = cell(x); times.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(times[3:]))
        p99_ms = float(np.percentile(times[3:], 99))
        label = f"{mb}MB" if mb < 1024 else f"{mb/1024:.2f}GB"
        results[label] = {
            "elems": elems, "avg_ms": round(avg_ms, 3), "p99_ms": round(p99_ms, 3),
            "throughput_GB_s": round(mb / 1000.0 / (avg_ms / 1000.0), 2)
        }
        print(f"  {op_name} {label:>10s}: avg={avg_ms:.2f}ms, p99={p99_ms:.2f}ms, thr={mb/1000/avg_ms*1000:.2f} GB/s")
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shared sizes
    sizes_mb = [1, 10, 50, 100, 500, 1024, 2048]

    # Q1: Pure Cast FP16->INT8
    print("=== Q1: Pure Cast (FP16->INT8) ===")
    class CastOnly(nn.Cell):
        def construct(self, x):
            return ops.Cast()(x, ms.int8)
    q1 = bench_op("Q1_cast", sizes_mb, CastOnly, args.device_id)

    # Q2: Full quant pipeline (Mul + Round + Clip + Cast)
    print("\n=== Q2: Full Quant (Mul*scale + Round + Clip + Cast) ===")
    class FullQuant(nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = ms.Parameter(Tensor([127.0], dtype=ms.float16), requires_grad=False)
        def construct(self, x):
            s = ops.Cast()(x, ms.float16) * self.scale
            r = ops.Rint()(s)
            c = ops.clip_by_value(r, -128, 127)
            return ops.Cast()(c, ms.int8)
    q2 = bench_op("Q2_full_quant", sizes_mb, FullQuant, args.device_id)

    # Q3: GPT-scale
    print("\n=== Q3: GPT-Scale ===")
    gpt_sizes_mb = [512, 1024, 2048, 3130]
    q3 = bench_op("Q3_gpt_scale", gpt_sizes_mb, CastOnly, args.device_id)

    result = {"Q1_cast_only": q1, "Q2_full_quant": q2, "Q3_gpt_scale": q3}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
